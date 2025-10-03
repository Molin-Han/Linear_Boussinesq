from firedrake import *
import numpy as np
from firedrake.output import VTKFile
from matplotlib import pyplot as plt
from petsc4py import PETSc
print = PETSc.Sys.Print

dtc = Constant(1.0)
delta_pc = Constant(1.0)

class HDivHelmholtzSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        k = as_vector([0., 0., 1.])
        W = u.function_space()
        uxz, uy, b = split(u)
        wxz, wy, q = split(v)
        velo = uxz + uy * as_vector([0., 1., 0.])
        w = wxz + wy * as_vector([0., 1., 0.])
        Jp = inner(velo, w) * dx
        Jp += dtc / delta_pc * div(velo) * div(w) * dx
        Jp -= dtc * inner(w, k) * b * dx
        Jp += q * b * dx + dtc * q * inner(k, velo) * dx
        # Boundary conditions
        bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
        bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
        bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)


def solve_LB_Slice(nx=10, length=1.0, height=1e-3, nlayers=20, delta=Constant(1.0), deltat=Constant(1.0)):
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    # m = UnitIntervalMesh(nx, distribution_parameters=distribution_parameters)
    m = PeriodicIntervalMesh(nx, length,distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(m, refinement_levels=2)
    layers = [nlayers] * (2+1)
    hierarchy = ExtrudedMeshHierarchy(mh, height, layers=layers, extrusion_type='uniform')
    mesh_old = hierarchy[-1]
    meshes = []
    for m in hierarchy:
        x, z = SpatialCoordinate(m)
        coord_fs = VectorFunctionSpace(m, "CG", 1, dim=3)
        new_coord = assemble(interpolate(as_vector([x, 0, z]), coord_fs))
        new_mesh = Mesh(new_coord)
        new_mesh.init_cell_orientations(as_vector([0,-1,0]))
        meshes.append(new_mesh)

    new_mh = HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)
    mesh = new_mh[-1] # Creating a new mesh.
    x, y, z = SpatialCoordinate(mesh) # Create a new mesh coordinate.
    finest_mesh_name = "finest"
    mesh.name = finest_mesh_name
    AR = height / length
    delta_x = length / nx
    delta_z = height / nlayers
    # Set up the function space using TensorProductElement to have flexibility in the element degrees.
    deg = 0 # TODO: The degree of the element
    CG_1 = FiniteElement("CG", interval, deg+1)
    DG_0 = FiniteElement("DG", interval, deg)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    RT_e = RT_horiz + RT_vert
    RT = FunctionSpace(mesh, RT_e) # RT1
    DG = FunctionSpace(mesh, 'DQ', deg)
    Vy_space = FunctionSpace(mesh, 'DQ', deg)
    # horizontal base spaces -- 2D
    S2 = FiniteElement("DG", interval, deg) # DG1 in 1D
    # vertical base spaces
    T0 = FiniteElement("CG", interval, deg+1) # CG2
    # Attempt to build the 2D element.
    Vv_elt = TensorProductElement(S2, T0) # DG horizontal and CG vertical
    Vb = FunctionSpace(mesh, Vv_elt)
    W = RT * Vy_space  * Vb * DG
    U = Function(W)
    u_slice, uy, b, p = split(U)
    w_slice, wy, q, phi = TestFunctions(W)

    n = FacetNormal(mesh)
    k = as_vector([0., 0., 1.])

    bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
    bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
    bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
    bcs = [bc1, bc2, bc3]

    # This is a 4 components function.
    VDG = VectorFunctionSpace(mesh, 'DQ', deg+1)
    DGf = FunctionSpace(mesh, 'DG', deg+1)
    pcg = PCG64(seed=1234567)
    rg = Generator(pcg)
    f_xz = rg.normal(VDG, 1.0, 2.0)
    f_y = rg.normal(DGf, 1.0, 2.0)
    f = f_xz + f_y * as_vector([0.,1.,0.])
    g = rg.normal(DGf, 1.0, 2.0)

    print('==================================================')
    print('Initial condition has been interpolated')

    # Equations
    def u_eqn(u, p, b, w):
        return (
            inner(w, u) * dx
            - deltat * div(w) * p * dx 
            - deltat * inner(w, k) * b * dx 
            - inner(w, f) * dx
        )

    def b_eqn(u, b, q):
        return (
            q * b * dx 
            + deltat * q * inner(k, u) * dx 
            - g * q * dx
        )

    def p_eqn(u, phi):
        return (
            phi * div(u) * dx
        )

    u = u_slice + uy * as_vector([0., 1., 0.])
    w = w_slice + wy * as_vector([0., 1., 0.])
    eqn = p_eqn(u, phi)
    eqn += u_eqn(u, p, b, w) #+ (unp1y - uny) * wy * dx # Modification to separate the y equation here.
    eqn += b_eqn(u, b, q)
    shift = eqn + delta * p * phi * dx
    Jp = derivative(shift, U)

    v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), W.sub(2), v_basis])

    # Parameters
    # TODO: Check if the Schur Complement is correct. The direct solve should converge in one iteration.
    helmholtz_schur_pc_params = {
                # 'ksp_type':'preonly',
                # 'pc_type': 'lu',
                # 'pc_factor_mat_solver_type': 'mumps',
                # 'ksp_monitor': None,
                # 'ksp_pc_type': 'lu',
                'pc_type':'ksp',
                'ksp_ksp_type': 'preonly', # preonly
                'ksp_pc_type':'lu',
                # 'ksp_ksp_monitor': None,
            }
    # helmholtz_schur_pc_params = {
    #     'ksp_type': 'preonly',
    #     'ksp_max_its': 30,
    #     'pc_type': 'mg',
    #     'pc_mg_type': 'full',
    #     'pc_mg_cycle_type':'v',
    #     'mg_levels': { 
    #         # 'ksp_type': 'gmres',
    #         'ksp_type':'richardson',
    #         # 'ksp_type': 'chebyshev',
    #         'ksp_richardson_scale': 0.2,
    #         'ksp_max_it': 3,
    #         # 'ksp_monitor':None,
    #         "pc_type": "python",
    #         "pc_python_type": "firedrake.ASMStarPC",
    #         "pc_star_construct_dim": 0,
    #         "pc_star_sub_sub_pc_type": "lu",
    #         # "pc_star_sub_sub_pc_type": "svd",
    #         # "pc_star_sub_sub_pc_svd_monitor": None,
    #     },
    #     'mg_coarse': {
    #         'ksp_type': 'preonly',
    #         'pc_type': 'lu',
    #     },
    # }

    params = {
        'mat_type': 'aij',
        # 'ksp_type': 'fgmres',
        'ksp_type': 'gmres',
        # 'ksp_type':'preonly',
        'snes_type':'ksponly',
        'ksp_atol': 0,
        'ksp_rtol': 1e-9,
        'ksp_view': ':SLBkspviewoutput.txt',
        'snes_monitor': None,
        # 'ksp_monitor': None,
        'ksp_monitor_true_residual': None,
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        'pc_fieldsplit_0_fields': '3',
        'pc_fieldsplit_1_fields': '0,1,2',
        'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu',
        },
        'fieldsplit_1': {
            'ksp_type':'gmres',
            'ksp_max_it':'30',
            # 'ksp_type':'preonly', #TODO: This will cause the first KSP residual increase!!!! using gmres will be fine ???
            'ksp_monitor_true_residual': None,
            'pc_type': 'python',
            'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
            'helmholtzschurpc': helmholtz_schur_pc_params,
            },
    }

    nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs, Jp=Jp)
    nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)

    nsolver.solve()

if __name__ == "__main__":
    delta = delta_pc
    deltat = dtc
    length = 3.0e5
    height = 1.0e2
    solve_LB_Slice(nx=20, length=length, height=height, nlayers=20, delta=delta, deltat=deltat)
    