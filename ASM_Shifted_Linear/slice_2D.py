from firedrake import *
import numpy as np
from firedrake.output import VTKFile
from matplotlib import pyplot as plt
from petsc4py import PETSc
print = PETSc.Sys.Print


# TODO: How to make it a intrinsic variable:
dtc = Constant(100.)
k = as_vector([0,0,1])
Omega = 7.292e-5
theta = pi / 3
omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])
N=1.0e-2

class HDivHelmholtzSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        W = u.function_space()
        uxz, uy, b = split(u) # TODO: This split is not working. now its [ux,uz], uy, b.
        wxz, wy, q = split(v)
        velo = uxz + uy * as_vector([0., 1., 0.])
        w = wxz + wy * as_vector([0., 1., 0.])
        Jp = (inner(velo, w) + dtc / 2 * div(velo) * div(w) + dtc * inner(cross(omega, velo), w))*dx
        Jp -= dtc/2 * inner(w, k) * b * dx
        Jp += inner(b, q)*dx
        Jp += dtc / 2 * N**2 * q * inner(k, velo) * dx
        #  Boundary conditions
        bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
        bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
        bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)


def solve_LB_Slice(nx=10, length=1.0, height=1e-3, nlayers=20, delta=Constant(1.0), dt=100., tmax=1000., xtest=False, ztest=False, artest=False):
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    m = PeriodicIntervalMesh(nx, length,distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(m, refinement_levels=2)
    hierarchy = ExtrudedMeshHierarchy(mh, height, base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
    mesh_old = hierarchy[-1]
    meshes = []
    for m in hierarchy:
        x, z = SpatialCoordinate(m)
        coord_fs = VectorFunctionSpace(m, "CG", 1, dim=3)
        new_coord = assemble(interpolate(as_vector([x, 0, z]), coord_fs))
        new_mesh = Mesh(new_coord)
        new_mesh.init_cell_orientations(as_vector([0,1,0]))
        meshes.append(new_mesh)

    new_mh = HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)
    mesh = new_mh[-1] # Creating a new mesh. # TODO: I need to rebuild MH another time.
    # mesh.init_cell_orientations(as_vector([0,1,0]))
    x, y, z = SpatialCoordinate(mesh) # Create a new mesh coordinate. TODO: need to check
    finest_mesh_name = "finest"
    mesh.name = finest_mesh_name
    AR = height / length
    delta_x = length / nx
    delta_z = height / nlayers
    # Set up the function space using TensorProductElement to have flexibility in the element degrees.
    CG_1 = FiniteElement("CG", interval, 1)
    DG_0 = FiniteElement("DG", interval, 0)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    RT_e = RT_horiz + RT_vert
    RT = FunctionSpace(mesh, RT_e) # RT1
    DG = FunctionSpace(mesh, 'DG', 0)
    Vy_space = FunctionSpace(mesh, 'DG', 0)
    # horizontal base spaces -- 2D
    S2 = FiniteElement("DG", interval, 0) # DG1 in 1D
    # vertical base spaces
    T0 = FiniteElement("CG", interval, 1) # CG2
    # Attempt to build the 2D element.
    Vv_elt = TensorProductElement(S2, T0) # DG horizontal and CG vertical
    Vb = FunctionSpace(mesh, Vv_elt) # Buoyancy space W theta
    W = RT * Vy_space * Vb * DG
    Un = Function(W)
    Unp1 = Function(W)
    un_slice, uny, bn, pn = split(Un)
    unp1_slice, unp1y, bnp1, pnp1 = split(Unp1)
    w_slice, wy, q, phi = TestFunctions(W)

    n = FacetNormal(mesh)
    k = as_vector([0., 0., 1.])
    Omega = 7.292e-5
    theta = pi / 3
    omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])
    dtc = Constant(dt)

    bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
    bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
    # bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
    # bcs = [bc1, bc2, bc3]
    bcs = [bc1, bc2]

    # Initial condition
    xc = Constant(length/2)
    yc = Constant(length/2)
    a = Constant(5000)
    U = Constant(0.)
    # TODO: this is a 4 components function.
    unic_slice, unyic, bnic, pnic = Un.subfunctions
    unp1ic_slice, unp1yic, bnp1ic, pnp1ic = Unp1.subfunctions
    unic_slice.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
    unp1ic_slice.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
    unyic.project(Constant(0.))
    unp1yic.project(Constant(0.))
    bnic.project(sin(pi*z/height)/(1+((x-xc)**2)/a**2) + N**2 * z)
    bnp1ic.project(sin(pi*z/height)/(1+((x-xc)**2)/a**2) + N**2 * z)
    pnic.project(0.5 * N**2 * z**2)
    pnp1ic.project(0.5 * N**2 * z**2)
    DG0 = FunctionSpace(mesh, 'DG', 0)
    One = Function(DG0).assign(1.0)
    area = assemble(One * dx)
    pnic_int = assemble(pn * dx)
    pnic.project(pn - pnic_int / area)
    pnp1ic_int = assemble(pnp1 * dx)
    pnp1ic.project(pnp1ic - pnp1ic_int/area)

    print('==================================================')
    print('Initial condition has been interpolated')
    name = 'ic'
    file_lb = VTKFile(name+'.pvd')
    u0, u0y, b0, P0 = Un.subfunctions
    file_lb.write(u0, u0y, b0, P0)
    print("Save initial condition.")

    # TODO: slice equation
    # TODO: Suggestion: divide into tendency term and the rest.

    # Equations
    def u_eqn(un, unp1, unph, pnp1, bnph):
        return (
            inner(w, (unp1 - un)) * dx 
            # + dtc * inner(w, 2 * cross(omega, unph)) * dx
            - dtc * div(w) * pnp1 * dx -
            dtc * inner(w, k) * bnph * dx
        )

    def b_eqn(bn, bnp1, unph):
        return (
            q * (bnp1 - bn) * dx +
            dtc * N**2 * q * inner(k, unph) * dx
        )

    def p_eqn(unph):
        return (
            phi * div(unph) * dx
        )

    un = un_slice #+ uny * as_vector([0., 1., 0.])
    unp1 = unp1_slice #+ unp1y * as_vector([0., 1., 0.])
    unph = 0.5 * (un+unp1)
    bnph = 0.5 * (bn+bnp1)
    pnph = 0.5 * (pn+pnp1)
    w = w_slice #+ wy * as_vector([0., 1., 0.])
    eqn = p_eqn(unph) + b_eqn(bn, bnp1, unph) + u_eqn(un, unp1, unph, pnp1, bnph) + (unp1y - uny) * wy * dx
    shift = eqn + delta * pnp1 * phi * dx
    Jp = derivative(shift, Unp1)

    v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), W.sub(2), v_basis])

    # Parameters
    # TODO: Check if the Schur Complement is correct. The direct solve should converge in one iteration.
    helmholtz_schur_pc_params = {
                'ksp_monitor': None,
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
            }
    # helmholtz_schur_pc_params = {
    #     'ksp_type': 'preonly',
    #     'ksp_max_its': 30,
    #     'pc_type': 'mg',
    #     'pc_mg_type': 'full',
    #     'pc_mg_cycle_type':'v',
    #     'mg_levels': {
    #         # 'ksp_type': 'gmres',
    #         # 'ksp_type':'richardson',
    #         'ksp_type': 'chebyshev',
    #         # 'ksp_richardson_scale': 0.2,
    #         'ksp_max_it': 1,
    #         # 'ksp_monitor':None,
    #         "pc_type": "python",
    #         "pc_python_type": "firedrake.ASMStarPC", # TODO: shall we use AssembledPC?
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

    # params = {
    #     # 'mat_type': 'aij',
    #     'ksp_type': 'gmres',
    #     'snes_type':'ksponly',
    #     'ksp_atol': 0,
    #     'ksp_rtol': 1e-9,
    #     # 'ksp_view': None,
    #     'snes_monitor': None,
    #     # 'ksp_monitor': None,
    #     'ksp_monitor_true_residual': None,
    #     'pc_type': 'fieldsplit',
    #     'pc_fieldsplit_type': 'schur',
    #     'pc_fieldsplit_schur_fact_type': 'full',
    #     'pc_fieldsplit_0_fields': '3',
    #     'pc_fieldsplit_1_fields': '0,1,2',
    #     'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
    #         'ksp_type': 'preonly',
    #         'pc_type': 'bjacobi',
    #         'sub_pc_type': 'ilu',
    #         # 'pc_factor_mat_solver_type': 'mumps',
    #     },
    #     'fieldsplit_1': {
    #         'ksp_type': 'preonly',
    #         'ksp_monitor': None,

    #         'pc_type': 'lu',
    #         'mat_type': 'aij',
    #         'pc_factor_mat_solver_type': 'mumps'
    #         # 'pc_type': 'python',
    #         # 'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
    #         # 'helmholtzschurpc': helmholtz_schur_pc_params,
    #         },
    # }
    # params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'} #TODO: This is working.
    params = {'snes_type':'ksponly','ksp_type': 'gmres', 'snes_monitor':None, 'ksp_monitor':None, 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'} # TODO: This is also working.

    nprob = NonlinearVariationalProblem(eqn, Unp1, bcs=bcs, Jp=Jp)
    nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)

    # Time Stepping
    name = 'lb_slice_imp_ASM'
    file_lb = VTKFile(name+'.pvd')
    un, uny, bn, pn = Un.subfunctions
    file_lb.write(un, uny, bn, pn)
    Unp1.assign(Un)
    t = 0.0
    dumpt = dt
    tdump = 0.
    dtc.assign(dt)
    i = 0
    while t < tmax - 0.5 * dt:
        print("The solver is currently solving for time", t)
        t += dt
        tdump += dt
        i += 1
        nsolver.solve()
        Un.assign(Unp1)
        if tdump > dumpt - dt*0.5:
            file_lb.write(un, uny, bn, pn)
            tdump -= dumpt

if __name__ == "__main__":
    solve_LB_Slice(nx=10, length=3.0e5, height=1e4, nlayers=20, delta=Constant(1.0), dt=100., tmax=1000., xtest=False, ztest=False, artest=False)
    