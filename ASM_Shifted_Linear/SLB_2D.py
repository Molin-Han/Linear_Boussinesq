from firedrake import *
import numpy as np
from firedrake.output import VTKFile
from matplotlib import pyplot as plt
from petsc4py import PETSc

print = PETSc.Sys.Print
deltat = Constant(1.0)
delta = Constant(1.0)

class HDivSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        k = as_vector([0,1])
        W = u.function_space()
        velo, b = split(u)
        w, q = split(v)
        Jp = (inner(velo, w) + deltat / delta * div(velo) * div(w))*dx # TODO: The delta shifting parameter enters here.
        Jp -= deltat * inner(w, k) * b * dx
        Jp += q * b *dx + deltat * q * inner(k, velo) * dx
        #  Boundary conditions
        bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)

def solve_SLB(nx=10, length=1.0, height=1e-3, nlayers=20, deltat=Constant(1.0), delta=Constant(1.0)):
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    m = PeriodicIntervalMesh(nx, length,distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(m, refinement_levels=2)
    layers = [nlayers] * (2+1)
    hierarchy = ExtrudedMeshHierarchy(mh, height, layers=layers, extrusion_type='uniform')
    mesh = hierarchy[-1]
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
    RT = FunctionSpace(mesh, RT_e)
    DG = FunctionSpace(mesh, 'DG', 0)
    # horizontal base spaces -- 2D
    S2 = FiniteElement("DG", interval, 0) # DG1 in 1D
    # vertical base spaces
    T0 = FiniteElement("CG", interval, 1) # CG2
    # Attempt to build the 2D element.
    Vv_elt = TensorProductElement(S2, T0) # DG horizontal and CG vertical
    Vb = FunctionSpace(mesh, Vv_elt) # Buoyancy space W theta
    W = RT * Vb * DG
    x, z = SpatialCoordinate(mesh)

    # Set up the RHS.
    U = Function(W)
    u, b, p = split(U)
    w, q, phi = TestFunctions(W)

    n = FacetNormal(mesh)
    k = as_vector([0,1])

    VDG = VectorFunctionSpace(mesh, 'DQ', 2)
    DG = FunctionSpace(mesh, 'DQ', 1)
    pcg = PCG64(seed=1234567)
    rg = Generator(pcg)
    # f = rg.normal(VDG, 1.0, 2.0)
    # g = rg.normal(DG, 1.0, 2.0)
    f = Function(RT).project(as_vector([sin(2 * pi *x), sin(2* pi *z/height)]))
    g = Function(DG).interpolate(sin(2*pi*x) + sin(2*pi*z))

    # One = Function(DG).assign(1.0)
    # area = assemble(One*dx)
    # f_int = assemble(inner(f, w)*dx)
    # print('the integral of f is ', norm(f_int.riesz_representation()))

    bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
    bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
    bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")

    bcs = [bc1, bc2, bc3]

    # Equations
    def u_eqn(u, p, b):
        return (
            inner(w, u) * dx
            - deltat * div(w) * p * dx 
            - deltat * inner(w, k) * b * dx 
            - inner(w, f) * dx
        )

    def b_eqn(u, b):
        return (
            q * b * dx 
            + deltat * q * inner(k, u) * dx 
            - g * q * dx
        )

    def p_eqn(u):
        return (
            phi * div(u) * dx
        )

    eqn = u_eqn(u,p,b) + b_eqn(u,b) + p_eqn(u)
    shift = eqn + delta * p * phi * dx
    Jp = derivative(shift, U)

    # pressure nullspace
    v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])

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
    #         'ksp_richardson_scale': 0.2, #TODO: tune.
    #         'ksp_max_it': 2,
    #         # 'ksp_monitor':None,
    #         "pc_type": "python",
    #         "pc_python_type": "firedrake.ASMStarPC", # TODO: shall we use AssembledPC?
    #         "pc_star_construct_dim": 0,
    #         "pc_star_sub_sub_pc_type": "lu",
    #         # "pc_star_sub_sub_pc_type": "svd",
    #         # "pc_star_sub_sub_pc_svd_monitor": None,

    #         # "pc_python_type": "firedrake.ASMVankaPC",
    #         # "pc_vanka_construct_dim": 0,
    #         # # "pc_vanka_sub_sub_pc_type": "lu",
    #         # "pc_vanka_sub_sub_pc_type": "svd",
    #         # "pc_vanka_sub_sub_pc_svd_monitor": None,
    #     },
    #     'mg_coarse': {
    #         'ksp_type': 'preonly',
    #         'pc_type': 'lu',
    #     },
    # }

    # # helmholtz_schur_pc_params = {
    # #             # 'ksp_monitor': None,
    # #             'ksp_type': 'preonly',
    # #             'pc_type': 'lu',
    # #             'pc_factor_mat_solver_type': 'mumps',
    # #         }

    # params_schur = {
    #     # 'mat_type': 'aij',
    #     'ksp_type': 'gmres',
    #     'snes_type':'ksponly',
    #     'ksp_atol': 0,
    #     'ksp_rtol': 1e-8,
    #     # 'ksp_view': None,
    #     'snes_monitor': None,
    #     # 'ksp_monitor': None,
    #     'ksp_monitor_true_residual': None,
    #     'pc_type': 'fieldsplit',
    #     'pc_fieldsplit_type': 'schur',
    #     'pc_fieldsplit_schur_fact_type': 'full',
    #     'pc_fieldsplit_0_fields': '2',
    #     'pc_fieldsplit_1_fields': '0,1',
    #     'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
    #         'ksp_type': 'preonly',
    #         'pc_type': 'bjacobi',
    #         'sub_pc_type': 'ilu',
    #         # 'pc_factor_mat_solver_type': 'mumps',
    #     },
    #     'fieldsplit_1': {
    #         'ksp_type': 'preonly',
    #         # 'ksp_monitor': None,
    #         'pc_type': 'python',
    #         'pc_python_type': __name__ + '.HDivSchurPC',
    #         'helmholtzschurpc': helmholtz_schur_pc_params,
    #         },
    # }
    params_schur = {
        'mat_type': 'aij',
        'snes_type':'ksponly', 
        'ksp_type': 'preonly', 
        'ksp_view': ':SLB2D.txt',
        'snes_monitor':None, 
        'ksp_monitor':None, 
        'ksp_error_if_not_converged': None,
        'pc_type':'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        'pc_fieldsplit_schur_precondition': 'full',
        'pc_fieldsplit_0_fields': '0,1',
        'pc_fieldsplit_1_fields': '2',
        'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
            'ksp_type': 'preonly',
            'ksp_error_if_not_converged': None,
            'pc_type': 'lu',
            # 'sub_pc_type': 'ilu',
            'pc_factor_mat_solver_type': 'mumps',
        },
        'fieldsplit_1': {
            'ksp_type': 'preonly',
            'ksp_error_if_not_converged': None,
            # 'ksp_monitor': None,
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            },
        }
    # nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs)
    nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs, Jp=Jp)
    nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params_schur)

    nsolver.solve()
    name = 'SLB_2D'
    file_lb = VTKFile(name+'.pvd')
    un, bn, pn = U.subfunctions
    un.rename("in-plane-vel")
    bn.rename("buoyancy")
    pn.rename("pressure")
    f.rename("f_RHS")
    g.rename("g_RHS")
    file_lb.write(un, bn, pn, f, g)

if __name__ == "__main__":
    nx=40
    length=1.0
    height=1e-1
    nlayers=20
    dt = deltat
    shift = delta
    solve_SLB(nx=nx, length=length, height=height, nlayers=nlayers, deltat=dt, delta=shift)
    
