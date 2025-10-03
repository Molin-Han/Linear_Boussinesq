from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from petsc4py import PETSc

print = PETSc.Sys.Print

class HDivHelmholtzSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        W = u.function_space()
        Jp = (inner(u, v) + div(v)*div(u))*dx
        #  Boundary conditions
        bc1 = DirichletBC(W, as_vector([0., 0.]), "top")
        bc2 = DirichletBC(W, as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(W, as_vector([0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)


def solve_Poisson(height=pi/40, nlayers=20, horiz_num=80, deltat=1.0, delta=1.0, refinement=3, monolithic=False, monitor=False, xtest=False, ztest=False, artest=False, ttest=False):
    AR = height
    delta_x = 1 / horiz_num
    delta_z = height / nlayers
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    # Extruded Mesh
    m = UnitIntervalMesh(horiz_num, name='interval', distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(m, refinement_levels=refinement)
    # hierarchy = ExtrudedMeshHierarchy(mh, height,base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
    layers = [nlayers] * (refinement+1)
    # layers = [1, nlayers, nlayers]
    hierarchy = ExtrudedMeshHierarchy(mh, height, layers=layers, extrusion_type='uniform')
    mesh = hierarchy[-1] # TODO: This is crucial.
    finest_mesh_name = "finest"
    mesh.name = finest_mesh_name
    # Mixed Finite Element Space
    deg = 0
    CG_1 = FiniteElement("CG", interval, deg+1)
    DG_0 = FiniteElement("DG", interval, deg)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    RT_e = RT_horiz + RT_vert
    RT = FunctionSpace(mesh, RT_e)
    DG = FunctionSpace(mesh, 'DG', deg)
    W = RT * DG # RT1 DG0 mixed finite element space
    VDG = VectorFunctionSpace(mesh, 'DQ', deg+1) # The VDG1 space for the corresponding velocity space.
    # Test Functions
    v, q = TestFunctions(W)

    # Solution Functions
    sol = Function(W) # solution in mixed space
    u, p = split(sol)

    x, y = SpatialCoordinate(mesh)

    pcg = PCG64(seed=123456789)
    rg = Generator(pcg)
    f = rg.normal(VDG, 1.0, 2.0)
    # f = rg.normal(DG, 1.0, 2.0)
    # One = Function(DG).assign(1.0)
    # area = assemble(One*dx)
    # f_int = assemble(f*dx)
    # f.interpolate(f - f_int/area)

    # F = (inner(u, v) - deltat * div(v) * p + div(u)*q)*dx - inner(f,q) * dx
    F = (inner(u, v) - deltat * div(v) * p + div(u)*q)*dx - deltat * inner(f,v) * dx
    shift = F + delta * p * q * dx
    Jp = derivative(shift, sol)

    # Boundary conditions
    bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
    bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
    bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
    bcs = [bc1, bc2, bc3]

    v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD) #pressure field nullspace
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
    trans_null = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    trans_nullspace = MixedVectorSpaceBasis(W, [W.sub(0), trans_null])

    if not monolithic:
        helmholtz_schur_pc_params = {'ksp_type':'preonly', 'pc_type':'lu'}
        # helmholtz_schur_pc_params = {
        #     'ksp_type': 'preonly',
        #     'ksp_max_its': 30,
        #     # 'ksp_monitor':None,
        #     # 'ksp_monitor_true_residual': None,
        #     'pc_type': 'mg',
        #     'pc_mg_type': 'full',
        #     'pc_mg_cycle_type':'v',
        #     'mg_levels': {
        #         # 'ksp_type': 'gmres',
        #         'ksp_type':'richardson',
        #         'ksp_richardson_scale': 1/4,
        #         'ksp_max_it': 1,
        #         # 'ksp_monitor':None,
        #         # 'ksp_monitor_true_residual':None,
        #         "pc_type": "python",
        #         "pc_python_type": "firedrake.ASMStarPC",
        #         "pc_star_construct_dim": 0,
        #         "pc_star_sub_sub_pc_type": "lu",
        #         # "pc_star_sub_sub_pc_type": "svd",
        #         # "pc_star_sub_sub_pc_svd_monitor": None,

        #         # "pc_python_type": "firedrake.ASMVankaPC",
        #         # "pc_vanka_construct_dim": 0,
        #         # "pc_vanka_sub_sub_pc_type": "lu",
        #     },
        #     'mg_coarse': {
        #         'ksp_type': 'preonly',
        #         'pc_type': 'lu',
        #     },
        # }
        params = {
            # 'mat_type': 'aij',
            'ksp_type': 'gmres',
            'snes_type':'ksponly',
            'ksp_atol': 0,
            'ksp_rtol': 1e-8,
            # 'ksp_view': None,
            'snes_monitor': None,
            # 'ksp_monitor': None,
            'ksp_monitor_true_residual': None,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_0_fields': '1',
            'pc_fieldsplit_1_fields': '0',
            'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu',
                # 'pc_factor_mat_solver_type': 'mumps',
            },
            'fieldsplit_1': {
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'ksp_monitor':None,
                'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
                'helmholtzschurpc': helmholtz_schur_pc_params,
            }
        }
        prob_w = NonlinearVariationalProblem(F, sol, bcs=bcs, Jp=Jp)
        solver_w = NonlinearVariationalSolver(prob_w, nullspace=nullspace, transpose_nullspace=trans_nullspace, solver_parameters=params, options_prefix='mixed_nonlinear')


    # if monolithic:
    #     params = {
    #         # 'mat_type': 'matfree',
    #         'ksp_type': 'gmres',
    #         'snes_monitor': None,
    #         # 'snes_type':'ksponly',
    #         'ksp_monitor': None,
    #         # "ksp_monitor_true_residual": None,
    #         'pc_type': 'mg',
    #         'pc_mg_type': 'full',
    #         'pc_mg_cycle_type':'v',
    #         "ksp_converged_reason": None,
    #         "snes_converged_reason": None,
    #         'mg_levels': {
    #             'ksp_type': 'richardson',
    #             # "ksp_monitor_true_residual": None,
    #             # "ksp_view": None,
    #             "ksp_atol": 1e-50,
    #             "ksp_rtol": 1e-10,
    #             'ksp_max_it': 1,
    #             "pc_type": "python",
    #             "pc_python_type": "firedrake.ASMVankaPC",
    #             "pc_vanka_construct_dim": 0,
    #             "pc_vanka_sub_sub_pc_type": "lu",
    #             # "pc_vanka_sub_sub_pc_type": "svd",
    #             # "pc_vanka_sub_sub_pc_svd_monitor": None,
    #             # 'pc_type': 'python',
    #             # 'pc_python_type': 'firedrake.AssembledPC',
    #             # 'assembled_pc_type': 'python',
    #             # 'assembled_pc_python_type': 'firedrake.ASMVankaPC',
    #             # 'assembled_pc_vanka_construct_dim': 0,
    #             # 'assembled_pc_vanka_sub_sub_pc_type': 'lu',
    #             # 'pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
    #         },
    #         'mg_coarse': {
    #             'ksp_type': 'preonly',
    #             'pc_type': 'lu',
    #         }
    #     }
    #     prob_w = NonlinearVariationalProblem(F, sol, bcs=bcs)
    #     solver_w = NonlinearVariationalSolver(prob_w, nullspace=nullspace, transpose_nullspace=trans_nullspace, solver_parameters=params, options_prefix='mixed_nonlinear')
    if not monitor:
        solver_w.solve()
    if monitor:
        # Checkpointing the mesh and the solution to make the plot.
        error_list = []
        sol_it = Function(W, name="sol_it")
        with CheckpointFile("sol_mesh.h5", "w") as chk:
            chk.save_mesh(mesh)

        def monitor(ksp, iteration_number, norm0):
            sol = ksp.buildSolution()
            with sol_it.dat.vec_wo as it_vec:
                sol.copy(result=it_vec) # sol_it is a firedrake function.
            if iteration_number == 0:
                with CheckpointFile(f"sol_its.h5", "w") as chk:
                    chk.save_function(sol_it, idx=iteration_number, name=f"sol_{iteration_number}")
                # print(f"saved its{iteration_number}")
            else: # Update
                with CheckpointFile(f"sol_its.h5", "a") as chk:
                    chk.save_function(sol_it, idx=iteration_number, name=f"sol_{iteration_number}")
                # print(f"saved its{iteration_number}")
        solver_w.snes.ksp.setMonitor(monitor)
        solver_w.solve()
        converged_it_num = solver_w.snes.ksp.getIterationNumber()
        with CheckpointFile("sol_mesh.h5", "r") as chk:
            mesh = chk.load_mesh(name=finest_mesh_name,distribution_parameters=distribution_parameters)
        with CheckpointFile("sol_its.h5", "r") as chk:
            sol_final = chk.load_function(mesh, f"sol_{converged_it_num}", idx=converged_it_num)
            for i in range(converged_it_num):
                sol_i = chk.load_function(mesh, f"sol_{i}", idx=i)
                err_i = norm(sol_i - sol_final) / norm(sol_final)
                error_list.append(err_i)
        print("Monitor is on and working.")
    
    if artest:
        # test for the aspect ratio
        np.savetxt(f'err_ar_{AR}.out', error_list)
    if xtest:
        # test for the different dx
        np.savetxt(f'err_dx_{delta_x}.out', error_list)
    if ztest:
        # test for the different dz
        np.savetxt(f'err_dz_{delta_z}.out', error_list)
    if ttest:
        # test for the different dt
        np.savetxt(f'err_dt_{deltat}.out', error_list)

if __name__ == "__main__":
    horiz_num = 40
    height = 1/1000
    nlayers = 40
    deltat = 1.0
    delta = deltat
    refinement = 2
    # monolithic = True
    monolithic = False
    solve_Poisson(height, nlayers, horiz_num, deltat, delta, refinement, monolithic=monolithic)
