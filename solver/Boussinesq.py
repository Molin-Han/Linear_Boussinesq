from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class Boussinesq:
    def __init__(self, N=1.0e-2, U=20., dT=600., nx=5e3, ny=1, Lx=1e3, Ly=1, height=1e4, nlayers=20, horiz_num=80, radius=2):

        # Extruded Mesh 3D
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.height = height
        self.U = U # steady velocity
        self.N = N # buoyancy frequency
        # self.m = UnitSquareMesh(self.nx, self.ny)
        self.m = PeriodicRectangleMesh(self.nx, self.ny, self.Lx, self.Ly, direction='both',quadrilateral=True)
        # Build the mesh hierarchy for the extruded mesh to construct vertically constant spaces.
        self.mh = MeshHierarchy(self.m, refinement_levels=0)
        self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='uniform')
        self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')

        # Mixed Finite Element Space
        horizontal_degree = 2
        vertical_degree = 2
        # TODO: need to check the base spaces.
        # horizontal base spaces
        S1 = FiniteElement("CG", interval, horizontal_degree) # CG2
        S2 = FiniteElement("DG", interval, horizontal_degree-1) # DG1

        S1_2d = FiniteElement("CG", quadrilateral, horizontal_degree) # CG2
        S2_2d = FiniteElement("DG", quadrilateral, horizontal_degree-1) # DG1
        # vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree) # CG2
        T1 = FiniteElement("DG", interval, vertical_degree-1) # DG1

        # Successful build of the 2D element.
        Vh_elt = TensorProductElement(S1, T1) # CG horizontal and DG vertical
        V_2h = HDivElement(Vh_elt)
        Vv_elt = TensorProductElement(S2, T0) # DG horizontal and CG vertical
        V_2v = HDivElement(Vv_elt)
        V_2d = V_2h + V_2v # quadrilateral RT element in 2D

        # Attempt to build the 3D element. #TODO: this has bug here.
        Vh_elt_3d = TensorProductElement(S1_2d, T1)
        Vh_3d = HDivElement(Vh_elt_3d)
        Vv_elt_3d = TensorProductElement(S2_2d, T0)
        Vv_3d = HDivElement(Vv_elt_3d)
        V_3d = Vh_3d + Vv_3d
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # TODO: Checked this expression. This is correct for triangular mesh.
        # P2i = FiniteElement("CG", interval, 2)
        # dP1t = FiniteElement("DG", triangle, 1)
        # dP1i = FiniteElement("DG", interval, 1)
        # RT2 = FiniteElement("RT", triangle, 2)
        # Hdiv_h = HDivElement(TensorProductElement(RT2, dP1i))
        # Hdiv_v = HDivElement(TensorProductElement(dP1t, P2i))
        # Hdiv_element = Hdiv_h + Hdiv_v
        # V = FunctionSpace(self.mesh, Hdiv_element, name="HDiv") # Velocity space RT(k)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        V = FunctionSpace(self.mesh, V_3d, name="HDiv") # Velocity space RT(k-1)
        Vb = FunctionSpace(self.mesh, Vv_elt, name="Buoyancy") # Buoyancy space
        Vp_elt = TensorProductElement(S2, T1) # DG horizontal and DG vertical
        Vp = FunctionSpace(self.mesh, Vp_elt, name="Pressure")

        self.W = V * Vp * Vb # velocity, pressure, buoyancy space
        self.x, self.y, self.z = SpatialCoordinate(self.mesh)

        # Setting up the solution variables.
        self.Un = Function(self.W)
        self.Unp1 = Function(self.W)
        self.un, self.pn, self.bn = split(self.Un)
        self.unp1, self.pnp1, self.bnp1 = split(self.Unp1)
        self.alpha, self.phi, self.gamma = TestFunctions(self.W)
        self.unph = 0.5*(self.un + self.unp1)
        self.bnph = 0.5*(self.bn + self.bnp1)
        self.pnph = 0.5*(self.pn + self.pnp1)

        self.n = FacetNormal(self.mesh)
        self.k = as_vector([0, 0, 1])
        Omega = 7.292e-5
        theta = pi / 3
        self.omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])

        self.dT = dT

        # Test Functions
        self.w, self.phi, self.q = TestFunctions(self.W)

    def build_initial_data(self):
        xc = self.Lx/2
        yc = self.Ly/2
        a = Constant(5000)
        U = Constant(self.U)
        self.un.interpolate(as_vector([U,0,0])) # TODO: need to check this.
        self.bn.interpolate(sin(pi*self.z/self.height)/(1+((self.x-xc)**2+(self.y-yc)**2)/a**2))


    def build_lu_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
    
    def build_ASM_params(self):
        self.params = {
                        'mat_type': 'matfree',
                        'ksp_type': 'gmres',
                        'snes_monitor': None,
                        # 'snes_type':'ksponly',
                        # 'ksp_monitor': None,
                        # "ksp_monitor_true_residual": None,
                        'pc_type': 'mg',
                        'pc_mg_type': 'full',
                        "ksp_converged_reason": None,
                        "snes_converged_reason": None,
                        'mg_levels': {
                                'ksp_type': 'richardson',
                                # "ksp_monitor_true_residual": None,
                                # "ksp_view": None,
                                "ksp_atol": 1e-50,
                                "ksp_rtol": 1e-10,
                                'ksp_max_it': 1,
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.AssembledPC',
                                'assembled_pc_type': 'python',
                                'assembled_pc_python_type': 'firedrake.ASMVankaPC',
                                'assembled_pc_vanka_construct_dim': 0,
                                'assembled_pc_vanka_sub_sub_pc_type': 'lu'
                                #'assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
                                },
                        'mg_coarse': {
                                'ksp_type': 'preonly',
                                'pc_type': 'lu'
                                }
                        }
    # TODO: need to build the parameters for the ASM solver.

    def build_boundary_condition(self):
        # Boundary conditions #TODO: need to check how to ensure the condition on pressure.
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "on_boundary")
        self.bcs = [bc1, bc2, bc3]

    def build_NonlinearVariationalSolver(self):
        # Simplify variable name
        un, pn, bn = self.un, self.pn, self.bn
        unp1, pnp1, bnp1 = self.unp1, self.pnp1, self.bnp1
        unph, pnph, bnph = self.unph, self.pnph, self.bnph
        w, phi, q = self.w, self.phi, self.q
        k = self.k
        dT = self.dT
        N = self.N
        omega = self.omega
        def u_eqn(w):# TODO: need to complete the velocity equation.
            return (
                w * (unp1 - un) * dx +
                dT * inner(w, 2 * outer(omega, unph)) * dx -
                dT * div(w) * pnph * dx - dT * inner(w, k) * bnph * dx
            )

        def b_eqn(q):
            return (
                q * (bnp1 - bn) * dx +
                dT * N**2 * q * inner(k, unph) * dx
            )

        def p_eqn(phi):
            return (
                phi * div(unph) * dx
            )
        
        eqn = u_eqn(w) + b_eqn(q) + p_eqn(phi)
        bcs = self.bcs
        self.nprob = NonlinearVariationalProblem(eqn, self.Unp1, bcs=bcs)

        # Nullspace for the problem
        v_basis = VectorSpaceBasis(constant=True) #pressure field nullspace
        self.nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), v_basis])
        trans_null = VectorSpaceBasis(constant=True)
        self.trans_nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), trans_null])
        self.nsolver = NonlinearVariationalSolver(
                                                    self.nprob,
                                                    nullspace=self.nullspace,
                                                    transpose_nullspace=self.trans_nullspace,
                                                    solver_parameters=self.params,
                                                    options_prefix='linear_boussinesq_ASM'
                                                    )

    def time_stepping(self, tmax=3600.0, dt=600.0):
        Un = self.Un
        Unp1 = self.Unp1

        name = "lb_imp"
        file_lb = File(name+'.pvd')
        un, Pin, bn = Un.split()
        file_lb.write(un, Pin, bn)
        Unp1.assign(Un)

        t = 0.0
        dumpt = 600.
        tdump = 0.
        self.dT.assign(dt)
        print('tmax=', tmax, 'dt=', dt)
        while t < tmax - 0.5*dt:
            print(t)
            t += dt
            tdump += dt

            self.nsolver.solve()
            print("The nonlinear solver is solved.")
            self.Un.assign(self.Unp1)

            if tdump > dumpt - dt*0.5:
                file_lb.write(un, Pin, bn)
                tdump -= dumpt


if __name__ == "__main__":
    N=1.0e-2
    U=20.
    dT=600.
    nx=10
    ny=1
    Lx=10
    Ly=1.0e-3
    height=1e4
    nlayers=20
    horiz_num=80
    radius=2

    eqn = Boussinesq(N=N, U=U, dT=dT, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    eqn.build_initial_data()
    eqn.build_lu_params()
    eqn.build_boundary_condition()
    eqn.build_NonlinearVariationalSolver()
    eqn.time_stepping()