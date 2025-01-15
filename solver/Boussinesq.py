from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class Boussinesq:
    def __init__(self, N= 1.0e-2, U=20., dT=600., nx=5e3, ny=1, Lx=1e3, Ly=1, height=1e4, nlayers=20, horiz_num=80, radius=2):

        # Extruded Mesh 3D
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.height = height
        self.U = U # steady velocity
        self.N = N # buoyancy frequency
        self.m = PeriodicRectangleMesh(self.nx, self.ny, self.Lx, self.Ly, direction='y',quadrilateral=True)
        self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')

        # Mixed Finite Element Space
        horizontal_degree = 2
        vertical_degree = 2
        # horizontal base spaces
        S1 = FiniteElement("CG", interval, horizontal_degree) # CG2
        S2 = FiniteElement("DG", interval, horizontal_degree-1) # DG1
        # vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree) # CG2
        T1 = FiniteElement("DG", interval, vertical_degree-1) # DG1

        Vh_elt = TensorProductElement(S1, T1) # CG horizontal and DG vertical
        V_horiz = HDivElement(Vh_elt)
        Vv_elt = TensorProductElement(S2, T0) # DG horizontal and CG vertical
        V_vert = HDivElement(Vv_elt)
        V_e = V_horiz + V_vert
        V = FunctionSpace(self.mesh, V_e, name="HDiv") # Velocity space RT(k-1)
        Vb = FunctionSpace(self.mesh, Vv_elt, name="Buoyancy") # Buoyancy space
        Vp_elt = TensorProductElement(S2, T1) # DG horizontal and DG vertical
        Vp = FunctionSpace(self.mesh, Vp_elt, name="Pressure")

        self.W = V * Vp * Vb # velocity, pressure, buoyancy space
        self.x, self.z = SpatialCoordinate(self.mesh)

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

        self.dT = dT

        # Test Functions
        self.w, self.phi, self.q = TestFunctions(self.W)

    def build_initial_data(self):
        xc = self.Lx/2
        a = Constant(5000)
        U = Constant(self.U)
        self.un.interpolate(as_vector([U,0,0])) # TODO: need to check this.
        self.bn.interpolate(sin(pi*self.z/self.height)/(1+(self.x-xc)**2/a**2))


    def build_lu_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
    
    def build_ASM_params(self):
        pass

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
        def u_eqn(w):
            pass

        def b_eqn(q):
            pass

        def p_eqn(phi):
            pass
        
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