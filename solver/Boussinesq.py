from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class Boussinesq:
    def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2, mesh="interval"):
        '''
        mesh : interval or circle to be extruded.
        '''
        self.ar = height/(2 * pi * radius)
        self.dx = 2 * pi * radius / horiz_num
        self.dz = height / nlayers
        print(f"The aspect ratio is {self.ar}")

        # Extruded Mesh
        if mesh == "interval":
            self.m = UnitIntervalMesh(horiz_num, name='interval')
            self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')
        if mesh == "circle":
            self.m = CircleManifoldMesh(horiz_num, radius=radius, name='circle')
            self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='radial')

        if mesh == '3d':
            self.m = 
        # Mixed Finite Element Space
        CG_1 = FiniteElement("CG", interval, 1)
        DG_0 = FiniteElement("DG", interval, 0)
        P1P0 = TensorProductElement(CG_1, DG_0)
        RT_horiz = HDivElement(P1P0)
        P0P1 = TensorProductElement(DG_0, CG_1)
        RT_vert = HDivElement(P0P1)
        RT_e = RT_horiz + RT_vert
        RT = FunctionSpace(self.mesh, RT_e)
        Wta = FunctionSpace(self.mesh, P0P1) # Buoyancy space
        DG = FunctionSpace(self.mesh, 'DG', 0)
        self.W = RT * DG * Wta


        # Setting up the solution variables.
        self.sol = Function(self.W)
        self.u, self.p, self.b = split(self.sol)
        self.alpha, self.phi, self.gamma = TestFunctions(self.W)

        self.x, self.z = SpatialCoordinate(self.mesh)

    def build_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
    

    def build_NonlinearVariationalSolver(self):
        

