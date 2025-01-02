from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class Boussinesq:
    def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2):
        self.ar = height/(2 * pi * radius)
        self.dx = 2 * pi * radius / horiz_num
        self.dz = height / nlayers
        print(f"The aspect ratio is {self.ar}")

        self.m = CircleManifoldMesh(horiz_num, radius=radius)
        # self.m = UnitIntervalMesh(horiz_num)
        # Extruded Mesh
        self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='radial')
        # self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')
