from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Boussinesq import Boussinesq

N=1.0e-2
U=0.
dt=600.0
tmax = 6000.0
nx=30
ny=1
Lx=3.0e5
Ly=1.0e-3 * Lx
height=1e4
nlayers=10

eqn = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
eqn.build_initial_data()
# eqn.build_lu_params()
eqn.build_ASM_MH_params()
# eqn.build_pure_Vanka_params()
eqn.build_boundary_condition()
eqn.build_NonlinearVariationalSolver()
eqn.time_stepping(tmax=tmax, dt=dt)
print("The simulation is completed.")