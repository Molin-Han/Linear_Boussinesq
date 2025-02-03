from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Boussinesq import Boussinesq

N=1.0e-2
U=0.
dt=600.0
tmax = 1200.0
nx=30
ny=1
Lx=3.0e5
Ly=1.0e-3 * Lx
height=1e4
nlayers_array = np.arange(2, 9, 2) * 10
fig, ax = plt.subplots()
ax.set_title("The solution error")
dz_list = []

for i in nlayers_array:
    print(i)
    nlayers = i
    dz = height / nlayers
    ar = height/ Lx
    print(f"Aspect ratio is {ar}")
    print(f"The dz is {dz}")
    dz_list.append(dz)

    # eqn = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    # eqn.build_initial_data()
    # eqn.build_ASM_MH_params()
    # eqn.build_boundary_condition()
    # eqn.build_NonlinearVariationalSolver()
    # eqn.time_stepping(tmax=tmax, dt=dt, monitor=False, artest=True)
    
    # eqn_monitor = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    # eqn_monitor.build_initial_data()
    # eqn_monitor.build_ASM_MH_params()
    # eqn_monitor.build_boundary_condition()
    # eqn_monitor.build_NonlinearVariationalSolver()
    # eqn_monitor.time_stepping(tmax=tmax, dt=dt, monitor=True, ztest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for ar = {ar}")


i = 0
time = 1200.0
for dz in dz_list:
    nlayer = nlayers_array[i]
    i += 1
    error = np.loadtxt(f'err_dz_{dz}_{int(time)}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"nlayer={nlayer}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dz}.png")

plt.savefig(f"error_final_dz_{dz}_{int(time)}.png")