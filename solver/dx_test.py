from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Boussinesq import Boussinesq


N=1.0e-2
U=0.
dt0=600.0
tmax = 1200.0
nx0=20
nx_array = np.arange(2, 11, 2) * 10
dt_array = dt0 * nx0 / nx_array
print(dt_array)
ny=1
Lx=3.0e5
Ly=1.0e-3 * Lx
height=1e4
nlayers=10

fig, ax = plt.subplots()
ax.set_title("The solution error")
dx_list = []
nx_list = []

index = 0
for i in nx_array:
    print(i)
    nx = i
    dx = Lx / nx
    ar = height/ Lx
    print(f"Aspect ratio is {ar}")
    print(f"The dx is {dx}")
    dx_list.append(dx)
    nx_list.append(nx)
    dt = dt_array[index]
    index += 1

    eqn = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    eqn.build_initial_data()
    eqn.build_ASM_MH_params()
    eqn.build_boundary_condition()
    eqn.build_NonlinearVariationalSolver()
    eqn.time_stepping(tmax=tmax, dt=dt0, monitor=False, xtest=True)
    
    eqn_monitor = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    eqn_monitor.build_initial_data()
    eqn_monitor.build_ASM_MH_params()
    eqn_monitor.build_boundary_condition()
    eqn_monitor.build_NonlinearVariationalSolver()
    eqn_monitor.time_stepping(tmax=tmax, dt=dt0, monitor=True, xtest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for dx = {dx}")


i = 2
time = 1200.0
j = 0
for dx in dx_list:
    nx = nx_list[j]
    j += 1
    error = np.loadtxt(f'err_dx_{dx}_{i}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"nx={nx}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dx}.png")

plt.savefig(f"error_final_dx_{dx}.png")