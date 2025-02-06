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
Lx=3.0e6
Ly=1.0e-3 * Lx
# height=1e4
height_array = np.arange(2, 10) * 1e2
nlayers=10
fig, ax = plt.subplots()
ax.set_title(f"The solution error, Lx={Lx}")
ar_list = []

for i in height_array:
    print(i)
    height = i
    ar = height/ Lx
    print(f"Aspect ratio is {ar}")
    ar_list.append(ar)
    eqn = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    eqn.build_initial_data()
    eqn.build_ASM_MH_params()
    eqn.build_boundary_condition()
    eqn.build_NonlinearVariationalSolver()
    eqn.time_stepping(tmax=tmax, dt=dt, monitor=False, artest=True)
    
    eqn_monitor = Boussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    eqn_monitor.build_initial_data()
    eqn_monitor.build_ASM_MH_params()
    eqn_monitor.build_boundary_condition()
    eqn_monitor.build_NonlinearVariationalSolver()
    eqn_monitor.time_stepping(tmax=tmax, dt=dt, monitor=True, artest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for ar = {ar}")


time = 1200.0
j = 0
for ratio in ar_list:
    height = height_array[j]
    j += 1
    error = np.loadtxt(f'err_ar_{ratio}_{int(time)}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"height={int(height)}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{ratio}.png")

plt.savefig(f"error_final_ar_{ratio}_{int(time)}.png")