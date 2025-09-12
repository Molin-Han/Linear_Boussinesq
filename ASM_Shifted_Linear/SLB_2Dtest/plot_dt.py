from firedrake import *
import numpy as np
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from twoD import solve_SLB
from petsc4py import PETSc

print = PETSc.Sys.Print

t_array = np.arange(1, 25, 4)
nx = 100
height = pi / 2000
nlayers = 100
length = 1.0
fig, ax = plt.subplots()
ax.set_title("The solution error for different $\Delta t$")
dt_list = []

for i in t_array:
    print(i)
    deltat = i
    delta = deltat
    dx = 1 / nx
    ar = height / length
    print(f"Aspect ratio is {ar}")
    print(f"The dt is {deltat}")
    dt_list.append(deltat)
    # solve_SLB(nx=nx, length=length, height=height, nlayers=nlayers, deltat=deltat, delta=delta, ttest=True)


j = 0
for dt in dt_list:
    delt = t_array[j]
    j += 1
    error = np.loadtxt(f'err_dt_{delt}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"$\Delta t$={delt}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dx}.png")

plt.savefig(f"dt_{deltat}.png")