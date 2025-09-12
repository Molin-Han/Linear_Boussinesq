from firedrake import *
import numpy as np
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from twoD import solve_SLB
from petsc4py import PETSc

print = PETSc.Sys.Print

nx_array = np.arange(12, 60, 5) * 5
height = pi / 2000
nlayers = 100
length = 1.0
fig, ax = plt.subplots()
ax.set_title("The solution error for different nx")
dx_list = []

for i in nx_array:
    print(i)
    nx = i
    dx = 1 / nx
    ar = height / length
    print(f"Aspect ratio is {ar}")
    print(f"The dx is {dx}")
    dx_list.append(dx)
    # solve_SLB(nx=nx, length=length, height=height, nlayers=nlayers, xtest=True)


j = 0
for dx in dx_list:
    horiz = nx_array[j]
    j += 1
    error = np.loadtxt(f'err_dx_{dx}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"nx={horiz}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dx}.png")

plt.savefig(f"dx_{dx}.png")