from firedrake import *
import numpy as np
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from twoD import solve_SLB
from petsc4py import PETSc

print = PETSc.Sys.Print

height = pi / 2000
nx = 100
length = 1.0
nlayers_array = np.exp(np.arange(2, 8)* 2/5) * 10
nlayers_array = nlayers_array.astype(int)

fig, ax = plt.subplots()
ax.set_title("The solution error for different nz")
dz_list = []

for i in nlayers_array:
    print(i)
    nlayers = i
    dz = height / nlayers
    ar = height/ length
    print(f"Aspect ratio is {ar}")
    print(f"The dz is {dz}")
    dz_list.append(dz)
    # solve_SLB(nx=nx, length=length, height=height, nlayers=nlayers, ztest=True)

j = 0
for dz in dz_list:
    nlayer = nlayers_array[j]
    j += 1
    error = np.loadtxt(f'err_dz_{dz}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"nz={nlayer}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dz}.png")
plt.savefig(f"dz_{dz}.png")