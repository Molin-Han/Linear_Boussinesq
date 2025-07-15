from firedrake import *
import numpy as np
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from twoD import solve_SLB
from petsc4py import PETSc

print = PETSc.Sys.Print

rate = 8000
height_array = np.exp(np.arange(7, -3, -2.0)) * pi / rate

nx = 100
nlayers = 200
length = 1.0
fig, ax = plt.subplots()
ax.set_title(f"The solution error with different AR.")
ar_list = []

for i in height_array:
    print(i)
    height = i
    ar = height / length
    print(f"Aspect ratio is {ar}")
    ar_list.append(ar)
    solve_SLB(nx=nx, length=length, height=height, nlayers=nlayers, artest=True)

j = 0
for ratio in ar_list:
    error = np.loadtxt(f'err_ar_{ratio}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"ar={round(ratio, 5)}")
    j+=1
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{ratio}.png")
plt.savefig(f"ar_{ratio}.png")