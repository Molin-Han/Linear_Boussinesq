from firedrake import *
from firedrake.output import VTKFile

m = UnitIntervalMesh(20, name='interval')
mesh = ExtrudedMesh(m, 10, layer_height=0.05, extrusion_type='uniform')

RT = FunctionSpace(mesh, 'RTCF', 1)
DG = FunctionSpace(mesh, 'DG', 0)
W = RT * DG
VDG = VectorFunctionSpace(mesh, 'DQ', 1)

v, q = TestFunctions(W)
sol = Function(W)
sol_shift = Function(W)
u, p = split(sol)
u_shift, p_shift = split(sol_shift)
x, y = SpatialCoordinate(mesh)
pcg = PCG64(seed=1234567)
rg = Generator(pcg)
g = rg.normal(DG, 1.0, 2.0)
f = rg.normal(VDG, 1.0, 2.0)
One = Function(DG).assign(1.0)
area = assemble(One*dx)
g_int = assemble(g*dx)
g.interpolate(g - g_int/area)
params = {
        'snes_type':'ksponly',
        'ksp_type': 'gmres',
        'snes_monitor':None, 
        'ksp_monitor':None, 
        'snes_rtol':1e-8,
        'snes_atol':0,
        'snes_stol':0,
        'ksp_rtol': 1e-10,
        'ksp_atol':0,
        'pc_type':'lu', 
        'mat_type': 'aij', 
        'pc_factor_mat_solver_type': 'mumps',
}

def F(u, p):
    return (inner(u,v) - div(v)*p + div(u)*q)*dx - g*q * dx - inner(f, v) * dx

shift = F(u_shift, p_shift) + p_shift*q*dx
Jp = derivative(shift, sol_shift)

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])


nprob = NonlinearVariationalProblem(F(u, p), sol, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)

nprob_shift = NonlinearVariationalProblem(F(u_shift, p_shift), sol_shift, bcs=bcs, Jp=Jp)
nsolver_shift = NonlinearVariationalSolver(nprob_shift, nullspace=nullspace, solver_parameters=params)

nsolver.solve()
nsolver_shift.solve()

name = '2ddiff_Poisson'
file = VTKFile(name + '.pvd')
diff = Function(W).assign(sol_shift-sol)
u_diff, p_diff = diff.subfunctions
file.write(u_diff, p_diff)
print(norm(diff))

sol_file = VTKFile('sol.pvd')
u_sol, p_sol = sol.subfunctions
sol_file.write(u_sol, p_sol)
