from firedrake import *
from firedrake.output import VTKFile

m = UnitIntervalMesh(80)
mh = MeshHierarchy(m, refinement_levels=2)
hierarchy = ExtrudedMeshHierarchy(mh, pi/20, base_layer=20, refinement_ratio=1, extrusion_type='uniform')
mesh = hierarchy[-1]
RT = FunctionSpace(mesh, 'RTCF', 1)
DG = FunctionSpace(mesh, 'DG', 0)
W = RT * DG
v, q = TestFunctions(W)
sol = Function(W)
u, p = split(sol)
x, y = SpatialCoordinate(mesh)
pcg = PCG64(seed=123456789)
rg = Generator(pcg)
f = rg.normal(DG, 1.0, 2.0) * Constant(10.0)
F = (inner(u, v) + div(v)*p + div(u)*q)*dx - f*q*dx

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
params = {
    'ksp_type': 'gmres',
    'ksp_monitor': None,
    'snes_monitor': None,
    'snes_type': 'ksponly',
    'pc_type': 'mg',
    'pc_mg_type': 'full',
    'mg_levels': {
        'ksp_type': 'chebyshev',
        'ksp_max_it': 1,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.ASMStarPC',
        'pc_star_construct_dim': 0,
        'pc_star_sub_sub_pc_type': 'lu',
    },
    'mg_coarse': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }
}

prob_w = NonlinearVariationalProblem(F, sol, bcs=bcs)
solver_w = NonlinearVariationalSolver(prob_w, nullspace=nullspace, solver_parameters=params)
solver_w.solve()
sol_u, sol_p = sol.subfunctions
sol_file = VTKFile('Poisson_sol.pvd')
sol_file.write(sol_u, sol_p)
