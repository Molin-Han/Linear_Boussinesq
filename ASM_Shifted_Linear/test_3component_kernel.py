from firedrake import *
from firedrake.output import VTKFile


# Basic Parameters 
nx=20
ny=20
Lx=2.0
Ly=1.0*Lx # TODO: horizontal AR issue happening.
height=2.0*1e-4
nlayers=10

# Setting up the extruded mesh
m = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='both', quadrilateral=True)
# m = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(m, nlayers, layer_height=height/nlayers, extrusion_type='uniform')
finest_mesh_name = "finest"
mesh.name = finest_mesh_name

# Set up the function space using TensorProductElement to have flexibility in the element degrees.
horizontal_degree = 2
vertical_degree = 2
# horizontal base spaces -- 2D
S1 = FiniteElement("RTCF", quadrilateral, horizontal_degree) # RT2 in 2D
S2 = FiniteElement("DQ", quadrilateral, horizontal_degree-1) # DG1 in 2D
# vertical base spaces
T0 = FiniteElement("CG", interval, vertical_degree) # CG2
T1 = FiniteElement("DG", interval, vertical_degree-1) # DG1
# Attempt to build the 3D element.
Vh_elt = TensorProductElement(S1, T1)
Vh = HDivElement(Vh_elt)
Vv_elt = TensorProductElement(S2, T0)
Vv = HDivElement(Vv_elt)
V_3d = Vh + Vv

V = FunctionSpace(mesh, V_3d) # Velocity space RT2
Vb = FunctionSpace(mesh, Vv_elt) # Buoyancy space W theta
Vp_elt = TensorProductElement(S2, T1) # DG horizontal and DG vertical
Vp = FunctionSpace(mesh, Vp_elt)
W = V * Vb * Vp
x, y, z = SpatialCoordinate(mesh)

# Set up the RHS.
U = Function(W)
Us = Function(W)
u, b, p = split(U)
us, bs, ps = split(Us)
w, q, phi = TestFunctions(W)

pcg = PCG64(seed=1234567)
rg = Generator(pcg)
f = rg.normal(Vp, 1.0, 2.0)
f_int = assemble(f*dx)
One = Function(Vp).assign(1.0)
area = assemble(One*dx)
f.project(f - f_int/area)
print(assemble(f*dx))
uic, bic, pic = U.subfunctions
pic.project(Constant(1000.0)*(z-0.5*height))
p_int = assemble(pic*dx)
pic.project(pic-p_int/area)

print(assemble(pic*dx))
file = VTKFile('IC.pvd')
file.write(uic,bic,pic,f)
print('write initial condition')

bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")

bcs = [bc1, bc2, bc3]

params = {
        'snes_type':'ksponly',
        'ksp_type': 'gmres',
        'snes_rtol':1e-8,
        'snes_atol':0,
        'snes_stol':0,
        'ksp_rtol': 1e-10,
        'ksp_atol':0,
        'snes_monitor':None,
        'ksp_monitor':None,
        'pc_type':'lu',
        'mat_type': 'aij',
        'pc_factor_mat_solver_type': 'mumps',
}

eqn = inner(u,w)*dx + q*b*dx + p*phi*dx

v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])
nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)


name = 'sol'
file1 = VTKFile(name+'.pvd')
nsolver.solve()
u_sol, b_sol, p_sol = U.subfunctions
file1.write(u_sol, b_sol, p_sol)
print(assemble(p_sol*dx))

