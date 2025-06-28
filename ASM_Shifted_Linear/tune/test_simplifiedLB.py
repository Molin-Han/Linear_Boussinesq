from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc
print = PETSc.Sys.Print

# Basic Parameters 
# nx=5
# ny=5
# Lx=3.0e5
# Ly=1.0*Lx # TODO: horizontal AR issue happening.
# height=1e4
# nlayers=10

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

n = FacetNormal(mesh)
k = as_vector([0,0,1])

VDG = VectorFunctionSpace(mesh, 'DQ', 2)
DG = FunctionSpace(mesh, 'DQ', 1)
pcg = PCG64(seed=1234567)
rg = Generator(pcg)
# f = rg.normal(VDG, 1.0, 2.0)
f = Function(V).project(as_vector([0.1 * z + sin(2*pi*x/Lx), 0,0]))
g = rg.normal(DG, 1.0, 2.0)

One = Function(DG).assign(1.0)
area = assemble(One*dx)
f_int = assemble(inner(f, w)*dx)
print('the integral of f is ', norm(f_int.riesz_representation()))
g_int = assemble(g*dx)
# f.interpolate(f - f_int/area)
# g.interpolate(g - g_int/area)

bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")

bcs = [bc1, bc2, bc3]
# bcs = []

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

# Equations
def u_eqn(u, p, b):
    return (
        inner(w, u) * dx -
        div(w) * p * dx 
        + inner(w, k) * b * dx 
        - inner(w, f) * dx
    )

def b_eqn(u, p, b):
    return (
        q * b * dx 
        + q * inner(k, u) * dx 
        - g * q * dx
    )

def p_eqn(u, p, b):
    return (
        phi * div(u) * dx
    )

eqn = p_eqn(u,p,b) + b_eqn(u,p,b) + u_eqn(u,p,b)
eqns = p_eqn(us,ps,bs) + b_eqn(us,ps,bs) + u_eqn(us,ps,bs)
shift = eqns + Constant(0.001) * ps * phi * dx
Jp = derivative(shift, Us)

# pressure nullspace
v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])

nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)
nprobs = NonlinearVariationalProblem(eqns, Us, bcs=bcs, Jp=Jp)
nsolvers = NonlinearVariationalSolver(nprobs, nullspace=nullspace, solver_parameters=params)

name = 'diff'
file = VTKFile(name + '.pvd')
nsolver.solve()
nsolvers.solve()
diff = Function(W).assign(Us - U)
u_sol, b_sol, p_sol = diff.subfunctions
file.write(u_sol, b_sol, p_sol)
print('relative norm is', norm(diff)/norm(U))
