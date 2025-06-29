from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc
print = PETSc.Sys.Print

nx=20
height=1
nlayers=20

m = PeriodicUnitIntervalMesh(nx)
mesh = ExtrudedMesh(m, nlayers, layer_height=height/nlayers, extrusion_type='uniform')

# Set up the function space using TensorProductElement to have flexibility in the element degrees.
CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
RT_e = RT_horiz + RT_vert
RT = FunctionSpace(mesh, RT_e)
DG = FunctionSpace(mesh, 'DG', 0)
# horizontal base spaces -- 2D
S2 = FiniteElement("DG", interval, 0) # DG1 in 1D
# vertical base spaces
T0 = FiniteElement("CG", interval, 1) # CG2
# Attempt to build the 2D element.
Vv_elt = TensorProductElement(S2, T0) # DG horizontal and CG vertical
Vb = FunctionSpace(mesh, Vv_elt) # Buoyancy space W theta
W = RT * Vb * DG
x, z = SpatialCoordinate(mesh)

# Set up the RHS.
U = Function(W)
Us = Function(W)
u, b, p = split(U)
us, bs, ps = split(Us)
w, q, phi = TestFunctions(W)

n = FacetNormal(mesh)
k = as_vector([0,1])

VDG = VectorFunctionSpace(mesh, 'DQ', 2)
DG = FunctionSpace(mesh, 'DQ', 1)
pcg = PCG64(seed=1234567)
rg = Generator(pcg)
f = rg.normal(VDG, 1.0, 2.0)
g = rg.normal(DG, 1.0, 2.0)

One = Function(DG).assign(1.0)
area = assemble(One*dx)
f_int = assemble(inner(f, w)*dx)
print('the integral of f is ', norm(f_int.riesz_representation()))

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")

bcs = [bc1, bc2, bc3]

params_direct = {
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

delta = Constant(1.0)

eqn = p_eqn(u,p,b) + b_eqn(u,p,b) + u_eqn(u,p,b)
eqns = p_eqn(us,ps,bs) + b_eqn(us,ps,bs) + u_eqn(us,ps,bs)
shift = eqns + delta * ps * phi * dx
Jp = derivative(shift, Us)

# pressure nullspace
v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])

nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params_direct)
nprobs = NonlinearVariationalProblem(eqns, Us, bcs=bcs, Jp=Jp)
nsolvers = NonlinearVariationalSolver(nprobs, nullspace=nullspace, solver_parameters=params_direct)

name = 'diff'
file = VTKFile(name + '.pvd')
nsolver.solve()
nsolvers.solve()
diff = Function(W).assign(Us - U)
u_sol, b_sol, p_sol = diff.subfunctions
file.write(u_sol, b_sol, p_sol)
print('relative norm is', norm(diff)/norm(U))