from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc
print = PETSc.Sys.Print

# Basic Parameters 
# nx=10
# ny=3
# Lx=3.0e5
# Ly=1.0e-3*Lx
# height=1e4
# nlayers=10

nx=20
ny=20
Lx=2.0e2
Ly=1.0*Lx # TODO: horizontal AR issue happening.
height=Lx * 1e-4
nlayers=10


N=1.0e-2
U=0.
dt=100.
dtc=Constant(dt)
tmax = 150.0


# Setting up the extruded mesh
# TODO: no mesh hierarchy yet.
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='both', quadrilateral=True,distribution_parameters=distribution_parameters)
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
# TODO: here we only solve for one time step.
Un = Function(W)
Unp1 = Function(W)
un, bn, pn = split(Un)
unp1, bnp1, pnp1 = split(Unp1)
unph = 0.5 * (un+unp1)
bnph = 0.5 * (bn+bnp1)
pnph = 0.5 * (pn+pnp1)

Uns = Function(W)
Unp1s = Function(W)
uns, bns, pns = split(Uns)
unp1s, bnp1s, pnp1s = split(Unp1s)
unphs = 0.5 * (uns+unp1s)
bnphs = 0.5 * (bns+bnp1s)
pnphs = 0.5 * (pns+pnp1s)

w, q, phi = TestFunctions(W)

n = FacetNormal(mesh)
k = as_vector([0,0,1])
Omega = 7.292e-5
theta = pi / 3
omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])

# Initial Condition
xc = Constant(Lx/2)
yc = Constant(Ly/2)
a = Constant(5000.)
U = Constant(0.)

unic, bnic, pnic = Un.subfunctions
unic.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
bnic.project(sin(pi*z/height)/(1+((x-xc)**2+(y-yc)**2)/a**2) + N**2 * z)
pnic.project(0.5 * N**2 * z**2)
DG0 = FunctionSpace(mesh, 'DG', 0)
One = Function(DG0).assign(1.0)
area = assemble(One * dx)
pn_int = assemble(pnic * dx)
pnic.project(pnic - pn_int/area)

unics, bnics, pnics = Uns.subfunctions
unics.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
bnics.project(sin(pi*z/height)/(1+((x-xc)**2+(y-yc)**2)/a**2) + N**2 * z)
pnics.project(0.5 * N**2 * z**2)
pns_int = assemble(pnics*dx)
pnics.project(pnics - pns_int/area)


# Visualise Initial Condition to confirm.
name = 'ic'
file_lb = VTKFile(name+'.pvd')
u0, b0, P0 = Un.subfunctions
file_lb.write(u0, b0, P0)
print("Save initial condition.")

bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

params = {
        'snes_type':'ksponly',
        'ksp_type': 'gmres',
        'snes_monitor':None,
        'ksp_monitor':None,
        'pc_type':'lu',
        'mat_type': 'aij',
        'pc_factor_mat_solver_type': 'mumps',
}

def u_eqn(un, unp1, unph, pnp1, bnph):
    return (
        inner(w, (unp1 - un)) * dx +
        dtc * inner(w, 2 * cross(omega, unph)) * dx -
        dtc * div(w) * pnp1 * dx - dtc * inner(w, k) * bnph * dx
    )

def b_eqn(bn, bnp1, unph):
    return (
        q * (bnp1 - bn) * dx +
        dtc * N**2 * q * inner(k, unph) * dx
    )

def p_eqn(unph):
    return (
        phi * div(unph) * dx
    )


eqn = u_eqn(un, unp1, unph, pnp1, bnph) + b_eqn(bn, bnp1, unph) + p_eqn(unph)
eqns = u_eqn(uns, unp1s, unphs, pnp1s, bnphs) + b_eqn(bns, bnp1s, unphs) + p_eqn(unphs) + Constant(0.001) * pnp1s * phi * dx
Jp = derivative(eqns, Unp1s)

v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])

nprob = NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)

nprobs = NonlinearVariationalProblem(eqns, Unp1s, bcs=bcs, Jp=Jp)
nsolvers = NonlinearVariationalSolver(nprobs, nullspace=nullspace, solver_parameters=params)

name = 'diff_ASMLB'
file = VTKFile(name + '.pvd')
name2 = 'ASMLB_direct'
file_sol = VTKFile(name2 + '.pvd')
name3 = 'ASMLB_shift'
file_shift = VTKFile(name3 + '.pvd')
Unp1.assign(Un)
Unp1s.assign(Uns)
nsolver.solve()
nsolvers.solve()
diff = Function(W).assign(Unp1s - Unp1)
u_d, b_d, p_d = diff.subfunctions
file.write(u_d, b_d, p_d)
u_sol, b_sol, p_sol = Unp1.subfunctions
file_sol.write(u_sol, b_sol, p_sol)
print('relative norm is', norm(diff)/norm(Unp1))
u_shift, b_shift, p_shift = Unp1s.subfunctions
file_shift.write(u_shift, b_shift, p_shift)
