from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc

print = PETSc.Sys.Print

N=1.0e-2
U=0.
dt=600.
tmax = 3000.0
nx=30
ny=1
Lx=3.0e5
Ly=1
height=1e4
nlayers=10

AR = height / Lx
deltax = Lx / nx
deltaz = height / nlayers
m = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='both', quadrilateral=True)
mh = MeshHierarchy(m, refinement_levels=2)
hierarchy = ExtrudedMeshHierarchy(mh, height, base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
# mesh = ExtrudedMesh(m, nlayers, layer_height=height/nlayers, extrusion_type='uniform')
mesh = hierarchy[-1]

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

Un = Function(W)
Unp1 = Function(W)
un, bn, pn = split(Un)
unp1, bnp1, pnp1 = split(Unp1)
w, q, phi = TestFunctions(W)

unph = 0.5 * (un+unp1)
bnph = 0.5 * (bn+bnp1)
pnph = 0.5 * (pn+pnp1)

n = FacetNormal(mesh)
k = as_vector([0,0,1])
Omega = 7.292e-5
theta = pi / 3
omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])
dtc = Constant(dt)

# Initial Condition
xc = Lx/2
yc = Ly/2
a = Constant(5000)
U = Constant(0.)

unic, bnic, pnic = Un.subfunctions
unp1ic, bnp1ic, pnp1ic = Unp1.subfunctions
unic.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
unp1ic.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
bnic.project(sin(pi*z/height)/(1+((x-xc)**2+(y-yc)**2)/a**2) + N**2 * z)
bnp1ic.project(sin(pi*z/height)/(1+((x-xc)**2+(y-yc)**2)/a**2) + N**2 * z)

pnic.project(0.5 * N**2 * z**2)
pnp1ic.project(0.5 * N**2 * z**2)

# Boundary Conditions
bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

# Parameters
params = {
    'snes_type':'ksponly',
    'ksp_type': 'gmres',
    # 'snes_monitor': None,
    'ksp_atol': 0,
    'ksp_rtol': 1e-8,
    'ksp_monitor': None,
    'pc_type': 'mg',
    'pc_mg_type': 'full',
    'mg_levels': {
        # 'ksp_type': 'richardson',
        'ksp_type': 'chebyshev',
        'ksp_max_it': 1,
        "pc_type": "python",
        # "pc_python_type": "firedrake.ASMStarPC", # TODO: shall we use AssembledPC?
        # "pc_star_construct_dim": 0,
        # "pc_star_sub_sub_pc_type": "lu",
        "pc_python_type": "firedrake.ASMVankaPC", # TODO: shall we use AssembledPC?
        "pc_vanka_construct_dim": 0,
        "pc_vanka_sub_sub_pc_type": "lu",
        # 'pc_type': 'python',
        # 'pc_python_type': 'firedrake.AssembledPC',
        # 'assembled_pc_type': 'python',
        # 'assembled_pc_python_type': 'firedrake.ASMStarPC',
        # 'assembled_pc_star_construct_dim': 0,
        # 'assembled_pc_star_sub_sub_pc_type': 'lu',
        # 'assembled_pc_star_sub_sub_pc_type': 'svd',
        # 'assembled_pc_star_sub_sub_pc_svd_monitor': None,
        },
    'mg_coarse': {'ksp_type': 'preonly','pc_type': 'lu'}
}
# params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
# Equations
def u_eqn(w):
    return (
        inner(w, (unp1 - un)) * dx +
        dtc * inner(w, 2 * cross(omega, unph)) * dx -
        dtc * div(w) * pnph * dx - dtc * inner(w, k) * bnph * dx
    )
def b_eqn(q):
    return (
        q * (bnp1 - bn) * dx +
        dtc * N**2 * q * inner(k, unph) * dx
    )
def p_eqn(phi):
    return (
        phi * div(unph) * dx
    )

eqn = p_eqn(phi) + b_eqn(q) + u_eqn(w)
nprob = NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)
v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)

# Time Stepping
name = 'lb_imp'
file_lb = VTKFile(name+'.pvd')
un, bn, Pin = Un.subfunctions
file_lb.write(un, bn, Pin)
Unp1.assign(Un)

t = 0.0
dumpt = dt
tdump = 0.
dtc.assign(dt)
i = 0
while t < tmax - 0.5 * dt:
    print("The solver is currently solving for time", t)
    t += dt
    tdump += dt
    i += 1
    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        file_lb.write(un, bn, Pin)
        tdump -= dumpt

