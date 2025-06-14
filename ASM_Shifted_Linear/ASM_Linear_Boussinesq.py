from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc

print = PETSc.Sys.Print

N=1.0e-2
U=0.
dt=100.
tmax = 150.0
nx=10
ny=3
Lx=3.0e5
Ly=1.0e-3*Lx
# Ly = Lx
height=1e4
nlayers=10

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='both', quadrilateral=True, distribution_parameters=distribution_parameters)
mh = MeshHierarchy(m, refinement_levels=2)
hierarchy = ExtrudedMeshHierarchy(mh, height, base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
mesh = hierarchy[-1]
# mesh = ExtrudedMesh(m, nlayers, layer_height=height/nlayers, extrusion_type='uniform')
finest_mesh_name = "finest"
mesh.name = finest_mesh_name

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
# pnph = pnp1

Un_shift = Function(W)
Unp1_shift = Function(W)
un_shift, bn_shift, pn_shift = split(Un_shift)
unp1_shift, bnp1_shift, pnp1_shift = split(Unp1_shift)

unph_shift = 0.5 * (un_shift+unp1_shift)
bnph_shift = 0.5 * (bn_shift+bnp1_shift)
pnph_shift = 0.5 * (pn_shift+pnp1_shift)
# pnph = pnp1

n = FacetNormal(mesh)
k = as_vector([0,0,1])
Omega = 7.292e-5
theta = pi / 3
omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])
dtc = Constant(dt)

# Initial Condition
xc = Constant(Lx/2)
yc = Constant(Ly/2)
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
DG0 = FunctionSpace(mesh, 'DG', 0)
One = Function(DG0).assign(1.0)
area = assemble(One * dx)
pnic_int = assemble(pn*dx)
pnic.project(pn - pnic_int/area)
pnp1ic_int = assemble(pnp1*dx)
pnp1ic.project(pnp1ic - pnp1ic_int/area)


unic, bnic, pnic = Un_shift.subfunctions
unp1ic, bnp1ic, pnp1ic = Unp1_shift.subfunctions
unic.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
unp1ic.project(as_vector([Constant(0.0),Constant(0.0),Constant(0.0)]))
bnic.project(sin(pi*z/height)/(1+((x-xc)**2+(y-yc)**2)/a**2) + N**2 * z)
bnp1ic.project(sin(pi*z/height)/(1+((x-xc)**2+(y-yc)**2)/a**2) + N**2 * z)
pnic.project(0.5 * N**2 * z**2)
pnp1ic.project(0.5 * N**2 * z**2)
DG0 = FunctionSpace(mesh, 'DG', 0)
One = Function(DG0).assign(1.0)
area = assemble(One * dx)
pnic_int = assemble(pn*dx)
pnic.project(pn - pnic_int/area)
pnp1ic_int = assemble(pnp1*dx)
pnp1ic.project(pnp1ic - pnp1ic_int/area)

# Visualise Initial Condition to confirm.
# name = 'ic'
# file_lb = VTKFile(name+'.pvd')
# u0, b0, P0 = Un.subfunctions
# file_lb.write(u0, b0, P0)
# print("Save initial condition.")
# Boundary Conditions
bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

# TODO: Mixed Version of this, need to be checked!!!
class HDivHelmholtzSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        W = u.function_space()
        velo, b = split(u)
        w, q = split(v)
        Jp = (inner(velo, w) + dtc / 2 * div(velo) * div(w) + dtc * inner(cross(omega, velo), w))*dx
        Jp -= dtc/2 * inner(w, k) * b * dx
        Jp += inner(b, q)*dx
        Jp += dtc / 2 * N**2 * q * inner(k, velo) * dx
        #  Boundary conditions
        bc1 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "top")
        bc2 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "bottom")
        bc3 = DirichletBC(W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)

# Parameters
# TODO: Check if the Schur Complement is correct. The direct solve should converge in one iteration.
helmholtz_schur_pc_params = {
            'ksp_monitor': None,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
# helmholtz_schur_pc_params = {
#     'ksp_type': 'preonly',
#     'ksp_max_its': 30,
#     'pc_type': 'mg',
#     'pc_mg_type': 'full',
#     'pc_mg_cycle_type':'v',
#     'mg_levels': {
#         # 'ksp_type': 'gmres',
#         # 'ksp_type':'richardson',
#         'ksp_type': 'chebyshev',
#         # 'ksp_richardson_scale': 0.2,
#         'ksp_max_it': 1,
#         # 'ksp_monitor':None,
#         "pc_type": "python",
#         "pc_python_type": "firedrake.ASMStarPC", # TODO: shall we use AssembledPC?
#         "pc_star_construct_dim": 0,
#         "pc_star_sub_sub_pc_type": "lu",
#         # "pc_star_sub_sub_pc_type": "svd",
#         # "pc_star_sub_sub_pc_svd_monitor": None,
#     },
#     'mg_coarse': {
#         'ksp_type': 'preonly',
#         'pc_type': 'lu',
#     },
# }
# params = {
#     # 'mat_type': 'aij',
#     'ksp_type': 'gmres',
#     'snes_type':'ksponly',
#     'ksp_atol': 0,
#     'ksp_rtol': 1e-5,
#     # 'ksp_view': None,
#     'snes_monitor': None,
#     # 'ksp_monitor': None,
#     'ksp_monitor_true_residual': None,
#     'pc_type': 'fieldsplit',
#     'pc_fieldsplit_type': 'schur',
#     'pc_fieldsplit_schur_fact_type': 'full',
#     'pc_fieldsplit_0_fields': '2',
#     'pc_fieldsplit_1_fields': '0,1',
#     'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
#         'ksp_type': 'preonly',
#         'pc_type': 'bjacobi',
#         'sub_pc_type': 'ilu',
#         # 'pc_factor_mat_solver_type': 'mumps',
#     },
#     'fieldsplit_1': {
#         'ksp_type': 'preonly',
#         'pc_type': 'python',
#         'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
#         'helmholtzschurpc': helmholtz_schur_pc_params,
#         },
# }
# params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
params = {'snes_type':'ksponly','ksp_type': 'gmres','snes_monitor':None, 'ksp_monitor':None, 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

# Equations
def u_eqn(w):
    return (
        inner(w, (unp1 - un)) * dx +
        dtc * inner(w, 2 * cross(omega, unph)) * dx -
        dtc * div(w) * pnp1 * dx - dtc * inner(w, k) * bnph * dx
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


def u_eqn_shift(w):
    return (
        inner(w, (unp1_shift - un_shift)) * dx +
        dtc * inner(w, 2 * cross(omega, unph_shift)) * dx -
        dtc * div(w) * pnp1_shift * dx - dtc * inner(w, k) * bnph_shift * dx
    )

def b_eqn_shift(q):
    return (
        q * (bnp1_shift - bn_shift) * dx +
        dtc * N**2 * q * inner(k, unph_shift) * dx
    )

def p_eqn_shift(phi):
    return (
        phi * div(unph_shift) * dx
    )


eqn = p_eqn(phi) + b_eqn(q) + u_eqn(w)
# shift = inner(unp1, w) * dx + dtc * inner(cross(omega, unp1), w) * dx - dtc * div(w) * pnp1 * dx - dtc / 2 * inner(w, k) * bnp1 *dx
# shift += bnp1 * q * dx + dtc * N**2 / 2 * inner(unp1, k) * q * dx
# shift += 1/2 * div(unp1) * phi * dx + pnp1 * phi * dx
eqn_shift = p_eqn_shift(phi) + b_eqn_shift(q) + u_eqn_shift(w)
shift = eqn_shift + Constant(1) * pnp1 * phi * dx
Jp = derivative(shift, Unp1_shift)

nprob = NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)
v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params)


nprob_shift = NonlinearVariationalProblem(eqn_shift, Unp1_shift, bcs=bcs, Jp=Jp)
nsolver_shift = NonlinearVariationalSolver(nprob_shift, nullspace=nullspace, solver_parameters=params)
# Time Stepping
name = 'lb_imp_ASM'
file_lb = VTKFile(name+'.pvd')
un, bn, pn = Un.subfunctions
file_lb.write(un, bn, pn)
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
    nsolver_shift.solve()
    Un.assign(Unp1)
    Un_shift.assign(Unp1_shift)
    if tdump > dumpt - dt*0.5:
        file_lb.write(un, bn, pn)
        tdump -= dumpt
    name = 'diff'
    diff = Function(W).assign(Unp1_shift - Unp1)
    u1, b1, p1 = diff.subfunctions
    file = VTKFile(name+'.pvd')
    file.write(u1, b1, p1)