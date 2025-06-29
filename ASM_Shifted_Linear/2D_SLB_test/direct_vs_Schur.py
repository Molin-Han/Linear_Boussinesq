from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc
print = PETSc.Sys.Print

nx=20
height=1
nlayers=20


distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = PeriodicUnitIntervalMesh(nx)
# mh = MeshHierarchy(m, refinement_levels=2)
# hierarchy = ExtrudedMeshHierarchy(mh, height, base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
# mesh = hierarchy[-1]
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

# TODO: This needs to be checked.
class HDivSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        W = u.function_space()
        velo, b = split(u)
        w, q = split(v)
        Jp = (inner(velo, w) + 1 / delta * div(velo) * div(w))*dx
        Jp += inner(w, k) * b * dx
        Jp += q * b *dx + q * inner(k, velo) * dx
        #  Boundary conditions
        bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)

# pressure nullspace
v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])


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


params_schur = {
    # 'mat_type': 'aij',
    'ksp_type': 'gmres',
    'snes_type':'ksponly',
    'ksp_atol': 0,
    'ksp_rtol': 1e-9,
    # 'ksp_view': None,
    'snes_monitor': None,
    # 'ksp_monitor': None,
    'ksp_monitor_true_residual': None,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_0_fields': '2',
    'pc_fieldsplit_1_fields': '0,1',
    'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        # 'pc_factor_mat_solver_type': 'mumps',
    },
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': __name__ + '.HDivSchurPC',
        'helmholtzschurpc': helmholtz_schur_pc_params,
        },
}
nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params_direct)
nprobs = NonlinearVariationalProblem(eqns, Us, bcs=bcs, Jp=Jp)
nsolvers = NonlinearVariationalSolver(nprobs, nullspace=nullspace, solver_parameters=params_schur)

name = 'diff'
file = VTKFile(name + '.pvd')
nsolver.solve() # TODO: Make a mesh Hierarchy will break the direct solver
nsolvers.solve()
diff = Function(W).assign(Us - U)
u_sol, b_sol, p_sol = diff.subfunctions
file.write(u_sol, b_sol, p_sol)
print('relative norm is', norm(diff)/norm(U))