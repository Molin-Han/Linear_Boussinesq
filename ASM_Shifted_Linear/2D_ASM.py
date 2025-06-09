from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc
print = PETSc.Sys.Print

N=1.0e-2
U=0.
dt=100.
tmax = 3600.0
nx=30
Lx=3.0e5
height=1e4
nlayers=10


m = IntervalMesh(nx, Lx)
mh = MeshHierarchy(m, refinement_levels=2)
hierarchy = ExtrudedMeshHierarchy(mh, height, base_layer= nlayers, refinement_ratio=1, extrusion_type='uniform')
mesh = hierarchy[-1]

CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
RT_e = RT_horiz + RT_vert
V = FunctionSpace(mesh, RT_e)
Vp = FunctionSpace(mesh, 'DG', 0)
Vb = FunctionSpace(mesh, P0P1)
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

n = FacetNormal(mesh)
k = as_vector([0,1])
Omega = 7.292e-5
theta = pi / 3
omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])
dtc = Constant(dt)
