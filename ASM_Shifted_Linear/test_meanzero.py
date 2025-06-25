from firedrake import *

# Basic Parameters 
nx=50
ny=50
Lx=10
Ly=1.0*Lx # TODO: horizontal AR issue happening.
height=10
nlayers=10

# Setting up the extruded mesh
# m = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='both', quadrilateral=True)
# m = UnitIntervalMesh(nx)
m = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(m, nlayers, layer_height=height/nlayers, extrusion_type='uniform')

DG = FunctionSpace(mesh, 'DQ', 1)
pcg = PCG64(seed=1234567)
rg = Generator(pcg)
f = rg.normal(DG, 1.0, 2.0)

f_int = assemble(f*dx)
One = Function(DG).assign(1.0)
area = assemble(One*dx)
f.project(f - f_int/area)
print(assemble(f*dx))