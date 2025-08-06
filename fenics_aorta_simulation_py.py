from mpi4py import MPI
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.fem import FunctionSpace, Constant, dirichletbc, locate_dofs_topological, Function, NonlinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import VectorElement, FiniteElement, TrialFunction, TestFunction, split, grad, div, inner, dx, dot
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver

# --- Load mesh ---
with XDMFFile(MPI.COMM_WORLD, "aorta_volume_with_surface.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

# --- Function spaces ---
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
VQ = FunctionSpace(mesh, P2 * P1)
VQ_trial = TrialFunction(VQ)
VQ_test = TestFunction(VQ)
u, p = split(VQ_trial)
v, q = split(VQ_test)

# --- Physical parameters ---
f = Constant(mesh, PETSc.ScalarType((0, 0, 0)))
mu = Constant(mesh, 0.0035)
rho = Constant(mesh, 1060.0)

# --- Define nonlinear variational problem ---
U = Function(VQ)  # current guess
U.x.array[:] = np.random.rand(U.x.array.shape[0]) * 1e-3
u_, p_ = split(U)
F = (
    mu * inner(grad(u_), grad(v)) * dx
    + rho * inner(dot(grad(u_), u_), v) * dx
    - div(v) * p_ * dx
    - q * div(u_) * dx
    - dot(f, v) * dx
)

# --- Inlet BC ---
def inlet_boundary(x):
    return np.logical_and.reduce((x[0] >= -50, x[0] <= 50, x[1] >= -250, x[1] <= -150, x[2] >= -850, x[2] <= -750))

inlet_facets = locate_entities_boundary(mesh, 2, inlet_boundary)
u_inlet_val_x = Constant(mesh, 0.0)
u_inlet_val_y = Constant(mesh, 0.0)
u_inlet_val_z = Constant(mesh, 1.0)
V = VQ.sub(0)
bc_x = dirichletbc(u_inlet_val_x, locate_dofs_topological(V.sub(0), 2, inlet_facets), V.sub(0))
bc_y = dirichletbc(u_inlet_val_y, locate_dofs_topological(V.sub(1), 2, inlet_facets), V.sub(1))
bc_z = dirichletbc(u_inlet_val_z, locate_dofs_topological(V.sub(2), 2, inlet_facets), V.sub(2))
bcs = [bc_x, bc_y, bc_z]

# --- Outlet pressure BCs (all 12 outlets) ---
outlet_defs = [
    lambda x: np.logical_and.reduce((x[0] >= -120, x[0] <= -100, x[1] >= -180, x[1] <= -160, x[2] >= -650)),
    lambda x: np.logical_and.reduce((x[0] >= -40, x[0] <= 30, x[1] >= -200, x[1] <= -180, x[2] >= -630)),
    lambda x: np.logical_and.reduce((x[0] >= -25, x[0] <= -10, x[1] >= -170, x[1] <= -150, x[2] >= -655)),
    lambda x: np.logical_and.reduce((x[0] >= 0, x[0] <= 15, x[1] >= -170, x[1] <= -160, x[2] >= -650)),
    lambda x: np.logical_and.reduce((x[0] >= 30, x[1] >= -200, x[1] <= -160, x[2] >= -650)),
    lambda x: np.logical_and.reduce((x[0] >= -70, x[0] <= -30, x[1] >= -200, x[1] <= -150, x[2] >= -895, x[2] <= -850)),
    lambda x: np.logical_and.reduce((x[0] >= -60, x[0] <= -40, x[1] <= -213, x[2] >= -950, x[2] <= -915)),
    lambda x: np.logical_and.reduce((x[0] >= 33, x[0] <= 40, x[1] >= -160, x[1] <= -140, x[2] >= -900, x[2] <= -850)),
    lambda x: np.logical_and.reduce((x[0] >= 37, x[0] <= 45, x[1] >= -170, x[1] <= -160, x[2] >= -1000, x[2] <= -930)),
    lambda x: np.logical_and.reduce((x[0] >= -25, x[0] <= -15, x[1] >= -240, x[1] <= -230, x[2] >= -1000, x[2] <= -970)),
    lambda x: np.logical_and.reduce((x[0] >= -40, x[0] <= -20, x[1] >= -205, x[1] <= -195, x[2] >= -1000, x[2] <= -960)),
    lambda x: np.logical_and.reduce((x[0] >= -80, x[0] <= 80, x[1] >= -200, x[1] <= -160, x[2] >= -1200, x[2] <= -1100)),
]

outlet_facets_combined = set()
for i, outlet_fn in enumerate(outlet_defs):
    outlet_facets = locate_entities_boundary(mesh, 2, outlet_fn)
    outlet_facets_combined.update(outlet_facets)
    print(f"Outlet {i+1} facets: {len(outlet_facets)}")
    bc = dirichletbc(Constant(mesh, 0.0), locate_dofs_topological(VQ.sub(1), 2, outlet_facets), VQ.sub(1))
    bcs.append(bc)

# --- Wall (no-slip) BC ---
def all_boundary(x):
    return np.full(x.shape[1], True, dtype=bool)

all_boundary_facets = locate_entities_boundary(mesh, 2, all_boundary)
wall_facets = np.setdiff1d(all_boundary_facets, np.array(list(outlet_facets_combined) + list(inlet_facets)))

bc_wall_x = dirichletbc(Constant(mesh, 0.0), locate_dofs_topological(V.sub(0), 2, wall_facets), V.sub(0))
bc_wall_y = dirichletbc(Constant(mesh, 0.0), locate_dofs_topological(V.sub(1), 2, wall_facets), V.sub(1))
bc_wall_z = dirichletbc(Constant(mesh, 0.0), locate_dofs_topological(V.sub(2), 2, wall_facets), V.sub(2))
bcs.extend([bc_wall_x, bc_wall_y, bc_wall_z])

# --- Nonlinear solve ---
problem = NonlinearProblem(F, U, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-6
n, converged = solver.solve(U)
print(f"Newton iterations: {n}, Converged: {converged}")

# --- Split solution ---
u_sol = U.sub(0)
p_sol = U.sub(1)
u_sol.name = "velocity"
p_sol.name = "pressure"

# --- Check solution values ---
print("Velocity range:", u_sol.x.array.min(), u_sol.x.array.max())
print("Pressure range:", p_sol.x.array.min(), p_sol.x.array.max())

# --- Export results ---
with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as ufile:
    ufile.write_mesh(mesh)
    ufile.write_function(u_sol)

with XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w") as pfile:
    pfile.write_mesh(mesh)
    pfile.write_function(p_sol)
