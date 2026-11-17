# toporlogy

**Topology optimisation you can take apart.** Density-based structural optimisation on regular 2-D and 3-D grids,
built as small replaceable pieces: geometry and boundary conditions, a differentiable design chain, linear FEM with an
exact adjoint, OC/MMA optimisers, manufacturing and connectivity constraints, a GPU multigrid solver, and a
post-processing pipeline that turns the density field into a checked, printable surface.

Derivatives come from PyTorch autograd plus one hand-written adjoint for the linear solve, so a new objective or
constraint is a forward computation and a finite-difference test, nothing more.

## Quickstart

```bash
git clone https://github.com/ayberkdt/toporlogy && cd toporlogy
pip install -e .                                  # numpy, scipy, torch, pyyaml
python -m toporlogy run examples/mbb2d.yaml --out runs/mbb
```

The run prints one line per iteration and ends with a `RESULT` line; the folder holds the resolved config, the
history and the final design. Then try a real part:

```bash
python -m toporlogy validate examples/lbracket_bore3d.yaml     # checks supports, loads, rigid-body modes
python -m toporlogy run examples/lbracket_bore3d.yaml --out runs/lbracket
```

That example builds an L-bracket from primitives, finds its bore, applies a bearing load, enforces a minimum and a
maximum member size, keeps the part in one body, and exports a smoothed STL with a quality report.

Optional extras: `pip install -e ".[post,fast,mesh,dev]"` adds pyvista/vtk/numba (surfaces), pypardiso (MKL direct solver),
gmsh (tetrahedral re-verification) and the test tools.

## What is inside

The library follows the optimisation pipeline; every stage is a module you can use alone or replace.

| Stage | Module | What it gives you |
|:---|:---|:---|
| Geometry | `geometry` | CSG primitives and STL voxelisation with partial volumes; automatic bore detection |
| Boundary conditions | `bc` | Selectors by geometry (bore faces, planes, shapes); 5 support types; 9 distributed load types incl. bearing and remote loads; load cases; resultant and rigid-body checks |
| Design chain | `chain`, `manufacturing` | Density filter, tanh projection, keep/void regions; AM overhang filter (any axis), symmetry, extrusion and pattern tying, casting/milling pull direction |
| Physics | `fem`, `solvers` | H8, H8IM and Q4 elasticity, heat conduction, modal analysis; PARDISO/SuperLU, or matrix-free multigrid on CUDA with direct fallback |
| Responses | `responses`, `robust`, `lengthscale`, `connectivity`, `dynamics` | Compliance, volume; robust minimum solid/void size; maximum thickness; one-body constraints; eigenfrequencies |
| Optimisation | `optim`, `study` | OC, MMA with any number of constraints, β continuation, move-limit control; run folders with git revision |
| Post-processing | `post` | Density → signed distance → topology-preserving fillets → exact functional faces → smooth surface → manufacturing checks → T10 re-check → STL/PLY |

Everything is also reachable from YAML: `python -m toporlogy list` shows every registered component with its
parameters and units; `docs/registry.md` explains how to add one.

## Keeping the part in one body

A compliance optimum often splits into separate load paths that are joined by nothing, or by a grey film that
disappears at thresholding. `connectivity.one_body()` stacks three defences: frozen minimal ties between the
interfaces, a conduction constraint on the eroded field, and a geometric gate that re-checks the result at several
thresholds and after erosion (with an optional bridge repair).

On 14 pre-registered scenarios plain optimisation passed none; the default plan passed all
(`docs/olcumler/robustness.md`).

| Method | Scenarios | PASS | FRAGILE | FAIL | Median compliance vs plain |
|:---|---:|---:|---:|---:|---:|
| Plain compliance | 14 | 0 | 2 | 12 | 1.00 |
| `one_body()` default | 14 | 14 | 0 | 0 | 1.30 |
| Frozen ties only | 14 | 14 | 0 | 0 | 1.25 |
| Conduction constraint only | 14 | 0 | 13 | 1 | 1.65 |

PASS means one body holding every interface at thresholds 0.3/0.5/0.7 and after a one-element erosion. Four
scenarios carry a large compliance penalty (26–53×); they are not yet analysed.

## Evidence

Each claim below links to the test or measurement that produced it. Regression oracles are earlier drivers of the
same repository, so they show the implementation is unchanged, not that it is independently correct.

| Check | Reference | Result | Source |
|:---|:---|:---|:---|
| 3-D optimisation history | earlier voxel driver (regression oracle) | ≤ 3.4e-11 relative, 7 scenarios | `tests/toporlogy/test_regression.py` |
| 2-D half-MBB history | repository's Python port of top88 | 1.4e-11 relative | `tests/toporlogy/test_regression.py` |
| Gradients of every link and response | central finite differences | ≤ 1e-6 relative | `tests/toporlogy/test_gradients.py` |
| Elements H8, H8IM, Q4, T10 | patch test, uniform tension | exact to 1e-12 (T10: 1e-9) | `tests/toporlogy/test_core.py`, `test_post_verify.py` |
| First bending frequency | Euler–Bernoulli cantilever | 0.997–1.001 (H8IM) | `tests/toporlogy/test_modal.py` |
| Multigrid solver | PARDISO, same system | 7.2× faster at 163 k DOF; 1 M elements solved to 1e-8 | `docs/olcumler/A2.md` |
| Post-processing, 12 inputs | pre-registered criteria | 9/12 pass all; 12/12 watertight with topology preserved | `docs/olcumler/robustness.md` |

Not yet done: comparison with the original DTU top88/Top3D codes, NAFEMS benchmarks, large benchmark sets, T10 mesh
convergence on optimised parts, three-dimensional stress constraints, and continuous integration. The plan is in
`docs/mimari_plan.md`.

## Python API

```python
import numpy as np
import toporlogy as T            # import before torch: it loads MKL PARDISO first

d = T.Domain((60, 20, 10), h=1.0)
X = d.node_coords()
fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
tip = np.flatnonzero((X[:, 0] == X[:, 0].max()) & (X[:, 2] == 0))
F = d.load_vector(); F[3 * tip + 2, 0] = -1.0 / len(tip)

chain = T.standard_chain(d, rmin=1.5)
study = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(),
                [T.Constraint(T.Volume(), 0.3)], T.MMA(), T.BetaContinuation(chain, 1, 16, 25))
result = study.run(max_iter=150)   # result.objective, result.xphys belong to the returned design
```

Boundary conditions by geometry instead of DOF numbers: `toporlogy.bc` (`BoundaryConditions`, `OnCylinder`,
`BearingLoad`, `RemoteForce`, `check()`). Surfaces: `toporlogy.post.finalize(...)`.

## Development

```bash
python -m pytest tests/toporlogy -m "not slow"
```

* `tests/toporlogy/topo_oracles/` holds frozen copies of earlier drivers used only as regression oracles.
* Optional native packages outside site-packages are found through `TOPORLOGY_PYLIB` (folders separated by the
  path separator); MKL DLLs through `TOPORLOGY_MKL_BIN`.
* On Windows, torch and MKL each ship `libiomp5md.dll`: import `toporlogy` before torch, otherwise the direct solver
  falls back to SuperLU.
* Under heavy CPU load, multithreaded MKL and torch can be 10–25× slower than single-threaded; set
  `torch.set_num_threads` in long runs.

The repository has no licence file yet, so no reuse rights are granted by default.
