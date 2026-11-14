# aerokit.topo — modular topology optimisation

Density-based topology optimisation on regular 2-D/3-D grids, written to be experimented with: every stage of the
pipeline is a small, replaceable object, derivatives come from PyTorch autograd (plus one hand-written adjoint for
the linear solve), and every new piece is checked against finite differences and against earlier validated drivers.

```
Domain ─► Chain (design → physical density) ─► Interpolation ─► Physics (K(E)u = f) ─► Responses ─► Optimizer ─► Study ─► Recorder ─► Post-processing
```

## Status

| Area | Module | State |
|---|---|---|
| Core: domain, chain (embed, density filter, tanh projection, passive), SIMP/RAMP, linear elasticity (H8, H8IM, Q4), compliance, volume, OC, Study loop (final design re-evaluated), run folders | `domain`, `chain`, `interp`, `fem/elastic`, `responses`, `optim/oc`, `study`, `record` | done; reproduces the repository's earlier 3-D voxel driver to ≤ 3e-11 and its Python port of top88 to 1.4e-11 (relative compliance history). These are regression oracles, not independent references: a comparison against the original DTU MATLAB code is still open |
| MMA (any number of constraints) | `optim/mma` | done; reproduces the earlier 2-D MMA driver to 2.4e-9 |
| Boundary conditions and loads: geometric selectors (bore faces with normal filter, planes, shapes), fixed / roller / symmetry / pinned bore / elastic supports, distributed point loads, traction, pressure, bearing, remote force & moment, torque, body force, load cases, per-case supports, rigid-body and resultant checks | `bc` | done |
| Geometry: CSG primitives, STL voxelisation with partial volumes, automatic bore detection | `geometry` | done |
| Length scale: robust (eroded / blueprint / dilated) minimum solid and void size, maximum thickness (local volume), grey level, thickness measurement | `robust`, `lengthscale` | done |
| Connectivity: one-body gate (thresholds, erosion, min-cut necks, enclosed voids), frozen minimum ties, conduction and coupling constraints, bridge repair, one-call plan `one_body()` | `connectivity`, `fem/heat` | done; on the splitting benchmark plain compliance TO gives 2 bodies, `one_body()` gives PASS |
| Manufacturing: AM overhang filter (±x/±y/±z), symmetry / extrusion / pattern tying, casting / milling pull direction | `manufacturing` | done |
| Eigenfrequency: consistent mass, differentiable eigenvalues, KS aggregation, two-range interpolation against spurious modes | `fem/modal`, `dynamics` | done |
| Post-processing: density → SDF, topology-preserving regularisation, exact functional faces, smoothing, mesh checks, T10 re-verification, export | `post/` | done; T10 mesh convergence on optimised parts not yet established |
| Declarative studies: YAML/JSON config, component registry (all feature modules), BC check in `validate`, CLI (`run`, `validate`, `show`, `compare`, `list`, sweeps) | `config`, `registry`, `cli` | done |
| Fast solver: matrix-free geometric multigrid (CUDA mixed precision, K-cycle, rigid-island deflation) with direct fallback, exact tolerance contract, fail-closed by default | `solvers/gmg`, `fem/elastic_gmg` | done; solver gate passed (`docs/olcumler/A2.md`; measured on a loaded machine, ratios are bounds) |

## Install and run

```bash
pip install -e .            # numpy, scipy, torch, pyyaml
pip install -e .[post,fast,mesh,dev]   # pyvista/vtk, pypardiso (MKL), gmsh, test tools
```

```bash
python -m aerokit.topo validate examples/topo/mbb2d.yaml
python -m aerokit.topo run examples/topo/mbb2d.yaml --out runs/mbb
python -m aerokit.topo run examples/topo/mbb2d.yaml --sweep chain.0.rmin=1.5,2.5 --sweep constraints.0.upper=0.4,0.5
python -m aerokit.topo compare runs/mbb runs/other
python -m aerokit.topo list
```

```python
import numpy as np
import aerokit.topo as T                      # import before torch (see below)

d = T.Domain((60, 20, 10), h=1.0)
X = d.node_coords()
fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
tip = np.flatnonzero((X[:, 0] == X[:, 0].max()) & (X[:, 2] == 0))
F = d.load_vector(); F[3 * tip + 2, 0] = -1.0 / len(tip)

chain = T.standard_chain(d, rmin=1.5, projection=True)
study = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(),
                [T.Constraint(T.Volume(), 0.3)], T.MMA(), T.BetaContinuation(chain, 1, 16, 25))
result = study.run(max_iter=150)
```

Boundary conditions by geometry instead of DOF numbers: `aerokit.topo.bc` (`BoundaryConditions`, `OnCylinder`,
`BearingLoad`, `RemoteForce`, `check()` / `summary()`); domains from shapes or STL: `aerokit.topo.geometry`.

## Notes

* **Import order (Windows, MKL):** torch and MKL PARDISO each ship `libiomp5md.dll`; loading MKL after torch aborts
  the process. `import aerokit.topo` loads PARDISO first; if torch was already imported, the direct solver falls back
  to SuperLU. Optional native packages can live in a separate folder given by `AEROKIT_PYLIB` (tests) and
  `AEROKIT_MKL_BIN` (MKL DLLs).
* **Threads:** under heavy CPU load, multithreaded MKL/torch can be 10–25× slower than single-threaded; set
  `torch.set_num_threads` accordingly.
* **Evidence:** measurements live in `docs/olcumler/`; the roadmap and decisions in `docs/mimari_plan.md`; how to add
  a component in `docs/registry.md`. Literature used in code is tied to keys of `aerokit.references` with `@cite`;
  docstrings state when a source was not read.

## Tests

```bash
python -m pytest tests/topo -m "not slow"
```

`tests/topo/topo_oracles/` holds frozen copies of earlier validated drivers used only as regression oracles.
Every differentiable link and response has a finite-difference gradient test.
