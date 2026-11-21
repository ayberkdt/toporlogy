# toporlogy

**Topology optimisation you can take apart.** Density-based structural optimisation on regular 2-D and 3-D grids,
built from small, replaceable pieces: geometry and boundary conditions, a differentiable design chain, linear FEM
with an exact adjoint, four optimisers (OC, MMA, GCMMA, SiMPL), stress, buckling, eigenfrequency, manufacturing and
connectivity constraints, a GPU multigrid solver, and a post-processing pipeline that turns the density field into a
checked, printable surface.

Derivatives come from PyTorch autograd plus one hand-written adjoint for the linear solve. A new objective or
constraint is therefore a forward computation and a finite-difference test, nothing more
([`docs/deriving_methods.md`](docs/deriving_methods.md) shows four worked derivations, all executed by a test).

> **Proprietary software — all rights reserved.** No licence to use, copy, modify or distribute this code is
> granted; see [LICENSE](LICENSE). Access to the repository does not imply permission.

| | |
|:---|:---|
| Version | 0.2.0 (research code; interfaces may change) |
| Python | ≥ 3.11 |
| Core dependencies | numpy, scipy, torch, pyyaml |
| Developed and tested on | Windows 11, CPU; CUDA for the multigrid solver |

**Contents:** [Install](#install) · [Quickstart](#quickstart) · [Configuration](#configuration) ·
[Python API](#python-api) · [What is inside](#what-is-inside) · [Methods of 21 Nov 2026](#methods-added-on-21-nov-2026) ·
[One body](#keeping-the-part-in-one-body) ·
[Evidence](#evidence) · [Limitations](#known-limitations) · [Development](#development) · [Licence](#licence)

## Install

```bash
pip install -e .                       # core: numpy, scipy, torch, pyyaml
pip install -e ".[post,fast,mesh,dev]" # everything
```

| Extra | Adds | Needed for |
|:---|:---|:---|
| `post` | pyvista, vtk, numba | surfaces, manufacturing checks, STL/PLY/OBJ export |
| `fast` | pypardiso | MKL PARDISO direct solver (SuperLU otherwise) |
| `mesh` | gmsh | tetrahedral (T10) re-analysis of the finished part |
| `plot` | matplotlib | plots |
| `dev` | pytest, pytest-timeout, hypothesis, mypy, ruff | tests and static checks |

On Windows, torch and MKL each ship `libiomp5md.dll`; loading MKL after torch aborts the process. `import toporlogy`
loads PARDISO first, so import it **before** torch. If torch came first, the direct solver falls back to SuperLU.

## Quickstart

```bash
python -m toporlogy validate examples/mbb2d.yaml     # supports, loads, resultants, rigid-body modes
python -m toporlogy run examples/mbb2d.yaml --out runs/mbb
```

`run` prints one line per iteration and ends with a `RESULT` line (status, objective, constraints, iterations,
time, grey fraction). The run folder holds everything needed to reproduce and inspect it:

| File | Content |
|:---|:---|
| `config.resolved.yaml` | the config with every default filled in, plus git revision and overrides |
| `setup.json` | environment (Python, NumPy, torch, machine) and study settings |
| `history.jsonl` | one row per iteration: objective, constraints, design change, schedule state |
| `result.npz` | final design vector `x` and physical density `xphys` |
| `result.json` | iterations, convergence flag, time, objective, constraint values, settings |
| `summary.json` | objective and constraints of the returned design, post-processing results, status |

A real part, from primitives to a printable surface:

```bash
python -m toporlogy run examples/lbracket_bore3d.yaml --out runs/lbracket
```

That example builds an L-bracket by CSG, finds its bore, applies a bearing load, enforces a minimum and a maximum
member size, keeps the part in one body, and exports a smoothed STL with a quality report.

| Command | Purpose |
|:---|:---|
| `run CONFIG [--set key.path=value] [--sweep key.path=v1,v2]` | one run, or every combination of a sweep (`sweep.csv`) |
| `validate CONFIG` | build the study and check the boundary conditions without optimising |
| `show RUN_DIR` | summary of a finished run |
| `compare RUN_DIR...` | several runs side by side |
| `list [KIND]` | every registered component with its parameters and units |

Exit codes: 0 ok, 1 run finished but failed a check (not converged, constraint violated, a post step failed),
2 configuration error.

## Configuration

A study is one YAML file. The half MBB beam (`examples/mbb2d.yaml`):

```yaml
name: mbb2d
domain: {type: box, shape: [60, 20], h: 1.0}
material: {E0: 1.0, nu: 0.3}
supports:
  - {type: fixed, where: {type: plane, axis: x, at: min}, dofs: x}        # symmetry line
  - {type: fixed, where: {type: point, at: [max, min]}, dofs: y}          # roller at the bottom right
load_cases:
  - name: centre
    loads:
      - {type: force, where: {type: point, at: [min, max]}, force: [0, -1]}
chain:
  - {type: density_filter, rmin: 2.0}
interpolation: {type: simp, penal: 3.0, emin_ratio: 1.0e-9}
objective: compliance
constraints:
  - {response: volume, upper: 0.5}
optimizer: {type: oc, move: 0.2, bisection_tol: 1.0e-3}
run: {max_iter: 300, tol: 0.01}
```

Seven examples cover 2-D and 3-D compliance, multiple load cases, eigenfrequency maximisation, symmetry with an
additive-manufacturing filter, connectivity and the full post-processing chain (`examples/`); five more in
`examples/research/` cover stress constraints (aggregated and augmented Lagrangian), buckling, a compliant inverter
and SiMPL with a PDE filter. 95 components in 13 kinds are available from YAML; `python -m toporlogy list` documents them and
[`docs/registry.md`](docs/registry.md) shows how to add one in a single line.

## Python API

```python
import numpy as np
import toporlogy as T            # before torch: loads MKL PARDISO first

d = T.Domain((60, 20, 10), h=1.0)
X = d.node_coords()
fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
tip = np.flatnonzero((X[:, 0] == X[:, 0].max()) & (X[:, 2] == 0))
F = d.load_vector(); F[3 * tip + 2, 0] = -1.0 / len(tip)

chain = T.standard_chain(d, rmin=1.5)
study = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(),
                [T.Constraint(T.Volume(), 0.3)], T.MMA(), T.BetaContinuation(chain, 1, 16, 25))
result = study.run(max_iter=150)
result.objective, result.constraint_values, result.xphys   # all of the returned design
```

Each `run()` starts from the schedule's initial state and a fresh optimizer, so running a study twice gives the
same result. Boundary conditions by geometry instead of DOF numbers: `toporlogy.boundary` (`BoundaryConditions`,
`OnCylinder`, `BearingLoad`, `RemoteForce`, `check()`). Surfaces: `toporlogy.post.finalize(...)`.

## What is inside

The library follows the optimisation pipeline; every stage is a module that can be used alone or replaced. Every
module, class and config component has one name; there are no alias modules or second names for old paths.

| Stage | Module | Provides |
|:---|:---|:---|
| Geometry | `geometry` | CSG primitives, STL voxelisation with partial volumes, bore detection |
| Boundary conditions | `boundary` | 18 geometric selectors; 5 support types (fixed, roller, symmetry, pinned bore, springs); nodal and 9 distributed load types incl. bearing and remote loads; load cases and combinations; resultant and rigid-body checks |
| Design chain | `design` (`chain`, `pde_filter`, `interp`, `manufacturing`) | cone density filter, Helmholtz (PDE) filter, tanh projection, keep/void regions; AM overhang filter, symmetry, extrusion and pattern tying, casting/milling pull direction |
| Physics | `fem`, `solvers` | H8, H8IM and Q4 elasticity, heat conduction, modal analysis, linear buckling; PARDISO/SuperLU, or matrix-free multigrid on CUDA with a direct fallback |
| Responses | `optimization` (`responses`, `robust`, `lengthscale`, `connectivity`, `dynamics`, `stress`, `buckling`, `compose`) | compliance, volume; robust minimum solid/void size; maximum thickness; one-body constraints; eigenfrequencies; von Mises stress; buckling load factor; displacement; sums and ratios of responses |
| Method toolkit | `optimization` (`aggregates`, `local`, `benchmark`) | six smooth aggregates with proven bounds; any local criterion as one aggregated constraint or as an augmented Lagrangian; evaluation of the thresholded design and comparison tables |
| Optimisation | `optimizers`, `optimization.study` | OC (one constraint), MMA and GCMMA (any number), SiMPL (one volume-type constraint), β continuation, multiplier updates, move-limit control; run folders with git revision |
| Post-processing | `post` | density → signed distance → topology-preserving opening/closing → exact functional faces → smooth surface → manufacturing checks → T10 re-analysis → STL/PLY/OBJ |

## Methods added on 21 Nov 2026

Stress, buckling, two optimizers, a PDE filter and a toolkit for deriving and comparing methods. Sources: the
network policy of the session that added them blocked publisher and arXiv pages, so **no paper text was read**;
every docstring says whether a formula was derived and verified here, read from public reference code, or written
from memory. The literature survey, the gaps it found and the verification status of every citation are in
[`docs/literatur_2026-11-21.md`](docs/literatur_2026-11-21.md); all numbers below are in
[`docs/olcumler/yontemler.md`](docs/olcumler/yontemler.md), measured on small 2-D problems under machine load.

| Addition | Verified by | Measured |
|:---|:---|:---|
| von Mises stress field, qp and polynomial relaxation (`optimization.stress`) | uniform tension exact to 1e-9 (2-D, 3-D); gradients ≤ 1e-6 | pure bending: centroid stress of Q4 1.8 % off |
| Aggregated constraint and augmented Lagrangian for any local field (`optimization.local`) | gradients ≤ 1e-6; small problem with known multipliers | L-bracket: both meet the relaxed constraint on the grey design; the thresholded design exceeds the limit by 71 % (aggregated) and 2.5 % (augmented Lagrangian) |
| Linear buckling (`fem.buckling`, `optimization.buckling`), sensitivity by autograd through a Rayleigh quotient | Euler column 2-D 1.125 → 1.031 → 1.007 → 1.0015 with refinement, 3-D H8IM 0.9975; gradients 8.5e-9 / 2.6e-9 | constraint λ₁ ≥ 0.02 raises λ₁ by 46 % for 21 % more compliance |
| GCMMA (`optimizers.gcmma`) | every accepted step conservative; no 2-cycle at an interior optimum | MBB: same result as MMA at twice the solves |
| SiMPL (`optimizers.simpl`) | known optima to 1e-10; descent with line search | MBB: within 0.1 % of MMA; without line search at the same number of solves |
| Helmholtz filter (`design.pde_filter`) | volume preserving, bounded, constants reproduced at walls; gradient ≤ 1e-6 | kernel variance equals the cone filter's only in 1-D: 11 % larger in 2-D, 25 % in 3-D |
| 0/1 evaluation (`optimization.benchmark`, post step `discrete_check`) | exact volume, deterministic | MBB: binary compliance 11–14 % below the grey value; ranking of methods changes |

The augmented Lagrangian did not work with its first settings (optimizer reset at every multiplier update) and its
present defaults were tuned on one problem; neither stress strategy yields a stress-feasible 0/1 structure.

## Keeping the part in one body

A compliance optimum often splits into separate load paths joined by nothing, or by a grey film that disappears at
thresholding. `optimization.connectivity.one_body()` stacks three defences: frozen minimal ties between the interfaces, a
conduction constraint on the eroded field, and a geometric gate that re-checks the result at several thresholds and
after erosion, with an optional bridge repair.

**On 14 pre-registered scenarios, plain optimisation kept none in one body; the default plan kept all 14**
([`docs/olcumler/robustness.md`](docs/olcumler/robustness.md)).

| Method | PASS | FRAGILE | FAIL | Median compliance vs plain |
|:---|---:|---:|---:|---:|
| `one_body()` default | 14 | 0 | 0 | 1.30 |
| Frozen ties only | 14 | 0 | 0 | 1.25 |
| Conduction constraint only | 0 | 13 | 1 | 1.65 |
| Plain compliance | 0 | 2 | 12 | 1.00 |

PASS: one body holding every interface at thresholds 0.3/0.5/0.7 and after a one-element erosion. Four scenarios
carry a large compliance penalty (26–53×) that has not been analysed. The bridge repair adds material without
re-optimising: repaired designs measured 1.07–2.61 × the volume bound. The gate fails a repair that breaks a constraint,
and one whose constraints cannot all be evaluated: robust and eroded-field responses need the design chain, which a
repaired density does not have (on `lbracket_bore3d` none of the constraints can be evaluated).

## Evidence

Every claim names the test or measurement behind it. The regression oracles are earlier drivers of this repository:
they show that the implementation did not change, not that it is independently correct.

| Check | Reference | Result | Source |
|:---|:---|:---|:---|
| 3-D optimisation history | earlier voxel driver | ≤ 3.4e-11 relative, 7 scenarios | `tests/toporlogy/test_regression.py` |
| 2-D half-MBB history | repository's Python port of top88 | 1.4e-11 relative | `tests/toporlogy/test_regression.py` |
| Gradients of every link and response | central finite differences | ≤ 1e-6 relative | `tests/toporlogy/test_gradients.py` |
| Elements H8, H8IM, Q4, T10 | patch test, uniform tension | exact to 1e-12 (T10: 1e-9) | `tests/toporlogy/test_core.py`, `test_post_verify.py` |
| First bending frequency | Euler–Bernoulli cantilever | 0.997–1.001 (H8IM) | `tests/toporlogy/test_modal.py` |
| Multigrid solver | PARDISO, same system | 7.2× faster at 163 k DOF; 10⁶ elements to 1e-8 | `docs/olcumler/A2.md` |
| Post-processing, 12 inputs | pre-registered criteria | 9/12 pass the geometric criteria, 7/12 all criteria incl. exact-solid volume and T10 mesh convergence; 12/12 watertight with topology preserved | `docs/olcumler/robustness.json`, `tests/toporlogy/test_sweep_post.py` |
| Reproducibility | two runs in separate processes | the seven examples bitwise identical (CPU, SuperLU) | `tests/toporlogy/test_baseline.py` |
| Stress, buckling, GCMMA, SiMPL, PDE filter | closed forms, invariants, finite differences | table above | `tests/toporlogy/test_stress.py`, `test_buckling.py`, `test_gcmma.py`, `test_simpl.py`, `test_pde_filter.py`, `test_aggregates.py` |

## Known limitations

- **No independent reference yet:** no comparison with the original DTU top88/Top3D codes or NAFEMS benchmarks.
- **T10 mesh convergence is not shown:** on the cantilever the T10/voxel compliance ratio changed by 12.4 % between
  the two tet sizes; on the MBB the finer tet size barely refined the mesh.
- **`Result.converged`** means "design change below `tol` and continuation finished"; it does not include constraint
  feasibility or optimality. Check `summary.json` → `status` and the constraint values.
- **No paper text behind the methods of 21 Nov 2026:** records and abstracts only; SiMPL and the augmented
  Lagrangian are this library's implementations of the published ideas, not verified reproductions.
- **Stress:** centroid stress of a voxel model (inaccurate at jagged boundaries, under-reported in grey elements);
  no regional aggregation, no robust (eroded/dilated) stress, no body-fitted check in the loop. Results satisfy the
  relaxed constraint on the grey design, not the stress limit on the thresholded one.
- **Buckling:** plain `Elasticity` only (direct solver, axis-aligned supports, one support set); no regularisation
  of the stress field against localised modes; the number of aggregated modes is fixed by the user.
- **Baseline on another platform:** in numerical mode on Linux / Python 3.13 six of the seven examples agree with
  the Windows reference to 1e-6; `lbracket_bore3d` differs by 0.6 % in the objective (measured 21 Nov 2026 on the
  unchanged code of `5f97b86`, cause not investigated).
- **MMA** is plain MMA: no global convergence guarantee; a two-cycle has been seen without an active constraint.
  GCMMA removes it at the cost of extra solves.
- **OC and SiMPL** handle exactly one constraint, and only one that needs no state solve (rejected at build
  otherwise).
- **Multigrid** is 3-D only, H8/H8IM only, and a separate physics class (`ElasticityGMG`), not a pluggable solver.
- **Geometry:** bore detection is axis-aligned only; CSG Booleans give a distance bound, not the exact distance.
- **Not provided:** thermoelasticity, nonlinear analysis, multi-material, de-homogenisation, neural
  parameterisations; STEP or other B-rep output (STL/PLY/OBJ only).
- **RAMP** interpolation is experimental and used by no default.

The full inventory, the open findings and the migration plan are in
[`docs/gelistirme_plani.md`](docs/gelistirme_plani.md); the method decisions in
[`docs/mimari_plan.md`](docs/mimari_plan.md); the literature survey and open problems in
[`docs/literatur_2026-11-21.md`](docs/literatur_2026-11-21.md).

## Development

```bash
python -m pytest tests/toporlogy -m "not slow"                  # main suite (5–10 min; 521 tests on 21 Nov 2026)
python -m pytest tests/toporlogy/test_baseline.py -m baseline   # bitwise baseline of the examples (~3.5 min)
python -m pytest tests/toporlogy -m slow                        # pre-registered sweeps (hours; gmsh for T10)
```

Guard rails for refactoring (`tests/toporlogy`):

| Test | Fails when |
|:---|:---|
| `test_baseline.py` | any example's objective, constraints, design or exported surface changes by one bit |
| `test_api_surface.py` | a public name disappears or points to another object; `toporlogy list` output changes |
| `test_architecture.py` | an import crosses the layer rules, or importing toporlogy loads torch before PARDISO |

After an intended change, rewrite the snapshot and say why in the commit:
`python tests/toporlogy/test_baseline.py --write`, `python tests/toporlogy/test_api_surface.py --write`.

- `tests/toporlogy/topo_oracles/` holds frozen copies of earlier drivers, used only as regression oracles.
- Optional native packages outside site-packages are found through `TOPORLOGY_PYLIB` (folders separated by the
  path separator), the MKL DLLs through `TOPORLOGY_MKL_BIN`.
- Under heavy CPU load, multithreaded MKL and torch can be 10–25× slower than single-threaded; set
  `torch.set_num_threads` in long runs.
- Static checks: `ruff check src tests` (clean); `mypy src/toporlogy` (existing errors; a change must not add any).

## Licence

Copyright © 2026 Ayberk. All rights reserved. This is proprietary software: no permission is granted to use, copy,
modify or distribute it without prior written permission. See [LICENSE](LICENSE).
