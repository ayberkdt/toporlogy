<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" width="100%"
       alt="toporlogy: topology optimisation you can take apart. A cantilever designed by toporlogy, clamped on the left and loaded at the right, coloured by strain energy density, next to the same run at iterations 6, 18 and 45.">
</picture>

**toporlogy** is density-based structural optimisation on regular 2-D and 3-D grids, built from small, replaceable
pieces: geometry and boundary conditions, a differentiable design chain, linear FEM with an exact adjoint, four
optimisers (OC, MMA, GCMMA, SiMPL), stress, buckling, eigenfrequency, manufacturing and connectivity constraints, a
GPU multigrid solver, and a post-processing pipeline that turns the density field into a checked, printable surface.
The banner is one of its results ([`docs/assets/make_banner.py`](docs/assets/make_banner.py) re-runs and redraws
it).

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
| `summary.json` | objective and constraints of the returned design, each constraint's verdict (PASS / FAIL / UNVERIFIED) and the source of its value, post-processing results, status |

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

Exit codes: 0 ok, 1 run finished but failed a check (not converged, constraint violated or unverified, objective
unverified, a post step failed), 2 configuration error. Only a value computed on the returned design can PASS or
FAIL; a value read from an older run folder or from the last iterate is UNVERIFIED and never shown as "ok".

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
and SiMPL with a PDE filter. 96 components in 13 kinds are available from YAML; `python -m toporlogy list` documents them and
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
| Physics | `fem`, `solvers` | H8, H8IM, Q4 and Q4IM elasticity (any material D, per element if needed), geometrically non-linear elasticity (Total Lagrangian), heat conduction, modal analysis, linear buckling; PARDISO/SuperLU (nested dissection order on the grid), or matrix-free multigrid on CUDA with a direct fallback |
| Responses | `optimization` (`responses`, `robust`, `lengthscale`, `connectivity`, `dynamics`, `stress`, `buckling`, `compose`, `nonlinear`) | compliance, end compliance (large displacements), volume, phase volume, mass, cost; robust minimum solid/void size; maximum thickness; one-body constraints; eigenfrequencies; von Mises stress; buckling load factor; displacement; sums and ratios of responses |
| Method toolkit | `optimization` (`aggregates`, `local`, `benchmark`) | six smooth aggregates with proven bounds; any local criterion as one aggregated constraint or as an augmented Lagrangian; evaluation of the thresholded design and comparison tables |
| Optimisation | `optimizers`, `optimization.study` | OC (one constraint), MMA and GCMMA (any number), SiMPL (one volume-type constraint), β continuation, multiplier updates, move-limit control; run folders with git revision |
| Post-processing | `post` | density → signed distance → topology-preserving opening/closing → exact functional faces → smooth surface (optionally clipped to the exact CAD envelope, `envelope: cad`) → manufacturing checks → T10 re-analysis with a mesh-convergence verdict → STL/PLY/OBJ; symmetry report of the design and the surface (`symmetry_check`) |

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

## Material models (25 Nov 2026)

The `material` section now describes more than E0 and ν. Measurements and the verification record:
[`docs/olcumler/malzeme_modelleri.md`](docs/olcumler/malzeme_modelleri.md).

```yaml
material:
  E0: 40000.0                       # with orthotropic: must equal E1 (the modulus the interpolation scales)
  plane: stress                     # 2-D only: stress | strain
  orthotropic: {E1: 40000, E2: 9000, E3: 9000, G12: 4000, G13: 4000, G23: 3200,
                nu12: 0.28, nu13: 0.28, nu23: 0.4, angle: 30}        # or axes: [[1,0,0],[0,1,0]] in 3-D
  strength: {Xt: 1500, Xc: 1200, Yt: 50, Yc: 200, S12: 70, safety_factor: 1.5}
  # isotropic strength: {tension: 300, compression: 450, safety_factor: 1.5}
constraints:
  - {response: {type: stress, criterion: tsai_wu}}                   # limit from material.strength
  - {response: {type: buckling, load_factor: 2.0, preload_case: 1}}  # case 1 held fixed, case 0 scaled
```

| Addition | Verified by |
|:---|:---|
| Orthotropic, rotated and plane-strain elasticity (`core.material`, `fem.elements.element_matrix_from_D`, `elasticity=` on `Elasticity`, `Modal`, `ElasticityGMG`) | isotropic case = the closed forms to 1e-14; rotation invariance and 1/E(n) = n n n n S; patch identities for any D; 2-D plane strain = 3-D slab with w = 0 to 1e-10; orthotropic strip gives E1, E2 exactly |
| Failure criteria von Mises, Tresca, Drucker–Prager, Raghava, Tsai–Wu (`optimization.strength`), field `FailureIndex` | index 1 at the uniaxial / shear / material-axis reference states; degree-1 homogeneous; rotation invariant (isotropic); von Mises field reproduced to 1e-12; gradients ≤ 3e-8 |
| Buckling: preload, cluster-cut warning, low-density mode share, first-failure load factor | preload c on the same pattern lowers λ by exactly c (1e-8); preload gradient 1.5e-8; square column double eigenvalue flagged; uniaxial first failure exact; orthotropic column → isotropic E_axial column, error 2.1 %, 0.59 %, 0.21 % at 4, 8, 16 elements across |
| 2-D incompatible-mode element `Q4IM` (Wilson's Q6); condensed modes in K_G (`LinearBuckling(modes_in_kg=True)`) | pure bending exact (Q4 −11 %); column of Ferrari & Sigmund 2019: Q4 20 × 2 reproduces their published error 0.119 (0.1187), Q4IM λ error 1.4e-3 against the converged 2-D value; gradient 5e-8 |
| Buckling localisation measure and `exclude` (no stress stiffness at singular points) | the published wall's λ1 does not converge (2.85 → 1.20 from 40² to 320²) because its mode sits at a singular stress; the measure rises with refinement there (0.88 → 0.95), 0.04 for an Euler column |
| Hashin–Shtrikman upper bound of a porous solid (`interpolation: hashin_shtrikman`), `simp_admissible_penal` | bulk bound = Lamé solution of a hollow sphere / disk to 1e-12; SIMP below both bounds exactly from p*(ν) on |

For ν = 0.3 the HS-admissible SIMP exponent is 3.08 in 2-D and 2.63 in 3-D; p = 3 is not admissible in 2-D at
ν = 0.3 (only at ν = 1/3). The HS interpolation does not penalise grey material: use it to judge a grey design.

## Physics added on 25 Nov 2026 (second round)

Four additions, each with its own pre-registered record. Three were developed in parallel on separate branches and
merged; the combinations the branches could not test alone have their own tests (`tests/toporlogy/test_merged_physics.py`,
`test_nonlinear_orientation.py`).

```yaml
material:
  E0: 40.0
  orthotropic: {E1: 40, E2: 9, E3: 9, G12: 4, G13: 4, G23: 3.2, nu12: 0.28, nu13: 0.28, nu23: 0.4,
                angle_field: fibre_angle.npy}          # degrees per element (2-D); axis_field in 3-D
  phases: [{name: soft, E: 10, density: 0.6, cost: 1}, {name: stiff, E: 40, density: 1.0, cost: 4}]
interpolation: multiphase_simp
physics: {type: nonlinear_elasticity, material: svk}   # or neo_hooke (isotropic)
objective: end_compliance
constraints:
  - {response: {type: phase_volume, phases: stiff}, upper: 0.2}
```

| Addition | Verified by | Record |
|:---|:---|:---|
| Per-element material orientation (`angle_field`, `axis_field`; `core.material.unit_elasticity_field`), accepted by `Elasticity`, `Modal`, `LinearBuckling`, `FailureIndex` and the T10 re-analysis | a uniform field reproduces the single-D path bit for bit; two-orientation strip in parallel / series = Voigt / Reuss closed forms to 1e-14; T10 and H8 off-axis bar E(θ) to 4e-15; gradients ≤ 2e-7 | [`eleman_yonu.md`](docs/olcumler/eleman_yonu.md) |
| Several solid phases + void sharing one unit D (`material.phases`, `interpolation: multiphase_simp`, the recursive SIMP of Gangl et al. 2026 §6.1); responses `phase_volume`, `mass`, `cost`; post step `phase_report` | one phase = SIMP bit for bit; two phases with the same E = one material; two-material bar in series / parallel to 1e-14; gradients ≤ 1.6e-7; no published compliance value to compare with (NOT VERIFIED) | [`cok_malzeme.md`](docs/olcumler/cok_malzeme.md) |
| Geometrically non-linear static analysis (`physics: nonlinear_elasticity`: Total Lagrangian, Q4 / H8, St. Venant–Kirchhoff with any D — also per element — or compressible neo-Hooke; energy interpolation for void elements) and `end_compliance` | uniaxial tension closed form 2e-14; linear solver at small load 1.4e-9; cantilever elastica −0.13 % (Richardson 0.04 %); tangent 1.9e-10; design gradient 1.2e-9; one pre-registered Newton-order check (neo-Hooke) failed, quadratic error contraction shown post hoc | [`geometrik_dogrusal_olmama.md`](docs/olcumler/geometrik_dogrusal_olmama.md) |
| Nested dissection elimination order for SuperLU on SPD grid systems (`solvers.ordering`; `Elasticity(ordering=...)`, heat conduction, modal analysis with σ ≤ 0, the preloaded buckling matrix) | same solution as the previous MMD order (compliance ≤ 6e-11, U ≤ 1e-10), also with partial volumes, keep/void and oblique supports; same gradient (8e-14); the seven examples within 1.7e-10 of the old baseline; modal and preloaded buckling eigenvalues within 9e-15 of the old order (σ > 0 keeps it) | [`cozucu_siralama.md`](docs/olcumler/cozucu_siralama.md), [`dogrulama_turu.md`](docs/olcumler/dogrulama_turu.md) §1.6 |

The direct solve was measured before the solver change: backward error ≤ 5e-17, one step of iterative refinement
changes the compliance by at most 1e-12, and the condition number (1e8–1e11 with Emin = 1e-6 … 1e-9) drops to 1e6
under Jacobi scaling, so its size is scale, not instability. The time is in the factorisation; the nested dissection
order cuts the fill by 26–33 % on the larger grids and the factorisation time by 2.1× at 53 k DOF (3-D), 2.8× at 90
k DOF and 2.5× at 241 k DOF (2-D). The same order makes the modal factorisation 2.6× faster at 53 k DOF (3-D) and the whole
preloaded buckling solve 1.3–1.5× faster.

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

On a symmetric problem the ties can be mirrored (`connectivity.symmetry`, or the planes of a `symmetry` chain link).
On the bracket of [`docs/olcumler/simetrik_baglar.md`](docs/olcumler/simetrik_baglar.md) this makes the frozen
cells symmetric but not the design (compliance +2.2 %); an exactly symmetric design needs `SymmetryTying`
(+1.5 %). Mirrored ties join every pair of interfaces through frozen cells, so the gate's narrowest-neck measure is
then undefined. The default is unchanged.

## Evidence

Every claim names the test or measurement behind it. The regression oracles are earlier drivers of this repository:
they show that the implementation did not change, not that it is independently correct.

| Check | Reference | Result | Source |
|:---|:---|:---|:---|
| Compliance problems against published codes | NumPy transcriptions of Top88 (`mbb2d.yaml`) and Top3D (24 × 8 × 4), tolerances written before the first run | objective and gradient ≤ 2.1e-11; 50- and 30-iteration histories ≤ 1e-11; same iteration count to convergence (158, 141) | `tests/toporlogy/test_independent_vv.py` |
| T10 mesh convergence | three-grid apparent order, Richardson value, fine-grid GCI | box cantilever: 0.8 / 0.5 / 0.32 CONVERGED (order 1.47, GCI 0.07 %), 2.0 … 0.5 NOT CONVERGED | `tests/toporlogy/test_t10_convergence.py` |
| 3-D optimisation history | earlier voxel driver | ≤ 3.4e-11 relative, 7 scenarios | `tests/toporlogy/test_regression.py` |
| 2-D half-MBB history | repository's Python port of top88 | 1.4e-11 relative | `tests/toporlogy/test_regression.py` |
| Gradients of every link and response | central finite differences | ≤ 1e-6 relative | `tests/toporlogy/test_gradients.py` |
| Elements H8, H8IM, Q4, T10 | patch test, uniform tension | exact to 1e-12 (T10: 1e-9) | `tests/toporlogy/test_core.py`, `test_post_verify.py` |
| First bending frequency | Euler–Bernoulli cantilever | 0.997–1.001 (H8IM) | `tests/toporlogy/test_modal.py` |
| Multigrid solver | PARDISO, same system | 7.2× faster at 163 k DOF; 10⁶ elements to 1e-8 | `docs/olcumler/A2.md` |
| Post-processing, 12 inputs | pre-registered criteria | 9/12 pass the geometric criteria, 7/12 all criteria incl. exact-solid volume and a T10 check at two tet sizes (an earlier criterion that the mesh-convergence test above replaces); 12/12 watertight with topology preserved | `docs/olcumler/robustness.json`, `tests/toporlogy/test_sweep_post.py` |
| Reproducibility | stored baseline of the seven examples | bitwise in the recorded reference environment; elsewhere scalars to 1e-6 relative (Linux: largest difference 1.5e-10) | `tests/toporlogy/test_baseline.py` |
| Stress, buckling, GCMMA, SiMPL, PDE filter | closed forms, invariants, finite differences | table above | `tests/toporlogy/test_stress.py`, `test_buckling.py`, `test_gcmma.py`, `test_simpl.py`, `test_pde_filter.py`, `test_aggregates.py` |
| Stiffness (Q4, H8, Q4IM, H8IM, orientation fields), mass, modal, conduction, centroid stress, von Mises and Tsai–Wu, buckling with and without preload | separately written dense implementation in the tests | matrices ≤ 4.6e-16, stresses and indices ≤ 1.1e-15, eigenvalues ≤ 3.2e-14, load factors ≤ 4e-13 | `test_independent_matrices.py`, `test_independent_stress.py`, `test_independent_buckling.py` |
| Manufactured solutions beyond isotropic elasticity | Q4IM; smoothly varying orthotropic axes (Q4, Q4IM, H8); heat conduction; SVK and neo-Hooke at max \|∇u\| = 0.2 | second order on three grids (rates 1.97–2.05); constant axes, wrong conductivity and a linear solve of the non-linear loads stall | `test_fem_mms_extended.py` |
| Gradients by the Taylor remainder | eleven responses incl. buckling, frequency, multi-phase, non-linear end compliance, multigrid | remainder order 2.00; a 1 % wrong gradient gives order 0.90–1.08 below the crossover step (the first, fixed-step version of that control failed for five responses: strict xfail) | `test_taylor_remainder.py` |
| Frame invariance and objectivity | 90° rotation and mirror with orientation fields; rigid rotation of large deformations | ≤ 7e-15 | `test_frame_invariance.py` |
| First-order optimality of the result | closed-form constrained problem; MBB with MMA, OC, GCMMA, SiMPL | 1.4e-16 at the closed-form solution; falls by 46–27 600× at convergence | `test_kkt_estimate.py` |

## Known limitations

- **Independent references:** compliance agrees with transcriptions of Top88 and Top3D to rounding. Stiffness
  (incompatible modes included), mass, conduction, centroid stress, failure indices and buckling agree to rounding
  with a second implementation written separately in the tests (`tests/toporlogy/_independent_fe.py`); it has the
  same author, so it is not another team's code. There is no NAFEMS benchmark (curved boundaries cannot be built on
  the regular grid).
- **T10 convergence of finished parts:** `post.verify.mesh_convergence` gives a verdict, but a finished surface is
  usually finer than the tet sizes worth solving. gmsh keeps a surface it cannot reparametrise ("discrete" mode), so
  only the interior refines; on `cantilever3d` the finest mesh of a converging sequence (about 750 000 T10 elements)
  does not fit in 16 GB. The R-01 experiment that needs it is pre-registered but not run
  ([`docs/olcumler/r01_onkayit.md`](docs/olcumler/r01_onkayit.md)).
- **`Result.converged`** means "design change below `tol` and continuation finished"; it does not include constraint
  feasibility or optimality. Check `summary.json` → `status`, the constraint values and `stationarity` (first-order
  optimality of the returned design with multipliers fitted to it, any optimizer; `kkt_residual` uses the
  optimizer's own multipliers of the last step).
- **No paper text behind the methods of 21 Nov 2026:** records and abstracts only; SiMPL and the augmented
  Lagrangian are this library's implementations of the published ideas, not verified reproductions.
- **Stress:** centroid stress of a voxel model (inaccurate at jagged boundaries, under-reported in grey elements);
  no regional aggregation, no robust (eroded/dilated) stress, no body-fitted check in the loop. Results satisfy the
  relaxed constraint on the grey design, not the stress limit on the thresholded one.
- **Buckling:** linear bifurcation only (no geometric non-linearity, imperfections or post-buckling); in 2-D the
  plane model cannot buckle out of its plane, so a thin plate's usually critical mode is not found; plain
  `Elasticity` only (direct solver, axis-aligned supports, one support set); no regularisation of the stress field
  against localised modes (the low-density share and the localisation measure only flag them; `exclude` moves a
  singular localisation rather than removing it); the number of aggregated modes is fixed by the user (a cut
  cluster is now warned about, not fixed). A λ governed by a stress singularity (end of a partly clamped edge,
  re-entrant corner, point load) has no mesh limit.
- **Material:** the fibre angle is data, not a design variable; no laminate theory, no damage progression after
  first failure; failure criteria at the element centroid like the von Mises field; the Tsai–Wu interaction
  f12 = −0.5 is a common choice, not a measured value; the connectivity coupling physics stays isotropic (ν); the
  multigrid solver takes one D only; the Hashin–Shtrikman interpolation scales E and keeps the solid's ν (its
  ν_HS(x) differs, 0.33 at x = 0.25 in 2-D).
- **Several phases:** the phases share ν and the material axes (one unit D scaled by E); OC and SiMPL, stress,
  buckling, modal, robust, connectivity, manufacturing ties and the geometry post steps (finalize, export,
  reverify) refuse several phases; no published multi-material value was reproduced.
- **Geometric non-linearity:** load control only (no arc-length: a limit point ends the solve with
  `NonlinearConvergenceError`); dead loads; homogeneous supports along x / y / z and one support set; direct
  solver only; no plasticity, contact, follower loads, incompatible-mode elements, or non-linear stress and
  buckling responses. The energy interpolation follows Wang et al. (2014) as written by a later paper (original
  not read); its adaptive cutoff can make the objective jump between designs.
- **MMA** is plain MMA: no global convergence guarantee; a two-cycle has been seen without an active constraint.
  GCMMA removes it at the cost of extra solves.
- **OC and SiMPL** handle exactly one constraint, and only one that needs no state solve (rejected at build
  otherwise). OC's bisection result depends on the constraint scale (Top3D comparison: rounding-level agreement
  when the scale is a power of 2, 1.9e-3 at 1/0.3).
- **Multigrid** is 3-D only, H8/H8IM only, and a separate physics class (`ElasticityGMG`), not a pluggable solver.
- **Geometry:** bore detection is axis-aligned only; CSG Booleans give a distance bound, not the exact distance.
- **Not provided:** thermoelasticity, plasticity, contact, post-buckling, de-homogenisation, neural
  parameterisations; STEP or other B-rep output (STL/PLY/OBJ only).
- **RAMP** interpolation is experimental and used by no default.

The current roadmap, the status of every item and the open decisions are in
[`docs/yol_haritasi_2026-11-18.md`](docs/yol_haritasi_2026-11-18.md); the full inventory and the migration plan in
[`docs/gelistirme_plani.md`](docs/gelistirme_plani.md); the method decisions in
[`docs/mimari_plan.md`](docs/mimari_plan.md); the literature survey and open problems in
[`docs/literatur_2026-11-21.md`](docs/literatur_2026-11-21.md); measurements in [`docs/olcumler/`](docs/olcumler/).

## Development

```bash
python -m pytest tests/toporlogy -m "not slow"                  # main suite (884 pass, 9 strict xfail on 25 Nov 2026)
python -m pytest tests/toporlogy/test_baseline.py -m baseline   # bitwise baseline of the examples (~3.5 min)
python -m pytest tests/toporlogy -m slow                        # pre-registered sweeps (hours; gmsh for T10)
```

Guard rails for refactoring (`tests/toporlogy`):

| Test | Fails when |
|:---|:---|
| `test_baseline.py` | any example's objective, constraints, design or exported surface changes by one bit (in the recorded reference environment; elsewhere a scalar changes by more than 1e-6 relative) |
| `test_api_surface.py` | a public name disappears or points to another object; `toporlogy list` output changes |
| `test_architecture.py` | an import crosses the layer rules, or importing toporlogy loads torch before PARDISO |

After an intended change, rewrite the snapshot and say why in the commit:
`python tests/toporlogy/test_baseline.py --write`, `python tests/toporlogy/test_api_surface.py --write`.

- `tests/toporlogy/topo_oracles/` holds frozen copies of earlier drivers, used only as regression oracles.
- Optional native packages outside site-packages are found through `TOPORLOGY_PYLIB` (folders separated by the
  path separator), the MKL DLLs through `TOPORLOGY_MKL_BIN`.
- Under heavy CPU load, multithreaded MKL and torch can be 10–25× slower than single-threaded; set
  `torch.set_num_threads` in long runs.
- Static checks: `ruff check src/toporlogy tests/toporlogy` and `mypy src/toporlogy`, both clean and both run by CI
  (`.github/workflows/tests.yml`); PARDISO, gmsh/T10 and the Linux baseline run in `backends.yml`. A test that needs
  a missing backend reports it as NOT VERIFIED, never as passed.

## Licence

Copyright © 2026 Ayberk. All rights reserved. This is proprietary software: no permission is granted to use, copy,
modify or distribute it without prior written permission. See [LICENSE](LICENSE).
