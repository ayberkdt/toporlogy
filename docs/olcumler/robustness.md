# Robustness sweeps: connectivity (one body) and post-processing (smoothing)

> **Status (15 Nov 2026, 00:0x): interrupted.** The connectivity sweep (section 3) is complete. The post-processing
> sweep ran (raw records in `robustness.json`), but its result section 4 and the deviation list in section 5 were not
> written before the run was stopped. The sweep tests in `tests/topo/test_sweep_*.py` are marked slow as a whole until
> they have been re-run on an idle machine.

Sections 1 and 2 were written **before any sweep run** (pre-registration, 2026-11-14 22:11). They are not edited
after the runs; deviations are listed in section 5. Results: section 3 onward. Raw records: `robustness.json`.

## 1. Pre-registration — connectivity sweep

Code: `tests/topo/test_sweep_connectivity.py` (scenario builders `cases()`, runner `run()`); single cases:
`scratch/robustness/conn_one.py <case> <method>`.

### 1.1 Scenarios (14; 10 × 2-D, 4 × 3-D)

All pads (support and load regions) are keep regions and interfaces. Support pad: all DOFs of its nodes fixed.
Load pad: per load case a force spread evenly over its nodes. SIMP p = 3, element size 1, ν = 0.3. With projection:
`standard_chain(rmin, beta=1)`, `BetaContinuation(1 → 16, every 20)`, `move_beta_ref = 2`. Without projection:
density filter only, no schedule. OC: `OC(dv_scale = n_active)`. Max iterations 150 (3-D: 120), tol 0.01.

| id | dim, grid | supports | loads | vf | rmin | projection / β | optimizer | feature under test |
|---|---|---|---|---|---|---|---|---|
| c01 | 2-D 60×24 | 2 | 2 | 0.15 | 2.0 | yes | MMA | two spans, pads far apart, weak pull apart (±0.02) |
| c02 | 2-D 72×24 | 2 | 3 | 0.12 | 2.0 | yes | MMA | three load pads between two supports |
| c03 | 2-D 80×20 | 4 | 4 | 0.10 | 1.5 | yes | MMA | four columns, each load above its own support |
| c04 | 2-D 48×24 | 2 | 2 | 0.20 | 2.0 | no | OC | pads close together (gap 8), grey-film risk |
| c05 | 2-D 60×24 | 2 | 2 | 0.08 | 1.5 | yes | MMA | c01 at very low volume fraction |
| c06 | 2-D 60×24 | 2 | 2 | 0.15 | 2.0 | yes | MMA | c01 with a void wall between the spans (path only over the top) |
| c07 | 2-D 48×48 L | 2 | 2 | 0.12 | 2.0 | yes | MMA | L-shaped part, each load next to its own support at the arm ends |
| c08 | 2-D 60×24 | 2 | 2 | 0.15 | 2.5 | no | OC | asymmetric loads (1 / 0.25), weak one-sided pull 0.005 |
| c09 | 2-D 60×24 | 2 | 2 | 0.15 | 2.0 | yes | MMA | three load cases (each pad alone, then a squeeze) |
| c10 | 2-D 60×24 | 1 | 2 | 0.12 | 2.0 | yes | MMA | control: one support, nothing to split |
| c11 | 3-D 30×4×12 | 2 | 2 | 0.15 | 1.5 | yes | MMA | 3-D two-span split (as `split_benchmark(3, 12)`) |
| c12 | 3-D 24×16×8 | 4 | 2 | 0.10 | 1.5 | yes | MMA | four corner supports, loads above two diagonal ones |
| c13 | 3-D 32×8×12 | 2 | 3 | 0.12 | 2.0 | no | MMA | three loads, two void walls between them |
| c14 | 3-D 24×6×12 | 2 | 2 | 0.20 | 2.0 | yes | OC | 3-D pads close together (gap 4) |

### 1.2 Methods (each scenario)

- **plain** — compliance minimisation with the volume bound only (scenario optimizer).
- **default** — `one_body(domain, interfaces)` with its defaults: frozen ties of section 1 + conduction constraint on
  the eroded field (η = 0.7, bound 20, sink = first interface). Needs MMA (two constraints): OC scenarios run it with
  MMA.
- **ties** — `one_body(..., ties=1, constraint=None)` (scenario optimizer).
- **conduction** — `one_body(..., ties=None, constraint="conduction", eta=0.7)` (MMA).
- OC scenarios also run **plain_mma** (plain with MMA), the reference for the compliance ratio of MMA-run methods.

### 1.3 Criteria

1. **Connectivity (primary)**: `analyse_connectivity(domain, xphys, interfaces, voids="warn")` verdict **PASS** on
   the optimised design (before any repair): one body at threshold 0.5 holding every interface, every interface in
   one body at thresholds 0.3 and 0.7 and after a one-element erosion. Enclosed 3-D voids are reported separately
   (also the verdict with `voids="fail"`).
2. **Volume feasibility**: final volume ≤ 1.01 vf (ties count as volume).
3. **Connectivity constraint feasibility** (default, conduction): value ≤ 1.01 bound.
4. **Repair**: `plan.finish(xphys)` ends PASS (the guarantee asserted by the slow test).

Recorded, not gated: compliance ratio vs plain (same optimizer), iterations, convergence, time, bodies, minimum
neck, tie cells. Reported: failure rate per method. Every FAIL of the default plan is investigated (cause fixed with a
regression test, or the limitation documented).

Expected (stated before the runs, not criteria): plain splits in c01–c09 and c11–c14; c10 stays one body.

## 2. Pre-registration — post-processing sweep

Code: `tests/topo/test_sweep_post.py`; `finalize(domain, xphys, preserve, PostParams())` with defaults unless
stated, then `manufacturing_checks`. All inputs 3-D (the pipeline is 3-D only), element size h = 1, h_f = h/4.

### 2.1 Inputs (12)

| id | input | preserve roles | special |
|---|---|---|---|
| p01 | sphere R 5.5 in 16³ | – | – |
| p02 | tube R 5.5 / 2.5, length 6 | Cut(inner cylinder) | bore accuracy |
| p03 | plate 16×16×4 with bore R 3.5 | Cut(bore), Bound(mounting plane z = 1) | bore + flatness |
| p04 | L-bracket (two boxes) | Keep pad | `r_fillet_min = 1.5 h`, concave-radius test |
| p05 | thin plate, thickness 1.25 h | – | near the wall limit, `min_wall = 1.0` |
| p06 | strut lattice (radius 1.0 h, pitch 6 h) | – | many tunnels (genus) |
| p07 | cube with an enclosed spherical cavity | – | `fill_voids=False` (cavity must stay) |
| r1 | cantilever 30×10×10, OC + β (Study) | Keep clamp plate, load pad | T10 re-check at 2 sizes |
| r2 | MBB 3-D 48×8×12, MMA + β (Study) | Keep support/load pads | T10 re-check at 2 sizes |
| r3 | bracket with two bores 32×6×20 (Study) | Cut(bores) functional, Keep(rings) | bore accuracy on a real result |
| r4 | robust (eroded/blueprint/dilated) cantilever 32×8×12 (Study) | Keep pads | minimum size |
| r5 | connectivity scenario c12 with the default one_body plan (Study) | Keep pads | one body survives smoothing |

### 2.2 Criteria (gating, per input)

1. watertight closed manifold (open = non-manifold = misoriented edges = non-manifold vertices = 0);
2. topology unchanged: mesh bodies and χ equal the thresholded fine-grid design after island removal
   (`opening.topology_before`); intentionally filled enclosed voids lower χ by one each and are reported (none
   expected: p07 runs with `fill_voids=False`);
3. |ΔV| = |V_mesh / V_target − 1| ≤ 1 % (V_target = Σ frac·x̄·h³; analytic inputs also vs the exact solid);
4. functional faces: bore radius error max over the full face ≤ 1 % of nominal; plane flatness ≤ 1e-3 h;
5. minimum triangle angle ≥ 15°;
6. Hausdorff distance to the raw iso-surface ≤ 0.6 h_f both ways.

Recorded, not gated: normal deviation (mean/p99/max), min angle p1, wall thickness (ray p1 / min, inscribed ball p1),
overhang fraction (45°, build direction +z), opening/closing radii used, islands, runtime. p04: concave-radius test at
1.5 h (pass if added depth ≤ h_f/2). p05: ray wall p5 within 10 % of 1.25 h.

T10 re-check (`reverify`, gmsh T10, E = 1, ν = 0.3) for r1 and r2 at tet sizes 1.0 h and 0.5 h: T10 / voxel
compliance ratio at both sizes; criterion: relative change of the ratio between the sizes ≤ 5 %.

Pass rate = inputs passing all gating criteria / inputs.

## 3. Results — connectivity

Source of every number in this section: `scratch/robustness/out/conn_2d.jsonl`, `conn_3d.jsonl` (copied into
`robustness.json` → `connectivity`), tables printed by `python scratch/robustness/report.py`. Command (2-D; 3-D with
`-k 3d` and `conn_3d.jsonl`):

```
PYTHONPATH=src AEROKIT_PYLIB=<pylib> AEROKIT_SWEEP_RECORD=scratch/robustness/out/conn_2d.jsonl \
  python -m pytest tests/topo/test_sweep_connectivity.py -m slow -k 2d -s
```

Run 2026-11-14 22:43–22:56 (2-D 3 min 32 s, 3-D 11 min 16 s; two processes in parallel, machine shared with other
agents — times are upper bounds). The sweep was run twice (22:16 before the `finish` fix, 22:43 after it, archived run:
`scratch/robustness/out/run1/`): all 59 records have identical compliance, volume, verdict and tie cells; only the
`finish` results differ (§3.3).

### 3.1 Pass rate per method (primary criterion, 14 scenarios)

| method | runs | PASS | FRAGILE | FAIL | volume feasible | finish PASS | median c/c0 | time s |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 14 | 0 | 2 | 12 | 14 | 14 | — | 184 |
| default | 14 | 14 | 0 | 0 | 13 | 14 | 1.30 | 240 |
| ties | 14 | 14 | 0 | 0 | 13 | 14 | 1.25 | 122 |
| conduction | 14 | 0 | 13 | 1 | 11 | 14 | 1.65 | 270 |

Failure rate (verdict ≠ PASS): plain 14/14, **default 0/14**, ties 0/14, conduction 14/14. Default plan, all
criteria together (connectivity PASS, volume ≤ 1.01 vf, constraint ≤ 1.01 bound, finish PASS): 13/14 — c03 is
volume-infeasible (§3.4). c/c0 = compliance / compliance of plain with the same optimizer (MMA-run methods in OC
scenarios: vs plain_mma). "finish PASS" is after the `finish` fix (§3.3).

### 3.2 Per scenario

FAIL (n) = verdict with n bodies at threshold 0.5; FRAG = FRAGILE; V! = volume > 1.01 vf. plain+finish = the plain
result repaired by `plan.finish` (not re-optimised; compliance at the same β, volume after repair).

| case | plain | default | default c/c0 | ties | ties c/c0 | conduction | conduction c/c0 | plain+finish | plain+finish c/c0 | plain+finish V/vf | cond+finish |
|:---|:---|:---|---:|:---|---:|:---|---:|:---|---:|---:|:---|
| c01 | FAIL (2) | PASS | 1.18 | PASS | 1.18 | FRAG | 1.24 | PASS | 1.00 | 1.53 | PASS |
| c02 | FRAG (1) | PASS | 1.74 | PASS | 1.74 | FRAG | 1.50 | PASS | 0.70 | 1.74 | PASS |
| c03 | FAIL (4) | PASS V! | 36.29 | PASS V! | 36.29 | FRAG V! | 470.94 | PASS | 0.58 | 2.61 | PASS |
| c04 | FAIL (2) | PASS | 1.13 | PASS | 1.13 | FRAG | 1.04 | PASS | 0.99 | 1.07 | PASS |
| c05 | FAIL (2) | PASS | 26.05 | PASS | 19.08 | FAIL V! | 755.32 | PASS | 0.98 | 2.08 | PASS |
| c06 | FAIL (2) | PASS | 1.24 | PASS | 1.24 | FRAG | 1.70 | PASS | 0.99 | 1.70 | PASS |
| c07 | FAIL (2) | PASS | 1.96 | PASS | 1.96 | FRAG | 2.65 | PASS | 0.90 | 1.84 | PASS |
| c08 | FAIL (2) | PASS | 1.34 | PASS | 1.34 | FRAG | 1.71 | PASS | 0.99 | 1.52 | PASS |
| c09 | FAIL (2) | PASS | 1.25 | PASS | 1.25 | FRAG | 1.36 | PASS | 0.96 | 1.50 | PASS |
| c10 | FAIL (10) | PASS | 52.89 | PASS | 52.90 | FRAG | 51.56 | PASS | 0.50 | 2.58 | PASS |
| c11 | FAIL (2) | PASS | 1.25 | PASS | 1.10 | FRAG | 1.47 | PASS | 0.98 | 1.75 | PASS |
| c12 | FAIL (4) | PASS | 34.36 | PASS | 1.13 | FRAG | 59.89 | PASS | 0.98 | 2.01 | PASS |
| c13 | FRAG (1) | PASS | 1.01 | PASS | 0.99 | FRAG V! | 1.60 | PASS | 0.29 | 2.01 | PASS |
| c14 | FAIL (2) | PASS | 1.03 | PASS | 1.01 | FRAG | 1.03 | PASS | 0.98 | 1.10 | PASS |

Iterations, convergence, time, volume, neck, tie cells and constraint value of every run: `robustness.json`
(`connectivity[*]`) and the "connectivity details" table of `report.py`. Not converged (change > 0.01 at the
iteration limit) — 19 of 59 runs: conduction in c01, c03, c05, c06, c07, c09, c11, c12, c14; c02 plain; c11 default;
all c12 and c14 runs; c13 default. Verdicts are of the design at the limit.

### 3.3 Source fix: `OneBodyPlan.finish` did not reach PASS (src/aerokit/topo/connectivity.py)

Failing case (run 1, 22:16): conduction-only results are FRAGILE (one body, split after the one-element erosion);
`plan.finish` returned them still FRAGILE in **14/14** scenarios. Two causes in the code: `bridge()` joins bodies only
at the nominal threshold, so a design that is one body there gets no bridge (0 cells added); and the default repair
section 2 cannot survive the gate's own one-element erosion (a 2-wide bar erodes away completely, as
`test_neck_min_cut_2d` already shows). Fix: after the nominal bridge, every realisation that still splits
(threshold 0.7, eroded solid) is bridged with section-3 paths set to x̄ = 1, up to three rounds. After the fix: 14/14
PASS for conduction+finish and for plain+finish (table above). Regression test:
`tests/topo/test_sweep_connectivity.py::test_finish_repairs_fragile_and_split_designs_to_pass` (a 2-wide link, a grey
link at 0.6 and a split design: all three ended FRAGILE before, PASS now). Existing tests `test_onebody.py`,
`test_connectivity.py` (fast): 28 passed after the change.

Price of repair (table, plain+finish): compliance ≤ plain (0.29–1.00) but volume 1.07–2.61 × vf, because bridges are
geometric and set solid; a repaired design is not volume-feasible without re-optimisation.

### 3.4 Default plan: no connectivity failure, but four large compliance penalties — causes

The default plan passed the connectivity criterion in 14/14 scenarios. Compliance penalty ≤ 2 in 10/14; four cases are
much worse, with two distinct mechanisms:

1. **Frozen ties eat the volume budget** (c03, c05, c10). Ties are frozen solid and count as volume. Pads + ties vs the
   budget vf·n_active (cells, from `tie_cells` and the case geometry): c03 88 + 82 = 170 > 160 (infeasible: the
   optimiser starts at x = 0 and cannot move — change 0, β doubled every iteration, stop after 5 iterations at
   V = 0.106), c05 44 + 58 = 102 of 115 (89 %), c10 38 + 78 = 116 of 173 (67 %). The rest of the design is starved;
   section-1 ties between pads that are not axis-aligned are Manhattan staircases (c10: one-cell staircase around the
   bottom left, ASCII dump by `scratch/robustness/ascii.py c10_2d_one_support ties`), which are very flexible.
   Ties alone show the same penalty (c05 19×, c10 53×), so the ties, not the constraint, cause it. **Limitation:**
   `one_body` does not know vf and does not warn when the frozen keep fraction approaches it.
2. **Conduction bound 20 too tight at low vf in 3-D** (c12). With ties present the conduction constraint is inactive
   in every 2-D case (default = ties to 4 digits) but active (value 20.0) in all four 3-D cases. In c12 (vf 0.10,
   four corner supports, two unloaded) it pushes 165 solid cells into the bottom layer to connect the unloaded
   supports thermally and leaves ~12 solid cells per layer for the load columns (`scratch/robustness/out/c12_probe.log`;
   a 250-iteration run converges at c = 15.3, still 18× ties). Probe (not pre-registered, `scratch/robustness/
   bound_probe.py`, `out/bound_probe*.log`): bound 40 → c12 c = 1.18 (PASS), bound 80 → 0.847 (= ties, PASS); bound 40
   on c05, c07, c10, c11: PASS in all, compliance equal (c05, c07, c10) or lower (c11 0.689 vs 0.783). The module notes
   say connected ratios scale like 1/vf × tortuosity; a fixed bound of 20 is near that value at vf = 0.1. Not changed
   in the library (calibration decision, not a bug): see §6.

### 3.5 Other observations

- **Conduction alone is never robust**: 13/14 FRAGILE, 1/14 FAIL (c05, also volume-infeasible). The constraint ends at
  its bound (19.997–20.06) with a one- or two-cell neck that the erosion removes (`min_neck_cells` 1–2). This confirms
  the note in `one_body`'s docstring; with the `finish` fix the repair makes them PASS (at extra volume).
- **Plain is never PASS** (14/14). The control c10 was expected to stay one body; it FAILs because its diagonal members
  are one-element staircases touching only at edges (face connectivity is required), ASCII dump as above.
- c02 and c13 plain are one body but FRAGILE (neck 1–2 cells), i.e. "connected" designs from plain TO can still fail
  the robustness checks.
- OC vs MMA (c04, c08, c14): plain splits with both; ties-only with OC passes in all three.
- Concurrency note: another agent changed `one_body` at 23:01 (interface cells added to the frozen tie mask), after
  both sweep runs. All pads in these scenarios are already keep regions, so the plan domain is unchanged; only
  `tie_cells` would count the pad cells too. Not re-run.

## 4. Results — post-processing

Not written yet (run interrupted). Raw records: `robustness.json` → post-processing entries.

## 5. Deviations from the pre-registration

Not written yet (run interrupted).
