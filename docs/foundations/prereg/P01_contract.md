# P01 — Physical time in the material-point contract (preregistration)

Preregistered 2026-11-27 01:14 +0300 together with E04, before the tests below were written or run. Base SHA
`f1ff694`. Module: `core/constitutive.py` (extended, not replaced: `TimeStep`, `MaterialRequest.step`,
`require_step`, `HistoryUpdate`, `small_strain_history_evaluate`, `drive_material_point(times=, temperatures=,
t0=, temperature0=, store=, strain0=)`), `fem/nonlinear.py` (refuses a model with `physical = True`).

Design and rules: `docs/foundations/MATERIAL_CONTRACT.md`.

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| CT1 | `TimeStep` rejects dt < 0, non-finite t or dt, temperature ≤ 0 K or non-finite, `temperature_prev` without `temperature`; `temperature_prev` defaults to `temperature` | — | exact (ValueError / equality) |
| CT2 | `require_step` raises ValueError without a step, and with `temperature=True` when the step has no temperature | — | exact |
| CT3 | `drive_material_point` without `times` reproduces a hand-written loop (StateStore + evaluate with `MaterialRequest(energy=True)`, Newton as before) on a J2 mixed path bitwise | test re-implementation of the old loop (regression, not independent) | `np.array_equal` |
| CT4 | with `times` and `temperatures`, a recording probe model receives `TimeStep(t_{k-1}, t_k − t_{k-1}, T_k, T_{k-1})` with t_0 = t0, T_0 = temperature0 in every evaluation of step k; decreasing times, wrong length, temperatures without times or without temperature0 → ValueError | the schedule itself | exact |
| CT5 | committed state is not changed inside the Newton loop of a stress-controlled step (probe model records the committed arrays at every evaluation; they are read-only and equal within a step); one commit per step | — | exact |
| CT6 | restart: 10 steps of J2 in one call equal 5 steps + a new call with `store` (restored from `checkpoint()`) and `strain0` for the last 5 steps | the uninterrupted run | `np.array_equal` on stress, strain and state |
| CT7 | `NonlinearElasticity(material=<model with physical=True>)` raises ValueError naming the element-modulus scaling | — | exact |
| CT8 | `small_strain_history_evaluate`: 2-D plane stress refused; 2-D plane strain equals the 3-D evaluation at ε_zz = 0 restricted to [xx, yy, xy] (probe linear model) | 3-D evaluation | `np.array_equal` |

Run: `python -m pytest tests/toporlogy/test_material_contract.py -q -p no:randomly`. First result appended below.

## First result (run 2026-11-27 between 01:14 and 01:27 +0300, SHA 25d8df6 + test file; the time was not read from the clock at the run — an earlier hand-written "01:35" was wrong and is corrected here)

CT1–CT8: 8 passed on the first run (`python -m pytest tests/toporlogy/test_material_contract.py -q -p no:randomly`).
Two test set-up corrections were made before that first run (a non-existent `Domain.box` call replaced by
`Domain(shape)`; the probe's Voigt packing replaced by `tensor.strain_to_voigt`). No threshold changed.
