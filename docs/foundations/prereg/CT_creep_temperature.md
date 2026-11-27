# CT — Creep topology optimisation under a temperature history (preregistration)

Preregistered 2026-11-27 05:41 +0300 (clock), before the change and the tests. Base `e85106c`.
Change: `CreepStructure(..., arrhenius=None, temperatures=None, temperature0=None)`: with an `ArrheniusFactor` each
step's creep coefficient is Δt · A · F_n, F_n = `arrhenius.step_average(T_n, T_{n+1})` (T linear in the step, as
`NortonCreep`); implemented as an effective step Δt_eff = Δt F_n in the local update (only the product enters). The
elastic moduli do not depend on T here. Without Arrhenius, Δt_eff = Δt · 1.0 (bitwise as before).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| CT1 | Arrhenius Q = 80 kJ/mol, T_ref = 300 K (synthetic), T ramp 300 → 330 K then hold, CO cantilever, w = 1 and 0.4: u at every step equals `HistoryProblem` + `NortonCreep(..., arrhenius)` with the same `TimeStep` temperatures | partially independent (different global algebra; same step-average quadrature) | rel 1e-9 |
| CT2 | gradient of J(t_s) under that history vs central FD | FD sweep, best | rel ≤ 1e-6 |
| CT3 | T = T_ref throughout equals the isothermal structure bitwise (objective and gradient) | isothermal | `==` / `array_equal` |
| CT4 | arrhenius without temperatures, temperatures without arrhenius, wrong length → ValueError | — | exact |

Run: `python -m pytest tests/toporlogy/test_creep_temperature.py -q -p no:randomly`.

## First result (recorded 2026-11-27 05:53 +0300)

An earlier invocation that also selected the slow CO6 test was stopped before its output was read (the CT tests had
probably run in it; nothing was seen). First observed run (with test_creep_topology, slow deselected): 12 passed,
1 xfailed (CO4). CT1: 4.3e-16 (w = 1) and 4.7e-16 (w = 0.4) against `HistoryProblem` + `NortonCreep(arrhenius)`.
CT2: best FD error 6.5e-10. CT3: bitwise. CT4: all refusals raise.
