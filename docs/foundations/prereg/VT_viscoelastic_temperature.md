# VT — Viscoelastic topology optimisation under a temperature history (preregistration)

Preregistered 2026-11-27 05:38 +0300 (clock), before the change and the tests. Base `de4dd3b`.
Change: `ViscoelasticStructure(..., temperatures=None, temperature0=None)`: with a `GeneralizedMaxwell` that has a
shift (WLF or Arrhenius) the step coefficients use the reduced increment Δξ = `shift.reduced_time_increment(T_n,
T_{n+1}, Δt)` (T linear in the step, as `GeneralizedMaxwell.update`); the recursion is otherwise unchanged and the
design dependence and adjoint are untouched. Without a shift the coefficients are keyed by Δξ = Δt (bitwise as before).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| VT1 | WLF shift (T_ref = 300 K, C1 = 17.4, C2 = 51.6 K, synthetic), temperature ramp 300 → 330 K then hold, VO cantilever, uniform w = 1 and 0.37: u at every step equals `HistoryProblem` + `GeneralizedMaxwell(shift)` with the same `TimeStep` temperatures | partially independent (different global algebra; same reduced-time quadrature) | rel 1e-10 |
| VT2 | gradient of J(t_s) under that temperature history vs central FD | FD sweep, best | rel ≤ 1e-6 |
| VT3 | constant T = T_ref with the shift equals the unshifted structure bitwise (objective and gradient) | unshifted | `==` / `array_equal` |
| VT4 | constant T = 320 K (≠ T_ref) equals the unshifted structure on the time grid t / a_T(320 K) (time–temperature superposition) | unshifted structure, scaled grid | rel 1e-12 |
| VT5 | a shifted material without temperatures, wrong length, or a temperature outside the shift's range → ValueError | — | exact |

Run: `python -m pytest tests/toporlogy/test_viscoelastic_temperature.py -q -p no:randomly`.
