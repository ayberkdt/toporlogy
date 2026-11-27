# NS — one Newton loop for cure_slab and nonlinear_heat (cleanup, behaviour neutral)

Preregistered 2026-11-27 09:40 +0300, base SHA `1f898bf`, before the shared helper is written and before the fingerprint test is run.

## Change

`CureSlab.step` and `NonlinearHeat._newton` carry the same Newton iteration on the free nodal temperatures (residual,
max-norm stop `rmax <= rtol * scale + noise`, non-finite refusal, `no convergence in N Newton iterations`, sparse LU
on the free block). It moves to one private helper `toporlogy.fem.mesh._newton.newton_free`; each module keeps its own
residual, scale and rounding-noise model (passed in as a callback), so the arithmetic of every iterate is unchanged.
The `max_iter` check (integer >= 1, bool refused) moves to `_validation.max_iterations`. No public name is added,
removed or renamed.

## Checks (thresholds fixed here)

| ID | Check | Threshold |
|---|---|---|
| NS1 | fingerprint of six fixed cases (cure slab Robin and Dirichlet runs, a cure step refused with `max_iter=1` and its unchanged state; nonlinear heat Q4 and T6 transient runs with Dirichlet + Robin + source and their steady solve, a heat step refused with `max_iter=1`): every field of every step result (times, fields, iteration counts, residual histories, energies), the committed state and the refusal messages | **bitwise** equal (sha256) to `tests/toporlogy/baseline/ns_newton_fingerprint_1f898bf.json`, written by the code at `1f898bf` before the change |
| NS2 | `tests/toporlogy/test_cure_slab.py` and `test_nonlinear_heat.py` | pass unchanged (no edit of either file) |
| NS3 | helper unit test: a 1-D problem R(x) = x^2 - 2 on one free unknown, J = 2x, start 1 | converges to sqrt(2) within 6 iterations, residual history strictly decreasing from iteration 1; with `max_iter=1` raises `StepRejected` matching "no convergence in 1 Newton iterations"; a NaN residual raises `StepRejected` "non-finite residual" |
| NS4 | ruff, mypy, layer map, API surface | clean; API surface snapshot unchanged |

Negative control for NS1: the test is run once against a copy of the baseline with one residual-history entry changed
in the last bit, and must fail (shows the comparison is bitwise, not tolerant).

## First result

Appended below after the first run.

### Set-up note (before the first test run; 2026-11-27 09:42 +0300)

The first draft of the refused heat step used dt = 1 s; the Dirichlet ramp 300 + 400 t then leaves the validity interval
of the synthetic k(T) (700 K > 450 K) and the step is refused by the property, not by Newton. Changed to dt = 0.02 s,
which is refused by Newton ("no convergence in 1 Newton iterations"). The baseline was rewritten by the unchanged code
at 1f898bf after that edit and written twice with identical digests (determinism). No threshold changed.

## First result (run 09:43–09:46 +0300 by the clock; recorded 2026-11-27 09:49 +0300)

Command: `python -m pytest tests/toporlogy/test_newton_shared.py tests/toporlogy/test_cure_slab.py tests/toporlogy/test_nonlinear_heat.py -q -p no:randomly`:
45 passed, 5 xfailed (the pre-existing NH7 strict xfails and their siblings) on the first run.

* **NS1 PASS**: all six fingerprints bitwise equal to the 1f898bf baseline.
* **NS2 PASS**: test_cure_slab.py and test_nonlinear_heat.py unchanged (`git diff` empty) and passing.
* **NS3 PASS**: sqrt(2) to 4 eps within 6 iterations, refusal messages as registered, input not modified.
* **NS4 PASS**: ruff and mypy clean (187 source files); API surface snapshot unchanged (`_newton` and
  `_validation` are private modules); test_architecture passes. test_catalog_ledger's risk matrix had to be
  re-rendered because NS3 adds one test to the counts (`_vv_layers.py --write`), not a behaviour change.
* **Negative control — deviation from the wording above**: the baseline stores sha256 digests, not raw values, so
  "change one entry of the baseline in the last bit" is not possible as written. Done instead: the returned
  residual history was moved by one ulp (`np.nextafter`) inside the solvers; the four cases that return a history
  (cure Robin, cure Dirichlet, heat Q4, heat T6) changed their digests, the two refused cases (which raise before
  returning) did not. So the comparison resolves one ulp.
