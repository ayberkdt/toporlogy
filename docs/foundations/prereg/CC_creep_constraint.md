# CC — Constraint on the accumulated creep strain in creep topology optimisation (preregistration)

Preregistered 2026-11-27 07:26 +0300 (clock), before the change and the tests. Base `dc37ed5`.
Change: (i) `CreepStructure.response` also returns `p_cr` (n_points,), the accumulated equivalent creep strain at the
last step (sum of the Δp of the torch local update, so differentiable); (ii) `optimize_viscoelastic(...,
extra_constraints=[(fn, limit), ...])` adds g_i = fn(result)/limit − 1 ≤ 0 for each callable fn(result) → scalar
tensor (MMA, gradients by autograd through the same march); (iii) a helper `pnorm_creep_strain(result, w, p)` =
(Σ_q w_q p_cr,q^P)^(1/P) with w_q the point volumes / total volume weighted by the element density x̄^q_e? — no: by
the point volume only (the solid creeps; void cells have negligible stress, hence negligible creep).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| CC1 | `p_cr` equals the sum of the Δp of `NortonCreep` driven through `HistoryProblem` (CO cantilever, uniform w = 1) | partially independent (CO2 set-up) | rel 1e-9 |
| CC2 | the gradient of the p-norm (P = 8) of p_cr in a random direction equals central FD | FD sweep, best | rel ≤ 1e-6 |
| CC3 | CO6 MBB set-up: minimise J(1000 s) with the extra constraint ‖p_cr‖_P ≤ 0.5 × its value at the unconstrained creep design (warm start from that design, KKT stop 1e-4, ≤ 1500 iterations): the final design satisfies the constraint to 1e-3 relative and its max point creep strain is lower than that of the unconstrained creep design; the price in J(1000 s) is reported | — | as stated |
| CC4 | `extra_constraints=[]` reproduces the previous behaviour (KS1 fingerprint) | fingerprint | bitwise |

Run: `python -m pytest tests/toporlogy/test_creep_constraint.py -q -p no:randomly -m "slow or not slow"` (CC3 slow).

## First result, quick checks (run 07:27 +0300 by the clock; recorded 2026-11-27 07:28 +0300)

CC1 PASS: p_cr equals the HistoryProblem + NortonCreep accumulated creep strain to 4.7e-15. CC2 PASS: best FD
error 6.4e-11 (P = 8, 5× load). CC4 PASS: KS1 fingerprint bitwise with extra_constraints=[]; min-max with extra
constraints refused. CC3 (slow) running; appended below when done.

## CC3 result (run 07:29–08:25 +0300 by the clock; recorded 2026-11-27 08:25 +0300)

**CC3 FAILED.** Unconstrained creep design (warm start from the elastic design, 1500 iterations, cap reached):
p-norm (P = 8) 1.1281e-2, max point creep strain 1.9593e-2, J(1000 s) 8.5960e6. Constrained run (limit 0.5 × p-norm
= 5.6403e-3, warm start from that design): cap reached (1500 iterations), p-norm 7.1697e-3 — **27 % above the
limit**, KKT residual 1.0 (infeasible), max creep strain 1.3293e-2 (lower, as hypothesised), J(1000 s) 1.6315e7
(+90 %). The halving target was set without a feasibility study; MMA did not reach a feasible design from this
start. Strict xfail; ledger `ledger_additions/CC_creep_constraint.json`. What holds: p_cr and its p-norm gradient
are exact (CC1, CC2); the constraint drives the design toward lower creep strain at a high compliance price.
Follow-up: continuation in the limit and a feasibility check before claiming a constrained optimum.

## Post hoc continuation in the limit (not preregistered; `scripts/cc_continuation_posthoc.py`, `CC_continuation_posthoc.json`; recorded 2026-11-27 09:36 +0300)

Warm start chain from the unconstrained creep design (p-norm 1.1281e-2, max 1.959e-2, J(1000 s) 8.596e6):

| limit factor | p-norm (limit) | feasible (1e-3) | max creep strain | J(1000 s) | ΔJ | iterations / KKT |
|---|---|---|---|---|---|---|
| 0.9 | 1.01526e-2 (1.01526e-2) | yes | 1.709e-2 | 8.637e6 | +0.5 % | 600 cap / 1.9e-3 |
| 0.8 | 9.02451e-3 (9.02451e-3) | yes | 1.462e-2 | 8.813e6 | +2.5 % | 600 cap / 3.8e-4 |
| 0.7 | 7.89768e-3 (7.89644e-3) | yes (+1.6e-4) | 1.303e-2 | 9.306e6 | +8.3 % | 600 cap / 2.5e-2 |

With continuation the constraint is met at 0.9, 0.8 and 0.7 of the unconstrained p-norm, buying a 13–33 % lower peak
creep strain for 0.5–8.3 % more service compliance; the direct jump to 0.5 (CC3) failed. All three runs stopped at
the 600-iteration cap without reaching KKT 1e-4: they are feasible designs on the trade-off, not converged optima.
