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
