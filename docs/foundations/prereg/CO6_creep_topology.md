# CO6 — Creep-optimised design vs elastic design (replacement preregistration of CO5)

Preregistered 2026-11-27 03:28 +0300 (clock), after the CO5 set-up was found self-contradictory and calibrated
(`CO_creep_topology.md`), before any optimisation run. Base `0b61055` + the CO module.

Set-up: MBB half-beam 30 × 10 Q4 (unit thickness, plane strain), K0 = 3.33 GPa, G0 = 0.77 GPa, Norton A = 1e-4 1/s,
σ_ref = 10 MPa, n = 3 (synthetic), load F = 5e6 at the top-left node applied at t = 0 (a zero-length elastic step)
and held: steps t = 0, 1, 3, 10, 30, 100, 300, 1000 s (uniform design ratio J(1000 s)/J(0) = 1.40, measured in the
calibration). V = 0.5, r_min = 1.5, SIMP p = 3, MMA with KKT stop 1e-4 (max 1500 iterations).

| ID | Hypothesis | Threshold |
|---|---|---|
| CO6a | elastic design: minimise J(0) (step 0, no creep) from the uniform start; creep design: minimise J(1000 s) warm-started from the elastic design. J(1000 s) of the creep design ≤ J(1000 s) of the elastic design | ≤ (no worse); gain reported |
| CO6b | both runs end KKT-stationary (≤ 1e-4) or report converged = False at the cap | exact |
| CO6c | the creep design pays at t = 0: J(0) of the creep design ≥ J(0) of the elastic design (the elastic design is the J(0) optimum it started from) | ≥ |

Run: `python -m pytest tests/toporlogy/test_creep_topology.py -q -p no:randomly -m slow -k co6 -s`.
