# H6 — The physical history models inside the FE history solver (preregistration)

Preregistered 2026-11-27 02:35 +0300 (clock), before the tests exist. Base `32c869c`. Uses `fem.mesh.history.
HistoryProblem` (H0–H5) with the physical models merged from the WP12–WP14 slices; no production code change
planned.

Set-up: homogeneous H8 bar 2 × 1 × 1 cells (length 2, unit section), roller supports as in H1 (x = 0 face u_x = 0,
y = 0 nodes u_y = 0, z = 0 nodes u_z = 0), an end force F(t) on the x = 2 face spread equally over its nodes,
physical time steps (log-spaced 0.1 … 1000 s after a ramp to t = 1 s). Reference: `drive_material_point` of the same
model with σ_xx = F/A prescribed and the other five stresses zero, same times (partially independent: same
constitutive kernel, independent global algebra and stress control).

| ID | Model (synthetic) | Hypothesis | Threshold |
|---|---|---|---|
| H6a | `NortonCreep(K, G, A = 1e-6 1/s, σ_ref = 10 MPa, n = 3)`, K = 3.33 GPa, G = 0.77 GPa, F/A = 15 MPa | end displacement = 2 ε_xx(material point) at every step; lateral strains equal | rel 1e-8 |
| H6b | `Perzyna(K, G, σ_y = 20 MPa, H = 100 MPa, τ0 = 10 s, σ_ref = 10 MPa, m = 1)`, F/A ramp to 25 MPa then hold | as H6a | rel 1e-8 |
| H6c | `GeneralizedMaxwell` (WP23 V set: G∞ = 1, G1 = 2 (τ 1 s), G2 = 3 MPa (τ 100 s); K∞ = 5, K1 = 4 MPa (τ 10 s)), F/A = 1 kPa | as H6a | rel 1e-8 |
| H6d | `PhysicalJ2(K, G, σ_y = 20 MPa, H = 100 MPa)` force ramp to 25 MPa (rate-independent; the step is ignored) | as H6a | rel 1e-8 |
| H6e | rollback: a creep step forced to fail (max_iter = 0) leaves the committed state of H6a unchanged and the repeated step equals the uninterrupted run | bitwise | `np.array_equal` |

Run: `python -m pytest tests/toporlogy/test_history_physical.py -q -p no:randomly`. First result appended below.
