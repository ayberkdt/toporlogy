# XF — The λ-based mixed solve refuses incompatible incompressible data (preregistration)

Preregistered 2026-11-27 03:07 +0300 (clock), before the change. Base: after the WP08 merge. Finding of X2d
(`prereg/X_mixed.md`): `MixedElasticity.solve` (λ = ∞, every boundary DOF prescribed, constant pressure in the
null space) accepts a boundary displacement with a net volume change and returns a solution with max|B u| = 3.1e-2,
i.e. it silently solves a different problem; `MixedKG.solve` raises. Rule (WP00 §4): an unsupported or impossible
combination fails explicitly.

Change: the compatibility check moves from `MixedKG.solve` into `MixedElasticity.solve` (shared by both); nothing
else changes; compatible data give bitwise the same results as before.

| ID | Hypothesis | Threshold |
|---|---|---|
| XF1 | `MixedElasticity` (λ-based, ν = 0.5) on the X2d net-inflow data raises ValueError naming the net volume change | exact |
| XF2 | the X2d characterisation of the old class (it returned a solution) is replaced by XF1; the λ-based fingerprint (X1d) and every existing mixed test pass unchanged | bitwise / pass |

Run: `python -m pytest tests/toporlogy/test_mixed_kg.py tests/toporlogy/test_mesh_locking.py tests/toporlogy/test_mesh_assembly.py -q -p no:randomly`.

## First result (run 2026-11-27 between 03:07 and 03:10 +0300; the "03:09" first written here was typed, not read from the clock — corrected)

XF1 PASS: `MixedElasticity(m, G, inf).solve` on the X2d net-inflow data raises "net volume change". XF2 PASS:
test_mixed_kg (incl. the λ-fingerprint X1d), test_mesh_locking, test_mesh_assembly: 129 passed, 6 xfailed
(recorded). The X2d characterisation of the old behaviour is replaced by the XF1 assertion; the measured old
violation (max|B u| = 3.1e-2) stays in `prereg/X_mixed.md`.
