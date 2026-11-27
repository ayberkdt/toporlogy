# H8 — The later material models inside the FE history solver (preregistration)

Preregistered 2026-11-27 06:47 +0300 (clock), before the test exists. Base `fef0879`. No
production change planned. Same H8 bar and supports as H6 (2 × 1 × 1 H8, rollers); reference `drive_material_point`
of the same model (partially independent: same constitutive kernel, independent global algebra and control).

| ID | Model (synthetic; builders of the slice's own tests) | Loading | Hypothesis | Threshold |
|---|---|---|---|---|
| H8a | `DruckerPrager(K, G, a = 0.3, b0 = 20 MPa, H = 100 MPa, a_flow = 0.1)` (non-associated: non-symmetric tangent) | axial compressive force ramp to −30 MPa | end displacement = 2 ε_xx(material point) per step | rel 1e-8 |
| H8b | `CreepPlasticJ2` (K, G of H6, A = 1e-6 1/s, σ_ref = 10 MPa, n = 3, σ_y = 20 MPa, H = 100 MPa) | force ramp to 25 MPa in 1 s, hold to 1000 s | as H8a | rel 1e-8 |
| H8c | `AdiabaticPerzyna` (test_adiabatic `good` set) | axial displacement ramp ε_xx = 0.03 over 10 s, lateral stresses free | axial stress and temperature state per step equal the material point under ε_xx control | rel 1e-8 |
| H8d | `CuringThermoElasticPoint` (test_cure `make_point`) | bar fixed axially at both ends, lateral free; temperature ramp 300 → 420 K over 600 s, 30 steps | axial stress and degree of cure per step equal the material point with ε_xx = 0 controlled | rel 1e-8 |
| H8e | `CrystallisingThermoElasticPoint` (test_thermoplastic `make_point`) | as H8d with its own temperature range (within the model's validity) | as H8d | rel 1e-8 |

Run: `python -m pytest tests/toporlogy/test_history_more_models.py -q -p no:randomly`.
