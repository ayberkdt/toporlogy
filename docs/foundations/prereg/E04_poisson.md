# E04 — Poisson convention of the homogenised cell (preregistration)

Preregistered 2026-11-27 01:14 +0300, before any code change. Base SHA `f1ff694` (branch `foundations/round-3`).
Responsible modules: `core/linear_elastic.py` (new `poisson_uniaxial`), `optimization/homogenization_responses.py`
(`EffectivePoisson`), `configuration/components/responses.py` (YAML `effective_poisson`).

## Finding (read from the code at f1ff694)

`EffectivePoisson.__call__` returns `C[0, 1] / C[0, 0]` and its docstring calls this ν12. The convention of
`core/anisotropy.poisson_ratio` and of `docs/foundations/MATHEMATICAL_CONVENTIONS.md` is ν_ij = −ε_j/ε_i under
uniaxial stress along i, i.e. ν_ij = −S_ji/S_ii with S = C⁻¹. For a 2-D cell without normal–shear coupling,
S = C⁻¹ gives −S_21/S_11 = C12/C22 (= ν12) and −S_12/S_22 = C12/C11 (= ν21). The legacy value is therefore ν21
for an uncoupled cell, and it is no Poisson ratio of any loading direction once C has normal–shear coupling. For a
square-symmetric cell (C11 = C22) ν12 = ν21 = C12/C11, so the historical square auxetic run is not physically wrong
in its target; its label is.

## Change

* `poisson_uniaxial(C, load, transverse)` (NumPy or torch, any Voigt size 3 or 6): ν = −S[transverse, load] /
  S[load, load] with S[:, load] from a linear solve of C x = e_load (no explicit inverse; differentiable in torch).
* `EffectivePoisson(convention, load=0, transverse=1, ...)`: `convention` is required. `"uniaxial"` = the physical
  ν_{load,transverse}; `"stiffness_ratio"` = the legacy C[transverse, load]/C[load, load] … exactly the old
  `C[0, 1]/C[0, 0]` for the defaults, kept under an explicit name, 2-D only as before. No default: an old config
  without the key fails with a message that names both conventions (no silent reinterpretation).
* YAML `effective_poisson` gets the keys `convention` (required), `load`, `transverse`. The in-repo configs and tests
  that use it (test_homogenization H15–H17, test_start_design ST4) are given `convention: stiffness_ratio` so their
  historical values are kept bitwise.
* `uniaxial` works in 3-D cells as well (6×6 C); `stiffness_ratio` keeps the 2-D restriction.

## Checks

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| PO1 | plane-stress orthotropic E1=40, E2=10, G12=4, ν12=0.3: `poisson_uniaxial(C,0,1)` = 0.3, `(C,1,0)` = 0.075; legacy C12/C11 = 0.075 | S written out by hand: S11=1/40, S22=1/10, S12=−0.3/40, S66=1/4, C = inv(S) | rel 1e-12 |
| PO2 | the same C rotated 27° about z (normal–shear coupled): ν_ij from `poisson_uniaxial` equals −ε_j/ε_i of `numpy.linalg.solve(C, e_i)` for (i, j) ∈ {(0,1), (1,0)}; reciprocity ν12/E1 = ν21/E2 with E_i = 1/S_ii | dense solve on the explicit 3×3 (a different code path from the torch solve); reciprocity identity | rel 1e-12 |
| PO2n | negative control: in the rotated C the legacy ratio C12/C11 differs from both ν12 and ν21 | — | relative difference > 1e-3 for both |
| PO3 | `EffectivePoisson("uniaxial")` gradient on a 10×10 cell (random grey x, seed 5) equals the central difference of the objective | central FD, steps 1e-3 … 1e-7 (5 steps); the best step is the reported one | rel ≤ 1e-6 at the best step |
| PO4 | YAML `effective_poisson` without `convention` → ValueError naming "uniaxial" and "stiffness_ratio"; `stiffness_ratio` gives exactly C[0,1]/C[0,0] (bitwise, `==`); unknown convention → ValueError | the old formula | exact |
| PO5 | homogeneous solid cell (x = 1 everywhere, 2-D plane stress and 3-D 3×3×3): `uniaxial` gives the material ν = 0.3 | material constant | abs 1e-10 |
| PO6 | load = transverse, or an index outside the normal components (2-D: {0, 1}, 3-D: {0, 1, 2}) → ValueError | — | exact |

## Migration measurement (no threshold; a recorded number)

From `docs/olcumler/auxetic_hucre.json` (historical run, C of the grey and the binary design stored there), compute
ν12 = −S21/S11 and ν21 = −S12/S22 and compare with the stored legacy value. Written to
`docs/foundations/E04_poisson_migration.json` with the script `docs/foundations/scripts/e04_migration.py`. The
historical json is not modified.

Run command: `python -m pytest tests/toporlogy/test_poisson_convention.py -q -p no:randomly`.
First result: appended below after the run.

## First result (2026-11-27 01:45 +0300, SHA 25d8df6 + E04 change)

`python -m pytest tests/toporlogy/test_poisson_convention.py -q -p no:randomly`: PO1, PO2, PO2n, PO3, PO5 (2-D and
3-D), PO6 passed on the first run. PO4 failed on the first run for a test set-up reason: the YAML layer reports
constructor errors as `configuration.registry.ConfigError`, which is not a `ValueError`; the message named both
conventions as required. The test now expects `ConfigError` (the production error type of the YAML layer); no
threshold changed, rerun passed. Affected existing tests (test_homogenization H15–H17, test_start_design ST4,
test_auxetic_cell) given `convention: stiffness_ratio` where they use the YAML response: 41 passed, 1 xfailed
(historical H7, unchanged).

Migration (`docs/foundations/E04_poisson_migration.json`): the historical auxetic cell is **not** square-symmetric
(C11 = 0.0200, C22 = 0.0226 grey; 0.0279 vs 0.0339 binary), so its reported value is ν21, not ν12:

| design | stored "ν12" (= C12/C11) | ν21 uniaxial | ν12 uniaxial |
|---|---|---|---|
| grey | −0.9506 | −0.9506 | −0.8425 |
| binary | −1.0102 | −1.0102 | −0.8307 |

ν21 and the legacy ratio agree to 3e-7 (grey; coupling |C16|/C11 = 9e-5) and exactly (binary; coupling 3e-13). The
historical optimisation therefore minimised ν21; the historical report `docs/olcumler/auxetic_hucre_onkayit.md` keeps
its text, and this note is the correction of its label. `tests/toporlogy/_auxetic.py` keeps its own C12/C11 (it is
the historical reproduction).
