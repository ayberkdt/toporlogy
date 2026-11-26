# Research examples

Configurations for the methods added on 21 Nov 2026 (stress, buckling, GCMMA, SiMPL, Helmholtz filter, mechanism
output, 0/1 check). They live in this sub-folder on purpose: the bitwise baseline (`tests/toporlogy/test_baseline.py`)
covers the files directly in `examples/`, whose numbers are tied to the reference environment; these are covered by
`tests/toporlogy/test_research_examples.py` (they build, run a few iterations, and their gradients are checked in
the module tests). Sizes are small so that each runs in well under a minute on a laptop.

| File | Shows |
|:---|:---|
| `stress_lbracket2d.yaml` | volume minimisation of the L-bracket with an aggregated von Mises constraint (P-norm, adaptive scale), GCMMA; solid load pad and one-body protection (without them the design drops the load into void, `docs/olcumler/gerilme.md` §3.6) |
| `stress_al_lbracket2d.yaml` | the same problem with local stress constraints by an augmented Lagrangian |
| `buckling_column2d.yaml` | compliance with a lower bound on the linear buckling load factor |
| `inverter2d.yaml` | compliant force inverter: output displacement as objective, springs at input and output |
| `mbb_simpl2d.yaml` | the half MBB beam with SiMPL, a Helmholtz filter and the 0/1 check |
| `units_cantilever2d.yaml` | a cantilever in mm and N: values with units (`"2.5 mm"`, `"-2 kN"`), a material record (`material: {catalog: ...}`), VTU field export and a checkpoint every 10 iterations (`--resume`); 60 iterations do not converge, it shows the configuration, not an optimum |
| `fiber_angle_cantilever2d.yaml` | cantilever of one orthotropic fibre material: density and the fibre angle of every element designed together (doubled-angle filter, `tensor_elasticity`; `docs/olcumler/malzeme_tasarimi.md`) |
| `fiber_matrix_angle_cantilever2d.yaml` | the same cantilever with two phases (isotropic matrix + orthotropic fibre) and void: both phase fields and one fibre angle per element designed together (`docs/olcumler/faz_aci_onkayit.md`) |
| `mesh_cantilever2d.yaml` | a cantilever on a general (perturbed Q4) mesh: `domain: {type: mesh}` takes its own build path (`configuration/mesh_build.py`), supports and `load_cases` on named node sets (`distribute: nodes | traction`) or a point, `rmin` in length units, `plane: strain`, `combinations`, per-case `supports` and a `vtu` post step available (`docs/olcumler/ag_topoloji_onkayit.md`, AY1-AY6, AX1-AX8, AC1-AC8) |

Run one: `python -m toporlogy run examples/research/stress_lbracket2d.yaml --out runs/stress`.
