# Research examples

Configurations for the methods added on 21 Nov 2026 (stress, buckling, GCMMA, SiMPL, Helmholtz filter, mechanism
output, 0/1 check). They live in this sub-folder on purpose: the bitwise baseline (`tests/toporlogy/test_baseline.py`)
covers the files directly in `examples/`, whose numbers are tied to the reference environment; these are covered by
`tests/toporlogy/test_research_examples.py` (they build, run a few iterations, and their gradients are checked in
the module tests). Sizes are small so that each runs in well under a minute on a laptop.

| File | Shows |
|:---|:---|
| `stress_lbracket2d.yaml` | volume minimisation of the L-bracket with an aggregated von Mises constraint (P-norm, adaptive scale), GCMMA |
| `stress_al_lbracket2d.yaml` | the same problem with local stress constraints by an augmented Lagrangian |
| `buckling_column2d.yaml` | compliance with a lower bound on the linear buckling load factor |
| `inverter2d.yaml` | compliant force inverter: output displacement as objective, springs at input and output |
| `mbb_simpl2d.yaml` | the half MBB beam with SiMPL, a Helmholtz filter and the 0/1 check |

Run one: `python -m toporlogy run examples/research/stress_lbracket2d.yaml --out runs/stress`.
