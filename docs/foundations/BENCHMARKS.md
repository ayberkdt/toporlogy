# CPU benchmarks (WP01 A03 / WP05 L07) — measurements, no claims

Script `scripts/benchmarks.py`; data `benchmarks_f12a243.json` (median and range over 3–5 repeats after one warm-up;
torch threads and versions in the file). Measured 2026-11-27 on a machine shared with other sessions and
processes, so absolute times are inflated and only orders of magnitude are meaningful. No speed-up is claimed.

| case | median |
|---|---|
| spectral log Fréchet derivative, batch 1 / 1 000 / 100 000 | 0.26 ms / 6.5 ms / 0.57 s |
| stress → Voigt, batch 1 / 1 000 / 100 000 | 0.08 ms / 0.20 ms / 22 ms |
| general-mesh Q4 40×20 / 120×60 (1 722 / 14 762 DOF) assemble + solve | 33 ms / 0.46 s |
| material point 10 000 points: legacy J2 / generalized Maxwell / Norton | 18 / 27 / 20 ms |
| viscoelastic adjoint gradient, 8×4 cantilever, 12 steps: full / checkpointed (k = 4) | 109 / 179 ms |
| SPD factor + 8 RHS, random sparse R Rᵀ + 10 I, n = 2 000 / 20 000 | 0.73 s / **371 s** |

The random-sparse SPD case is **not representative of FE matrices**: R Rᵀ of a random sparse R has no locality, so
the factor fills in almost completely (n = 20 000 took 6 minutes, while the structured FE matrix of 14 762 DOF
solves in under half a second). It is kept as a worst-case fill measurement, not a solver benchmark. The checkpointed
adjoint costs 1.65× the time of the full one for 0.057× the saved memory (D3).
