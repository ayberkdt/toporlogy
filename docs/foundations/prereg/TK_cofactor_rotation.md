# TK — Cofactor, determinant derivative and the rotation logarithm near π (preregistration)

Preregistered 2026-11-27 02:30 +0300 (clock), before the code and the tests. Base `a2b50a5`. Module
`core/tensor.py`: new `cofactor(F)` (explicit adjugate, no inverse: cof F = (adj F)ᵀ, valid for singular F) and
`det_derivative(F, H) = cof(F) : H` (WP03 T01). Rotation logarithm `rotvec_from_rotation` is existing code; T6 is a
new check of it.

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| TK1 | random F (3×3 and 2×2, seed 31, cond ≤ 1e3): cofactor equals det(F) F⁻ᵀ | numpy inv and det (different code path) | rel 1e-12 |
| TK2 | rank-2 and rank-1 3×3 F (built from SVD with zero singular values): cofactor equals the mpmath adjugate at 50 digits (rank-1: cofactor = 0 exactly up to rounding) | mpmath (independent arithmetic) | abs 1e-14 · ‖F‖² |
| TK3 | det_derivative(F, H) equals the central difference of det(F + hH) (h sweep 1e-2…1e-7, best) for invertible, rank-2 and rank-1 F | FD | rel 1e-8 at the best step (abs 1e-12 · ‖F‖² for rank-1, where the derivative is 0) |
| TK4 | rotation logarithm at angles 1e-10, 1e-4, 0.8, π − 1e-6 about random axes (seed 32): exp(log R) = R to 1e-12; ‖log R − w‖ ≤ 1e-10 for angles ≤ 0.8; for π − 1e-6 the returned angle equals the true angle to 1e-6 and the axis equals ±a (the double cover / branch is reported, not hidden) | the input rotation vector | as stated |
| TK5 | negative controls: a reflection (det −1) is refused by `rotvec_from_rotation`; cofactor of a non-square or wrong-shape input raises | — | ValueError |

Run: `python -m pytest tests/toporlogy/test_tensor_cofactor.py -q -p no:randomly`. First result appended below.
