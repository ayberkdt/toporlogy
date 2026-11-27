# F2fix — Default quadrature degree of curved T10 cells (preregistration)

Preregistered 2026-11-27 05:27 +0300 (clock), before the change. Base `01f32e3`. Finding
(`F4_RATES.md`, R05, F2-1 to F2-4): on a curved quadratic tetrahedron the affine patch integrand cof(J)·∇N has
degree 3 in ξ; the default rule (degree 2 = `stiffness_degree` for affine simplices) misses it, giving patch errors
of 1e-4 (u) on curved meshes, while degree ≥ 3 is exact to rounding. T6 in 2-D: cof(J) is linear, cof(J)·∇N has degree
2, the default is already sufficient (not changed).

Change: `kernels.default_degree(ref, X)` = `stiffness_degree(ref)`, plus 1 for T10 when any cell is curved (a
mid-edge node off the chord midpoint by more than 1e-12 of the cell scale); used where the default degree is chosen
(`problem._Linear`, `history.HistoryProblem`). Affine T10 meshes and every other element keep their rule bitwise.

| ID | Hypothesis | Threshold |
|---|---|---|
| FX1 | the curved T10 patch test of `test_mesh_assembly` (g2c mesh, curved = 0.1) with the default degree now reproduces the affine field to rounding | u and stress errors ≤ 1e-12 (relative, as g2b) |
| FX2 | affine T10 and all other elements: default degree unchanged, element matrices bitwise equal to before | `np.array_equal` |
| FX3 | the existing mesh tests (test_mesh_assembly, test_mesh_elements, test_mesh_verification, test_fem_rates_investigation, test_mesh_foundations) pass unchanged | pass |

## First result (recorded 2026-11-27 05:30 +0300)

FX1 PASS: curved T10 (g2c mesh) with the default rule (now degree 3): u 1.5e-15, stress 3.1e-14. FX2 PASS: T3, T6,
Q4, Q9, T4, T10 (affine), H8 keep `stiffness_degree` and bitwise equal element matrices. FX3: 277 passed, 5 xfailed
(historical), 2 failed — `test_f2_2` and `test_f2_4` of the F4 investigation measure the *degree-2* rule and had
taken it from the default; they now pass `degree=2` explicitly (their subject is the degree-2 rule; values and
thresholds unchanged) and pass (7 passed). The g2c measurement test now prints the degree-3 numbers.
