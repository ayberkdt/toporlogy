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
