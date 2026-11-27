# Adaptivity: a posteriori indicators and newest-vertex bisection (T3, Poisson)

Status: WP09 slice `A_adaptivity`. Code: `src/toporlogy/fem/mesh/adaptivity.py`. Tests:
`tests/toporlogy/test_adaptivity.py`. Preregistration and first results: `prereg/A_adaptivity.md`. Ledger entry:
`ledger_additions/A_adaptivity.json` (one failed prediction, A2d).

## Scope and what is not claimed

Problem `−∇·(κ∇u) = s` on a conforming T3 mesh, Dirichlet data on the **whole** boundary (`HeatProblem`). The
estimators are **indicators**: all weights are 1, no reliability or efficiency constant is proved, nothing is robust
against contrast in κ, and no guaranteed upper bound is claimed. Not implemented: elasticity (traction jumps,
near-incompressible weights), natural boundary data, T6/Q4, 3-D, curved boundaries (midpoints are straight),
goal-oriented (dual weighted residual) estimators, history-dependent state transfer, wall-time comparison.

## Indicators

Residual (`residual_indicators`, squared, energy units):
`η_T² = (h_T²/κ_T)‖s‖²_T + ½ Σ_{E interior}(h_E/κ̄_E)‖[κ∇u_h·n]‖²_E + Σ_{E ⊂ Γ_D} h_E‖∂_s(g − I_h g)‖²_E`,
`h_T` the longest edge, `h_E` the edge length, `κ̄_E` the larger neighbouring κ. In a T3 cell with constant κ,
`div(κ∇u_h) = 0`, so the element residual is the source. The data term needs the exact trace and its gradient
(`dirichlet_data = (g, grad_g)`); it is what lets the estimator see the interpolation of curved or non-linear
boundary data, and is zero for linear data.
ZZ (`zz_indicators`): area-weighted patch average of the constant gradients, P1 interpolation, `κ_T‖G_h − ∇u_h‖²_T`.
`dorfler_mark(eta2, θ)`: smallest set with mass ≥ θ·total, stable ties, ascending cell numbers.

## Refinement

`newest_vertex_labelling` rotates every cell so that `cells[:,0]` is opposite the longest edge. `refine(mesh, marked)`
bisects the refinement edge `(c1,c2)` into children `(m,c0,c1)`, `(m,c2,c0)`; if the neighbour's refinement edge is a
different edge the neighbour is bisected first (iterative closure, guarded: an unlabelled mesh raises
`RuntimeError`). Orientation is preserved, there are no hanging nodes (A3a). Cell tags are inherited, facet sets split
into child facets (boundary facets only), a node set receives the midpoint when both ends belong to it and the edge
is on the boundary. `Refinement.transfer_nodal` gives each new node the mean of its two edge ends (the P1 interpolant
is preserved: exact for affine fields, interpolation only for others); `inherit_cell` copies parent values to the
children. `mesh_state`/`mesh_from_state` checkpoint a mesh as plain arrays.

`solve_adaptive_poisson` runs solve → estimate → mark → refine with caps (`max_dofs`, `max_iterations`, `target`),
a stop reason and an `on_level` hook; it is a pure function of the mesh, so a restart from a checkpoint reproduces the
run bitwise (A3d). Estimators: `residual`, `zz`, `residual_no_jump` (negative control only).

## Measured (synthetic benchmarks; details in the prereg record)

* Smooth MMS `u = sin πx sin πy` on the unit square: the closed-form local model of the prereg predicts residual
  effectivity `4√2 = 5.657` and `|e| = 3.489 h`; measured 5.645 and 3.489 at n = 64. ZZ effectivity 1.003. These are
  properties of this mesh pattern and these unit weights, not of the estimator in general.
* L-shape, `u = r^{2/3} sin(2θ/3)` (harmonic; data = trace at boundary nodes): uniform bisection slope −0.326 in the
  DOFs (theory −1/3), adaptive (θ = 0.5) −0.504 (theory −1/2). At N ≈ 1.0–1.2·10⁴ the adaptive energy error is 3.9
  times smaller than uniform. Residual effectivity along the adaptive run 2.7 → 3.45, ZZ ≈ 1.0.
* Negative control (jump term dropped): error stays at 0.276 (initial 0.298), slope 0.000, the corner cell is never
  refined (diameter 0.71 against 6·10⁻⁵).
* Failed prediction A2d: at θ = 0.5 on the uniform h = 1/16 mesh the marked set (6 cells) holds 47.9 % of the true
  squared error and misses 2 of the 6 cells touching the corner; at θ = 0.8 it holds 85.3 % and contains them all.

## Use

```python
from toporlogy.fem.mesh.adaptivity import newest_vertex_labelling, solve_adaptive_poisson
res = solve_adaptive_poisson(newest_vertex_labelling(mesh), source, 1.0, (g, grad_g), 0.5, max_dofs=20000)
res.stop_reason, res.levels[-1].eta          # actual estimator at the stop, not a target that was reached
```
