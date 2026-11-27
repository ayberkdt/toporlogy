# Mixed elasticity, incompressibility and locking (WP08)

Module `toporlogy.fem.mesh.mixed`; tests `tests/toporlogy/test_mixed_kg.py` (and the earlier `test_mesh_locking.py`);
preregistration and first results `docs/foundations/prereg/X_mixed.md`. All numbers below are from synthetic unit-square
problems (E = 1), not material data.

## Two mixed formulations, two pressures

| | `MixedElasticity(mesh, mu, lam)` | `MixedKG(mesh, G, K)` |
|---|---|---|
| displacement block | `A = ∫ 2μ ε:ε` | `A_dev = ∫ 2G ε:ε − (2G/3) ∫ div u div v` (= `∫ 2G dev ε:dev ε`) |
| pressure | `p_λ = −λ div u` | `p_K = −K div u = −tr σ / 3` (mean pressure) |
| pressure block | `−M/λ` | `−M/K` |
| valid for | λ > −2μ/3, **λ ≠ 0** (ν = 0 refused) | K > 0, G > 0 (ν = 0, ν < 0 valid) |
| exact constraint | λ = ∞ | K = ∞ (`div u = 0`) |

Both are the same continuous problem with `σ = 2G dev ε + K tr ε I`; they differ in the discrete problem and in the
pressure variable: `p_K = p_λ − (2G/3) div u`. They coincide in the displacement only if `div(V_h)` lies in the pressure
space (e.g. T3 with P0). The λ-based class is unchanged (its construction was split into a private `_build` helper and the
pressure-block modulus into `_block_modulus`; outputs are bitwise those recorded before the change, X1d). 3-D physical
strain is used throughout; plane strain embeds ε_zz = 0, so `dev ε = ε − (div u/3) I₃`. Plane stress is refused (its
effective modulus stays bounded, there is no incompressibility constraint). Scalar G, K only: a heterogeneous or
anisotropic material needs its own derived volumetric split.

Weak form (all v, q): `∫ 2G dev ε(u):ε(v) − ∫ p div v = f(v)`, `−∫ q div u − ∫ q p/K = 0`; matrix
`[[A_dev, Bᵀ], [B, −M/K]]`, `B = −∫ψ div φ`, `M = ∫ψψ`; symmetric indefinite, LU with partial pivoting; PCG is refused.

## What was verified (X_mixed, all PASS unless stated)

* **Equivalence (X1).** T3–P0 K-based equals the displacement T3 solution to 3e-15 (u) and 4e-14 (p = −K div u_h cellwise) at
  ν = −0.5, 0, 0.3. For Taylor–Hood the two differ (T6 7.7e-2, Q9 2.5e-2 at n = 4) and converge together: u error rate 3.0,
  difference rate 3.9, pressure rate 2.0–2.1 (T6 and Q9, ν = 0 and 0.3). The old class refuses λ = 0; the new one does not.
  p_K equals the mean stress pressure `−(1+ν)(σxx+σyy)/3` of the displacement solution to 1.5e-14.
* **Incompressibility (X2).** ν = 0.4999 and K = ∞ with T6/Q9: u rate 2.9, p rate 2.0, ‖div u_h‖ rate 1.9 (K = ∞), the weak
  constraint holds to 1e-10 of its scale, and the error at ν = 0.4999 is within 4% of that at ν = 0.3 (no locking, n = 16).
  Gauge: with every boundary DOF fixed the constant pressure is in the null space; the class adds `∫p = 0`; the result equals
  `solve_with_gauge` on the unbordered matrix (u to 1e-14, p up to a constant to 3e-13). Inertia of the saddle matrix (T6 4×4
  clamped): (n_u, n_p, 0) for finite K, (n_u, n_p − 1, 1) at K = ∞ without gauge, (n_u + 1, n_p, 0) with the gauge row.
  **Compatibility:** on a fully constrained boundary at K = ∞ the data must satisfy ∮ u·n = 0. `MixedKG.solve` raises
  `ValueError` otherwise; **the λ-based class does not** (it silently returns a solution with `max|B u| = 3e-2`; not changed,
  reported). Inf-sup (dense β): T6 0.332 → 0.331 and Q9 0.453 → 0.437 for n = 4 → 16, one zero mode (the constant); equal-order
  Q4–Q1 has 8 zero modes at every n and Q4–P0 has 2 (checkerboard), β falling with n: negative controls.
* **Solver compatibility.** MINRES with the block-diagonal SPD preconditioner `diag(A_dev,ff, (1/G+1/K) M)` at ν = 0.4999
  converges in 15 and 9 iterations (n = 8, 16). The plain MINRES control did not reach its tolerance (120 and 148 iterations,
  true residuals 4e-4, 2e-3): the preregistered growth claim failed (strict xfail, ledger entry), so nothing is claimed about plain MINRES except
  that it did not converge to 1e-6 in these runs.
* **Locking vs conditioning (X3).** Cantilever, L/D = 32, plane stress, closed-form energy by quadrature. Q4 deficit
  `d = 1/R − 1` follows the parasitic-shear estimate `d_pred = ((1−ν)/2)(hx/D)²` within a factor 1.1–1.4 (d/d_pred = 1.17, 1.12,
  1.10 at hx/D = 1, 2, 4; d scales ×3.8–3.95 per doubling of hx). Q9 stays at R ≥ 0.998. Q4IM (square elements, ny = 1, 2, 4)
  gives R ≥ 0.9996 where Q4 gives 0.674, 0.892, 0.971. At fixed element aspect 4 and refinement ny = 2, 4, 8 the
  condition number rises (2.1e7, 1.2e8, 5.7e8) while the deficit falls (1.56, 0.39, 0.098): the deficit is the discretisation,
  not the conditioning; scaling or one refinement step changes u by < 1e-10.
* **Reduced integration (X4).** One-point Q4 plane strain, unconstrained n×n patches: 5, 6, 6, 6 zero-energy modes for
  n = 1…4 (full integration 3, selective reduced integration 3); checkerboard displacement vectors have zero energy
  (‖Kz‖/‖K‖ ≤ 2e-16). Production `ElasticProblem(degree=1)` has the same counts as the dense reference.

## Decisions and limits

* Use `MixedKG` for ν near 0.5, for ν ≤ 0 and wherever the pressure is reported as a pressure; use Taylor–Hood (T6–P1,
  Q9–Q1 continuous vertex pressure). Equal-order and Q4–P0 pairs are kept only as negative controls.
* Locking remedy for Q4 meshes: incompatible modes (existing `Q4IM`, rectangular/square elements) or a quadratic element;
  one-point reduced integration is refused as a remedy (hourglass modes). No B-bar/EAS element was added in this slice.
* Not done here: 3-D tests of the K-based class (implemented; in 3-D `A_dev` alone is singular on pure dilatations, the saddle
  matrix is not), heterogeneous G, K per cell, derivatives with respect to K and G (WP08 X03), finite-strain mixed paths,
  mesh distortion sweeps of the slender-beam law, ν sweeps of the cantilever. The X3 law constant is an estimate, not a proof.
