# F2 / F4 — separating the causes of two historical failed convergence predictions (R04, R05): preregistration

Preregistered 2026-11-27 04:41 +0300, before any new numerical test was written or run. Base SHA `cf4b174` (worktree started at
`f1ff694`, fast-forwarded to `cf4b174`). Test file (written after this commit): `tests/toporlogy/test_fem_rates_investigation.py`.
Record (written after the results): `docs/foundations/F4_RATES.md`. Production code is changed only for a confirmed
defect. Historical xfails and thresholds in `tests/toporlogy/test_mesh_verification.py` / `docs/vv_kutugu.json` are
not touched; every finding below is a new test.

Sources read in this session: `tests/toporlogy/test_mesh_verification.py` (V3, `_mms_errors`, `_rates`, `MMS_LEVELS`,
`_V3_XFAIL`), `tests/toporlogy/test_mesh_assembly.py` (`_mesh`, `_patch`, G2b, G2c), `docs/olcumler/fem_genel.md` §2-3,
`docs/olcumler/fem_genel_onkayit.md` §2-3, `docs/vv_kutugu.json` (the two strict xfails), `docs/foundations/risk_register.json`
(R04, R05), `fem/mesh/manufactured.py`, `fem/mesh/quadrature.py`, `fem/mesh/problem.py` (`errors`, `solve`),
`fem/mesh/mesh.py` (`perturb`), `fem/mesh/kernels.py` (`stiffness_degree`: T10 default rule degree 2 = Keast 4-point).

## Historical facts to reproduce first (stated as read, to be re-measured, not assumed)

* V3 T4, n = 4, 8, 16, seeds 100 + n, vertices +-0.2 h, c = 1, nu = 0.3: L2 rate 1.796 (threshold 1.8), energy 0.92.
* V3 T10, n = 2, 4, 8, same seeds rule: energy rate 1.788 (threshold 1.8), L2 2.92.
* Post hoc (historical): T4 16 -> 24: 1.92 / 0.97; T10 8 -> 12: 3.00 / 1.89.
* G2b: curved (mid nodes +-0.1 h) T10, rule degree 3: u error 9.9e-16. G2c: T10, default 4-point rule: u 1.7e-4, stress 2.6e-3
  (mesh `_mesh("T10", curved=0.1)`, n = (4, 3, 2), seed 3, vertices +-0.2 h).

## R05 (acceptance gate F2): curved T10 patch test against quadrature degree. Derivation of the expectation

Patch field u = A x + a. Isoparametric map x(xi) = sum N_i x_i with the same P2 functions as the displacement, so
u_h = sum N_i u_i = A x(xi) + a for any map (x itself is in the space): u = A x + a is in the discrete space on every
(curved) mesh, strain B u_e = A at every point, stress sigma = D eps constant (this holds at every quadrature
degree). The only way the discrete solution can miss it is an inexact stiffness / internal-force integral.

Nodal internal force of node i: f_i = integral over the cell of (grad_x N_i)^T sigma dV = sigma-contracted
integral of grad_x N_i detJ dxi, with grad_x N_i = J^-T grad_xi N_i and J^-T detJ = cof(J) (cofactor matrix, 3-D).
With sigma constant, the integrand to be integrated exactly is cof(J) grad_xi N_i (no division by detJ appears: the
rational parts cancel). Degrees in xi for the quadratic map x = x_aff + c b(xi) (x_aff the affine map of the four
vertices, b(xi) the quadratic bubble built from the mid-node displacements d_k of size c, c = curved amplitude):
J = J0 + c J1(xi), J0 constant, J1 degree 1. cof(J) = cof0 + c cof1 (degree 1) + c^2 cof2 (degree 2), since each cofactor
entry is a product of two J entries. grad_xi N_i is degree 1. Integrand degree: c^0 term 1, c^1 term 2, c^2 term 3.
Consequences:

* A rule exact to degree 3 integrates the integrand exactly; the patch test then holds up to rounding for every
  curved T10 cell (G2b used degree 3: consistent).
* The degree-2 Keast rule integrates the c^0 and c^1 terms exactly and misses only the c^2 term: the patch error
  at degree 2 is proportional to c^2 to leading order (doubling c multiplies the quadrature error of the cell integral by
  exactly 4; the assembled u error by 4 up to O(c) relative corrections). At c = 0 (straight edges, affine tets) it is
  rounding.
* Hence "never exact" is false for this field: the patch test on a curved P2 tetrahedron becomes exact at rule degree 3,
  and degree 2 (production default, `stiffness_degree` = 2 (p-1)) is one degree short. For nonconstant stress (general
  fields) the integrand is rational (division by detJ in the stress gradient terms) and no polynomial rule is exact:
  that is outside this derivation and not claimed.
* For T6 (2-D) the same count gives cof degree 1 times grad N degree 1 = 2, equal to the default degree 2(p-1) = 2:
  consistent with G2b passing at the default (not re-run here).

| ID | Hypothesis | Reference | Metric | Threshold |
|---|---|---|---|---|
| F2-1 | For one curved T10 cell the integrand cof(J) grad_xi N_i has total polynomial degree exactly 3; its degree-3 part is proportional to c^2 (zero at c = 0). The 4-point Keast rule misses the cell integral by an amount that scales exactly as c^2; the degree-3 collapsed rule does not miss it. | sympy: own P2 shape functions (barycentric formulas), own map, own cofactor, exact monomial integrals a! b! c! / (a+b+c+3)!; Keast constants typed in the test | polynomial degree; max abs of degree-3 part at c = 0; relative quadrature error of the cell force at degree 2 and degree 3; ratio of the degree-2 errors at c and c/2 | degree = 3; degree-3 part at c = 0 exactly 0; degree-2 relative error > 1e-6 at c = 0.1 h; degree-3 error <= 1e-12; ratio in [3.99, 4.01] |
| F2-2 | Reproduction: `_mesh("T10", curved=0.1)` seed 3, degree 2: u error 1.7e-4, stress error 2.6e-3 (two significant digits); degree 3: u error <= 1e-12 | production, same call as G2b / G2c | relative u error, relative stress error | as stated |
| F2-3 | Patch error against rule degree 2, 3, 4, 6, 8 on the same mesh family, seeds 3, 4, 5, 6, 7 (curved 0.1 h, vertices 0.2 h): degree 2 fails detectably, degree >= 3 passes to rounding | exact field A x + a (no independent solver needed) | max u error relative to max u; max stress error relative | degree 2: u error in [1e-6, 1e-2] for every seed; degree >= 3: u error <= 1e-10 and stress error <= 1e-9 for every seed and every degree |
| F2-4 | At degree 2 the u error is proportional to c^2: c = 0, 0.025, 0.05, 0.1 (units of h), seed 3 | exact field | u error | c = 0: <= 1e-10; ratios e(0.05)/e(0.025) and e(0.1)/e(0.05) each in [3, 5] |

Seed 3, 4, ... : `_mesh(name, seed=s, curved=c)` as in `test_mesh_assembly.py` (a different seed gives different vertex
and mid-node draws; the mid-node draws are `uniform(-c h, c h)` so for equal seed they scale linearly with c).
A detJ inversion at some seed (the mesh is refused) is a test-set-up event, recorded, and the seed is replaced by the next.

## R04 (acceptance gate F4): causes of the two missed rates

Candidate causes and the measurement that separates each:

* (a) reference error: the exact solution u = curl psi + c grad psi, gradients and body force are exact polynomials
  (module notes); possible defects are an implementation slip in `Manufactured` or in `problem.errors`.
  Measured by F4-5 (sympy for Manufactured; an independent dense error integral for `errors`).
* (b) mesh generator and shape regularity: the V3 meshes draw a new random vertex field at every level
  (seed 100 + n); a pair rate therefore contains the ratio of two unrelated mesh-quality draws.
  Measured by F4-3 (other seed families, undistorted mesh).
* (c) quadrature of the error integral (degree 2k + 4: T4 6, T10 8) and of the load (`body_force`, degree 2k + 4).
  The exact error integrand is a polynomial of degree up to 22 (u has total degree 11 per component in 3-D) so these
  rules are not exact. Measured by F4-2 (rate recomputed with degree +6, +12; with the load degree +6).
* (d) pre-asymptotic behaviour: the rate is still below its limit at the historical levels. Measured by F4-1 (the
  interpolation error of the exact solution in the same space on the same meshes, no solve, to much finer levels) and
  by F4-4 (further solved levels, as memory permits).

Pair rate p(n1, n2) = ln(e(n1)/e(n2)) / ln(n2/n1) exactly as `_rates` (last two levels). Gap = 1.8 - p_hist
(T4 L2: 0.004, T10 energy: 0.012).

| ID | Hypothesis | Reference | Metric | Threshold |
|---|---|---|---|---|
| F4-0 | Reproduction: the historical settings give the ledger numbers, and two runs are bitwise identical | the ledger / `docs/olcumler/fem_genel.md`; second run | T4 (4, 8, 16): L2 p, energy p; T10 (2, 4, 8): L2 p, energy p; `==` on all error floats between two runs | T4 L2 1.796 (+-0.0006), T4 energy 0.92 (+-0.006), T10 L2 2.92 (+-0.006), T10 energy 1.788 (+-0.0006); bitwise equal |
| F4-1 | (d) The interpolation error of the exact field (nodal interpolation into the same space, no solve, `problem.errors` of the nodal vector) shows the same pre-asymptotic rates as the Galerkin solution at the historical pairs, and reaches the asymptotic window on finer levels. T4: n = 8, 16, 24, 32; T10: n = 4, 8, 12, 16, 24; seeds 100 + n | the exact field itself; consistency with F4-5 | pair rates: T4 L2, T10 energy | (i) T4 L2 interpolation p(8, 16) within 0.10 of the Galerkin 1.796 and T10 energy interpolation p(4, 8) within 0.10 of 1.788; (ii) last pair >= 1.9 (T4 (24, 32) L2; T10 (16, 24) energy); (iii) last pair > first pair + 0.05 |
| F4-2 | (c) Quadrature does not cause the miss. The pair rate recomputed with error-norm degree +6 and +12, and with load degree +6, differs from the historical by less than half the gap and stays below 1.8 | the same solves, `errors(degree=...)`, `body_force(degree=...)` | pair rate change | abs(dp) <= 0.002 (T4 L2), <= 0.006 (T10 energy) for every variant; p < 1.8 for every variant |
| F4-3 | (b) The historical miss is within the scatter of the mesh draw: with base seeds 200, 300, 400, 500 (seed = base + n) at least 2 of the 4 give the historical pair rate >= 1.8; the undistorted structured mesh gives >= 1.8 | same pipeline, other seeds; `box(...)` with no perturbation | T4 L2 p(8, 16), T10 energy p(4, 8) | count >= 2 of 4; undistorted >= 1.8 (two separate checks) |
| F4-4 | (d) Further solved levels move toward the asymptote: T4 p(16, 24) L2 >= 1.9 (hist. 1.92, reproduce +-0.01), then p(24, 32) >= 1.8; T10 p(8, 12) energy 1.89 (+-0.01), then p(12, 16) >= 1.8. Levels beyond 12 / 24 are run only under the memory rule below | solves | pair rates | as stated |
| F4-5 | (a) `Manufactured` and `problem.errors` are correct. (i) `u`, `grad`, `body_force` equal sympy derivatives of psi = prod g(x_d) at 20 random points in 3-D (c = 1, lam = 1.15385, mu = 0.76923); (ii) an independent dense error integral (monomial fit of the nodal field per cell, own collapsed Gauss-Legendre rule 12 points per direction on the physical tet, own Lame energy) equals `problem.errors(degree=24)` on T4 n = 2 and T10 n = 2 (random nodal field, straight-sided cells) | sympy; dense numpy in the test | max relative difference of u, grad, f; relative difference of l2 and energy | (i) <= 1e-11; (ii) <= 1e-9 |

Memory rule (shared machine, ~3 GB): solved levels run in order; the DOF cap is about 110 k (T4 n = 32: 107 811 DOF;
T10 n = 16: 107 811 DOF) and a level is skipped, and said to be skipped, when the previous level's measured peak
working set times 2.5 exceeds the free memory at that moment or the previous level took > 600 s. T10 n >= 20
(> 200 k DOF) is not attempted. Skipped levels are recorded as "not resolvable here"; they are not guessed.

Verdict rules for the record `F4_RATES.md` (fixed now): R04 is "resolved cause" for the cause whose test fails to be a
null result by the thresholds above and whose removal restores the rate >= 1.8; "still pre-asymptotic" if F4-1 and F4-4
agree and F4-2, F4-3, F4-5 are null; "not resolvable here" if the finer levels cannot be reached. R05 is "resolved cause"
if F2-1..F2-4 hold. If a check fails, its test stays as strict xfail with the measured value and a ledger entry in
`docs/foundations/ledger_additions/F4_rates.json`.

Run: `python -m pytest tests/toporlogy/test_fem_rates_investigation.py -q -p no:randomly [-m slow]`.
First results are appended below (each with a clock time).
