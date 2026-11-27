# M_mesh — general mesh topology, validation, quality, point location (preregistration)

Preregistered 2026-11-27 03:54 +0300, base SHA `b993b6e` (branch of `foundations/round-3`), before any test of the new modules was
written or run. Plan source: WP07 (`07_MESH_TOPOLOGY_IO.md`) and acceptance rows M1-M5 of `23_ACCEPTANCE_CASES.md`.
Modules (new, nothing existing edited): `src/toporlogy/fem/mesh/topology.py` (validation, renumbering, hanging
nodes), `quality.py` (quality metrics, Jacobian certificate), `locate.py` (inverse isoparametric map, point
location, interpolation). Tests: `tests/toporlogy/test_mesh_foundations.py`. Design: `docs/foundations/MESH_TOPOLOGY.md`.

Read before writing this record (this session, from source): `mesh.py` (a `Mesh` holds ONE element type; a mapping
of cells is refused with "mixed cell types are not supported"), `reference.py`, `kernels.py` (signed detJ,
`element_scale`, `DETJ_TOL = 1e-12`), `core/vtu.py` (ASCII writer `write_cells_vtu` and a reader `read_vtu`, stdlib
only). `import gmsh` fails (ModuleNotFoundError) and `import meshio` fails: no Gmsh/meshio reader can be run here.

## Scope statements fixed in advance

* Mixed-cell-type meshes are UNSUPPORTED by the existing `Mesh`; M5 tests the explicit refusal, not an affine patch
  test on a mixed mesh. The only M5 content that is implemented: detection of hanging nodes (2-D) as a
  non-conforming interface that no MPC resolves.
* M3 Gmsh physical-group import: NOT VERIFIED (dependency absent, no test). VTU roundtrip: a reader exists
  (`read_vtu`, stdlib); the test compares against an independent reader written in the test file.
* M6 (periodicity) is not part of this slice.
* Quality is *sampled* (reference nodes and a quadrature rule of degree 2k); the Jacobian certificate is a separate
  quantity (Bernstein lower bound), and no universal quality threshold is prescribed: flagging takes an explicit
  threshold argument.

## Definitions (fixed before the run)

Mean-ratio / shape quality. Let J = ∂x/∂ξ at a sample point and A the Jacobian of the map from the reference cell
to the ideal (regular) cell: A = I for Q4, H8 (the reference square/cube is already regular); for the reference
triangle (0,0),(1,0),(0,1) → equilateral with unit edges, A = [[1, 1/2],[0, √3/2]]; for the reference tetrahedron →
regular with unit edges, A = [[1, 1/2, 1/2],[0, √3/2, √3/6],[0, 0, √(2/3)]] (columns are the images of e1, e2, e3;
|col3|² = 1/4 + 1/12 + 2/3 = 1, col2·col3 = 1/4 + 1/4 = 1/2, so all six edges have length 1). S = J A⁻¹. With
d the dimension:

* mean ratio `q = d · sgn(det S) |det S|^{2/d} / ‖S‖_F²` (= 1 iff S is a similarity, ≤ 1 by the AM-GM inequality
  on the singular values, negative iff detJ < 0);
* scaled Jacobian `SJ = det S / Π_j ‖S e_j‖` (columns);
* condition `κ = σmax(S) / σmin(S)` (inf if singular);
* edge ratio = longest / shortest straight vertex-to-vertex edge;
* sampled `detJ/s_e` with s_e = (longest edge)^d as in `kernels.element_scale`.

Per cell the minimum of q, SJ, detJ/s_e and the maximum of κ over the samples are reported. All of q, SJ, κ are
invariant under x → c R x + t (c > 0, R a rotation): J → c R J, S → c R S.

Jacobian certificate. detJ(ξ) is a polynomial: for simplices of degree m = d(k−1) (T3, T4: m = 0; T6: 2; T10: 3) in
the barycentric coordinates; for tensor cells of degree n per variable (Q4: 1, Q9: 3, H8: 2; derived: each entry of J
has degree ≤ 1 (Q4) / ≤ 2 (Q9) / ≤ 1 per variable in the two other columns (H8) and the determinant picks one entry
per column). The Bernstein coefficients c_a of that polynomial are obtained exactly from its values on the lattice
of degree m (n), and `min_a c_a ≤ min_cell detJ`. `certified` = (min c_a / s_e > DETJ_TOL). An uncertified cell
with positive sampled detJ is "uncertain"; with a negative sample, "inverted".

Hanging node (2-D): a used vertex that lies on a boundary facet strictly between its end vertices, within
`tol · facet length`, and is not an end vertex of it.

Point location. For a physical point p and a candidate cell, Newton's method on r(ξ) = x(ξ) − p from the reference
centroid with step halving. Let g(ξ) = min(ξ₁..ξ_d, 1 − Σξ) (simplex) or 1 − max|ξ_i| (tensor cell). Interior if
g > tol, boundary if |g| ≤ tol, outside if g < −tol, tol = 1e-10 in reference coordinates. One interior cell:
INSIDE; ≥ 2 interior cells: AMBIGUOUS (overlap), lowest id; only boundary hits: BOUNDARY, lowest id; no hit:
OUTSIDE, cell −1, ξ = NaN, interpolation NaN (no extrapolation).

Renumbering. New cell k is old cell `cell_perm[k]`; new node j is old node `node_perm[j]`; `cells_new[k] =
node_inv[cells[cell_perm[k]]]` with `node_inv[node_perm[j]] = j`; a facet set entry (c, f) becomes
(`cell_inv[c]`, f), a node set entry n becomes `node_inv[n]`. Node-major DOF d = ncomp·node + comp, so
`u_new[ncomp·j + c] = u_old[ncomp·node_perm[j] + c]`.

## Checks

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| M1a | T3, Q4, T6, T4, H8 perturbed meshes: the elastic solution (fixed left, traction on right, constant body force) of a random cell+node renumbering, mapped back with `to_old_nodal`, equals the original-numbering solution; compliance fᵀu equal | the same solver in the original numbering (an invariance test, NOT an independent reference) | max abs diff / max abs u ≤ 1e-12; compliance rel ≤ 1e-12 |
| M1b | cell tags and facet/node sets follow the renumbering: tags[cell_perm], facet nodes' coordinates and node-set coordinates are the same point sets | coordinates of the original mesh | exact equality |
| M1c | validation reports, with exact ids: coincident nodes (pair), duplicate cell (also with permuted node order), a facet shared by 3 cells, unused node, negative-orientation cell, repeated node in a cell; out-of-range index is refused by `Mesh` itself; `require_valid` raises `MeshValidationError` naming them | meshes built by hand with known defects | exact ids |
| M1d | valid generated meshes (rectangle T3/Q4/T6, box T4/H8/T10, perturbed) report no finding (negative control); `renumber` refuses a non-permutation | — | exact |
| M1e | negative control: mapping back with the wrong direction (`to_new_nodal`) differs by > 1e-3 relative | — | > 1e-3 |
| M2a | regular elements give the ideal value: equilateral T3, regular T4 (also as straight T6, T10), unit square Q4, unit cube H8: q = SJ = κ = 1 | definitions above | abs ≤ 1e-12 |
| M2b | closed forms: right isosceles T3 (0,0),(1,0),(0,1): q = √3/2, SJ = 2/√5 (derivation: S = A⁻¹ = [[1, −1/√3],[0, 2/√3]], det S = 2/√3, ‖S‖_F² = 8/3, column norms 1 and √(5/3)); rectangle a×1 Q4 (J = diag(a/2, 1/2)): q = 2a/(a²+1), κ = a, SJ = 1 (a = 1, 2, 10); box a×b×c H8: q = 3(abc)^{2/3}/(a²+b²+c²); sliver tet (regular tet, apex height × ε): J = diag(1,1,ε)·A, S = diag(1,1,ε): q = 3 ε^{2/3}/(2+ε²), κ = 1/ε, **SJ = 1 (SJ does not see this sliver)** | the closed forms | rel ≤ 1e-12 |
| M2c | invariance: random perturbed T3/Q4/T4/H8 meshes under x → s R x (s = 1e-6, 1, 1e3; random rotation) and one translation by 3 at s = 1: min q, min SJ, max κ, edge ratio unchanged | the same mesh unscaled | rel ≤ 1e-12 |
| M2d | flagging with an explicit threshold: the sliver (ε = 1e-3) has q < 0.05 and is flagged, the regular tet is not; a graded Q4 mesh reports edge ratio and κ matching closed forms | closed forms | exact flags; rel 1e-12 |
| M2e | inverted cells found by id: T3 with two nodes swapped has q = −q_orig and is in `negative_cells`; Q4 "dart" (reentrant corner) and H8 with a node pulled through the cell are reported inverted, certified False | by construction | exact ids |
| M2f | Jacobian certificate: straight T3, T4, T6, T10 and a parallelogram Q4 are certified with lower bound equal to the constant detJ (rel 1e-12); for 200 000 random T6 cells (reference vertices, midside nodes = straight midpoint + U(−0.4, 0.4)² each component, `default_rng(12345)`) the bound is ≤ the dense-lattice minimum of detJ (41×41 barycentric lattice, own T6 shape functions in the test) + 1e-12 s_e for every cell; every certified cell has dense min > 0; **at least one cell has all samples positive, dense min < 0 and certified False** (the inversion between sample points that sampling misses) | dense lattice with test-side shape functions | bound property exact up to 1e-12 s_e; existence count ≥ 1 |
| M4a | known ξ recovered: affine and non-affine Q4, H8, T3, T4, T6 (curved), cell id equal and ξ error ≤ 1e-12 (physical points built from test-side shape functions) | test-side formulas for x(ξ) | abs ≤ 1e-12 |
| M4b | classification: midpoint of a boundary facet and a vertex shared by several cells → BOUNDARY, cell = lowest id among those containing it; the same point moved 1e-6 outward → OUTSIDE (cell −1, interpolation NaN); interior points INSIDE | geometry of the rectangle | exact |
| M4c | interpolation of an affine field a + B x at located points on distorted Q4 and on curved T6 meshes is exact | a + B p | rel ≤ 1e-12 |
| M4d | negative controls: the affine (3-vertex) inverse on a distorted Q4 errs by > 1e-3 in ξ; interpolation of x² on Q4 is not exact (> 1e-6) | — | > thresholds |
| M5a | `Mesh` with a mapping of T3 and Q4 cell blocks raises ValueError containing "mixed" (mixed meshes UNSUPPORTED; no patch test) | — | exact |
| M5b | hanging node: the 3-cell mesh L (0,0),(0.5,−1),(1,0), U1 (0,0),(0.5,0),(0.25,1), U2 (0.5,0),(1,0),(0.75,1) (points ordered (0,0),(0.5,−1),(1,0),(0.5,0),(0.25,1),(0.75,1); cells L = 0,1,2, U1 = 0,3,4, U2 = 3,2,5) has node 3 = (0.5,0) hanging on L's boundary facet (cell 0, local facet 2); conforming generated T3/Q4/T6 meshes have none; for 3-D meshes the field is None (NOT CHECKED) | by construction | exact |
| M3a | VTU roundtrip of a renumbered mesh with a nodal vector field and cell tags for T3, Q4, T4, H8: `write_cells_vtu` then an independent ElementTree reader in the test recovers points (zero padded to 3), connectivity, types, fields bit for bit | the test's own parser | `np.array_equal` |
| M3b | Gmsh physical-group import | — | NOT VERIFIED, no test (no `gmsh`, no `meshio`) |

Run: `python -m pytest tests/toporlogy/test_mesh_foundations.py -q -p no:randomly`. First result appended below.

## First result (2026-11-27 04:05 +0300, base SHA `b993b6e` + the three new modules and the test file)

`python -m pytest tests/toporlogy/test_mesh_foundations.py -q -p no:randomly`: 37 passed on the first run, no
threshold or hypothesis changed, no test set-up correction needed (the only edits after the first run: a
`@pytest.mark.slow` marker on M2f, which takes about 105 s, and type annotations in `locate.py` for mypy).

| ID | Result | Measured |
|---|---|---|
| M1a | PASS (T3, Q4, T6, T4, H8) | solution max rel diff 6.2e-16, 8.7e-16, 4.5e-15, 1.8e-15, 1.1e-15; compliance 3.6e-16, 1.8e-16, 8.7e-16, 0, 2.8e-16 |
| M1b, M1c, M1d | PASS | exact ids as predicted |
| M1e | PASS | wrong-direction map > 1e-3 relative |
| M2a | PASS | q, SJ, κ, edge ratio = 1 within 1e-12 (T3, T6, T4, T10, Q4, H8; all certified) |
| M2b | PASS | closed forms within 1e-12 rel; SJ = 1 for the sliver (documented blindness confirmed) |
| M2c | PASS | worst relative change 2.3e-15 (T3), 3.0e-15 (Q4), 6.0e-15 (T4), 4.1e-15 (H8) |
| M2d, M2e | PASS | flags and inverted ids as predicted |
| M2f | PASS | max(bound − dense min)/s_e = 1.3e-15; 31 524 of 200 000 random T6 certified; 2 360 cells had all samples positive, dense min < 0 and were not certified (prediction was ≥ 1) |
| M4a | PASS | worst ξ error 1.4e-15 (Q4, H8, T3, T4, T6; affine and non-affine) |
| M4b | PASS | boundary / outside / tie rule as predicted |
| M4c | PASS | interpolation error 3.0e-16 (Q4), 5.0e-16 (curved T6) |
| M4d | PASS | negative controls exceed their thresholds |
| M5a, M5b | PASS | mixed refusal; hanging node (0, 2, 3) found, none in conforming meshes, 3-D reports None |
| M3a | PASS | bitwise VTU roundtrip for T3, Q4, T4, H8 with the test's own reader |
| M3b | NOT VERIFIED | no gmsh, no meshio |

Not claimed: the Bernstein bound is exact for the determinant polynomial of the element's own map; it says nothing
about boundary-curved geometry beyond the mesh; the T6 inversion count is for one seeded random family, not a rate.
