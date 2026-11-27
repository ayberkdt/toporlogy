# Mesh topology, validation, quality and point location (WP07 slice)

Record: `docs/foundations/prereg/M_mesh.md`; tests: `tests/toporlogy/test_mesh_foundations.py`. Modules (all in
`src/toporlogy/fem/mesh/`, none exported from the package `__init__`; import them by module):

| Module | Content |
|---|---|
| `topology.py` | `validate_mesh`, `require_valid`, `MeshReport`, `MeshValidationError`, `hanging_nodes`, `renumber`, `Renumbering` |
| `quality.py` | `cell_quality`, `quality_from_nodes`, `CellQuality`, `flag_cells`, `sampled_detj`, `ideal_jacobian` |
| `locate.py` | `locate`, `inverse_map`, `interpolate`, `Location`, `INSIDE`, `BOUNDARY`, `OUTSIDE`, `AMBIGUOUS` |

## What is and is not supported

* A `Mesh` holds one element type. **Mixed cell blocks (T3 + Q4, tet + hex) are unsupported and refused** with
  `ValueError("... mixed cell types are not supported")`; there is no mixed-mesh patch test, and none was faked.
* **Gmsh / meshio import is NOT VERIFIED here** (`gmsh` and `meshio` are not importable). VTU output exists
  (`core/vtu.py`, T3/Q4/T4/H8 only); its roundtrip is tested against a reader written in the test.
* Non-conforming interfaces: detected only as *hanging nodes* in 2-D (a vertex strictly inside a boundary facet).
  3-D reports `hanging_nodes = None` (not checked). No multi-point constraints exist, so a hanging-node mesh must be
  rejected, not solved.
* Periodicity (M6) and field transfer between meshes (L² projection, conservation) are not in this slice.

## Validation (`validate_mesh`)

Report only; nothing is merged, flipped or repaired. Defects, all by id: coincident nodes (pair; `same_piece` tells
whether both lie in one connected piece, so deliberately disconnected coincident geometry is distinguishable),
duplicate cells (same node set in any order), a node repeated within a cell, facets shared by more than two cells,
unused nodes, negative signed detJ (sampled at the reference nodes and a degree-2k rule), degenerate cells,
hanging nodes. An out-of-range node index never reaches a `Mesh` (the constructor refuses it).
`require_valid` raises `MeshValidationError` whose message and `report` name the ids.

## Renumbering

`renumber(mesh, cell_perm, node_perm)`: new cell k is old cell `cell_perm[k]`, new node j is old node
`node_perm[j]`; local node and facet order are untouched, so facet sets `(cell, local facet)` only change the cell
number. Tags, facet sets and node sets follow. `Renumbering.to_old_nodal(u, ncomp)` maps a node-major DOF vector
back (`u_old[ncomp·node_perm[j] + c] = u_new[ncomp·j + c]`); `to_new_nodal` is the inverse; `to_new_cell` /
`to_old_cell` do the same for per-cell fields. Check M1a: the elastic solution is the same to 5e-15 relative for
T3, Q4, T6, T4, H8 (an invariance test, not an independent reference).

## Quality

With S = J A⁻¹ (A = identity for Q4/H8; unit-edge equilateral triangle / regular tetrahedron for T3, T6 / T4, T10):

* mean ratio q = d · sgn(det S) |det S|^{2/d} / ‖S‖²_F (1 ideal, → 0 sliver, negative inverted);
* scaled Jacobian SJ = det S / Π‖S eⱼ‖ (**blind to a sliver tetrahedron that keeps orthogonal columns:** SJ = 1 for
  the apex-flattened regular tet, while q = 3ε^{2/3}/(2+ε²) and κ = 1/ε);
* condition κ = σmax/σmin of S, edge ratio, sampled detJ/s_e.

q, SJ, κ, edge ratio are invariant under scaling, rotation and translation (measured ≤ 6e-15 relative). They are
*sampled* quantities (reference nodes plus a degree-2k rule). `flag_cells(quality, mean_ratio)` takes the threshold
from the caller: there is no universal acceptance value, and a valid mesh can still be inaccurate or badly
conditioned.

### Jacobian certificate (sampled ≠ certified)

detJ is a polynomial of degree d(k−1) in the barycentric coordinates (simplices) or 1 / 3 / 2 per variable (Q4 / Q9 /
H8). Its Bernstein coefficients are computed exactly from its values on the corresponding lattice; their minimum
`bound_rel` (relative to s_e) bounds detJ from below over the whole cell. `certified` ⇔ `bound_rel > DETJ_TOL`.
Cells with positive samples but no certificate are `uncertain`. Measured on 200 000 seeded random curved T6 cells:
the bound was never above the dense-lattice minimum (max excess 1.3e-15 s_e), 31 524 cells certified, and 2 360 cells
had all samples positive yet a negative dense minimum, i.e. an inversion that sampling alone does not see.

## Point location (`locate`)

KD-tree on padded per-cell boxes, then Newton on x(ξ) = p with step halving for every cell type (affine or not).
Status per point: `INSIDE` (one cell with ξ strictly inside, tolerance 1e-10 in reference coordinates), `BOUNDARY`
(on a facet / edge / vertex, lowest cell id wins, `n_hits` counts all), `AMBIGUOUS` (several strictly interior cells,
an overlapping invalid mesh), `OUTSIDE` (cell −1, ξ NaN). `interpolate` returns NaN outside: no extrapolation. Known
ξ is recovered to 1.4e-15 (affine and non-affine Q4, H8, T3, T4, curved T6); an affine field interpolates exactly
(3e-16 on Q4, 5e-16 on curved T6). Limits: the candidate search is a bounding-box heuristic (a 10 % pad), not a proof
for extremely curved cells; Newton uses the reference centroid as the start.
