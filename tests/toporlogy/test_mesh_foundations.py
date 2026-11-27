"""WP07 general mesh foundations (docs/foundations/prereg/M_mesh.md): validation, renumbering, quality and Jacobian
certificate, point location, hanging nodes, VTU roundtrip. Gmsh / meshio import is NOT VERIFIED (neither installed)."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from toporlogy.core.linear_elastic import isotropic_plane_strain, isotropic_stiffness
from toporlogy.core.vtu import write_cells_vtu
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, Mesh, box, get, perturb, rectangle
from toporlogy.fem.mesh.locate import BOUNDARY, INSIDE, OUTSIDE, interpolate, locate
from toporlogy.fem.mesh.quality import cell_quality, flag_cells, quality_from_nodes
from toporlogy.fem.mesh.topology import (MeshValidationError, hanging_nodes, renumber, require_valid,
                                         validate_mesh)

S3 = np.sqrt(3.0)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(b), 1e-300)))


# ------------------------------------------------------------------------------------------------ test-side maps
_CORNERS_H8 = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_CORNERS_Q4 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)


def _N(name: str, xi: np.ndarray) -> np.ndarray:
    """Shape functions written independently of the production reference elements."""
    if name == "Q4":
        return np.prod(1 + _CORNERS_Q4 * xi, axis=1) / 4
    if name == "H8":
        return np.prod(1 + _CORNERS_H8 * xi, axis=1) / 8
    if name == "T3":
        return np.array([1 - xi.sum(), xi[0], xi[1]])
    if name == "T4":
        return np.array([1 - xi.sum(), xi[0], xi[1], xi[2]])
    if name == "T6":
        a, b, c = 1 - xi.sum(), xi[0], xi[1]
        return np.array([a * (2 * a - 1), b * (2 * b - 1), c * (2 * c - 1), 4 * a * b, 4 * b * c, 4 * c * a])
    raise KeyError(name)


def _t6_dN(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n, 6, 2) derivatives of the T6 shape functions at (s, t), hand-derived (L1 = 1-s-t, L2 = s, L3 = t)."""
    a = 1 - s - t
    z = np.zeros_like(s)
    d = np.stack([
        np.stack([-(4 * a - 1), -(4 * a - 1)], -1),
        np.stack([4 * s - 1, z], -1),
        np.stack([z, 4 * t - 1], -1),
        np.stack([4 * (a - s), -4 * s], -1),
        np.stack([4 * t, 4 * s], -1),
        np.stack([-4 * t, 4 * (a - t)], -1)], axis=1)
    return d


def _random_xi(name: str, rng: np.random.Generator) -> np.ndarray:
    dim = 3 if name in ("H8", "T4") else 2
    if name in ("Q4", "H8"):
        return rng.uniform(-0.8, 0.8, dim)
    lam = rng.dirichlet(np.ones(dim + 1))
    c = np.full(dim, 1.0 / (dim + 1))
    return c + 0.8 * (lam[1:] - c)


# ------------------------------------------------------------------------------------------------ M1
def _solve(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    dim = mesh.dim
    D = isotropic_plane_strain(1.0, 0.3) if dim == 2 else isotropic_stiffness(1.0, 0.3)
    P = ElasticProblem(mesh, D)
    f = P.body_force([0.1, -0.2, 0.05][:dim]) + P.traction(mesh.facet_sets["right"], [1.0, 0.3, 0.2][:dim])
    s = P.solve(f, Dirichlet.fixed(mesh.dofs(mesh.node_sets["left"])))
    return s.u[:, 0], f


def _meshes() -> dict[str, Mesh]:
    return {"T3": perturb(rectangle(4, 3, element="T3"), 0.05, seed=1),
            "Q4": perturb(rectangle(4, 3, element="Q4"), 0.05, seed=2),
            "T6": perturb(rectangle(3, 2, element="T6"), 0.05, seed=3, curved=0.02),
            "T4": perturb(box(2, 2, 2, element="T4"), 0.06, seed=4),
            "H8": perturb(box(2, 2, 2, element="H8"), 0.06, seed=5)}


@pytest.mark.parametrize("name", ["T3", "Q4", "T6", "T4", "H8"])
def test_m1a_renumbering_gives_the_same_solution(name: str) -> None:
    m = _meshes()[name]
    rng = np.random.default_rng(10)
    new, ren = renumber(m, rng.permutation(m.n_cells), rng.permutation(m.n_nodes))
    u0, f0 = _solve(m)
    u1, f1 = _solve(new)
    back = ren.to_old_nodal(u1, ncomp=m.dim)
    err = np.abs(back - u0).max() / np.abs(u0).max()
    cerr = abs(f1 @ u1 - f0 @ u0) / abs(f0 @ u0)
    print(f"M1a {name}: solution {err:.2e}, compliance {cerr:.2e}")
    assert err <= 1e-12
    assert cerr <= 1e-12
    # the forward map is the exact inverse of the backward map
    assert np.array_equal(ren.to_new_nodal(back, ncomp=m.dim), u1)
    # negative control (M1e): the wrong direction is far off
    wrong = ren.to_new_nodal(u1, ncomp=m.dim)
    assert np.abs(wrong - u0).max() / np.abs(u0).max() > 1e-3


@pytest.mark.parametrize("name", ["T3", "Q4", "T6", "T4", "H8"])
def test_m1b_tags_and_sets_follow_the_renumbering(name: str) -> None:
    m = _meshes()[name]
    m.cell_tags = np.arange(m.n_cells) * 7 + 1
    rng = np.random.default_rng(11)
    cp, npm = rng.permutation(m.n_cells), rng.permutation(m.n_nodes)
    new, ren = renumber(m, cp, npm)
    assert np.array_equal(new.cell_tags, m.cell_tags[cp])
    assert np.array_equal(ren.to_old_cell(new.cell_tags), m.cell_tags)
    # cell k of the new mesh has the node coordinates of old cell cp[k], in the same local order
    assert np.array_equal(new.points[new.cells], m.points[m.cells[cp]])
    for key in m.facet_sets:
        a = m.points[m.facet_nodes(m.facet_sets[key])]
        b = new.points[new.facet_nodes(new.facet_sets[key])]
        order_a = np.lexsort(a.reshape(len(a), -1).T)
        order_b = np.lexsort(b.reshape(len(b), -1).T)
        assert np.array_equal(a[order_a], b[order_b])
        assert np.array_equal(m.cell_tags[m.facet_sets[key][:, 0]].sum(), new.cell_tags[new.facet_sets[key][:, 0]].sum())
    for key in m.node_sets:
        ca = m.points[m.node_sets[key]]
        cb = new.points[new.node_sets[key]]
        assert np.array_equal(ca[np.lexsort(ca.T)], cb[np.lexsort(cb.T)])


def test_m1c_validation_reports_defects_with_ids() -> None:
    base = rectangle(2, 2, element="T3")
    P, C = base.points, base.cells
    # coincident nodes: node 9 duplicates node 4 and replaces it in one cell
    c0 = int(np.flatnonzero((C == 4).any(axis=1))[0])
    C2 = C.copy()
    C2[c0][C2[c0] == 4] = 9
    r = validate_mesh(Mesh(np.vstack([P, P[4]]), C2, "T3"))
    assert r.duplicate_nodes.tolist() == [[4, 9]]
    assert r.duplicate_node_same_piece.tolist() == [True]
    assert not r.ok and "coincident nodes [[4, 9]]" in r.summary()
    # deliberate disconnected coincident geometry: two squares with their own nodes along x = 1
    Pq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [2, 0], [2, 1], [1, 1]], float)
    r = validate_mesh(Mesh(Pq, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), "Q4"))
    assert r.duplicate_nodes.tolist() == [[1, 4], [2, 7]]
    assert r.duplicate_node_same_piece.tolist() == [False, False]
    # duplicate cell with permuted node order
    r = validate_mesh(Mesh(P, np.vstack([C, C[3][[1, 2, 0]]]), "T3"))
    assert r.duplicate_cells.tolist() == [[3, 8]]
    # a facet shared by three cells (edge 0-1)
    Pn = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, -1], [0.5, 2]], float)
    r = validate_mesh(Mesh(Pn, np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]]), "T3"))
    assert len(r.nonmanifold_facets) == 1
    assert sorted(map(tuple, r.nonmanifold_facets[0].tolist())) == [(0, 0), (1, 2), (2, 0)]
    # unused node
    r = validate_mesh(Mesh(np.vstack([P, [[5.0, 5.0]]]), C, "T3"))
    assert r.unused_nodes.tolist() == [9]
    # negative orientation by cell id
    C3 = C.copy()
    C3[5] = C3[5][[0, 2, 1]]
    r = validate_mesh(Mesh(P, C3, "T3"))
    assert r.negative_cells.tolist() == [5]
    with pytest.raises(MeshValidationError, match=r"\[5\]") as ei:
        require_valid(Mesh(P, C3, "T3"))
    assert ei.value.report.negative_cells.tolist() == [5]
    # a node named twice in a cell
    C4 = C.copy()
    C4[2] = [C4[2, 0], C4[2, 1], C4[2, 1]]
    r = validate_mesh(Mesh(P, C4, "T3"))
    assert r.repeated_node_cells.tolist() == [2] and 2 in r.degenerate_cells.tolist()
    # an out-of-range index never gets into a Mesh
    with pytest.raises(ValueError, match="out of range"):
        Mesh(P, np.array([[0, 1, 99]]), "T3")


def test_m1d_valid_meshes_report_nothing_and_bad_permutations_are_refused() -> None:
    good = [rectangle(3, 3, element=e) for e in ("T3", "Q4", "T6")] + [box(2, 2, 2, element=e)
                                                                      for e in ("T4", "H8", "T10")]
    good += list(_meshes().values())
    for m in good:
        r = validate_mesh(m)
        assert r.ok, f"{m.element}: {r.summary()}"
    m = rectangle(2, 2, element="Q4")
    with pytest.raises(ValueError, match="permutation"):
        renumber(m, cell_perm=np.array([0, 0, 1, 2]))
    with pytest.raises(ValueError, match="permutation"):
        renumber(m, node_perm=np.arange(m.n_nodes + 1))


# ------------------------------------------------------------------------------------------------ M2
def _tri_nodes(name: str, Xv: np.ndarray) -> np.ndarray:
    edges = {"T3": (), "T6": ((0, 1), (1, 2), (2, 0)), "T4": (),
             "T10": ((0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1))}[name]
    return np.vstack([Xv] + [[(Xv[i] + Xv[j]) / 2 for i, j in edges]]) if edges else Xv


REG_TRI = np.array([[0, 0], [1, 0], [0.5, S3 / 2]])
REG_TET = np.array([[0, 0, 0], [1, 0, 0], [0.5, S3 / 2, 0], [0.5, S3 / 6, np.sqrt(2 / 3)]])


@pytest.mark.parametrize("name,Xv", [("T3", REG_TRI), ("T6", REG_TRI), ("T4", REG_TET), ("T10", REG_TET),
                                     ("Q4", _CORNERS_Q4 / 2 + 0.5), ("H8", _CORNERS_H8 / 2 + 0.5)])
def test_m2a_regular_element_has_the_ideal_quality(name: str, Xv: np.ndarray) -> None:
    X = _tri_nodes(name, Xv) if name in ("T6", "T10") else Xv
    q = quality_from_nodes(get(name), X[None])
    assert abs(q.mean_ratio_min[0] - 1) <= 1e-12
    assert abs(q.scaled_jacobian_min[0] - 1) <= 1e-12
    assert abs(q.condition_max[0] - 1) <= 1e-12
    assert abs(q.edge_ratio[0] - 1) <= 1e-12
    assert bool(q.certified[0])


def test_m2b_closed_forms() -> None:
    q = quality_from_nodes(get("T3"), np.array([[[0, 0], [1, 0], [0, 1]]], float))
    assert _rel(q.mean_ratio_min, np.array([S3 / 2])) <= 1e-12
    assert _rel(q.scaled_jacobian_min, np.array([2 / np.sqrt(5)])) <= 1e-12
    for a in (1.0, 2.0, 10.0):
        X = np.array([[[0, 0], [a, 0], [a, 1], [0, 1]]], float)
        q = quality_from_nodes(get("Q4"), X)
        assert _rel(q.mean_ratio_min, np.array([2 * a / (a * a + 1)])) <= 1e-12
        assert _rel(q.condition_max, np.array([a])) <= 1e-12
        assert abs(q.scaled_jacobian_min[0] - 1) <= 1e-12
    for a, b, c in ((1, 1, 1), (1, 2, 3), (1, 10, 10)):
        X = (_CORNERS_H8 / 2 + 0.5) * np.array([a, b, c])
        q = quality_from_nodes(get("H8"), X[None])
        assert _rel(q.mean_ratio_min, np.array([3 * (a * b * c) ** (2 / 3) / (a * a + b * b + c * c)])) <= 1e-12
        assert _rel(q.condition_max, np.array([max(a, b, c) / min(a, b, c)])) <= 1e-12
    for eps in (1e-1, 1e-3):
        X = REG_TET.copy()
        X[3, 2] *= eps
        q = quality_from_nodes(get("T4"), X[None])
        assert _rel(q.mean_ratio_min, np.array([3 * eps ** (2 / 3) / (2 + eps ** 2)])) <= 1e-12
        assert _rel(q.condition_max, np.array([1 / eps])) <= 1e-12
        assert abs(q.scaled_jacobian_min[0] - 1) <= 1e-12     # SJ does not see this sliver (documented)


def _distorted() -> dict[str, Mesh]:
    return {"T3": perturb(rectangle(3, 3, element="T3"), 0.08, seed=1),
            "Q4": perturb(rectangle(3, 3, element="Q4"), 0.08, seed=2),
            "T4": perturb(box(2, 2, 2, element="T4"), 0.1, seed=3),
            "H8": perturb(box(2, 2, 2, element="H8"), 0.1, seed=4)}


def _rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(dim, dim)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@pytest.mark.parametrize("name", ["T3", "Q4", "T4", "H8"])
def test_m2c_scale_rotation_translation_invariance(name: str) -> None:
    m = _distorted()[name]
    q0 = cell_quality(m)
    rng = np.random.default_rng(20)
    R = _rotation(m.dim, rng)
    worst = 0.0
    for s, t in ((1e-6, 0.0), (1.0, 0.0), (1e3, 0.0), (1.0, 3.0)):
        m2 = Mesh(s * m.points @ R.T + t, m.cells, name)
        q1 = cell_quality(m2)
        for f in ("mean_ratio_min", "scaled_jacobian_min", "condition_max", "edge_ratio"):
            worst = max(worst, _rel(getattr(q1, f), getattr(q0, f)))
    print(f"M2c {name}: worst relative change {worst:.2e}")
    assert worst <= 1e-12


def test_m2d_flags_use_an_explicit_threshold_and_graded_meshes_match_closed_forms() -> None:
    X = np.stack([REG_TET, REG_TET * np.array([1, 1, 1e-3])])
    q = quality_from_nodes(get("T4"), X)
    assert flag_cells(q, 0.05).tolist() == [1]
    assert flag_cells(q, 0.0).tolist() == []
    xs = np.array([0.0, 1.0, 3.0, 7.0])
    P = np.array([[x, y] for x in xs for y in (0.0, 1.0)])
    cells = np.array([[2 * i, 2 * (i + 1), 2 * (i + 1) + 1, 2 * i + 1] for i in range(3)])
    q = cell_quality(Mesh(P, cells, "Q4"))
    w = np.array([1.0, 2.0, 4.0])
    assert _rel(q.edge_ratio, w) <= 1e-12
    assert _rel(q.condition_max, w) <= 1e-12
    assert _rel(q.mean_ratio_min, 2 * w / (w * w + 1)) <= 1e-12
    assert q.worst("mean_ratio_min") == 2 and q.worst("edge_ratio") == 2
    counts, _ = q.histogram("mean_ratio_min", bins=4)
    assert counts.sum() == 3


def test_m2e_inverted_cells_are_found_by_id() -> None:
    X = np.array([[[0, 0], [0, 1], [1, 0]]], float)                 # right isosceles, two nodes swapped
    q = quality_from_nodes(get("T3"), X)
    assert _rel(q.mean_ratio_min, np.array([-S3 / 2])) <= 1e-12
    assert q.inverted.tolist() == [0] and not q.certified[0]
    m = Mesh(X[0], np.array([[0, 1, 2]]), "T3")
    assert validate_mesh(m).negative_cells.tolist() == [0]
    dart = np.array([[[0, 0], [1, 0], [0.2, 0.2], [0, 1]]], float)  # reentrant corner at node 2 (hand-derived detJ < 0)
    q = quality_from_nodes(get("Q4"), dart)
    assert q.inverted.tolist() == [0] and not q.certified[0]
    cube = (_CORNERS_H8 / 2 + 0.5).copy()
    cube[6] = 0.1                                                   # corner 6 pulled through the cell (detJ = -1.7/8)
    q = quality_from_nodes(get("H8"), cube[None])
    assert q.inverted.tolist() == [0] and not q.certified[0]


def _dense_min_detj_t6(delta: np.ndarray) -> np.ndarray:
    """Min detJ of T6 cells (reference vertices, midside = straight midpoint + delta (n, 3, 2)) on a 41 x 41
    barycentric lattice, test-side shape-function derivatives."""
    g = np.linspace(0.0, 1.0, 41)
    s, t = np.meshgrid(g, g, indexing="ij")
    keep = s + t <= 1 + 1e-12
    dN = _t6_dN(s[keep], t[keep])                                   # (nl, 6, 2)
    V = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
    X = np.broadcast_to(V, (len(delta), 6, 2)).copy()
    X[:, 3:] += delta
    J = np.einsum("eni,qnj->eqij", X, dN)
    return (J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]).min(axis=1)


@pytest.mark.slow
def test_m2f_jacobian_certificate() -> None:
    rng = np.random.default_rng(5)
    # affine cells: the bound equals the constant detJ
    Xt = rng.uniform(0, 1, (6, 3, 2))
    Xt[:, 1:, :] += np.array([[1, 0], [0, 1]]) * 2
    for name in ("T3", "T6"):
        Xn = np.stack([_tri_nodes(name, x) for x in Xt])
        q = quality_from_nodes(get(name), Xn)
        det = (Xt[:, 1, 0] - Xt[:, 0, 0]) * (Xt[:, 2, 1] - Xt[:, 0, 1]) - (Xt[:, 1, 1] - Xt[:, 0, 1]) * (Xt[:, 2, 0] - Xt[:, 0, 0])
        L = np.max([np.linalg.norm(Xt[:, i] - Xt[:, j], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
        assert _rel(q.bound_rel, det / L ** 2) <= 1e-12 and q.certified.all()
    for name in ("T4", "T10"):
        Xv = np.stack([REG_TET * rng.uniform(0.5, 2) + rng.normal(0, 0.1, (4, 3)) for _ in range(5)])
        Xn = np.stack([_tri_nodes(name, x) for x in Xv])
        q = quality_from_nodes(get(name), Xn)
        det = np.linalg.det(np.stack([Xv[:, 1] - Xv[:, 0], Xv[:, 2] - Xv[:, 0], Xv[:, 3] - Xv[:, 0]], axis=1))
        pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        L = np.max([np.linalg.norm(Xv[:, i] - Xv[:, j], axis=1) for i, j in pairs], axis=0)
        assert _rel(q.bound_rel, det / L ** 3) <= 1e-12 and q.certified.all()
    par = np.array([[[0, 0], [2, 0], [3, 1], [1, 1]]], float)
    q = quality_from_nodes(get("Q4"), par)
    assert _rel(q.bound_rel, np.array([0.5 / 4])) <= 1e-12 and q.certified[0]
    # random curved T6: bound property, soundness of the certificate, and inversion between samples
    n, chunk = 200_000, 2_000
    delta_all = np.random.default_rng(12345).uniform(-0.4, 0.4, (n, 3, 2))
    V = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
    worst_gap = -np.inf
    n_hidden = 0
    n_cert = 0
    for k in range(0, n, chunk):
        dl = delta_all[k:k + chunk]
        X = np.broadcast_to(V, (len(dl), 6, 2)).copy()
        X[:, 3:] += dl
        q = quality_from_nodes(get("T6"), X)
        dense = _dense_min_detj_t6(dl)
        se = 2.0                                                     # longest vertex edge² of the reference triangle
        worst_gap = max(worst_gap, float((q.bound_rel - dense / se).max()))
        assert (dense[q.certified] > 0).all()
        n_cert += int(q.certified.sum())
        n_hidden += int(((q.detj_min_rel > 0) & (dense < 0) & ~q.certified).sum())
    print(f"M2f: max(bound - dense min)/s_e = {worst_gap:.3e}; certified {n_cert}; positive-sampled but inverted "
          f"cells (uncertified): {n_hidden}")
    assert worst_gap <= 1e-12
    assert n_hidden >= 1


# ------------------------------------------------------------------------------------------------ M4
def _case_meshes() -> list[tuple[str, Mesh]]:
    return [("Q4", rectangle(3, 3, element="Q4")), ("Q4", perturb(rectangle(3, 3, element="Q4"), 0.08, seed=1)),
            ("H8", box(2, 2, 2, element="H8")), ("H8", perturb(box(2, 2, 2, element="H8"), 0.1, seed=2)),
            ("T3", perturb(rectangle(3, 3, element="T3"), 0.08, seed=3)),
            ("T4", perturb(box(2, 2, 2, element="T4"), 0.1, seed=4)),
            ("T6", perturb(rectangle(2, 2, element="T6"), 0.08, seed=5, curved=0.05))]


def test_m4a_known_reference_coordinates_are_recovered() -> None:
    rng = np.random.default_rng(30)
    worst = 0.0
    for name, m in _case_meshes():
        xi0 = np.array([_random_xi(name, rng) for _ in range(m.n_cells)])
        x = np.array([_N(name, xi0[c]) @ m.points[m.cells[c]] for c in range(m.n_cells)])
        loc = locate(m, x)
        assert (loc.status == INSIDE).all(), name
        assert np.array_equal(loc.cell, np.arange(m.n_cells)), name
        worst = max(worst, float(np.abs(loc.xi - xi0).max()))
    print(f"M4a: worst xi error {worst:.2e}")
    assert worst <= 1e-12


def test_m4b_boundary_outside_and_tie_rules() -> None:
    for name in ("Q4", "T3", "T6"):
        m = rectangle(2, 2, element=name)
        right = m.facet_nodes(m.facet_sets["right"])[:, :2]
        mid = m.points[right].mean(axis=1)
        loc = locate(m, mid)
        assert (loc.status == BOUNDARY).all()
        assert np.array_equal(loc.cell, m.facet_sets["right"][:, 0])
        out = mid + np.array([1e-6, 0.0])
        lo = locate(m, out)
        assert (lo.status == OUTSIDE).all() and (lo.cell == -1).all() and np.isnan(lo.xi).all()
        u = interpolate(m, m.points[:, 0], lo)
        assert np.isnan(u).all()
        # the central vertex (0.5, 0.5) is shared by several cells: lowest id wins, all hits are counted
        node = int(np.flatnonzero(np.abs(m.points - 0.5).max(axis=1) < 1e-12)[0])
        owners = np.flatnonzero((m.cells == node).any(axis=1))
        lc = locate(m, np.array([[0.5, 0.5]]))
        assert lc.status[0] == BOUNDARY and lc.cell[0] == owners.min() and lc.n_hits[0] == len(owners)
        # centroids are interior
        lin = locate(m, m.cell_centroids())
        assert (lin.status == INSIDE).all() and np.array_equal(lin.cell, np.arange(m.n_cells))


@pytest.mark.parametrize("name", ["Q4", "T6"])
def test_m4c_affine_field_interpolates_exactly(name: str) -> None:
    m = perturb(rectangle(4, 4, element=name), 0.06, seed=7, curved=0.03 if name == "T6" else 0.0)
    pts = np.random.default_rng(31).uniform(0.02, 0.98, (500, 2))
    loc = locate(m, pts)
    assert (loc.status != OUTSIDE).all()
    a, B = np.array([0.3, -1.2]), np.array([[2.0, -0.5], [0.7, 1.9]])
    u = interpolate(m, a + m.points @ B.T, loc)
    ref = a + pts @ B.T
    err = np.abs(u - ref).max() / np.abs(ref).max()
    print(f"M4c {name}: interpolation error {err:.2e}")
    assert err <= 1e-12


def test_m4d_negative_controls() -> None:
    m = perturb(rectangle(3, 3, element="Q4"), 0.08, seed=1)
    xi0 = np.array([0.6, 0.6])
    err_aff = 0.0
    for c in range(m.n_cells):
        X = m.points[m.cells[c]]
        x = _N("Q4", xi0) @ X
        A = np.column_stack([(X[1] - X[0]) / 2, (X[3] - X[0]) / 2])        # affine map of the vertices 0, 1, 3 only
        xi_aff = np.linalg.solve(A, x - X[0]) - 1.0
        err_aff = max(err_aff, float(np.abs(xi_aff - xi0).max()))
    assert err_aff > 1e-3
    pts = np.random.default_rng(1).uniform(0.05, 0.95, (200, 2))
    loc = locate(m, pts)
    u = interpolate(m, m.points[:, 0] ** 2, loc)
    assert np.abs(u - pts[:, 0] ** 2).max() > 1e-6


# ------------------------------------------------------------------------------------------------ M5
def test_m5a_mixed_cell_types_are_refused() -> None:
    P = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]], float)
    with pytest.raises(ValueError, match="mixed"):
        Mesh(P, {"Q4": np.array([[0, 1, 2, 3]]), "T3": np.array([[1, 4, 2]])}, "Q4")  # type: ignore[arg-type]


def test_m5b_hanging_node_is_a_nonconforming_interface() -> None:
    P = np.array([[0, 0], [0.5, -1], [1, 0], [0.5, 0], [0.25, 1], [0.75, 1]], float)
    m = Mesh(P, np.array([[0, 1, 2], [0, 3, 4], [3, 2, 5]]), "T3")
    assert hanging_nodes(m).tolist() == [[0, 2, 3]]
    r = validate_mesh(m)
    assert r.hanging_nodes is not None and r.hanging_nodes.tolist() == [[0, 2, 3]] and not r.ok
    for e in ("T3", "Q4", "T6"):
        assert len(hanging_nodes(perturb(rectangle(3, 3, element=e), 0.05, seed=1))) == 0
    assert validate_mesh(box(2, 2, 2, element="T4")).hanging_nodes is None       # 3-D: NOT CHECKED
    with pytest.raises(ValueError, match="2-D"):
        hanging_nodes(box(1, 1, 1, element="H8"))


# ------------------------------------------------------------------------------------------------ M3
def _read_vtu_independent(path: str) -> dict[str, np.ndarray]:
    """Minimal ASCII reader written for this test (does not use toporlogy.core.vtu.read_vtu)."""
    root = ET.parse(path).getroot()
    piece = root.find("UnstructuredGrid/Piece")
    assert piece is not None
    out: dict[str, np.ndarray] = {}
    for section in ("PointData", "CellData", "Cells"):
        for el in piece.iterfind(f"{section}/DataArray"):
            k = int(el.attrib["NumberOfComponents"])
            words = (el.text or "").split()
            a = np.array([float(w) for w in words]) if el.attrib["type"].startswith("Float") else \
                np.array([int(w) for w in words])
            out[f"{section}/{el.attrib['Name']}"] = a.reshape(-1, k) if k > 1 else a
    el = piece.find("Points/DataArray")
    assert el is not None
    out["Points"] = np.array([float(w) for w in (el.text or "").split()]).reshape(-1, 3)
    return out


@pytest.mark.parametrize("name,vtk", [("T3", 5), ("Q4", 9), ("T4", 10), ("H8", 12)])
def test_m3a_vtu_roundtrip_of_a_renumbered_mesh(tmp_path, name: str, vtk: int) -> None:
    m = _meshes()[name]
    m.cell_tags = np.arange(m.n_cells) % 3 + 10
    rng = np.random.default_rng(40)
    new, ren = renumber(m, rng.permutation(m.n_cells), rng.permutation(m.n_nodes))
    u = rng.normal(size=(new.n_nodes, new.dim))
    path = write_cells_vtu(tmp_path / "m.vtu", new.points, new.cells, vtk, point_data={"u": u},
                           cell_data={"tag": new.cell_tags})
    d = _read_vtu_independent(path)
    pts = np.zeros((new.n_nodes, 3))
    pts[:, :new.dim] = new.points
    assert np.array_equal(d["Points"], pts)
    assert np.array_equal(d["Cells/connectivity"].reshape(new.cells.shape), new.cells)
    assert np.array_equal(d["Cells/types"], np.full(new.n_cells, vtk))
    assert np.array_equal(d["PointData/u"], u)
    assert np.array_equal(d["CellData/tag"], new.cell_tags)
    # and the tags survive the trip back to the old numbering
    assert np.array_equal(ren.to_old_cell(d["CellData/tag"]), m.cell_tags)
