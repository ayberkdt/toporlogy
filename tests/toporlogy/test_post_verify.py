"""toporlogy.post.verify: T10 element (patch test; the older aerokit.bracket.tetfem as oracle when installed) and the
gmsh subprocess mesher."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.post.verify import TetMesh, solve_t10, stiffness_t10, tet_mesh

_EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]


def _box_t10(n: tuple[int, int, int], L: tuple[float, float, float]) -> TetMesh:
    """Structured T10 mesh of [0, Lx]×[0, Ly]×[0, Lz]: Kuhn split of each hex into 6 tets + edge midpoints."""
    nx, ny, nz = n
    g = np.stack(np.meshgrid(*[np.linspace(0, L[k], n[k] + 1) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    kuhn = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [nid(i + a, j + b, k + cc) for a in (0, 1) for b in (0, 1) for cc in (0, 1)]
                for t in kuhn:
                    tets.append([c[q] for q in t])
    T4 = np.array(tets)
    P = g[T4]
    det = np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])
    T4[det < 0] = T4[det < 0][:, [0, 2, 1, 3]]
    edges = np.sort(np.concatenate([T4[:, [a, b]] for a, b in _EDGES]), axis=1)
    ue, inv = np.unique(edges, axis=0, return_inverse=True)
    X = np.vstack([g, 0.5 * (g[ue[:, 0]] + g[ue[:, 1]])])
    mid = len(g) + inv.reshape(6, -1).T
    T = np.hstack([T4, mid])
    # boundary faces (each face of a tet: 3 vertices + 3 midside nodes in gmsh T6 order)
    loc = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    epos = {e: i for i, e in enumerate(_EDGES)}
    epos.update({(b, a): i for (a, b), i in list(epos.items())})
    F = []
    for a, b, c in loc:
        F.append(np.column_stack([T[:, a], T[:, b], T[:, c], T[:, 4 + epos[(a, b)]], T[:, 4 + epos[(b, c)]],
                                  T[:, 4 + epos[(c, a)]]]))
    F = np.vstack(F)
    key = np.sort(F[:, :3], axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return TetMesh(X, T, F[idx[cnt == 1]], {})


def test_t10_patch_uniaxial_tension() -> None:
    m = _box_t10((4, 2, 2), (2.0, 1.0, 1.0))
    E, nu, P = 210.0, 0.3, 5.0
    sup = [(lambda X: X[:, 0] < 1e-9, [0]), (lambda X: X[:, 1] < 1e-9, [1]), (lambda X: X[:, 2] < 1e-9, [2])]
    r = solve_t10(m, sup, [(lambda X: X[:, 0] > 2 - 1e-9, (P, 0.0, 0.0))], E, nu)
    c = r["cases"][0]
    s = P / 1.0
    assert abs(c["compliance"] / (P * s / E * 2.0) - 1) < 1e-9
    umax = np.sqrt((2 * s / E) ** 2 + 2 * (nu * s / E) ** 2)
    assert abs(c["max_displacement"] / umax - 1) < 1e-9
    assert abs(c["vm_max"] / s - 1) < 1e-8
    assert abs(r["volume"] - 2.0) < 1e-12
    assert np.allclose(c["reaction_sum"], [-P, 0, 0], atol=1e-9)


def test_t10_stiffness_matches_old_tetfem() -> None:
    tetfem = pytest.importorskip("aerokit.bracket.tetfem")
    m = _box_t10((2, 2, 1), (1.0, 1.3, 0.7))
    rng = np.random.default_rng(0)
    m.X = m.X + 0.03 * rng.standard_normal(m.X.shape)          # distorted, curved-free elements
    K = stiffness_t10(m, 7.0, 0.27)
    K_old, _ = tetfem._stiffness(tetfem.TetMesh(m.X, m.T, m.faces, {}), 7.0, 0.27)
    assert abs(K - K_old).max() <= 1e-12 * abs(K_old).max()


@pytest.mark.slow
@pytest.mark.gmsh
def test_gmsh_tet_mesh_of_sphere() -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("pyvista")
    import pyvista as pv
    s = pv.Sphere(radius=5.0, theta_resolution=48, phi_resolution=48).triangulate().clean()
    V = np.asarray(s.points, float)
    F = np.asarray(s.faces).reshape(-1, 4)[:, 1:]
    from toporlogy.post.mesh import enclosed_volume
    if enclosed_volume(V, F) < 0:
        F = F[:, ::-1]
    m = tet_mesh(V, F, size=1.2, timeout=120)
    assert m.T.shape[1] == 10 and len(m.T) > 200
    assert abs(m.volume / enclosed_volume(V, F) - 1) < 0.02
