"""IA1-IA4: auxetic cell on a general periodic mesh with projection (docs/foundations/prereg/IA_auxetic_cell.md,
thresholds committed in a76da90 before the code change). Formulation of docs/olcumler/auxetic_hucre_onkayit.md.
Synthetic, dimensionless."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from test_cell_projection import test_ip2_unprojected_path_is_unchanged as _ip2_check
from test_cell_topology import EMIN, NU, PENAL, square

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.optimization.cell_topology import PeriodicDensityFilter, _evaluate, _project, optimize_cell

BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
C11_MIN = 0.02


def test_ia1_gradients_of_nu_and_c11() -> None:
    m, L = square(12)
    D = D_plane_stress(NU)
    filt = PeriodicDensityFilter(m, L, 0.15)
    x = np.random.default_rng(1).uniform(0.2, 0.9, m.n_cells)

    def both(x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        xh, dp = _project(filt(x), 8.0, 0.5)
        v, g, C, dc11 = _evaluate(m, L, D, xh, PENAL, EMIN, "poisson")
        return v, float(C[0, 0]), filt.backward(g * dp), filt.backward(dc11 * dp)

    _, _, gv, gc = both(x)
    wv = wc = 0.0
    for i in np.random.default_rng(2).choice(m.n_cells, 6, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        a, b = both(xp), both(xm)
        wv = max(wv, abs((a[0] - b[0]) / 2e-6 - gv[i]) / np.abs(gv).max())
        wc = max(wc, abs((a[1] - b[1]) / 2e-6 - gc[i]) / np.abs(gc).max())
    print(f"\n  IA1: nu {wv:.1e}, C11 {wc:.1e}")
    assert wv <= 1e-6 and wc <= 1e-6


def test_ia4_bulk_path_fingerprint_unchanged() -> None:
    _ip2_check()


def _one_body(m, solid: np.ndarray, L: np.ndarray) -> bool:
    """Solid cells form one body when cells sharing a node are linked, across the periodic faces as well."""
    from scipy.spatial import cKDTree
    P = m.points
    key = np.arange(m.n_nodes)
    tree = cKDTree(P)
    for a in L:                                               # identify periodic partner nodes
        d, j = tree.query(P + a)
        ok = d < 1e-9
        key[j[ok]] = np.minimum(key[j[ok]], key[np.flatnonzero(ok)])
    for _ in range(3):
        key = key[key]
    cells = np.flatnonzero(solid)
    rows, cols = [], []
    node_cells: dict[int, list[int]] = {}
    for c in cells:
        for n in key[m.cells[c]]:
            node_cells.setdefault(int(n), []).append(int(c))
    for lst in node_cells.values():
        rows += [lst[0]] * (len(lst) - 1)
        cols += lst[1:]
    import scipy.sparse as sp
    G = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m.n_cells, m.n_cells))
    n, lab = connected_components(G, directed=False)
    return len(np.unique(lab[cells])) == 1


@pytest.mark.slow
def test_ia2_ia3_auxetic_square_cell() -> None:
    m, L = square(40)
    c = m.cell_centroids()
    x0 = 0.4 + 0.2 * np.cos(2 * np.pi * c[:, 0]) * np.cos(2 * np.pi * c[:, 1])
    D = D_plane_stress(NU)
    r = optimize_cell(m, L, D, 0.4, 0.06, objective="poisson", penal=PENAL, emin=EMIN, move=0.2, x0=x0,
                      kkt_tol=1e-4, projection=(0.5, BETAS), stage_iter=100, min_C11=C11_MIN)
    C = r.C
    grey = float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9)))
    nu12 = C[0, 1] / C[0, 0]
    solid = r.x_phys >= 0.5
    hb = homogenize_mesh_elasticity(m, D, L, np.where(solid, 1.0, EMIN))
    nu_b = hb.C[0, 1] / hb.C[0, 0]
    min_eig = float(np.linalg.eigvalsh(hb.C).min())
    print(f"\n  IA2: nu12 {nu12:.4f}, nu21 {C[0, 1] / C[1, 1]:.4f}, C11 {C[0, 0]:.5f}, C33 {C[2, 2]:.3e}, "
          f"volume {r.volume:.8f}, grey {grey:.4f}, it {r.iterations}, kkt {r.kkt}")
    print(f"  IA3: thresholded nu12 {nu_b:.4f}, nu21 {hb.C[0, 1] / hb.C[1, 1]:.4f}, min eig {min_eig:.3e}, "
          f"one periodic body {_one_body(m, solid, L)}")
    assert r.volume <= 0.4 * (1 + 1e-6)
    assert C[0, 0] >= C11_MIN * (1 - 1e-6)
    assert grey <= 0.05
    assert nu12 <= -0.5
    assert nu_b <= -0.3 and min_eig > 0
