"""IC0-IC3: auxetic cell with E1, E2, G12 bounded and the solid's connectivity checked
(docs/foundations/prereg/IAc_auxetic_bounded.md, thresholds committed in 040f61f before the code change).
Synthetic, dimensionless."""
from __future__ import annotations

import numpy as np
import pytest

from test_cell_auxetic import _one_body
from test_cell_topology import EMIN, NU, PENAL, square

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.optimization.cell_topology import PeriodicDensityFilter, _moduli_gradients, _project, optimize_cell

BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
LIMITS = (0.01, 0.01, 0.002)


def test_ic0_connectivity_check() -> None:
    m, L = square(12)
    c = m.cell_centroids()
    full = np.ones(m.n_cells, bool)
    stripes = ((c[:, 1] > 0.1) & (c[:, 1] < 0.3)) | ((c[:, 1] > 0.6) & (c[:, 1] < 0.8))
    d = np.mod(c[:, 0] - c[:, 1], 1.0)
    diagonal = (d < 0.15) | (d > 0.85)
    got = (_one_body(m, full, L), _one_body(m, stripes, L), _one_body(m, diagonal, L))
    print(f"\n  IC0: full {got[0]}, two stripes {got[1]}, wrapping diagonal {got[2]}")
    assert got == (True, False, True)


def test_ic1_gradients_of_the_three_moduli() -> None:
    m, L = square(12)
    D = D_plane_stress(NU)
    filt = PeriodicDensityFilter(m, L, 0.15)
    x = np.random.default_rng(1).uniform(0.2, 0.9, m.n_cells)

    def mod(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xh, dp = _project(filt(x), 8.0, 0.5)
        v, g = _moduli_gradients(m, L, D, xh, PENAL, EMIN)
        return v, np.stack([filt.backward(gi * dp) for gi in g])

    _, G = mod(x)
    worst = np.zeros(3)
    for i in np.random.default_rng(2).choice(m.n_cells, 6, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        fd = (mod(xp)[0] - mod(xm)[0]) / 2e-6
        worst = np.maximum(worst, np.abs(fd - G[:, i]) / np.abs(G).max(axis=1))
    print(f"\n  IC1: E1 {worst[0]:.1e}, E2 {worst[1]:.1e}, G12 {worst[2]:.1e}")
    assert (worst <= 1e-6).all()


@pytest.mark.slow
@pytest.mark.timeout(2400)   # 600 MMA iterations with extra homogenisations: 7-14 min alone, longer on a shared CPU
def test_ic2_ic3_bounded_auxetic_cell() -> None:
    m, L = square(40)
    c = m.cell_centroids()
    x0 = 0.4 + 0.2 * np.cos(2 * np.pi * c[:, 0]) * np.cos(2 * np.pi * c[:, 1])
    D = D_plane_stress(NU)
    r = optimize_cell(m, L, D, 0.4, 0.06, objective="poisson_stress", penal=PENAL, emin=EMIN, move=0.2, x0=x0,
                      kkt_tol=1e-4, projection=(0.5, BETAS), stage_iter=100, min_moduli=LIMITS)
    S = np.linalg.inv(r.C)
    mods = 1.0 / np.diag(S)
    nu12, nu21 = -S[1, 0] / S[0, 0], -S[0, 1] / S[1, 1]
    grey = float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9)))
    solid = r.x_phys >= 0.5
    Cb = homogenize_mesh_elasticity(m, D, L, np.where(solid, 1.0, EMIN)).C
    Sb = np.linalg.inv(Cb)
    nb12 = -Sb[1, 0] / Sb[0, 0]
    eb = float(np.linalg.eigvalsh(Cb).min())
    body = _one_body(m, solid, L)
    print(f"\n  IC2: nu12 {nu12:.4f}, nu21 {nu21:.4f}, E1 E2 G12 {mods.round(5).tolist()}, volume {r.volume:.8f}, "
          f"grey {grey:.4f}, it {r.iterations}, kkt {r.kkt}")
    print(f"  IC3: thresholded nu12 {nb12:.4f}, min eig {eb:.3e}, one periodic body {body}, "
          f"moduli {(1 / np.diag(Sb)).round(5).tolist()}")
    assert r.volume <= 0.4 * (1 + 1e-6)
    assert (mods >= np.asarray(LIMITS) * (1 - 1e-6)).all()
    assert grey <= 0.05
    assert nu12 <= -0.2
    assert eb >= 1e-4 and nb12 <= -0.1 and body
