"""IB1-IB4: auxetic cell with the compliance-based Poisson ratio nu12 = -S21/S11
(docs/foundations/prereg/IAb_auxetic_compliance.md, thresholds committed in 796f96e before the code change).
Synthetic, dimensionless."""
from __future__ import annotations

import numpy as np
import pytest

from test_cell_auxetic import _one_body
from test_cell_auxetic import test_ia1_gradients_of_nu_and_c11 as _ia1_check
from test_cell_projection import test_ip2_unprojected_path_is_unchanged as _ip2_check
from test_cell_topology import EMIN, NU, PENAL, square

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.optimization.cell_topology import (PeriodicDensityFilter, _e1_gradient, _evaluate, _project,
                                                  optimize_cell)

BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
E1_MIN = 0.01


def nus(C: np.ndarray) -> tuple[float, float]:
    S = np.linalg.inv(C)
    return float(-S[1, 0] / S[0, 0]), float(-S[0, 1] / S[1, 1])


def test_ib1_gradients_of_nu_stress_and_e1() -> None:
    m, L = square(12)
    D = D_plane_stress(NU)
    filt = PeriodicDensityFilter(m, L, 0.15)
    x = np.random.default_rng(1).uniform(0.2, 0.9, m.n_cells)

    def both(x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        xh, dp = _project(filt(x), 8.0, 0.5)
        v, g, _, _ = _evaluate(m, L, D, xh, PENAL, EMIN, "poisson_stress")
        e1, de1 = _e1_gradient(m, L, D, xh, PENAL, EMIN)
        return v, e1, filt.backward(g * dp), filt.backward(de1 * dp)

    _, _, gv, ge = both(x)
    wv = we = 0.0
    for i in np.random.default_rng(2).choice(m.n_cells, 6, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        a, b = both(xp), both(xm)
        wv = max(wv, abs((a[0] - b[0]) / 2e-6 - gv[i]) / np.abs(gv).max())
        we = max(we, abs((a[1] - b[1]) / 2e-6 - ge[i]) / np.abs(ge).max())
    print(f"\n  IB1: nu {wv:.1e}, E1 {we:.1e}")
    assert wv <= 1e-6 and we <= 1e-6


def test_ib4_other_paths_unchanged() -> None:
    _ip2_check()
    _ia1_check()


@pytest.mark.slow
def test_ib2_ib3_auxetic_under_uniaxial_stress() -> None:
    m, L = square(40)
    c = m.cell_centroids()
    x0 = 0.4 + 0.2 * np.cos(2 * np.pi * c[:, 0]) * np.cos(2 * np.pi * c[:, 1])
    D = D_plane_stress(NU)
    r = optimize_cell(m, L, D, 0.4, 0.06, objective="poisson_stress", penal=PENAL, emin=EMIN, move=0.2, x0=x0,
                      kkt_tol=1e-4, projection=(0.5, BETAS), stage_iter=100, min_E1=E1_MIN)
    C = r.C
    nu12, nu21 = nus(C)
    e1 = 1.0 / np.linalg.inv(C)[0, 0]
    grey = float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9)))
    solid = r.x_phys >= 0.5
    Cb = homogenize_mesh_elasticity(m, D, L, np.where(solid, 1.0, EMIN)).C
    nb12, nb21 = nus(Cb)
    print(f"\n  IB2: nu12 {nu12:.4f}, nu21 {nu21:.4f}, E1 {e1:.5f}, C13/C11 {C[0, 2] / C[0, 0]:.3e}, "
          f"C23/C22 {C[1, 2] / C[1, 1]:.3e}, volume {r.volume:.8f}, grey {grey:.4f}, it {r.iterations}, kkt {r.kkt}")
    print(f"  IB3: thresholded nu12 {nb12:.4f}, nu21 {nb21:.4f}, min eig {np.linalg.eigvalsh(Cb).min():.3e}, "
          f"one periodic body {_one_body(m, solid, L)}")
    assert r.volume <= 0.4 * (1 + 1e-6)
    assert e1 >= E1_MIN * (1 - 1e-6)
    assert grey <= 0.05
    assert nu12 <= -0.3
    assert nb12 <= -0.2 and np.linalg.eigvalsh(Cb).min() > 0
