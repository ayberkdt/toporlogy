"""IP1-IP4: Heaviside projection with beta continuation for the IH cells (docs/foundations/prereg/IP_projection.md,
thresholds committed in c152684 before the change; the IP2 baseline was written by the code before the change,
12c05ad). Synthetic, dimensionless."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_cell_topology import EMIN, NU, PENAL, SQRT3, hole_start, kappa_hs, rhombus, square

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.optimization.cell_topology import PeriodicDensityFilter, _project, cell_objective, optimize_cell

BASE = Path(__file__).resolve().parent / "baseline" / "ip2_cell_fingerprint_c152684.json"
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def test_ip1_gradient_through_the_projection() -> None:
    m, L = square(12)
    D = D_plane_stress(NU)
    filt = PeriodicDensityFilter(m, L, 0.15)
    x = np.random.default_rng(1).uniform(0.2, 0.9, m.n_cells)

    def kappa(x: np.ndarray) -> tuple[float, np.ndarray]:
        xh, dp = _project(filt(x), 8.0, 0.5)
        v, g, _ = cell_objective(m, L, D, xh, PENAL, EMIN)
        return v, filt.backward(g * dp)

    _, g = kappa(x)
    worst = 0.0
    for i in np.random.default_rng(2).choice(m.n_cells, 6, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        worst = max(worst, abs((kappa(xp)[0] - kappa(xm)[0]) / 2e-6 - g[i]) / np.abs(g).max())
    print(f"\n  IP1: worst {worst:.1e}")
    assert worst <= 1e-6


def test_ip2_unprojected_path_is_unchanged() -> None:
    want = json.loads(BASE.read_text(encoding="utf-8"))
    m, L = square(40)
    r = optimize_cell(m, L, D_plane_stress(NU), 0.5, 0.06, penal=PENAL, emin=EMIN, max_iter=5, move=0.2,
                      x0=hole_start(m, np.array([0.5, 0.5])), kkt_tol=1e-4)
    assert hashlib.sha256(np.ascontiguousarray(r.x).tobytes()).hexdigest() == want["sha256_x"]
    assert float(r.objective).hex() == want["objective_hex"] and r.iterations == want["iterations"]


def void_hole(m: Mesh, centre: np.ndarray, area: float) -> np.ndarray:
    r = np.sqrt(0.5 * area / np.pi)
    return np.where(np.linalg.norm(m.cell_centroids() - centre, axis=1) <= r, 0.0, 1.0)


def _run(m: Mesh, L: np.ndarray, x0: np.ndarray):
    return optimize_cell(m, L, D_plane_stress(NU), 0.5, 0.06, penal=PENAL, emin=EMIN, move=0.2, x0=x0,
                         kkt_tol=1e-4, projection=(0.5, BETAS), stage_iter=100)


def _check(tag: str, r) -> None:
    grey = float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9)))
    ratio = r.objective / kappa_hs(0.5)
    C = r.C
    print(f"\n  {tag}: kappa {r.objective:.6f}, ratio {ratio:.4f}, volume {r.volume:.8f}, grey {grey:.4f}, "
          f"it {r.iterations}, last stop {r.stop}, kkt {r.kkt}, "
          f"isotropy defect {abs(C[0, 0] - C[0, 1] - 2 * C[2, 2]) / C[0, 0]:.3e}")
    assert r.volume <= 0.5 * (1 + 1e-6)
    assert r.objective <= kappa_hs(0.5) * (1 + 1e-6)
    assert grey <= 0.05
    assert ratio >= 0.85


@pytest.mark.slow
def test_ip3_square_cell_with_projection() -> None:
    m, L = square(40)
    _check("IP3", _run(m, L, void_hole(m, np.array([0.5, 0.5]), 1.0)))


@pytest.mark.slow
def test_ip4_rhombic_cell_with_projection() -> None:
    m, L = rhombus(40)
    _check("IP4", _run(m, L, void_hole(m, np.array([0.75, SQRT3 / 4]), SQRT3 / 2)))
