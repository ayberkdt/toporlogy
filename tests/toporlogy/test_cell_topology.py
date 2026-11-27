"""IH1-IH5: inverse homogenisation of bulk-modulus cells on general periodic meshes
(docs/foundations/prereg/IH_inverse_homogenization.md, thresholds committed in decc95c before this file was run).
Reference: the 2-D Hashin-Shtrikman upper bound of a porous solid, written here. Synthetic, dimensionless."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.mesh import Mesh, rectangle
from toporlogy.optimization.cell_topology import PeriodicDensityFilter, cell_objective, optimize_cell

NU, E0 = 0.3, 1.0
PENAL, EMIN = 3.5, 1e-9
SQRT3 = np.sqrt(3.0)


def kappa_hs(f: float) -> float:
    """2-D HS upper bound of a porous solid (plane stress): f κ μ / ((1 − f) κ + μ)."""
    k, m = E0 / (2 * (1 - NU)), E0 / (2 * (1 + NU))
    return f * k * m / ((1 - f) * k + m)


def test_reference_limits() -> None:
    k = E0 / (2 * (1 - NU))
    assert abs(kappa_hs(1.0) - k) <= 1e-15 and kappa_hs(0.0) == 0.0
    assert all(kappa_hs(f) <= f * k for f in np.linspace(0, 1, 11))


def square(n: int, element: str = "T3") -> tuple[Mesh, np.ndarray]:
    return rectangle(n, n, 1.0, 1.0, element), np.eye(2)


def rhombus(n: int) -> tuple[Mesh, np.ndarray]:
    m0 = rectangle(n, n, 1.0, 1.0, "T3")
    A = np.array([[1.0, 0.5], [0.0, SQRT3 / 2]])
    return replace(m0, points=m0.points @ A.T), A.T.copy()


def hole_start(mesh: Mesh, centre: np.ndarray, f: float = 0.5) -> np.ndarray:
    x = np.full(mesh.n_cells, f)
    x[np.linalg.norm(mesh.cell_centroids() - centre, axis=1) <= 0.2] = f / 2
    return x


# ---------------------------------------------------------------------------------------------- IH1
def test_ih1_gradient_against_central_differences() -> None:
    m, L = square(12)
    D = D_plane_stress(NU) * E0
    filt = PeriodicDensityFilter(m, L, 0.15)
    x = np.random.default_rng(1).uniform(0.2, 0.9, m.n_cells)

    def kappa(x: np.ndarray) -> tuple[float, np.ndarray]:
        v, g, _ = cell_objective(m, L, D, filt(x), PENAL, EMIN)
        return v, filt.backward(g)

    _, g = kappa(x)
    idx = np.random.default_rng(2).choice(m.n_cells, 6, replace=False)
    worst = 0.0
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        fd = (kappa(xp)[0] - kappa(xm)[0]) / 2e-6
        worst = max(worst, abs(fd - g[i]) / np.abs(g).max())
    print(f"\n  IH1: worst {worst:.1e}")
    assert worst <= 1e-6


# ---------------------------------------------------------------------------------------------- IH2
def test_ih2_periodic_filter() -> None:
    n = 12
    m, L = square(n, "Q4")
    F = PeriodicDensityFilter(m, L, 0.2)
    rows = np.asarray(F.matrix.sum(axis=1)).ravel()
    assert np.abs(rows - 1).max() <= 1e-14
    assert np.abs(F(np.full(m.n_cells, 0.37)) - 0.37).max() <= 1e-14
    c = m.cell_centroids()
    ij = np.rint(c * n - 0.5).astype(int)                     # column i (x), row j (y) of each cell
    where = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}
    x = np.random.default_rng(3).uniform(size=m.n_cells)
    shift = np.array([where[((a + 1) % n, b)] for a, b in ij])  # cell k -> its right neighbour (periodic)
    xs = np.empty_like(x)
    xs[shift] = x                                             # field moved one column to the right
    Fxs = F(xs)
    Fx_shifted = np.empty_like(x)
    Fx_shifted[shift] = F(x)
    e_per = float(np.abs(Fxs - Fx_shifted).max())
    y = np.random.default_rng(4).uniform(size=m.n_cells)
    e_adj = abs(float(F(x) @ y) - float(x @ F.backward(y))) / abs(float(F(x) @ y))
    print(f"\n  IH2: periodicity {e_per:.1e}, adjoint {e_adj:.1e}")
    assert e_per <= 1e-12 and e_adj <= 1e-12


# ---------------------------------------------------------------------------------------------- IH3-IH5
def _run(mesh: Mesh, L: np.ndarray, centre: np.ndarray, penal: float = PENAL, max_iter: int = 300):
    return optimize_cell(mesh, L, D_plane_stress(NU) * E0, 0.5, 0.06, penal=penal, emin=EMIN, max_iter=max_iter,
                         move=0.2, x0=hole_start(mesh, centre), kkt_tol=1e-4)


def _report(tag: str, r) -> None:
    grey = float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9)))
    C = r.C
    print(f"\n  {tag}: kappa {r.objective:.6f}, HS {kappa_hs(0.5):.6f}, ratio {r.objective / kappa_hs(0.5):.4f}, "
          f"volume {r.volume:.8f}, it {r.iterations}, stop {r.stop}, kkt {r.kkt}, grey {grey:.3f}, "
          f"C11/C22 {C[0, 0] / C[1, 1]:.4f}, isotropy defect {abs(C[0, 0] - C[0, 1] - 2 * C[2, 2]) / C[0, 0]:.3e}")


@pytest.mark.slow
def test_ih3_square_cell_bulk_modulus() -> None:
    m, L = square(40)
    r = _run(m, L, np.array([0.5, 0.5]))
    _report("IH3", r)
    assert r.volume <= 0.5 * (1 + 1e-6)
    assert r.objective <= kappa_hs(0.5) * (1 + 1e-6)
    assert r.objective / kappa_hs(0.5) >= 0.80


@pytest.mark.slow
def test_ih4_rhombic_cell_bulk_modulus() -> None:
    m, L = rhombus(40)
    r = _run(m, L, np.array([0.75, SQRT3 / 4]))
    _report("IH4", r)
    assert r.volume <= 0.5 * (1 + 1e-6)
    assert r.objective <= kappa_hs(0.5) * (1 + 1e-6)
    assert r.objective / kappa_hs(0.5) >= 0.80


@pytest.mark.slow
def test_ih5_negative_control_unpenalised_material_exceeds_the_bound() -> None:
    m, L = square(40)
    r = _run(m, L, np.array([0.5, 0.5]), penal=1.0, max_iter=100)
    _report("IH5", r)
    assert r.objective > 1.05 * kappa_hs(0.5)
