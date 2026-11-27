"""MH1-MH8: periodic homogenisation on a general mesh with sensitivities (``toporlogy.fem.mesh.homogenization``).
Thresholds: docs/foundations/prereg/MH_mesh_homogenization.md (committed before this file was run). Synthetic,
dimensionless moduli. Independent references: the exact affine solution (MH1), the regular-grid implementation
``toporlogy.fem.homogenization`` (MH2, MH6; a separate assembly and a master-slave periodic map), the layered
average written here (MH3), central differences (MH4a), phase Voigt/Reuss (MH5)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import D_iso, D_plane_stress
from toporlogy.fem.homogenization import homogenize_conduction, homogenize_elasticity
from toporlogy.fem.mesh.constraints import ConstraintError
from toporlogy.fem.mesh.homogenization import homogenize_mesh_conduction, homogenize_mesh_elasticity
from toporlogy.fem.mesh.mesh import Mesh, box, rectangle
from toporlogy.fem.mesh.topology import renumber

NU = 0.3
SHEAR = 0.35


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def skew2(element: str, n: tuple[int, int] = (4, 3)) -> tuple[Mesh, np.ndarray]:
    m0 = rectangle(n[0], n[1], 1.0, 0.8, element)
    Aff = np.array([[1.0, SHEAR], [0.0, 1.0]])
    m = replace(m0, points=m0.points @ Aff.T)
    return m, np.array([[1.0, 0.0], Aff @ np.array([0.0, 0.8])])


def skew3(element: str) -> tuple[Mesh, np.ndarray]:
    m0 = box(2, 2, 2, 1.0, 1.0, 1.0, element)
    Aff = np.array([[1.0, 0.2, 0.1], [0.05, 1.0, 0.15], [0.1, 0.05, 1.0]])
    m = replace(m0, points=m0.points @ Aff.T)
    return m, np.array([Aff @ np.eye(3)[i] for i in range(3)])


def centroids(m: Mesh) -> np.ndarray:
    return m.points[m.cells[:, :4] if m.dim == 2 and m.cells.shape[1] >= 4 else m.cells].mean(axis=1)


# ---------------------------------------------------------------------------------------------- MH1
@pytest.mark.parametrize("element", ["Q4", "T3", "T6", "Q9"])
def test_mh1_homogeneous_skew_cell_2d(element: str) -> None:
    m, L = skew2(element)
    D = D_plane_stress(NU)
    h = homogenize_mesh_elasticity(m, D, L)
    print(f"\n  MH1 {element}: {rel(h.C, D):.1e}")
    assert rel(h.C, D) <= 1e-10


@pytest.mark.parametrize("element", ["H8", "T4"])
def test_mh1_homogeneous_skew_cell_3d(element: str) -> None:
    m, L = skew3(element)
    D = D_iso(NU)
    h = homogenize_mesh_elasticity(m, D, L)
    print(f"\n  MH1 {element}: {rel(h.C, D):.1e}")
    assert rel(h.C, D) <= 1e-10


# ---------------------------------------------------------------------------------------------- MH2
def _grid_match(nx: int, ny: int) -> tuple[Mesh, Domain, np.ndarray]:
    """Mesh Q4 rectangle and grid of the same cell; ``idx[e]`` = mesh cell of grid element e (by centroid)."""
    m = rectangle(nx, ny, float(nx), float(ny), "Q4")
    d = Domain((nx, ny))
    cg = d.element_centers()[d.active_elements]
    cm = m.points[m.cells].mean(axis=1)
    idx = np.array([int(np.argmin(np.linalg.norm(cm - c, axis=1))) for c in cg])
    assert np.abs(cm[idx] - cg).max() <= 1e-12 and len(np.unique(idx)) == m.n_cells
    return m, d, idx


def test_mh2_matches_the_regular_grid_implementation() -> None:
    m, d, idx = _grid_match(12, 10)
    Eg = np.random.default_rng(3).uniform(0.1, 2.0, d.n_active)
    Em = np.empty(m.n_cells)
    Em[idx] = Eg
    g = homogenize_elasticity(d, Eg, NU, sensitivities=True)
    h = homogenize_mesh_elasticity(m, D_plane_stress(NU), [[12.0, 0.0], [0.0, 10.0]], Em, sensitivities=True)
    assert g.dC is not None and h.dC is not None
    eC, edC = rel(h.C, g.C), rel(h.dC[idx], g.dC)
    print(f"\n  MH2: C {eC:.1e}, dC {edC:.1e}")
    assert eC <= 1e-10 and edC <= 1e-9


# ---------------------------------------------------------------------------------------------- MH3
def _layered(Cs: list[np.ndarray], f: list[float]) -> np.ndarray:
    """Exact average of layers normal to y (2-D Voigt [xx, yy, xy]): σ_yy, σ_xy and ε_xx are continuous."""
    t, n = [0], [1, 2]
    Ann = sum(fi * np.linalg.inv(C[np.ix_(n, n)]) for C, fi in zip(Cs, f, strict=True))
    Cnn = np.linalg.inv(Ann)
    Btn = sum(fi * C[np.ix_(t, n)] @ np.linalg.inv(C[np.ix_(n, n)]) for C, fi in zip(Cs, f, strict=True))
    Ctt = sum(fi * (C[np.ix_(t, t)] - C[np.ix_(t, n)] @ np.linalg.inv(C[np.ix_(n, n)]) @ C[np.ix_(n, t)])
              for C, fi in zip(Cs, f, strict=True))
    out = np.zeros((3, 3))
    out[np.ix_(n, n)] = Cnn
    out[np.ix_(t, n)] = Btn @ Cnn
    out[np.ix_(n, t)] = (Btn @ Cnn).T
    out[np.ix_(t, t)] = Ctt + Btn @ Cnn @ Btn.T
    return out


@pytest.mark.parametrize("element", ["T3", "Q9"])
def test_mh3_layered_cell_equals_the_layered_average(element: str) -> None:
    m0 = rectangle(4, 12, 1.0, 1.0, element)
    P = m0.points.copy()
    P[:, 0] += 0.04 * np.sin(np.pi * P[:, 0]) * np.sin(2 * np.pi * P[:, 1])     # x only: layers stay conforming
    m = replace(m0, points=P)
    y = m0.points[m0.cells].mean(axis=1)[:, 1]
    E = np.where(y < 3 / 12, 1.0, np.where(y < 8 / 12, 7.0, 3.0))
    D = D_plane_stress(NU)
    ref = _layered([1.0 * D, 7.0 * D, 3.0 * D], [3 / 12, 5 / 12, 4 / 12])
    h = homogenize_mesh_elasticity(m, D, np.eye(2), E)
    print(f"\n  MH3 {element}: {rel(h.C, ref):.1e}")
    assert rel(h.C, ref) <= 1e-9


# ---------------------------------------------------------------------------------------------- MH4
def _mh4_case() -> tuple[Mesh, np.ndarray, np.ndarray, np.ndarray]:
    m, L = skew2("T6", (4, 4))
    E = np.random.default_rng(5).uniform(0.2, 2.0, m.n_cells)
    return m, L, E, D_plane_stress(NU)


def test_mh4a_sensitivities_against_central_differences() -> None:
    m, L, E, D = _mh4_case()
    h = homogenize_mesh_elasticity(m, D, L, E, sensitivities=True)
    assert h.dC is not None
    cells = np.random.default_rng(11).choice(m.n_cells, 8, replace=False)
    worst = 0.0
    for e in cells:
        step = 1e-6 * E[e]
        Ep, Em = E.copy(), E.copy()
        Ep[e] += step
        Em[e] -= step
        fd = (homogenize_mesh_elasticity(m, D, L, Ep).C - homogenize_mesh_elasticity(m, D, L, Em).C) / (2 * step)
        worst = max(worst, float(np.abs(fd - h.dC[e]).max() / np.abs(h.dC).max()))
    print(f"\n  MH4a: worst {worst:.1e}")
    assert worst <= 1e-6


def test_mh4b_euler_identity() -> None:
    m, L, E, D = _mh4_case()
    h = homogenize_mesh_elasticity(m, D, L, E, sensitivities=True)
    assert h.dC is not None
    e = rel(np.einsum("e,eij->ij", E, h.dC), h.C)
    print(f"\n  MH4b: {e:.1e}")
    assert e <= 1e-12


# ---------------------------------------------------------------------------------------------- MH5
def test_mh5_two_phase_cell_between_voigt_and_reuss() -> None:
    m = rectangle(10, 10, 1.0, 1.0, "T3")
    E = np.ones(m.n_cells)
    E[np.random.default_rng(2).permutation(m.n_cells)[: m.n_cells // 2]] = 10.0
    D = D_plane_stress(NU)
    h = homogenize_mesh_elasticity(m, D, np.eye(2), E)
    f_hi = 0.5                                     # equal-area cells (uniform T3 rectangle), exactly half are stiff
    CV = (f_hi * 10.0 + (1 - f_hi) * 1.0) * D
    CR = np.linalg.inv(f_hi * np.linalg.inv(10.0 * D) + (1 - f_hi) * np.linalg.inv(D))
    lo = float(np.linalg.eigvalsh(h.C - CR).min())
    hi = float(np.linalg.eigvalsh(CV - h.C).min())
    print(f"\n  MH5: min eig (C - C_R) {lo:.3e}, (C_V - C) {hi:.3e}")
    tol = -1e-10 * float(np.abs(h.C).max())
    assert lo >= tol and hi >= tol


# ---------------------------------------------------------------------------------------------- MH6
def test_mh6_conduction_homogeneous_and_against_the_grid() -> None:
    m, L = skew2("T6")
    k = homogenize_mesh_conduction(m, 2.5, L)
    e1 = rel(k.C, 2.5 * np.eye(2))
    mg, d, idx = _grid_match(12, 10)
    kg = np.random.default_rng(4).uniform(0.1, 2.0, d.n_active)
    km = np.empty(mg.n_cells)
    km[idx] = kg
    g = homogenize_conduction(d, kg)
    h = homogenize_mesh_conduction(mg, 1.0, [[12.0, 0.0], [0.0, 10.0]], km)
    e2 = rel(h.C, g.C)
    print(f"\n  MH6: homogeneous {e1:.1e}, grid {e2:.1e}")
    assert e1 <= 1e-10 and e2 <= 1e-10


# ---------------------------------------------------------------------------------------------- MH7
def test_mh7_renumbering_invariance() -> None:
    m, L, E, D = _mh4_case()
    h = homogenize_mesh_elasticity(m, D, L, E, sensitivities=True)
    rng = np.random.default_rng(9)
    cp, npm = rng.permutation(m.n_cells), rng.permutation(m.n_nodes)
    m2, _ = renumber(m, cp, npm)
    h2 = homogenize_mesh_elasticity(m2, D, L, E[cp], sensitivities=True)
    assert h.dC is not None and h2.dC is not None
    eC, edC = rel(h2.C, h.C), rel(h2.dC, h.dC[cp])
    print(f"\n  MH7: C {eC:.1e}, dC {edC:.1e}")
    assert eC <= 1e-12 and edC <= 1e-12


# ---------------------------------------------------------------------------------------------- MH8
def test_mh8_refusals() -> None:
    m, L = skew2("Q4")
    D = D_plane_stress(NU)
    with pytest.raises(ConstraintError):
        homogenize_mesh_elasticity(m, D, [[1.0, 0.0], [0.0, 0.8]])        # second vector misses the skew
    with pytest.raises(ValueError, match="one value per cell"):
        homogenize_mesh_elasticity(m, D, L, np.ones(m.n_cells + 1))
    with pytest.raises(ValueError, match="> 0"):
        homogenize_mesh_elasticity(m, D, L, np.zeros(m.n_cells))
    with pytest.raises(ValueError, match="linearly dependent"):
        homogenize_mesh_elasticity(m, D, [[1.0, 0.0], [2.0, 0.0]])


# ---------------------------------------------------------------------------------------------- MH9
def test_mh9_sub_period_lattice_is_refused() -> None:
    m = rectangle(12, 10, 12.0, 10.0, "Q4")
    with pytest.raises(ValueError, match="mesh measure"):
        homogenize_mesh_elasticity(m, D_plane_stress(NU), [[6.0, 0.0], [0.0, 10.0]])
    with pytest.raises(ValueError, match="mesh measure"):
        homogenize_mesh_conduction(m, 1.0, [[6.0, 0.0], [0.0, 10.0]])


# ---------------------------------------------------------------------------------------------- MH10
def test_mh10_conductivity_must_be_symmetric_positive_definite() -> None:
    m, L = skew2("Q4")
    with pytest.raises(ValueError, match="positive definite"):
        homogenize_mesh_conduction(m, np.diag([1.0, -1.0]), L)
    with pytest.raises(ValueError, match="symmetric"):
        homogenize_mesh_conduction(m, np.array([[1.0, 0.5], [0.0, 1.0]]), L)
    k = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert rel(homogenize_mesh_conduction(m, k, L).C, k) <= 1e-10
    assert rel(homogenize_mesh_conduction(m, 2.5, L).C, 2.5 * np.eye(2)) <= 1e-10


# ---------------------------------------------------------------------------------------------- MH11
@pytest.mark.parametrize("t", [0.0, -1.0, float("nan")])
def test_mh11_non_positive_thickness_is_refused(t: float) -> None:
    m, L = skew2("Q4")
    with pytest.raises(ValueError, match="thickness"):
        homogenize_mesh_elasticity(m, D_plane_stress(NU), L, thickness=t)
    with pytest.raises(ValueError, match="thickness"):
        homogenize_mesh_conduction(m, 1.0, L, thickness=t)
