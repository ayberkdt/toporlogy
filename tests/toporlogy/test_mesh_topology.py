"""Density optimisation on a general mesh (docs/olcumler/ag_topoloji_onkayit.md, MS1-MS5; MS6 is the baseline)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.domain import Domain
from toporlogy.design.chain import DensityFilter
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh import ElasticProblem, Mesh, perturb, rectangle
from toporlogy.optimization.mesh_topology import MeshCompliance, MeshDensityFilter, optimize_mesh

NU = 0.3


def _grid():
    d = Domain((12, 6), h=0.5)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    F = np.zeros(d.ndof)
    tip = int(np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9))[0])
    F[2 * tip + 1] = -1.0
    return d, fixed, F


def test_ms1_filter_equals_grid_filter() -> None:
    d, _, _ = _grid()
    m = Mesh.from_domain(d)
    assert np.array_equal(np.asarray(m.cell_tags), np.arange(d.ne))
    for rmin in (1.5, 2.5):
        A = MeshDensityFilter(m, rmin * d.h).matrix.toarray()
        B = DensityFilter(d, rmin).matrix().toarray()
        assert np.abs(A - B).max() <= 1e-14


def test_ms2_compliance_equals_grid() -> None:
    d, fixed, F = _grid()
    m = Mesh.from_domain(d)
    E = 0.1 + np.random.default_rng(2).random(d.ne)
    c, dc, _ = MeshCompliance(ElasticProblem(m, D_plane_stress(NU), thickness=d.thickness), F, fixed)(E)
    el = Elasticity(d, fixed, nu=NU)
    U = el.solve_numpy(E, F)
    cg = float(F @ U[:, 0])
    dcg = -el.element_energy(U)[:, 0] * el.geom
    print(f"MS2: value {abs(c - cg) / abs(cg):.2e}, grad {np.abs(dc - dcg).max() / np.abs(dcg).max():.2e}")
    assert abs(c - cg) <= 1e-12 * abs(cg)
    assert np.abs(dc - dcg).max() <= 1e-12 * np.abs(dcg).max()


def _cantilever(element: str, amplitude: float):
    m = rectangle(16, 8, 2.0, 1.0, element)
    if amplitude:
        m = perturb(m, amplitude, seed=3)
    X = m.points
    fixed = m.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = int(np.argmin(np.abs(X[:, 0] - 2.0) + np.abs(X[:, 1] - 0.0)))
    F = np.zeros(2 * m.n_nodes)
    F[2 * tip + 1] = -1.0
    return m, fixed, F


@pytest.mark.parametrize("element,amp", [("Q4", 0.03), ("T3", 0.03)])
def test_ms3_chain_gradient_on_distorted_meshes(element, amp) -> None:
    m, fixed, F = _cantilever(element, amp)
    comp = MeshCompliance(ElasticProblem(m, D_plane_stress(NU)), F, fixed)
    filt = MeshDensityFilter(m, 0.2)
    p, emin = 3.0, 1e-9

    def f(x):
        xp = filt(x)
        c, dcdE, _ = comp(emin + xp ** p * (1 - emin))
        return c, filt.backward(dcdE * p * xp ** (p - 1) * (1 - emin))
    x = 0.2 + 0.7 * np.random.default_rng(5).random(m.n_cells)
    v = np.random.default_rng(6).normal(size=m.n_cells)
    c, g = f(x)
    s = 1e-6
    fd = (f(x + s * v)[0] - f(x - s * v)[0]) / (2 * s)
    print(f"MS3 {element}: rel {abs(g @ v - fd) / abs(fd):.2e}")
    assert abs(g @ v - fd) <= 1e-6 * abs(fd)


def test_ms4_optimisation_on_a_distorted_mesh() -> None:
    m, fixed, F = _cantilever("Q4", 0.03)
    res = optimize_mesh(m, D_plane_stress(NU), F, fixed, volume=0.4, rmin=0.2, max_iter=30)
    print(f"MS4: compliance {res.history['f0'][0]:.4e} -> {res.objective:.4e}, volume {res.volume:.4f}, "
          f"iterations {len(res.history['f0'])}")
    assert res.objective < res.history["f0"][0]
    assert res.volume <= 0.4 * (1 + 1e-3)


def test_ms5_refusals() -> None:
    m, fixed, F = _cantilever("Q4", 0.0)
    with pytest.raises(ValueError, match="rmin"):
        MeshDensityFilter(m, 0.0)
    with pytest.raises(ValueError, match="volume"):
        optimize_mesh(m, D_plane_stress(NU), F, fixed, volume=1.2, rmin=0.2)
    with pytest.raises(ValueError, match="load"):
        MeshCompliance(ElasticProblem(m, D_plane_stress(NU)), F[:-1], fixed)
