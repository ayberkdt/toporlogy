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


# ============================================================================== MD: a general mesh inside Study
def _studies_on_grid_mesh():
    from toporlogy.design.chain import Chain, Embed, Passive
    from toporlogy.design.interp import SIMP
    from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity, MeshFilterLink, MeshVolume
    from toporlogy.optimization.responses import Compliance, Volume
    from toporlogy.optimization.study import Constraint, Study
    from toporlogy.optimizers.mma import MMA
    d, fixed, F = _grid()
    grid = Study(d, Elasticity(d, fixed, nu=NU), F, Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)]),
                 SIMP(d, penal=3.0, emin_ratio=1e-9), Compliance(), [Constraint(Volume(), 0.4)], MMA())
    md = MeshDomain(Mesh.from_domain(d))
    mesh = Study(md, MeshElasticity(md, D_plane_stress(NU), fixed, thickness=d.thickness), F,
                 Chain([Embed(md), MeshFilterLink(md, 1.5 * d.h), Passive(md)]),
                 SIMP(md, penal=3.0, emin_ratio=1e-9), Compliance(), [Constraint(MeshVolume(md), 0.4)], MMA())
    return grid, mesh


def test_md1_mesh_study_equals_grid_study() -> None:
    grid, mesh = _studies_on_grid_mesh()
    x = 0.2 + 0.7 * np.random.default_rng(9).random(grid.n_var)
    eg, em = grid.evaluate(x), mesh.evaluate(x)
    print(f"MD1: value {abs(em.f0 - eg.f0) / abs(eg.f0):.2e}, grad {np.abs(em.df0 - eg.df0).max() / np.abs(eg.df0).max():.2e}, "
          f"volume {abs(em.values['volume'] - eg.values['volume']):.2e}")
    assert abs(em.f0 - eg.f0) <= 1e-12 * abs(eg.f0)
    assert np.abs(em.df0 - eg.df0).max() <= 1e-12 * np.abs(eg.df0).max()


def test_md2_histories_agree() -> None:
    grid, mesh = _studies_on_grid_mesh()
    hg, hm = grid.run(max_iter=10, tol=0.0).history["f0"], mesh.run(max_iter=10, tol=0.0).history["f0"]
    rel = max(abs(a - b) / abs(a) for a, b in zip(hg, hm, strict=True))
    print(f"MD2: max history difference {rel:.2e} over {len(hg)} iterations")
    assert rel <= 1e-8


def _mesh_study(element: str):
    from toporlogy.design.chain import Chain, Embed, Passive
    from toporlogy.design.interp import SIMP
    from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity, MeshFilterLink, MeshVolume
    from toporlogy.optimization.responses import Compliance
    from toporlogy.optimization.study import Constraint, Study
    from toporlogy.optimizers.mma import MMA
    m, fixed, F = _cantilever(element, 0.03)
    md = MeshDomain(m)
    return Study(md, MeshElasticity(md, D_plane_stress(NU), fixed), F,
                 Chain([Embed(md), MeshFilterLink(md, 0.2), Passive(md)]), SIMP(md, penal=3.0, emin_ratio=1e-9),
                 Compliance(), [Constraint(MeshVolume(md), 0.4)], MMA())


@pytest.mark.parametrize("element", ["Q4", "T3"])
def test_md3_study_gradient_on_distorted_meshes(element) -> None:
    st = _mesh_study(element)
    x = 0.2 + 0.7 * np.random.default_rng(10).random(st.n_var)
    v = np.random.default_rng(11).normal(size=st.n_var)
    ev = st.evaluate(x)
    s = 1e-6
    fd = (st.evaluate(x + s * v, False).f0 - st.evaluate(x - s * v, False).f0) / (2 * s)
    print(f"MD3 {element}: rel {abs(ev.df0 @ v - fd) / abs(fd):.2e}")
    assert abs(ev.df0 @ v - fd) <= 1e-6 * abs(fd)


def test_md4_study_run_on_a_distorted_mesh() -> None:
    st = _mesh_study("Q4")
    res = st.run(max_iter=30)
    c0 = res.history["f0"][0]
    print(f"MD4: compliance {c0:.4e} -> {res.objective:.4e}, volume {res.constraint_values['volume']:.4f}")
    assert res.objective < c0 and res.constraint_values["volume"] <= 0.4 * (1 + 1e-3)
