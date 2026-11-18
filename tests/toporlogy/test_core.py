"""Unit tests of the A1 core (docs/mimari_plan.md §3, §5 A1): Domain numbering, element physics (patch test),
direct solvers, filter against the old implementations, Study input checks and the Recorder run folder."""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from topo_oracles import topo3d
from topo_oracles.voxfem import H8_CORNER, VoxelMesh
from topo_oracles import topology
from toporlogy import (MMA, OC, SIMP, Compliance, Constraint, DensityFilter, Domain, Elasticity, Recorder, Response,
                          Study, Volume, standard_chain)
from toporlogy.solvers import DirectSolver


# ---------------------------------------------------------------------------------------------- Domain
def test_domain_3d_numbering_matches_voxelmesh():
    shape, h, origin = (3, 2, 4), 0.5, (1.0, -2.0, 0.25)
    d = Domain(shape, h, origin=origin)
    assert (d.ne, d.nn, d.ndof) == (24, 4 * 3 * 5, 3 * 60)
    # element number e = (ix·ny + iy)·nz + iz, node n = (ix·(ny+1) + iy)·(nz+1) + iz
    ijk = d.element_ijk()
    assert np.array_equal((ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2], np.arange(d.ne))
    X = d.node_coords()
    idx = np.rint((X - origin) / h).astype(int)
    assert np.array_equal(d.node_number(idx), np.arange(d.nn))
    # corner k of every element sits at origin + h·(ijk + (H8_CORNER[k] + 1)/2), DOFs node-major (x, y, z)
    nodes = d.edof[:, ::3] // 3
    assert np.array_equal(d.edof[:, 1::3], d.edof[:, ::3] + 1) and np.array_equal(d.edof[:, 2::3], d.edof[:, ::3] + 2)
    expect = origin + h * (ijk[:, None, :] + (H8_CORNER[None] + 1) / 2)
    assert np.allclose(X[nodes], expect, atol=1e-14)
    # identical to the oracle mesh on a full box
    mesh = VoxelMesh(np.ones(shape, bool), h, origin=np.array(origin))
    assert np.array_equal(mesh.edof, d.edof) and np.allclose(mesh.X, X)
    assert np.array_equal(np.argwhere(np.ones(shape, bool)), ijk)


def test_domain_2d_numbering_matches_grid2d():
    nelx, nely = 5, 3
    d = Domain((nelx, nely), 1.0)
    X = d.node_coords()
    ijk = d.element_ijk()
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])        # counter-clockwise from lower-left (ke_q4 order)
    assert np.allclose(X[d.edof[:, ::2] // 2], ijk[:, None, :] + corners[None])
    # top88 Grid2D: element i·nely + j with j counted from the TOP, node n with y = nely − n % (nely+1)
    g = topology.Grid2D(nelx, nely)
    old_node = np.arange((nelx + 1) * (nely + 1))
    new_node = (old_node // (nely + 1)) * (nely + 1) + (nely - old_node % (nely + 1))
    assert np.allclose(g.X, X[new_node])
    old_dof_to_new = (2 * new_node[:, None] + np.arange(2)).ravel()
    for e_old in range(g.nel):
        i, j = divmod(e_old, nely)
        e_new = i * nely + (nely - 1 - j)
        assert np.array_equal(d.edof[e_new], old_dof_to_new[g.edof[e_old]])


def test_domain_selections_and_vectors():
    shape = (4, 3, 2)
    frac = np.ones(shape)
    frac[3, 2, :] = 0.0                                         # one column outside the part
    frac[0, 0, 0] = 0.4
    keep = np.zeros(shape, bool)
    keep[0, :, :] = True
    void = np.zeros(shape, bool)
    void[1, 1, :] = True
    void[0, 0, 1] = True                                        # overlaps keep: keep wins
    d = Domain(shape, 2.0, frac=frac, keep=keep, void=void)
    assert d.n_active == 22 and d.keep.sum() == 6 and d.void.sum() == 2 and d.n_design == 14
    assert not d.void[0, 0, 1]
    # corner node (4, 3, z) belongs only to the removed column
    corner = d.nodes_where(lambda X: (X[:, 0] == 8.0) & (X[:, 1] == 6.0))
    assert corner.size == 0
    assert d.nodes_where(lambda X: (X[:, 0] == 8.0) & (X[:, 1] == 6.0), active_only=False).size == 3
    assert len(d.active_dofs) == d.ndof - 3 * 3
    n = np.array([5, 2])
    assert np.array_equal(d.dofs(n), [6, 7, 8, 15, 16, 17])
    assert np.array_equal(d.dofs(n, 1), [7, 16])
    assert np.array_equal(d.dofs(n, [2, 0]), [6, 8, 15, 17])
    assert d.dofs(np.array([], int)).size == 0
    # uniform start (topo3d rule): keep counts 1, void 0, the free share the rest → mean over the part = V*
    x0 = d.uniform_start(0.5)
    assert np.allclose(x0, (0.5 * 22 - 6) / 14)
    assert np.isclose(d.to_grid(x0)[d.active].mean(), 0.5)
    assert np.all(d.uniform_start(0.1) == 0.0) and np.all(d.uniform_start(1.0) == 1.0)
    # to_grid: design entries in ascending element order, keep / void / outside filled
    v = np.arange(1, d.n_design + 1) / 100.0
    g = d.to_grid(v)
    assert np.array_equal(g.ravel()[d.design_elements], v)
    assert np.all(g[d.keep] == 1.0) and np.all(g[d.void] == 0.0) and np.all(g[~d.active] == 0.0)
    assert np.allclose(d.element_centers()[0], [1.0, 1.0, 1.0])


def test_domain_rejects_bad_input():
    with pytest.raises(ValueError, match='shape\\ must\\ be'):
        Domain((3,))
    with pytest.raises(ValueError, match='shape\\ must\\ be'):
        Domain((3, 0, 2))
    with pytest.raises(ValueError, match='h\\ >\\ 0'):
        Domain((2, 2), h=0.0)
    with pytest.raises(ValueError, match='frac\\ must\\ lie'):
        Domain((2, 2), frac=np.full((2, 2), 1.5))
    with pytest.raises(ValueError, match='no\\ element\\ with\\ frac'):
        Domain((2, 2), frac=np.zeros((2, 2)))


# ---------------------------------------------------------------------------------------------- filter vs old code
def test_density_filter_equals_topo3d_on_masked_part():
    shape = (7, 5, 4)
    mask = np.ones(shape, bool)
    mask[4:, 3:, :] = False
    mask[2, 1:3, 1:3] = False
    d = Domain(shape, 1.0, frac=mask.astype(float))
    mesh = VoxelMesh(mask, 1.0)
    x = np.random.default_rng(0).random(mesh.ne)
    for rmin in (1.5, 2.3):
        H, Hs = topo3d.density_filter(mesh, rmin)
        g = np.zeros(shape)
        g[mask] = x
        new = DensityFilter(d, rmin)(torch.as_tensor(g)).numpy()
        assert np.max(np.abs(new[mask] - (H @ x) / Hs)) < 1e-14
        assert np.all(new[~mask] == 0.0)


def test_density_filter_2d_equals_top88():
    nelx, nely = 9, 5
    g = topology.Grid2D(nelx, nely)
    H, Hs = g.filter_matrix(2.4)
    x_old = np.random.default_rng(1).random(g.nel)
    x_new = g.to_array(x_old)[::-1].T                          # (nely, nelx) top row first → (nelx, nely) y up
    out = DensityFilter(Domain((nelx, nely)), 2.4)(torch.as_tensor(x_new.copy())).numpy()
    assert np.max(np.abs(out.T[::-1] - g.to_array((H @ x_old) / Hs))) < 1e-14


# ---------------------------------------------------------------------------------------------- physics: patch tests
def _face_traction(d: Domain, sigma: float) -> np.ndarray:
    """Consistent nodal forces of a uniform traction σ (x direction) on the face x = L."""
    F = d.load_vector()
    ix = d.shape[0] - 1
    for e in np.flatnonzero(d.element_ijk()[:, 0] == ix):
        nodes = d.edof[e, ::d.dim] // d.dim
        on_face = d.node_coords()[nodes, 0] > d.origin[0] + d.shape[0] * d.h - 1e-9
        share = sigma * (d.h ** 2 / 4 if d.dim == 3 else d.h * d.thickness / 2)
        F[d.dim * nodes[on_face], 0] += share
    return F


@pytest.mark.parametrize("element", ["H8", "H8IM"])
def test_patch_uniform_tension_3d(element):
    E, nu, sigma = 2.5, 0.3, 3.0
    d = Domain((3, 2, 2), 0.5, origin=(0.2, -0.1, 0.3))
    o = d.origin
    fixed = np.concatenate([d.dofs(d.nodes_where(lambda X, k=k: abs(X[:, k] - o[k]) < 1e-9), k) for k in range(3)])
    phys = Elasticity(d, fixed, nu=nu, element=element)
    F = _face_traction(d, sigma)
    U = phys.solve_numpy(np.full(phys.n_elements, E), F)[:, 0]
    R = d.node_coords() - o
    exact = np.column_stack([sigma / E * R[:, 0], -nu * sigma / E * R[:, 1], -nu * sigma / E * R[:, 2]]).ravel()
    assert np.max(np.abs(U - exact)) <= 1e-12 * np.max(np.abs(exact))
    # uniform stress σ: every element stores uᵀKu = σ²/E · h³; total compliance σ² V / E
    ue = phys.element_energy(U)[:, 0] * phys.geom * E
    assert np.allclose(ue, sigma ** 2 / E * d.h ** 3, rtol=1e-12)
    V = np.prod(d.shape) * d.h ** 3
    assert np.isclose(F[:, 0] @ U, sigma ** 2 * V / E, rtol=1e-12)


def test_patch_uniform_tension_2d_plane_stress():
    E, nu, sigma, t = 1.7, 0.25, 2.0, 0.1
    d = Domain((4, 3), 0.25, thickness=t)
    fixed = np.concatenate([d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9), 0),
                            d.dofs(d.nodes_where(lambda X: X[:, 1] < 1e-9), 1)])
    phys = Elasticity(d, fixed, nu=nu, element="Q4")
    F = _face_traction(d, sigma)
    U = phys.solve_numpy(np.full(phys.n_elements, E), F)[:, 0]
    X = d.node_coords()
    exact = np.column_stack([sigma / E * X[:, 0], -nu * sigma / E * X[:, 1]]).ravel()
    assert np.max(np.abs(U - exact)) <= 1e-12 * np.max(np.abs(exact))
    ue = phys.element_energy(U)[:, 0] * phys.geom * E
    assert np.allclose(ue, sigma ** 2 / E * d.h ** 2 * t, rtol=1e-12)


def test_elasticity_rejects_bad_fixed():
    d = Domain((2, 2))
    with pytest.raises(ValueError, match='out\\ of\\ range'):
        Elasticity(d, [d.ndof])
    with pytest.raises(ValueError, match='no\\ free\\ DOF'):
        Elasticity(d, np.arange(d.ndof))
    with pytest.raises(ValueError, match='2\\-D\\ element'):
        Elasticity(d, [0], element="H8")


# ---------------------------------------------------------------------------------------------- direct solvers
def test_pardiso_equals_superlu():
    shape = (8, 5, 4)
    mask = np.ones(shape, bool)
    mask[5:, 3:, :] = False
    d = Domain(shape, 0.01, frac=mask.astype(float))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    rng = np.random.default_rng(2)
    E = 1e-3 + rng.random(d.n_active)
    F = rng.standard_normal((d.ndof, 3))
    pard, slu = DirectSolver("auto"), DirectSolver("splu")
    from toporlogy.solvers import pardiso
    if pard.method != "pardiso" and str(pardiso.state()["reason"]).startswith("ModuleNotFoundError"):
        pytest.skip(f"pypardiso not installed ({pardiso.state()['reason']}); set TOPORLOGY_PYLIB to test it")
    # any other reason (torch imported before toporlogy: MKL refused) is the failure this test exists for
    assert pard.method == "pardiso", f"PARDISO not loaded: {pardiso.state()['reason']}"
    assert slu.method == "splu"
    U1 = Elasticity(d, fixed, solver=pard).solve_numpy(E, F)
    U2 = Elasticity(d, fixed, solver=slu).solve_numpy(E, F)
    rel = np.max(np.abs(U1 - U2)) / np.max(np.abs(U2))
    print(f"\n  PARDISO vs SuperLU: max rel diff {rel:.2e}")
    assert rel <= 1e-10
    # residual of the reduced system
    phys = Elasticity(d, fixed, solver=slu)
    A = phys.matrix(E)
    K = A + A.T - sp.diags(A.diagonal())                       # full matrix from the upper triangle
    r = K @ U1[phys.free] - F[phys.free]
    assert np.max(np.abs(r)) <= 1e-9 * np.max(np.abs(F))
    with pytest.raises(ValueError, match="method: auto"):
        DirectSolver("cholmod")
    with pytest.raises(RuntimeError, match="factor"):
        DirectSolver("splu").solve(np.ones(3))


# ---------------------------------------------------------------------------------------------- Study input checks
class _UResponse(Response):
    name = "u_dof"
    needs = frozenset({"u"})

    def __call__(self, ctx):
        return ctx.U[0, 0]


def _small_2d():
    d = Domain((6, 3))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    F = d.load_vector()
    F[d.dofs(d.nodes_where(lambda X: (X[:, 0] == 6) & (X[:, 1] == 0)), 1), 0] = -1.0
    return d, Elasticity(d, fixed), F, standard_chain(d, 1.5, projection=False)


def test_study_rejects_bad_inputs():
    d, phys, F, chain = _small_2d()
    vol = Constraint(Volume(), 0.4)
    with pytest.raises(ValueError, match="rows"):
        Study(d, phys, F[:-2], chain, SIMP(d), Compliance(), [vol], OC())
    with pytest.raises(ValueError, match="at most 1"):
        Study(d, phys, F, chain, SIMP(d), Compliance(), [vol, Constraint(Volume(weighted=True), 0.5)], OC())
    st = Study(d, phys, F, chain, SIMP(d), Compliance(), [vol], OC())
    with pytest.raises(ValueError, match="x0"):
        st.run(x0=np.full(d.n_design + 1, 0.4), max_iter=1)
    with pytest.raises(ValueError, match="weights"):
        Study(d, phys, F, chain, SIMP(d), Compliance(weights=[1.0, 2.0]), [vol], OC()).evaluate(np.full(d.n_design, .4))
    # OC needs a gradient-free constraint function; a U-dependent constraint cannot give one: rejected when the
    # study is built (B-07; before, the run failed at its first step)
    with pytest.raises(ValueError, match="state solve"):
        Study(d, phys, F, chain, SIMP(d), Compliance(), [Constraint(_UResponse(), 1.0)], OC())
    st_u = Study(d, phys, F, chain, SIMP(d), Compliance(), [Constraint(_UResponse(), 1.0)], MMA())
    with pytest.raises(ValueError, match='needs\\ U'):
        st_u.constraint_values(np.full(d.n_design, 0.4))
    with pytest.raises(ValueError, match="constraint_fn"):                 # OC itself still refuses to step blind
        OC().step(np.full(4, 0.5), 1.0, -np.ones(4), np.array([-0.1]), np.ones((1, 4)))
    with pytest.raises(ValueError, match='move'):
        OC(move=0.0)


def test_study_evaluate_values():
    d, phys, F, chain = _small_2d()
    st = Study(d, phys, F, chain, SIMP(d, emin_ratio=1e-9), Compliance(), [Constraint(Volume(), 0.4)], OC())
    x = np.full(d.n_design, 0.4)
    ev = st.evaluate(x)
    U = phys.solve_numpy(np.full(d.n_active, 1e-9 + 0.4 ** 3 * (1 - 1e-9)), F)
    assert np.isclose(ev.f0, float(F[:, 0] @ U[:, 0]), rtol=1e-12)
    assert np.isclose(ev.g[0], 0.0, atol=1e-15) and np.isclose(ev.values["volume"], 0.4)
    assert np.allclose(ev.dg[0].sum(), 1.0 / 0.4)               # Σ ∂mean/∂x = 1, scaled by 1/V*
    assert ev.df0.shape == (d.n_design,) and np.all(ev.df0 < 0)


# ---------------------------------------------------------------------------------------------- Recorder
def test_recorder_run_folder(tmp_path):
    d, phys, F, chain = _small_2d()
    folder = tmp_path / "run1"
    rec = Recorder(folder, snapshot_every=2, meta={"case": "unit-test"})
    st = Study(d, phys, F, chain, SIMP(d), Compliance(), [Constraint(Volume(), 0.4, scale=1.0)],
               OC(dv_scale=d.n_active), recorder=rec)
    res = st.run(max_iter=5, tol=0.0)
    assert res.iterations == 5
    setup = json.loads((folder / "setup.json").read_text(encoding="utf-8"))
    assert setup["meta"] == {"case": "unit-test"} and "git" in setup and setup["settings"]["optimizer"] == "OC"
    assert setup["settings"]["links"] == ["Embed", "DensityFilter", "Passive"]
    rows = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["it"] for r in rows] == [1, 2, 3, 4, 5]
    assert np.allclose([r["f0"] for r in rows], res.history["f0"], rtol=0, atol=0)
    assert np.allclose([r["volume"] for r in rows], res.history["volume"])
    with np.load(folder / "result.npz") as z:
        assert np.array_equal(z["x"], res.x) and np.array_equal(z["xphys"], res.xphys)
    out = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert out["iterations"] == 5 and np.isclose(out["objective"], res.objective)
    with np.load(folder / "snapshots.npz") as z:
        assert sorted(z.files) == ["it00002", "it00004"] and z["it00004"].dtype == np.float16


def test_default_start_meets_the_weighted_volume_bound():
    """B-15: with partial voxels, Volume(weighted=True) is Σ frac·x̄ / Σ frac, but the start design was always built
    from the element count; the weighted volume of the start missed the bound. Now it starts exactly on it."""
    shape = (12, 6)
    frac = np.ones(shape)
    frac[:, -2:] = 0.3                                       # a band of partial voxels
    keep = np.zeros(shape, bool)
    keep[0, :2] = True
    d = Domain(shape, 1.0, frac=frac, keep=keep)
    from toporlogy import Chain, Embed
    for weighted in (False, True):
        st = Study(d, Elasticity(d, d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))), d.load_vector(),
                   Chain([Embed(d)]), SIMP(d), Compliance(), [Constraint(Volume(weighted=weighted), 0.4)], OC())
        v = st.evaluate(st.default_start(), gradients=False).values["volume"]
        assert v == pytest.approx(0.4, rel=1e-12), (weighted, v)
