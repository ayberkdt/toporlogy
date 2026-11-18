"""toporlogy.bc: selectors, supports (incl. oblique constraints and springs), loads with exact resultants,
load cases, per-case supports, keep pads, checks and an L-bracket mesh-convergence run."""
import math

import numpy as np
import pytest
import torch

from toporlogy import SIMP, Compliance, Constraint, Elasticity, OC, Study, Volume, standard_chain, study_gradcheck
from toporlogy import bc
from toporlogy import geometry as g
from toporlogy.checks import gradcheck
from toporlogy.domain import Domain

BORE = ([34, 5, -1], [34, 5, 11], 3.0)


def lbracket(h: float) -> Domain:
    s = g.csg([("add", g.Box([0, 0, 0], [40, 10, 10])), ("add", g.Box([0, 0, 0], [10, 40, 10])),
               ("cut", g.Cylinder(*BORE))])
    return g.voxelize(s, h)


@pytest.fixture(scope="module")
def lb1() -> Domain:
    return lbracket(1.0)


def block(n=(6, 4, 4), h=1.0) -> Domain:
    return Domain(n, h)


# ------------------------------------------------------------------------------------------------ selectors
def test_bore_selector_excludes_plane_faces(lb1):
    info = bc.part_info(lb1)
    bore = bc.BoreFace(*BORE)
    n = bore.select(lb1)
    _, r, _ = bore.coords(info.X[n])
    assert len(n) > 100
    assert r.max() <= 3.0 + 0.75 + 1e-9                                  # never on the end-face annulus
    X = info.X[n]
    on_end = (np.abs(X[:, 2]) < 1e-9) | (np.abs(X[:, 2] - 10) < 1e-9)
    assert np.all(r[on_end] <= 3.0 + 0.5)                     # end-face nodes only on the staircase mouth edge
    # the same band without the normal filter picks up the plane faces at the bore mouths (the old bug)
    naive = bc.OnCylinder(*BORE, band=1.5, facing=None).select(lb1)
    Xn = info.X[naive]
    _, rn, _ = bore.coords(Xn)
    mouth = ((np.abs(Xn[:, 2]) < 1e-9) | (np.abs(Xn[:, 2] - 10) < 1e-9)) & (rn > 3.0 + 0.5)
    assert mouth.sum() > 20


def test_selector_set_operations_and_errors():
    d = block()
    top = bc.OnPlane([0, 0, 4], 2)
    left = bc.InBox([0, -1, -1], [2, 9, 9])
    a, b = top.select(d), left.select(d)
    assert np.array_equal((top | left).select(d), np.union1d(a, b))
    assert np.array_equal((top & left).select(d), np.intersect1d(a, b))
    assert np.array_equal((top - left).select(d), np.setdiff1d(a, b))
    assert len(top.select(d)) == 7 * 5
    assert len(bc.AtPoints([[0, 0, 0], [6, 4, 4]]).select(d)) == 2
    assert len(bc.Nearest([[3.2, 2.1, 9.0]]).select(d)) == 1
    with pytest.raises(bc.EmptySelectionError, match="selected no node"):
        bc.OnPlane([0, 0, 50], 2).select(d)
    with pytest.raises(bc.EmptySelectionError, match="AtPoints"):
        bc.AtPoints([[0.5, 0, 0]]).select(d)


# ------------------------------------------------------------------------------------------------ loads
def test_bearing_load_resultant_and_no_axial_component(lb1):
    info = bc.part_info(lb1)
    F = np.array([300.0, -1000.0, 0.0])
    for on in ("surface", "nodes"):
        for mode in ("parallel", "radial"):
            a = bc.BearingLoad(bc.BoreFace(*BORE), F, mode=mode, on=on).apply(info)
            assert np.allclose(a.force, F, rtol=0, atol=1e-12 * 1000)
            assert np.abs(a.forces[:, 2]).max() <= 1e-12 * 1000          # zero component along the axis
            _, _, er = bc.BoreFace(*BORE).coords(info.X[a.nodes])
            if on == "surface":                    # end faces count half: no bending moment about the bore centre
                assert np.abs(a.moment(X=info.X)[:2]).max() <= 1e-9 * 1000 * 3
            if on == "nodes":                                            # only the loaded half carries load
                loaded = np.linalg.norm(a.forces, axis=1) > 0
                assert np.all(er[loaded] @ (F / np.linalg.norm(F)) >= -1e-12) and loaded.sum() < len(loaded)
    # an axial component is carried exactly as well
    Fa = np.array([0.0, -1000.0, 200.0])
    a = bc.BearingLoad(bc.BoreFace(*BORE), Fa).apply(info)
    assert np.allclose(a.force, Fa, rtol=0, atol=1e-9)


def test_remote_moment_exact_and_force_free():
    d = block((8, 4, 4))
    info = bc.part_info(d)
    M = np.array([1.0, -2.0, 5.0])
    a = bc.RemoteMoment(bc.OnPlane([8, 0, 0], 0), M, point=[8, 2, 2]).apply(info)
    assert np.allclose(a.force, 0, atol=1e-13 * np.linalg.norm(M))
    assert np.allclose(a.moment(X=info.X), M, rtol=0, atol=1e-13 * np.linalg.norm(M))
    # 2-D: scalar moment
    d2 = Domain((10, 4), 1.0)
    i2 = bc.part_info(d2)
    a2 = bc.RemoteMoment(bc.OnPlane([10, 0], 0), 7.0).apply(i2)
    assert np.allclose(a2.force, 0, atol=1e-13) and a2.moment(X=i2.X)[0] == pytest.approx(7.0, abs=1e-12)
    # collinear nodes cannot carry a moment about their own line
    with pytest.raises(ValueError, match="cannot carry"):
        bc.RemoteMoment(bc.AtPoints([[8, 0, 0], [8, 0, 4]]), [0, 0, 1.0], point=[8, 0, 2],
                        weights="uniform").apply(info)


def test_point_and_remote_force_statically_equivalent():
    d = block((8, 4, 4))
    info = bc.part_info(d)
    F = np.array([0.0, 2.0, -10.0])
    p = np.array([7.3, 2.0, 4.0])                                        # off the node grid, on the top face
    a = bc.PointLoad(p, F).apply(info)
    assert len(a.nodes) >= 4
    assert np.allclose(a.force, F, atol=1e-13 * 10) and np.allclose(a.moment(X=info.X), 0, atol=1e-12)
    r = bc.RemoteForce(bc.OnPlane([8, 0, 0], 0), F, point=[12, 2, 2], moment=[0, 0, 3.0]).apply(info)
    assert np.allclose(r.force, F, atol=1e-12)
    assert np.allclose(r.moment([12, 2, 2], info.X), [0, 0, 3.0], atol=1e-12)
    with pytest.raises(bc.EmptySelectionError, match="no surface node within"):
        bc.PointLoad([4, 2, 20], F).apply(info)


def test_pressure_traction_surface_force():
    d = block((6, 4, 4), h=0.5)
    info = bc.part_info(d)
    closed = bc.Pressure(bc.SurfaceNodes(), 3.0).apply(info)             # closed surface: zero net force
    assert np.allclose(closed.force, 0, atol=1e-12)
    top = bc.Pressure(bc.OnPlane([0, 0, 2], 2), 3.0).apply(info)
    A = 3.0 * 2.0
    assert np.allclose(top.force, [0, 0, -3.0 * A], atol=1e-12)
    t = bc.Traction(bc.OnPlane([0, 0, 2], 2), [1.0, 0, 0]).apply(info)
    assert np.allclose(t.force, [A, 0, 0], atol=1e-12)
    s = bc.SurfaceForce(bc.OnPlane([3, 0, 0], 0), [0, 5.0, 0]).apply(info)
    assert np.allclose(s.force, [0, 5.0, 0], atol=1e-13)
    corner = s.forces[np.argmin(np.linalg.norm(info.X[s.nodes] - [3, 0, 0], axis=1)), 1]
    middle = s.forces[np.argmin(np.linalg.norm(info.X[s.nodes] - [3, 1, 1], axis=1)), 1]
    assert middle == pytest.approx(4 * corner)                            # consistent A/4 per face corner


# ------------------------------------------------------------------------------------------------ supports, checks
def test_rigid_body_check_flags_roller_cantilever():
    d = block((20, 4, 4))
    tip = [bc.LoadCase("tip", [bc.PointLoad([20, 2, 4], [0, 0, -1.0])])]
    B = bc.BoundaryConditions(d, [bc.Roller(bc.OnPlane([0, 0, 0], 0), 0)], tip)
    rep = B.check()
    assert "rigid-body" in rep.codes("error")
    msg = next(i.message for i in rep.errors if i.code == "rigid-body")
    assert msg.startswith("3 unrestrained") and "rotation about the axis (1, 0, 0)" in msg
    B2 = bc.BoundaryConditions(d, [bc.Fixed(bc.OnPlane([0, 0, 0], 0))], tip)
    assert B2.check().ok
    d2 = Domain((20, 4), 1.0)                                              # 2-D: pin at one point = 1 free mode
    B3 = bc.BoundaryConditions(d2, [bc.Fixed(bc.AtPoints([[0, 0]]))],
                               [bc.LoadCase("t", [bc.PointLoad([20, 2], [0, -1.0])])])
    msg = next(i.message for i in B3.check().errors if i.code == "rigid-body")
    assert msg.startswith("1 unrestrained") and "rotation about the point (0, 0)" in msg


def test_pinned_bore_free_modes(lb1):
    load = [bc.LoadCase("t", [bc.BearingLoad(bc.BoreFace(*BORE), [0, -1.0, 0])])]
    pin = bc.PinnedBore(bc.BoreFace([5, 35, -1], [5, 35, 11], 3.0))
    s = g.csg([("add", g.Box([0, 0, 0], [40, 10, 10])), ("add", g.Box([0, 0, 0], [10, 40, 10])),
               ("cut", g.Cylinder(*BORE)), ("cut", g.Cylinder([5, 35, -1], [5, 35, 11], 3.0))])
    d = g.voxelize(s, 1.0)
    msg = next(i.message for i in bc.BoundaryConditions(d, [pin], load).check().errors if i.code == "rigid-body")
    assert msg.startswith("2 unrestrained")
    pin_ax = bc.PinnedBore(bc.BoreFace([5, 35, -1], [5, 35, 11], 3.0), axial=True)
    msg = next(i.message for i in bc.BoundaryConditions(d, [pin_ax], load).check().errors if i.code == "rigid-body")
    assert msg.startswith("1 unrestrained") and "rotation about the axis (0, 0, 1) through (5, 35," in msg
    pin_t = bc.PinnedBore(bc.BoreFace([5, 35, -1], [5, 35, 11], 3.0), axial=True, tangential=True)
    B = bc.BoundaryConditions(d, [pin_t], load)
    assert B.check().ok


def test_pinned_bore_solution_radial_zero_2d():
    s = g.Box([0, 0], [30, 10]) - g.Circle([5, 5], 2.5)
    d = g.voxelize(s, 0.5)
    pin = bc.PinnedBore(bc.BoreFace([5, 5], None, 2.5))
    stop = bc.Roller(bc.AtPoints([[30, 0]]), 1)                           # stops the rotation about the pin
    B = bc.BoundaryConditions(d, [pin, stop], [bc.LoadCase("t", [bc.SurfaceForce(bc.OnPlane([30, 0], 0),
                                                                                  [1.0, 0.0])])])
    assert B.check().ok
    phys = B.physics()
    assert isinstance(phys, bc.ConstrainedElasticity)
    E = d.frac.ravel()[d.active_elements]
    F = B.F()
    U = phys.solve_numpy(E, F)[:, 0].reshape(-1, 2)
    n = pin.sel.select(d)
    _, _, er = pin.sel.coords(bc.part_info(d).X[n])
    assert np.abs(np.einsum("ij,ij->i", U[n], er)).max() <= 1e-10 * np.abs(U).max()
    assert np.abs(np.einsum("ij,ij->i", U[n], np.c_[-er[:, 1], er[:, 0]])).max() > 1e-3 * np.abs(U).max()
    R = bc.reaction_forces(phys, E, U.ravel(), F[:, 0])
    assert np.allclose(R.sum(axis=0), -F[:, 0].reshape(-1, 2).sum(axis=0), atol=1e-9)


def test_oblique_constraint_matches_penalty():
    d = block((6, 3, 3))
    d1 = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    bottom = bc.OnPlane([0, 0, 0], 2)
    side = bc.Roller(bc.OnPlane([0, 0, 0], 0), 0)
    load = [bc.LoadCase("t", [bc.SurfaceForce(bc.OnPlane([0, 0, 3], 2), [1.0, 0.5, -1.0])])]
    B = bc.BoundaryConditions(d, [bc.Fixed(bottom, [d1, [0, 0, 1]]), side], load)
    assert B.check().ok
    cs = B.constraints("t")
    assert len(cs.frames) > 0
    phys = B.physics()
    E = np.ones(d.n_active)
    U = phys.solve_numpy(E, B.F())[:, 0]
    nb = bottom.select(d)
    Un = U.reshape(-1, 3)[nb]
    assert np.abs(Un @ d1).max() < 1e-13 and np.abs(Un[:, 2]).max() < 1e-13
    k = 1e10
    pen = bc.BoundaryConditions(d, [bc.ElasticSupport(bottom, direction=d1, total=k * 9),
                                    bc.ElasticSupport(bottom, direction=[0, 0, 1], total=k * 9), side], load)
    Up = pen.physics().solve_numpy(E, pen.F())[:, 0]
    assert np.linalg.norm(Up - U) / np.linalg.norm(U) < 1e-6
    R = bc.reaction_forces(phys, E, U, B.F()[:, 0])
    assert np.allclose(R.sum(axis=0), -B.F()[:, 0].reshape(-1, 3).sum(axis=0), atol=1e-10)


def test_springs_exact_column():
    """ν = 0 column on springs (per area) under a top pressure: w_top = p/k + p H / E exactly."""
    E0, p, k, H = 200.0, 2.0, 50.0, 4.0
    d = block((3, 3, 4))
    B = bc.BoundaryConditions(d, [bc.ElasticSupport(bc.OnPlane([0, 0, 0], 2), k, direction="all")],
                              [bc.LoadCase("p", [bc.Pressure(bc.OnPlane([0, 0, H], 2), p)])])
    assert B.check().ok                                                    # springs restrain all rigid modes
    phys = B.physics(nu=0.0)
    U = phys.solve_numpy(np.full(d.n_active, E0), B.F())[:, 0].reshape(-1, 3)
    top = bc.OnPlane([0, 0, H], 2).select(d)
    assert np.allclose(U[top, 2], -(p / k + p * H / E0), rtol=1e-10)
    assert np.abs(U[:, :2]).max() < 1e-12


def test_symmetry_half_model_gives_half_compliance():
    full = Domain((8, 4, 4), 1.0)
    half = Domain((4, 4, 4), 1.0)
    cf = {}
    for name, d, sup in (("full", full, [bc.Fixed(bc.OnPlane([0, 0, 0], 2))]),
                         ("half", half, [bc.Fixed(bc.OnPlane([0, 0, 0], 2)), bc.Symmetry([4, 0, 0], 0)])):
        B = bc.BoundaryConditions(d, sup, [bc.LoadCase("p", [bc.Pressure(bc.OnPlane([0, 0, 4], 2), 1.0)])])
        F = B.F()
        U = B.physics().solve_numpy(np.ones(d.n_active), F)
        cf[name] = float((F * U).sum())
    assert cf["half"] == pytest.approx(cf["full"] / 2, rel=1e-10)


def test_load_on_support_and_overlap_warnings():
    d = block()
    face = bc.OnPlane([0, 0, 0], 0)
    B = bc.BoundaryConditions(d, [bc.Fixed(face), bc.Roller(bc.OnPlane([0, 0, 0], 2), 2)],
                              [bc.LoadCase("bad", [bc.SurfaceForce(face, [0, 0, -1.0]),
                                                   bc.NodalForces(bc.AtPoints([[6, 4, 4]]), [0, 0, -1.0])])])
    codes = B.check().codes("warning")
    assert {"load-on-support", "overlapping-supports", "single-node-load"} <= set(codes)


def test_empty_selection_message_names_the_case():
    d = block()
    with pytest.raises(bc.EmptySelectionError, match="load case 'lc'"):
        bc.BoundaryConditions(d, [bc.Fixed(bc.OnPlane([0, 0, 0], 0))],
                              [bc.LoadCase("lc", [bc.SurfaceForce(bc.OnPlane([0, 0, 99], 2), [0, 0, 1.0])])])


def test_cases_combinations_and_per_case_supports():
    d = block((8, 3, 3))
    left, right = bc.Fixed(bc.OnPlane([0, 0, 0], 0)), bc.Fixed(bc.OnPlane([8, 0, 0], 0))
    l1 = bc.SurfaceForce(bc.OnPlane([8, 0, 0], 0), [0, 0, -1.0])
    l2 = bc.SurfaceForce(bc.OnPlane([0, 0, 3], 2), [0, 1.0, 0])
    B = bc.BoundaryConditions(d, [left], [bc.LoadCase("a", [l1]), bc.LoadCase("b", [l2]),
                                          bc.LoadCase("c", [bc.SurfaceForce(bc.OnPlane([0, 0, 0], 0), [0, 0, 1.0])],
                                                      supports=[right])],
                              combinations=[bc.Combination("a+2b", {"a": 1.0, "b": 2.0})])
    F = B.F()
    assert F.shape == (d.ndof, 4) and np.allclose(F[:, 3], F[:, 0] + 2 * F[:, 1])
    phys = B.physics()
    assert isinstance(phys, bc.PerCaseSupports)
    E0 = np.ones(d.n_active)
    U = phys.solve(torch.as_tensor(E0), F).numpy()
    ref_left = Elasticity(d, B.fixed_dofs("a")).solve_numpy(E0, F[:, [0, 1, 3]])
    ref_right = Elasticity(d, B.fixed_dofs("c")).solve_numpy(E0, F[:, [2]])
    assert np.allclose(U[:, [0, 1, 3]], ref_left, rtol=1e-12, atol=1e-14)
    assert np.allclose(U[:, [2]], ref_right, rtol=1e-12, atol=1e-14)
    res = gradcheck(lambda x: (torch.as_tensor(F) * phys.solve(0.5 + x, F)).sum(), np.full(d.n_active, 0.5))
    assert res["max_rel_error"] < 1e-6
    with pytest.raises(ValueError, match="different supports"):
        bc.BoundaryConditions(d, [left], [bc.LoadCase("a", [l1]), bc.LoadCase("c", [l2], supports=[right])],
                              combinations=[bc.Combination("x", {"a": 1.0, "c": 1.0})])


def test_keep_mask_and_clearance():
    d = block((10, 4, 4), h=0.5)
    B = bc.BoundaryConditions(d, [bc.Fixed(bc.OnPlane([0, 0, 0], 0))],
                              [bc.LoadCase("t", [bc.SurfaceForce(bc.OnPlane([5, 0, 0], 0), [0, 0, -1.0])])])
    keep = B.keep_mask(thickness=0.1)                                     # clamped to 2 h = 1.0
    ix = np.argwhere(keep)[:, 0]
    assert keep[:2].all() and keep[-2:].all() and set(np.unique(ix)) == {0, 1, 8, 9}
    d2 = Domain(d.shape, d.h, frac=d.frac, keep=keep)
    assert d2.keep.sum() == keep.sum()
    void = bc.clearance_mask(d, g.Cylinder([2.5, 1, -1], [2.5, 1, 3], 0.6))
    assert void.any() and not (void & keep).any()


def test_body_force_matrix_and_design_dependent_compliance():
    d = block((6, 2, 3))
    G = bc.body_force_matrix(d, [0, 0, -9.81], density=2.0)
    assert np.allclose(np.asarray(G.sum(axis=1)).reshape(-1, 3).sum(axis=0), [0, 0, -9.81 * 2.0 * 36])
    bf = bc.BodyForce([0, 0, -9.81], 2.0).apply(bc.part_info(d))
    assert np.allclose(bf.force, bf.intended_force, rtol=1e-14)
    B = bc.BoundaryConditions(d, [bc.Fixed(bc.OnPlane([0, 0, 0], 0))],
                              [bc.LoadCase("tip", [bc.SurfaceForce(bc.OnPlane([6, 0, 0], 0), [0, 0, -1.0])])])
    F0 = B.F()
    resp = bc.BodyForceCompliance(F0, G, factors=[0.05])
    study = Study(d, B.physics(), F0, standard_chain(d, 1.5), SIMP(d, penal=3.0), resp,
                  [Constraint(Volume(), 0.5)], OC())
    x = np.random.default_rng(1).uniform(0.3, 0.8, d.n_design)
    res = study_gradcheck(study, x)
    assert res["max_rel_error"] < 1e-6
    # the value, rebuilt independently: F = F0 + s G x̄ (x̄ on the active elements), U from solve_numpy, c = Fᵀ U
    ev = study.evaluate(x, gradients=False)
    act = d.active_elements
    xbar = ev.xphys.reshape(-1)[act]
    E = SIMP(d, penal=3.0)(torch.as_tensor(ev.xphys)).numpy().reshape(-1)[act]
    Fd = F0.reshape(-1) + 0.05 * (G @ xbar)
    U = B.physics().solve_numpy(E, Fd[:, None])[:, 0]
    assert ev.f0 == pytest.approx(float(Fd @ U), rel=1e-10)
    assert abs(float(Fd @ U) - float(F0.reshape(-1) @ U)) > 1e-3 * ev.f0      # the body force term is not negligible
    # with zero body force it equals the plain compliance
    study0 = Study(d, B.physics(), F0, standard_chain(d, 1.5), SIMP(d, penal=3.0), Compliance(), [], OC())
    resp0 = bc.BodyForceCompliance(F0, G, factors=[0.0])
    e0 = study0.evaluate(x, gradients=False).f0
    study.objective = resp0
    assert study.evaluate(x, gradients=False).f0 == pytest.approx(e0, rel=1e-12)


def test_summary_table_plain_and_short(lb1):
    B = bc.BoundaryConditions(lb1, [bc.Fixed(bc.OnPlane([0, 40, 0], 1))],
                              [bc.LoadCase("down", [bc.BearingLoad(bc.BoreFace(*BORE), [0, -1000.0, 0])]),
                               bc.LoadCase("torque", [bc.Torque(bc.BoreFace(*BORE), 500.0)])])
    rep = B.check()
    assert rep.ok, str(rep)
    s = B.summary()
    assert s.isascii()
    assert 'Load case "down"' in s and 'Load case "torque"' in s and "resultants match" in s
    header = next(line for line in s.splitlines() if line.strip().startswith("Load "))
    assert len(header.split("  ")) - header.split("  ").count("") <= 7     # at most 7 columns
    assert s.splitlines()[-1].startswith("Check OK")
    tq = B.applied["torque"][0]
    assert np.allclose(tq.force, 0, atol=1e-10) and tq.moment(X=B.info.X)[2] == pytest.approx(500.0, abs=1e-9)


def test_lbracket_bearing_load_mesh_convergence():
    """L-bracket from CSG with a bore, top face clamped, bearing load: compliance finite and converging.

    Measured (E = 70e3, ν = 0.3, F = 1000, embedded bearing load): h = 2 → 587.60, h = 1.25 → 601.34,
    h = 1 → 604.90, h = 2/3 → 608.93 (grids aligned with the 10-thick plates). The node-based bearing load
    (``on="nodes"``) gave 590.15 / 594.80 / 606.60 / 610.14, not monotone. h = 0.8 (plates 12.5 elements thick,
    a half-filled outer layer) gives 592.76: the partial-volume ersatz stiffens a misaligned outer layer.
    """
    c = {}
    for h in (2.0, 1.25):
        d = lbracket(h)
        B = bc.BoundaryConditions(d, [bc.Fixed(bc.OnPlane([0, 40, 0], 1))],
                                  [bc.LoadCase("down", [bc.BearingLoad(bc.BoreFace(*BORE), [0, -1000.0, 0])])])
        B.check().raise_on_error()
        phys = B.physics(nu=0.3)
        assert type(phys) is Elasticity
        F = B.F()
        E = 70e3 * d.frac.ravel()[d.active_elements]
        U = phys.solve_numpy(E, F)
        c[h] = float((F * U).sum())
        R = bc.reaction_forces(phys, E, U[:, 0], F[:, 0])
        assert np.allclose(R.sum(axis=0), [0, 1000.0, 0], atol=1e-6)
    assert all(np.isfinite(v) and v > 0 for v in c.values())
    assert c[1.25] == pytest.approx(601.34, rel=1e-3) and c[2.0] == pytest.approx(587.60, rel=1e-3)
    assert abs(c[1.25] / c[2.0] - 1) < 0.05


def test_summary_of_a_case_without_loads():
    """B-09: BoundaryConditions accepts a load case with an empty load list (zero load vector), but summary() failed
    on it with max() of an empty sequence; the case is now listed as having no loads."""
    d = block((6, 3, 3))
    l1 = bc.SurfaceForce(bc.OnPlane([6, 0, 0], 0), [0, 0, -1.0])
    B = bc.BoundaryConditions(d, [bc.Fixed(bc.OnPlane([0, 0, 0], 0))], [bc.LoadCase("a", [l1]), bc.LoadCase("empty", [])])
    s = B.summary()
    assert 'Load case "empty": no loads' in s and 'Load case "a": 1 load(s)' in s
    assert not B.F()[:, 1].any()


def test_remote_moment_does_not_modify_the_load_object():
    """B-18: RemoteMoment.apply overwrote the object's F and point (its zero force and the part's centroid), so a
    load applied to one part changed what it held for the next. Applying it now leaves the object as built and
    gives each part its own centroid."""
    rm = bc.RemoteMoment(bc.OnPlane([0, 0, 0], 0), [0.0, 0.0, 5.0])
    F0, p0 = list(rm.F), list(rm.point)
    a1 = rm.apply(bc.part_info(Domain((4, 2, 2))))
    a2 = rm.apply(bc.part_info(Domain((4, 6, 8))))
    assert list(rm.F) == F0 and list(rm.point) == p0
    assert np.allclose(a1.ref_point, [0, 1, 1]) and np.allclose(a2.ref_point, [0, 3, 4])
    assert np.allclose(a1.force, 0.0) and np.allclose(a2.force, 0.0)


def test_equal_supports_built_separately_are_one_set():
    """B-11: support sets were told apart by the identity of the support objects, so the same clamp built twice made
    two sets (two factorisations) and a combination of their cases raised "mixes cases with different supports"."""
    d = block((8, 3, 3))
    l1 = bc.SurfaceForce(bc.OnPlane([8, 0, 0], 0), [0, 0, -1.0])
    l2 = bc.SurfaceForce(bc.OnPlane([0, 0, 3], 2), [0, 1.0, 0])
    clamp = lambda: [bc.Fixed(bc.OnPlane([0, 0, 0], 0))]                        # noqa: E731
    B = bc.BoundaryConditions(d, clamp(), [bc.LoadCase("a", [l1], supports=clamp()),
                                           bc.LoadCase("b", [l2], supports=clamp())],
                              combinations=[bc.Combination("a+b", {"a": 1.0, "b": 1.0})])
    assert len(B._sets) == 1
    F = B.F()
    assert np.allclose(F[:, 2], F[:, 0] + F[:, 1])
    other = [bc.Fixed(bc.OnPlane([8, 0, 0], 0))]                                  # really different: still refused
    with pytest.raises(ValueError, match="different supports"):
        bc.BoundaryConditions(d, clamp(), [bc.LoadCase("a", [l1]), bc.LoadCase("b", [l2], supports=other)],
                              combinations=[bc.Combination("a+b", {"a": 1.0, "b": 1.0})])


def test_design_dependent_body_force_with_per_case_supports():
    """B-10: with load cases that have different supports, physics() is PerCaseSupports, which offers only solve();
    BodyForceCompliance (design_load_solve) needed solve_numpy/_solve_free/factor and raised AttributeError. Each
    support group is now solved by its own physics. Reference: the two cases solved as two separate problems."""
    d = block((6, 2, 3))
    G = bc.body_force_matrix(d, [0, 0, -9.81], density=2.0)
    left, right = bc.Fixed(bc.OnPlane([0, 0, 0], 0)), bc.Fixed(bc.OnPlane([6, 0, 0], 0))
    la = bc.SurfaceForce(bc.OnPlane([6, 0, 0], 0), [0, 0, -1.0])
    lb = bc.SurfaceForce(bc.OnPlane([0, 0, 0], 0), [0, 0, -1.0])
    B = bc.BoundaryConditions(d, [left], [bc.LoadCase("a", [la]), bc.LoadCase("b", [lb], supports=[right])])
    assert isinstance(B.physics(), bc.PerCaseSupports)
    x = np.random.default_rng(3).uniform(0.3, 0.8, d.n_design)

    def value(Bx, F0, factors):
        st = Study(d, Bx.physics(), F0, standard_chain(d, 1.5), SIMP(d, penal=3.0),
                   bc.BodyForceCompliance(F0, G, factors=factors), [Constraint(Volume(), 0.5)], OC())
        return st, st.evaluate(x, gradients=False).f0

    st, both = value(B, B.F(), [0.05, 0.05])
    sa, a = value(bc.BoundaryConditions(d, [left], [bc.LoadCase("a", [la])]), B.F()[:, :1], [0.05])
    sb, b = value(bc.BoundaryConditions(d, [right], [bc.LoadCase("b", [lb])]), B.F()[:, 1:], [0.05])
    assert both == pytest.approx(a + b, rel=1e-12)
    g = st.evaluate(x).df0
    assert np.max(np.abs(g - sa.evaluate(x).df0 - sb.evaluate(x).df0)) <= 1e-12 * np.max(np.abs(g))
    # h = 1e-4: at the default step the FD truncation error is ~2e-6 here, and 7.7e-6 for case "a" alone (measured)
    assert study_gradcheck(st, x, h=1e-4)["max_rel_error"] < 1e-6
