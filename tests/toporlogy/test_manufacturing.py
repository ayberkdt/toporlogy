"""Manufacturing-constraint links (toporlogy.design.manufacturing): AM overhang filter, tying, casting, composition.

Pre-registered thresholds: oracle equality of the AM forward pass ≤ 1e-12 (max abs); finite-difference gradients
≤ 1e-6 relative (:func:`toporlogy.optimization.diagnostics.gradcheck`, float64); tied optimisation results exact to 1e-12.

Oracles: :class:`topo_oracles.am_filter.AMFilter2D` (2-D, array (nely, nelx), row 0 at the top) and
:class:`topo_oracles.topo3d.AMFilter3D` (3-D, VoxelMesh of the full box = C-order of :class:`Domain`). Side names
of the old 2-D filter, measured here: base "S" ↔ build +y, "N" ↔ −y, "E" ↔ +x, "W" ↔ −x (the old module's E/W
labels are the reverse of its docstring: ``np.rot90(a, 1)`` brings the *west* column to the bottom).
The old 3-D ``overhang_violations`` applies the support stencil over the wrong axes (layer axis and first in-layer
axis); the 3-D count is therefore checked against a brute-force loop instead.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import (OC, SIMP, Compliance, Constraint, DensityFilter, Domain, Elasticity, Embed,
                          HeavisideProjection, Passive, Study, Volume, gradcheck)
from toporlogy.design.chain import Chain
from toporlogy.solvers import DirectSolver
from toporlogy.design.manufacturing import (AMFilter, Casting, Extrusion, GroupAverage, Pattern, Symmetry, Tying,
                                        combine_tying, manufacturing_chain, overhang_map, overhang_violations,
                                        parse_direction, undercut_map, undercuts, validate_chain)

TOL = 1e-6
ORACLE = 1e-12
NDIR = 4


def _interior(rng, shape):
    return 0.2 + 0.6 * rng.random(shape)


def _masked_3d() -> Domain:
    shape = (7, 6, 5)
    mask = np.ones(shape, bool)
    mask[5:, 4:, :] = False
    mask[2:4, 2:4, 1:3] = False
    keep = np.zeros(shape, bool)
    keep[-1, :2, 2:] = True
    void = np.zeros(shape, bool)
    void[3, 0, :2] = True
    return Domain(shape, 0.5, frac=mask.astype(float), keep=keep, void=void)


def _masked_2d() -> Domain:
    shape = (10, 8)
    mask = np.ones(shape, bool)
    mask[7:, 5:] = False
    mask[3:6, 2] = False
    keep = np.zeros(shape, bool)
    keep[-1, :2] = True
    void = np.zeros(shape, bool)
    void[1, 6] = True
    return Domain(shape, 1.0, frac=mask.astype(float), keep=keep, void=void)


def _lin(rng, shape):
    w = torch.as_tensor(rng.standard_normal(shape), dtype=torch.float64)
    return lambda y: (w * y).sum() + 0.3 * (y ** 2).sum()


# ================================================================================================== AM
_OLD2D = [("S", "+y"), ("N", "-y"), ("E", "+x"), ("W", "-x")]


@pytest.mark.parametrize("P,eps", [(40.0, 1e-4), (80.0, 1e-3)])
def test_am_2d_oracle(P, eps):
    from topo_oracles.am_filter import AMFilter2D
    from topo_oracles.am_filter import overhang_violations as old_ov
    rng = np.random.default_rng(1)
    d = Domain((13, 8))
    x = rng.random(d.shape)
    for side, build in _OLD2D:
        old = np.flipud(AMFilter2D(side, P=P, eps=eps).forward(np.flipud(x.T))).T
        new = AMFilter(d, build, P=P, eps=eps)(torch.as_tensor(x)).numpy()
        err = np.abs(old - new).max()
        print(f"\n  AM 2-D oracle {side}<->{build} P={P}: max|d| = {err:.1e}")
        assert err <= ORACLE
        assert overhang_violations(x, build) == old_ov(np.flipud(x.T), side)


def _brute_overhang(x, axis, sign, thr=0.5):
    s, n = x > thr, 0
    for idx in np.ndindex(x.shape):
        kb = idx[axis] - sign
        if not s[idx] or not 0 <= kb < x.shape[axis]:
            continue
        sup = False
        for o in range(x.ndim):
            for dd in ([0] if o == axis else [-1, 0, 1]):
                j = list(idx)
                j[axis] = kb
                if o != axis:
                    j[o] += dd
                if all(0 <= j[i] < x.shape[i] for i in range(x.ndim)) and s[tuple(j)]:
                    sup = True
        n += not sup
    return n


def test_am_3d_oracle():
    from topo_oracles.topo3d import AMFilter3D
    from topo_oracles.voxfem import VoxelMesh
    rng = np.random.default_rng(2)
    d = Domain((6, 5, 4))
    mesh = VoxelMesh(np.ones(d.shape, bool), 1.0)
    x = rng.random(d.shape)
    worst = 0.0
    for axis in range(3):
        for sign in (1, -1):
            old = AMFilter3D(mesh, axis, sign).forward(x.ravel()).reshape(d.shape)
            new = AMFilter(d, (axis, sign))(torch.as_tensor(x)).numpy()
            worst = max(worst, np.abs(old - new).max())
            assert overhang_violations(x, (axis, sign)) == _brute_overhang(x, axis, sign)
    print(f"\n  AM 3-D oracle, 6 directions: max|d| = {worst:.1e}")
    assert worst <= ORACLE


def test_parse_direction():
    assert parse_direction("+z", 3) == (2, 1)
    assert parse_direction("-x", 2) == (0, -1)
    assert parse_direction("y", 3) == (1, 1)
    assert parse_direction((2, -1), 3) == (2, -1)
    with pytest.raises(ValueError, match='not\\ valid\\ in\\ 2\\-D'):
        parse_direction("+z", 2)


def test_am_masked_convention():
    """Outside cells support nothing; keep cells are printed solid and support the layer above."""
    shape = (7, 6)
    frac = np.ones(shape)
    frac[1:4, 1] = 0.0                                 # three outside cells under (2, 2)
    keep = np.zeros(shape, bool)
    keep[5, 2] = True                                  # keep cell floating above empty cells
    d = Domain(shape, frac=frac, keep=keep)
    x = np.zeros(shape)
    x[2, 2] = 1.0                                      # solid above the hole
    x[5, 3] = 1.0                                      # solid on top of the keep cell
    xi = AMFilter(d, "+y")(torch.as_tensor(x)).numpy()
    assert xi[2, 2] < 0.01                             # unsupported: the outside cells are not printed
    assert np.all(xi[~d.active] == 0.0)
    assert xi[5, 2] == 1.0                             # keep: printed solid
    assert xi[5, 3] > 0.99                             # supported by the keep cell below
    om = overhang_map(x, "+y", d)
    assert om[2, 2] and not om[5, 3]


def test_am_base_layer():
    rng = np.random.default_rng(3)
    d = Domain((5, 4, 9), h=0.5, origin=(0, 0, 1.0))
    x = torch.as_tensor(rng.random(d.shape))
    f = AMFilter(d, "-z", base_layer=2)
    y = f(x).numpy()
    np.testing.assert_array_equal(y[:, :, -3:], x.numpy()[:, :, -3:])       # plate layers printed as designed
    assert np.any(np.abs(y[:, :, :-3] - x.numpy()[:, :, :-3]) > 1e-3)
    g = AMFilter(d, "-z", base_coordinate=1.0 + 0.5 * 6.2)                   # coordinate inside layer index 6
    assert g.base_layer == 2
    np.testing.assert_array_equal(g(x).numpy(), y)


@pytest.mark.parametrize("dim,build", [(2, "-x"), (3, "+z"), (3, "-y")])
def test_am_numpy_adjoint_equals_autograd(dim, build):
    """The CPU path (numpy forward + hand-written adjoint) equals the pure-torch autograd path."""
    rng = np.random.default_rng(14)
    d = _masked_2d() if dim == 2 else _masked_3d()
    fa, fb = AMFilter(d, build, base_layer=1), AMFilter(d, build, base_layer=1, autograd=True)
    x0 = _interior(rng, d.shape)
    w = torch.as_tensor(rng.standard_normal(d.shape))
    out = []
    for f in (fa, fb):
        x = torch.as_tensor(x0).requires_grad_(True)
        y = f(x)
        (g,) = torch.autograd.grad((w * y).sum(), x)
        out.append((y.detach().numpy(), g.numpy()))
    ey, eg = np.abs(out[0][0] - out[1][0]).max(), np.abs(out[0][1] - out[1][1]).max() / np.abs(out[1][1]).max()
    print(f"\n  AM numpy vs torch {dim}-D {build}: forward {ey:.1e}, gradient (rel) {eg:.1e}")
    assert ey <= 1e-13 and eg <= 1e-12


@pytest.mark.parametrize("build,base", [("+y", 0), ("-x", 2)])
def test_am_gradcheck_2d(build, base):
    rng = np.random.default_rng(4)
    d = _masked_2d()
    f = AMFilter(d, build, base_layer=base)
    obj = _lin(rng, d.shape)
    r = gradcheck(lambda x: obj(f(x)), _interior(rng, d.shape), n_dirs=NDIR)
    print(f"\n  gradcheck AM 2-D {build}: {r['max_rel_error']:.2e}")
    assert r["max_rel_error"] <= TOL


@pytest.mark.parametrize("build", ["+z", "-y", "+x"])
def test_am_gradcheck_3d(build):
    rng = np.random.default_rng(5)
    d = _masked_3d()
    f = AMFilter(d, build, base_layer=1)
    obj = _lin(rng, d.shape)
    r = gradcheck(lambda x: obj(f(x)), _interior(rng, d.shape), n_dirs=NDIR)
    print(f"\n  gradcheck AM 3-D {build}: {r['max_rel_error']:.2e}")
    assert r["max_rel_error"] <= TOL


# ================================================================================================== tying
def _closed(t: Tying) -> bool:
    d = t.domain
    pos = np.full(d.ne, -1)
    pos[d.design_elements] = np.arange(d.n_design)
    for m in t.maps:
        p = m[d.design_elements]
        ok = p >= 0
        ok[ok] = d.design.ravel()[p[ok]]
        if np.any(t.group_np[ok] != t.group_np[pos[p[ok]]]):
            return False
    return True


def _is_projection(t: Tying, rng) -> float:
    x, y = rng.random(t.domain.n_design), rng.random(t.domain.n_design)
    px, py = t.project(x), t.project(y)
    return max(np.abs(t.project(px) - px).max(), abs(float((x - px) @ py)))


def test_symmetry_off_grid_plane_and_two_planes():
    rng = np.random.default_rng(6)
    d = Domain((10, 9, 5))
    t = Symmetry(d, {"y": 3.7})                        # 2c = 7.4 → partner of iy is 6 − iy; iy = 7, 8 have none
    rep = t.report()
    print(f"\n  symmetry y=3.7: {rep}")
    assert rep["unpaired"]["outside_box"] == 2 * 10 * 5
    assert rep["n_groups"] == 10 * 5 * 6                 # per line: pairs (0,6) (1,5) (2,4), single 3, singles 7, 8
    assert _closed(t) and _is_projection(t, rng) < 1e-12
    t2 = Symmetry(d, [("y", 4.5), ("x", 5.0)])
    g = t2.group_np.reshape(d.shape)
    np.testing.assert_array_equal(g, g[:, ::-1, :])
    np.testing.assert_array_equal(g, g[::-1, :, :])
    sizes = np.bincount(np.bincount(t2.group_np))
    assert sizes[2] == 5 * 5 and sizes[4] == 5 * 4 * 5 and sizes.sum() - sizes[2] - sizes[4] == 0  # iy = 4 on plane
    assert _closed(t2)


def test_symmetry_gradient_is_group_sum():
    rng = np.random.default_rng(7)
    d = _masked_3d()
    t = Symmetry(d, {"y": 1.4, "z": 1.25})
    rep = t.report()
    print(f"\n  symmetry on masked domain: {rep}")
    assert rep["unpaired"]["outside_part"] > 0 and rep["unpaired"]["keep"] > 0
    assert _closed(t) and _is_projection(t, rng) < 1e-12
    w = torch.as_tensor(rng.standard_normal(d.n_design))
    fn = lambda z: (w * t(z) ** 3).sum()               # noqa: E731 - non-symmetric response
    z = torch.as_tensor(_interior(rng, t.n_in), dtype=torch.float64).requires_grad_(True)
    (gz,) = torch.autograd.grad(fn(z), z)
    x = t(z).detach().requires_grad_(True)
    (gx,) = torch.autograd.grad((w * x ** 3).sum(), x)
    np.testing.assert_allclose(gz.numpy(), np.bincount(t.group_np, gx.numpy(), t.n_in), rtol=1e-13, atol=1e-15)
    r = gradcheck(fn, z.detach().numpy(), n_dirs=NDIR)
    assert r["max_rel_error"] <= TOL
    np.testing.assert_allclose(t.reduce(t.expand(z.detach().numpy())), z.detach().numpy(), rtol=0, atol=1e-15)


def test_pattern_cyclic_closure_non_mutual():
    """Nearest-cell rotation by 60° on a square grid is not an involution; the closure still gives a projection."""
    rng = np.random.default_rng(8)
    d = Domain((12, 12))
    t = Pattern.cyclic(d, 6, center=(6.0, 6.0))
    rep = t.report()
    print(f"\n  cyclic n=6: {rep}")
    assert rep["non_mutual_relations"] > 0
    assert _closed(t) and _is_projection(t, rng) < 1e-12
    t4 = Pattern.cyclic(Domain((8, 8, 3)), 4, center=(4.0, 4.0, 0.0), axis="z")
    g = t4.group_np.reshape(8, 8, 3)
    np.testing.assert_array_equal(g, np.rot90(g, 1, axes=(0, 1)))      # 90° is exact on the grid
    assert t4.report()["non_mutual_relations"] == 0


def test_pattern_translational_and_extrusion():
    rng = np.random.default_rng(9)
    d = Domain((12, 4, 3))
    t = Pattern.translational(d, (4.0, 0.0, 0.0))
    assert set(np.bincount(t.group_np)) == {3} and t.n_in == 4 * 4 * 3
    g = t.group_np.reshape(d.shape)
    np.testing.assert_array_equal(g[:4], g[4:8])
    np.testing.assert_array_equal(g[:4], g[8:])
    dm = _masked_3d()
    e = Extrusion(dm, "z")
    rep = e.report()
    print(f"\n  extrusion z on masked domain: {rep}")
    assert rep["unpaired"]["outside_part"] > 0 and rep["unpaired"]["keep"] > 0 and rep["unpaired"]["void"] > 0
    gz = np.full(dm.shape, -1)
    gz.ravel()[dm.design_elements] = e.group_np
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            col = gz[i, j][gz[i, j] >= 0]
            assert len(set(col)) <= 1
    assert _closed(e) and _is_projection(e, rng) < 1e-12
    c = combine_tying(Symmetry(dm, {"y": 1.5}), e)
    assert _closed(c) and c.n_in < e.n_in
    w = torch.as_tensor(rng.standard_normal(dm.n_design))
    r = gradcheck(lambda z: (w * c(z) ** 2).sum(), _interior(rng, c.n_in), n_dirs=NDIR)
    assert r["max_rel_error"] <= TOL


def test_group_average_gradcheck():
    rng = np.random.default_rng(10)
    d = _masked_3d()
    t = Extrusion(d, "x")
    ga = GroupAverage(t)
    flt = DensityFilter(d, 1.8)
    obj = _lin(rng, d.shape)
    x = torch.as_tensor(_interior(rng, d.shape))
    y = ga(x).numpy()
    np.testing.assert_allclose(t.reduce(y.ravel()[d.design_elements])[t.group_np], y.ravel()[d.design_elements],
                               atol=1e-15)
    r = gradcheck(lambda v: obj(ga(flt(v))), x.numpy(), n_dirs=NDIR)
    assert r["max_rel_error"] <= TOL


# ================================================================================================== casting
@pytest.mark.parametrize("direction,parting", [("+z", None), ("-x", None), ("+y", 1.3)])
def test_casting_gradcheck(direction, parting):
    rng = np.random.default_rng(11)
    d = _masked_3d()
    c = Casting(d, direction, parting=parting)
    obj = _lin(rng, d.shape)
    r = gradcheck(lambda x: obj(c(x)), _interior(rng, d.shape), n_dirs=NDIR)
    print(f"\n  gradcheck casting {direction} parting={parting}: {r['max_rel_error']:.2e}")
    assert r["max_rel_error"] <= TOL


@pytest.mark.parametrize("direction,parting", [("+z", None), ("-z", None), ("+z", 1.1), ("+x", 1.6)])
def test_casting_no_undercut(direction, parting):
    rng = np.random.default_rng(12)
    d = _masked_3d()
    x = rng.random(d.shape)
    c = Casting(d, direction, parting=parting)
    y = c(torch.as_tensor(x)).numpy()
    n_in = sum(undercuts(x, direction, d, parting, thr) for thr in (0.3, 0.5, 0.7))
    inherent = undercut_map(np.zeros(d.shape), direction, d, parting)    # keep cells beyond a void / hole
    n_out = [int((undercut_map(y, direction, d, parting, thr) & ~inherent).sum())
             for thr in np.linspace(0.05, 0.95, 19)]
    print(f"\n  casting {direction} parting={parting}: undercuts in {n_in} (3 thresholds), out {sum(n_out)} beyond the "
          f"{int(undercut_map(y, direction, d, parting).sum())} unavoidable (keep behind a void/hole) "
          f"(19 thresholds), max xhat = {y.max():.4f}")
    assert n_in > 0 and sum(n_out) == 0
    ok = ~c.zero.numpy() & ~d.keep
    assert np.all(y[ok] >= x[ok] - 1e-15)                                # cumulative max ≥ own value where allowed
    assert y.max() <= d.shape[parse_direction(direction, 3)[0]] ** (1 / c.p) + 1e-12


def test_casting_blocking_and_undercut_map():
    shape = (3, 6)
    void = np.zeros(shape, bool)
    void[1, 2] = True
    frac = np.ones(shape)
    frac[2, 0] = 0.0                                   # leading outside cell: the part starts at iy = 1
    d = Domain(shape, frac=frac, void=void)
    x = np.full(shape, 0.0)
    x[:, 4] = 1.0
    y = Casting(d, "+y")(torch.as_tensor(x)).numpy()
    np.testing.assert_allclose(y[0, :5], 1.0, atol=1e-12)                # filled down to the parting surface
    assert np.all(y[1, 2:] == 0.0)                                       # beyond the void cell: blocked
    assert np.all(y[1, :2] < 1e-150)                                     # nothing beyond the void to fill for
    np.testing.assert_allclose(y[2, 1:5], 1.0, atol=1e-12)
    assert y[2, 0] == 0.0
    assert undercuts(y, "+y", d) == 0
    um = undercut_map(x, "+y", d)
    assert um[0, :4].all() and not um[2, 0]


# ================================================================================================== composition
def test_manufacturing_chain_order():
    d = _masked_3d()
    t = Symmetry(d, {"y": 1.5})
    ch = manufacturing_chain(d, 1.5, tying=t, am={"build": "+z"}, casting=None)
    assert [type(m).__name__ for m in ch.links] == ["Symmetry", "Embed", "DensityFilter", "GroupAverage",
                                                    "AMFilter", "HeavisideProjection", "Passive"]
    with pytest.raises(ValueError, match='must\\ be\\ the\\ first\\ link'):
        validate_chain(Chain([Embed(d), t]))
    with pytest.raises(ValueError, match='out\\ of\\ order'):
        validate_chain(Chain([Embed(d), DensityFilter(d, 1.5), HeavisideProjection(), AMFilter(d), Passive(d)]))
    with pytest.raises(ValueError, match='need\\ a\\ DensityFilter'):
        validate_chain(Chain([Embed(d), AMFilter(d), Passive(d)]))
    rng = np.random.default_rng(13)
    ch2 = manufacturing_chain(d, 1.5, tying=[t, Extrusion(d, "x")], casting={"direction": "-z"})
    obj = _lin(rng, d.shape)
    r = gradcheck(lambda z: obj(ch2(z)), _interior(rng, ch2[0].n_in), n_dirs=NDIR)
    print(f"\n  gradcheck full manufacturing chain: {r['max_rel_error']:.2e}")
    assert r["max_rel_error"] <= TOL


# ================================================================================================== optimisation
def _cantilever_3d(shape=(24, 12, 8)) -> tuple[Domain, np.ndarray, np.ndarray]:
    """x = 0 clamped, line load −z on x = L at mid-height: symmetric about the mid-planes y = ny/2 and z = nz/2
    (the z-mirror flips the load sign, which leaves the compliance unchanged)."""
    d = Domain(shape)
    eps = 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - shape[0]) < eps) & (abs(X[:, 2] - shape[2] / 2) < eps))
    F = d.load_vector()
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    return d, F, fixed


def _run(d, F, fixed, chain, iters, volfrac=0.3, move=0.2):
    # 2-D: SuperLU (PARDISO's threads are slow for tiny systems when many processes share the CPU); 3-D: default
    solver = DirectSolver("splu") if d.dim == 2 else None
    study = Study(d, Elasticity(d, fixed, solver=solver), F, chain, SIMP(d), Compliance(), [Constraint(Volume(), volfrac, 1.0)],
                  OC(move=move, dv_scale=d.n_active))
    return study.run(max_iter=iters, tol=0.0)


@pytest.fixture(scope="module")
def cantilever():
    return _cantilever_3d()


def test_symmetric_optimisation_exact(cantilever):
    d, F, fixed = cantilever
    free = _run(d, F, fixed, manufacturing_chain(d, 1.5, projection=False), 6)
    asym_free = np.abs(free.xphys - free.xphys[:, ::-1, :]).max()
    one = _run(d, F, fixed, manufacturing_chain(d, 1.5, tying=Symmetry(d, {"y": 6.0}), projection=False), 6)
    two = _run(d, F, fixed, manufacturing_chain(d, 1.5, tying=Symmetry(d, {"y": 6.0, "z": 4.0}),
                                               projection=False), 6)
    a1 = np.abs(one.xphys - one.xphys[:, ::-1, :]).max()
    a2 = max(np.abs(two.xphys - two.xphys[:, ::-1, :]).max(), np.abs(two.xphys - two.xphys[:, :, ::-1]).max())
    print(f"\n  24x12x8, 6 it: untied asym_y = {asym_free:.1e} (c = {free.objective:.4f}); y-sym {a1:.1e} "
          f"(c = {one.objective:.4f}, {one.seconds:.1f} s); y+z-sym {a2:.1e} (c = {two.objective:.4f})")
    assert a1 <= 1e-12 and a2 <= 1e-12
    # a symmetric problem: the tie costs nothing (measured 18 Nov 2026: the same c to 4 digits)
    assert one.objective == pytest.approx(free.objective, rel=1e-6)
    # The symmetric load alone cannot tell a working tie from a no-op: the untied design is symmetric too, up to
    # round-off (measured 6.8e-12). With an off-centre load the untied design is clearly asymmetric and the tie must
    # force the mirror image against the physics.
    X = d.node_coords()
    side = d.nodes_where(lambda X: (abs(X[:, 0] - d.shape[0]) < 1e-6) & (abs(X[:, 1] - 2.0) < 1e-6)
                         & (abs(X[:, 2] - d.shape[2] / 2) < 1e-6))
    assert len(side) == 1 and X[side[0], 1] == 2.0
    Fa = F.copy()
    Fa[d.dofs(side, 2), 0] += -1.0
    free_a = _run(d, Fa, fixed, manufacturing_chain(d, 1.5, projection=False), 6)
    tied_a = _run(d, Fa, fixed, manufacturing_chain(d, 1.5, tying=Symmetry(d, {"y": 6.0}), projection=False), 6)
    asym_a = np.abs(free_a.xphys - free_a.xphys[:, ::-1, :]).max()
    tied_asym = np.abs(tied_a.xphys - tied_a.xphys[:, ::-1, :]).max()
    print(f"  off-centre load: untied asym_y = {asym_a:.2e} (c = {free_a.objective:.4f}); tied {tied_asym:.1e} "
          f"(c = {tied_a.objective:.4f})")
    assert asym_a > 1e-2 and tied_asym <= 1e-12


def test_extrusion_and_pattern_optimisation_exact():
    d, F, fixed = _cantilever_3d((16, 8, 6))
    ex = _run(d, F, fixed, manufacturing_chain(d, 1.5, tying=Extrusion(d, "y"), projection=False), 6)
    a_ex = np.abs(ex.xphys - ex.xphys[:, :1, :]).max()
    pt = _run(d, F, fixed, manufacturing_chain(d, 1.5, tying=Pattern.translational(d, (8.0, 0, 0)),
                                              projection=False), 6)
    a_pt = np.abs(pt.xphys[:8] - pt.xphys[8:]).max()
    print(f"\n  16x8x6, 6 it: extrusion y: max |xbar - xbar(y=0)| = {a_ex:.1e} (c = {ex.objective:.4f}); "
          f"pattern x/8: max copy diff = {a_pt:.1e} (c = {pt.objective:.4f})")
    assert a_ex <= 1e-12 and a_pt <= 1e-12


def _cantilever_2d(nx=48, ny=24):
    d = Domain((nx, ny))
    eps = 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - nx) < eps) & (abs(X[:, 1] - ny / 2) < eps))
    F = d.load_vector()
    F[d.dofs(tip, 1), 0] = -1.0
    return d, F, fixed


def test_am_and_casting_2d_runs():
    d, F, fixed = _cantilever_2d()
    iters = 60
    base = _run(d, F, fixed, manufacturing_chain(d, 2.0, projection=False), iters, volfrac=0.4, move=0.1)
    am = _run(d, F, fixed, manufacturing_chain(d, 2.0, am={"build": "+y"}, projection=False), iters, volfrac=0.4,
              move=0.1)
    cast = _run(d, F, fixed, manufacturing_chain(d, 2.0, casting={"direction": "+y"}, projection=False), iters,
                volfrac=0.4, move=0.1)
    ov_base, ov_am = overhang_violations(base.xphys, "+y", d), overhang_violations(am.xphys, "+y", d)
    uc_base, uc_cast = undercuts(base.xphys, "+y", d), undercuts(cast.xphys, "+y", d)
    print(f"\n  2-D 48x24 cantilever, {iters} it, V = 0.4: overhang violations (build +y) without AM {ov_base}, "
          f"with AM {ov_am}; c = {base.objective:.2f} / {am.objective:.2f}")
    print(f"  undercuts (pull +y) without casting {uc_base}, with casting {uc_cast}; c = {cast.objective:.2f}; "
          f"seconds {base.seconds:.1f} / {am.seconds:.1f} / {cast.seconds:.1f}")
    assert ov_base > 0 and ov_am <= 0.25 * ov_base
    assert uc_base > 0 and uc_cast == 0
