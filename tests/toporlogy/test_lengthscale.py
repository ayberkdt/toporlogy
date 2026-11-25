"""Length-scale measurement, grey level and local-volume (maximum size) response — toporlogy.optimization.lengthscale.

1. Analytic shapes: plate, bar, round hole, round solid disc, 3-D plate; residues of opening / closing.
2. Calibration of the local-volume bound: a member of thickness t_max passes, one element thicker fails.
3. Finite-difference gradients (threshold 1e-6, float64) of LocalVolume (p-mean and with padding rules) and GreyLevel.
4. 2-D cantilever with a maximum-thickness constraint (MMA): measured maximum member thickness ≤ target + one element,
   while the unconstrained design exceeds it.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.optimization.lengthscale import (GreyLevel, LocalVolume, ball_kernel, grey_measure, local_volume_field,
                                      max_thickness_settings, measure_lengthscale, slab_fraction)
from toporlogy.optimizers.mma import MMA

TOL = 1e-6


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """One torch / MKL thread while this module runs (restored afterwards). On a machine shared by several CPU-bound
    jobs the multithreaded PARDISO / torch kernels oversubscribe the cores: one robust MBB evaluation took 5.0-5.4 s
    multithreaded vs 0.16-0.25 s single-threaded (measured 14 Nov 2026, 100 % CPU load from other processes)."""
    n_torch = torch.get_num_threads()
    torch.set_num_threads(1)
    lib = None
    try:
        from toporlogy.solvers.pardiso import load as pardiso_module
        pp = pardiso_module()
        if pp is not None:
            lib = pp.PyPardisoSolver().libmkl
            n_mkl = lib.MKL_Get_Max_Threads()
            lib.MKL_Set_Num_Threads(1)
    except Exception:  # noqa: BLE001 - optional (SuperLU fallback has no thread setting)
        lib = None
    yield
    torch.set_num_threads(n_torch)
    if lib is not None:
        lib.MKL_Set_Num_Threads(n_mkl)


# ---------------------------------------------------------------------------------------------- measurement
def test_plate_thickness_exact_2d_and_3d():
    x = np.zeros((40, 30))
    x[10:16, :] = 1.0                                   # plate 6 elements thick, infinite along y (reflect)
    r = measure_lengthscale(x, h=0.5, outside="reflect", radii=(1.0, 2.0))
    assert r.solid_min == pytest.approx(3.0) and r.solid_max == pytest.approx(3.0)
    assert r.void_min == pytest.approx(10.0)            # 10 void elements to a symmetry face = 20-element gap
    assert r.opening_residue[1.0] == 0.0 and r.opening_residue[2.0] == 1.0   # radius 2 > half thickness 1.5
    assert r.closing_residue[1.0] == 0.0
    assert r.grey_fraction == 0.0 and r.m_nd == 0.0
    x3 = np.zeros((12, 8, 10))
    x3[:, :, 3:7] = 1.0
    r3 = measure_lengthscale(x3, 2.0, outside="reflect")
    assert r3.solid_min == pytest.approx(8.0) and r3.solid_max == pytest.approx(8.0)
    assert r3.void_min == pytest.approx(12.0)           # 3 void elements to a symmetry face → 6 elements = 12


def test_bar_with_free_ends_and_free_faces():
    """A 4-element bar with void faces everywhere: max = thickness; the square ends are convex corners, so the
    minimum local thickness is small (opening definition) while the median (50th percentile) is the thickness."""
    x = np.zeros((40, 20))
    x[5:35, 8:12] = 1.0
    r = measure_lengthscale(x, 1.0, outside="void")
    assert r.solid_max == pytest.approx(4.0)
    assert r.solid_percentiles[50.0] == pytest.approx(4.0)
    assert r.solid_min < 4.0                            # corners
    assert r.void_max > 10.0                            # the outside is void: the void reaches past the faces


def test_round_hole_and_disc():
    n = 48
    c = (np.arange(n) + 0.5) - n / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    hole = np.where(X ** 2 + Y ** 2 <= 9.0 ** 2, 0.0, 1.0)   # hole of diameter 18 elements
    r = measure_lengthscale(hole, 1.0, outside="reflect")
    assert abs(r.void_max - 18.0) <= 1.0                # digitised circle: within one element
    assert r.void_percentiles[50.0] > 14.0
    disc = 1.0 - hole
    rd = measure_lengthscale(disc, 1.0, outside="void")
    assert abs(rd.solid_max - 18.0) <= 1.0


def test_thin_member_and_knife_edge_residue():
    x = np.zeros((40, 40))
    x[5:35, 10:20] = 1.0                                # thick block
    x[5:35, 25] = 1.0                                   # 1-element fin
    r = measure_lengthscale(x, 1.0, outside="void", radii=(1.0,))
    assert r.solid_min <= 1.0 + r.resolution
    assert r.opening_residue[1.0] > 0.08                # the fin (30 of 330 elements) and block corners removed


def test_grey_measure_values():
    assert grey_measure(np.array([0.25, 1.0])) == pytest.approx(0.375)
    r = measure_lengthscale(np.full((10, 10), 0.5), 1.0)
    assert r.m_nd == pytest.approx(1.0) and r.grey_fraction == 1.0


# ---------------------------------------------------------------------------------------------- local volume
def test_ball_kernel_and_local_volume_rules():
    k = ball_kernel(2.0, 2)
    assert k.sum() == 13                                 # centroid distance ≤ 2
    x = torch.ones(6, 5, dtype=torch.float64)
    assert torch.allclose(local_volume_field(x, 2.0, "solid"), torch.ones_like(x))
    assert torch.allclose(local_volume_field(x, 2.0, "ignore"), torch.ones_like(x))
    assert torch.allclose(local_volume_field(x, 2.0, "reflect"), torch.ones_like(x))
    v = local_volume_field(x, 2.0, "void")
    assert float(v[0, 0]) == pytest.approx(6.0 / 13.0)   # corner: 6 of 13 kernel cells inside


@pytest.mark.parametrize("dim,member", [(2, "bar"), (3, "plate"), (3, "rod")])
def test_max_thickness_calibration(dim, member):
    """Member of thickness t_max: local volume ≤ α; one element thicker: > α (discrete ball, exact coverage)."""
    t = 4
    s = max_thickness_settings(t, 1.0, dim, member=member)
    n = 25
    shape = (n,) * dim

    def member_field(tt):
        g = np.indices(shape) - n // 2 + 0.5 * ((tt + 1) % 2)     # centre on an element (odd) or a face (even)
        if member == "rod":
            return (g[1] ** 2 + g[2] ** 2 <= (tt / 2) ** 2).astype(float)
        return (np.abs(g[0]) <= tt / 2 - 1e-9).astype(float)
    if member == "rod":                                  # digitised rods are not exact: test the continuum relation
        assert s["continuum"] == pytest.approx(slab_fraction(t / (2 * s["radius"]), 3, "rod"))
        return
    vmax = lambda tt: float(local_volume_field(member_field(tt), s["radius"], "reflect").max())
    assert vmax(t) <= s["alpha"] + 1e-12
    assert vmax(t + 1) > s["alpha"] + 0.02


def _cantilever(nx=48, ny=24):
    d = T.Domain((nx, ny), 1.0)
    eps = 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    F = d.load_vector()
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - nx) < eps) & (abs(X[:, 1] - ny / 2) < 1.0 + eps))
    F[d.dofs(tip, 1), 0] = -1.0 / len(tip)
    return d, fixed, F


def test_gradcheck_local_volume_and_grey():
    d = T.Domain((14, 9), 1.0, keep=np.pad(np.ones((1, 2), bool), ((13, 0), (0, 7))))
    chain = T.standard_chain(d, 2.0, projection=True, beta=4.0)
    rng = np.random.default_rng(3)
    x0 = 0.2 + 0.6 * rng.random(d.n_design)
    for resp in (LocalVolume(3.0, p=16.0), LocalVolume(2.5, p=8.0, outside=[("reflect", "void"), ("ignore", "solid")]),
                 LocalVolume(2.0, aggregate="max"), GreyLevel()):
        def fn(xt, resp=resp):
            v = xt
            fields = []
            for link in chain.links:
                v = link(v)
                fields.append(v)
            ctx = T.EvalContext(d, v, v, x=xt, fields=fields, chain=chain)
            return resp(ctx)
        g = T.gradcheck(fn, x0, n_dirs=4)
        assert g["max_rel_error"] < TOL, (type(resp).__name__, g)


# ---------------------------------------------------------------------------------------------- optimisation
def _run_cantilever(d, fixed, F, local=None, iters=160):
    chain = T.standard_chain(d, 1.5, projection=True, beta=1.0)
    cons = [T.Constraint(T.Volume(), 0.5)]
    if local is not None:
        cons.append(T.Constraint(local[0], local[1]))
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(), cons, MMA(move=0.2),
                 T.BetaContinuation(chain, 1.0, 16.0, 30, 0.002))
    return st.run(max_iter=iters, tol=0.002)


def test_max_thickness_cantilever():
    """Local-volume constraint (t_max = 4 elements): measured max member thickness ≤ t_max + 1 element; the design
    without it has thicker members (both printed)."""
    t0 = time.perf_counter()
    d, fixed, F = _cantilever()
    t_max = 4.0
    s = max_thickness_settings(t_max, 1.0, 2)
    free = _run_cantilever(d, fixed, F)
    lv = LocalVolume(s["radius"], p=16.0, adaptive=0.5, outside="void")
    con = _run_cantilever(d, fixed, F, (lv, s["alpha"]))
    m_free = measure_lengthscale(free.xphys, 1.0, outside="void")
    m_con = measure_lengthscale(con.xphys, 1.0, outside="void")
    lmax = float(local_volume_field(torch.as_tensor(con.xphys), s["radius"], "void").max())
    print(f"\n[max thickness] R {s['radius']:.2f} alpha {s['alpha']:.4f}; free: c {free.objective:.4g}, "
          f"{m_free.summary()}\n  constrained: c {con.objective:.4g}, max local volume {lmax:.4f}, {m_con.summary()}"
          f"\n  ({time.perf_counter() - t0:.1f} s)")
    assert m_free.solid_max > t_max + 1.0
    assert m_con.solid_max <= t_max + 1.0
    assert lmax <= s["alpha"] + 0.01                              # the true maximum, not only the p-mean
    assert m_con.grey_fraction < 0.05
