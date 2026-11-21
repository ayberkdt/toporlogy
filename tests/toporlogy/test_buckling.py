"""Linear buckling (fem/buckling.py, optimization/buckling.py).

Element level: Gauss-point operators reproduce the element stiffness; stress stiffness of rigid motions is zero and
is linear in the load. Closed form: Euler column, clamped-free, P_cr = π² E I / (4 L²), 2-D (three meshes, error must
fall) and 3-D (H8IM). Gradients: pre-registered 1e-6 (Richardson, as test_gradients), including the term through the
displacements. A short optimisation shows the constraint raising the load factor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.fem.buckling import LinearBuckling, gauss_operators
from toporlogy.fem.elements import D_iso, D_plane_stress, element_matrix
from toporlogy.optimization.buckling import BucklingKS, BucklingLoadFactor, inverse_load_factors
from toporlogy.optimization.diagnostics import study_gradcheck

TOL = 1e-6


@pytest.mark.parametrize("dim,element", [(2, "Q4"), (3, "H8")])
def test_gauss_operators_reproduce_element_stiffness(dim: int, element: str) -> None:
    dN, B, w = gauss_operators(dim)
    D = D_plane_stress(0.3) if dim == 2 else D_iso(0.3)
    K = np.einsum("g,gvk,vw,gwl->kl", w, B, D, B)
    assert np.abs(K - element_matrix(dim, element, 0.3)).max() < 1e-14
    assert abs(w.sum() - 1.0) < 1e-15 and np.abs(dN.sum(axis=2)).max() < 1e-15      # partition of unity


def _column(shape, h, element=None, E0=1000.0, nu=0.0):
    d = T.Domain(shape, h)
    X = d.node_coords()
    ax = d.dim - 1 if d.dim == 3 else 1                     # the long axis: last entry of shape
    fixed = d.dofs(np.flatnonzero(X[:, ax] == 0))
    top = np.flatnonzero(np.isclose(X[:, ax], shape[ax] * h))
    F = d.load_vector()
    w = np.ones(len(top))
    for k in range(d.dim):
        if k != ax:
            w *= np.where(np.isclose(X[top, k], 0) | np.isclose(X[top, k], shape[k] * h), 0.5, 1.0)
    F[d.dim * top + ax, 0] = -w / w.sum()                   # unit compressive resultant
    m = T.Elasticity(d, fixed, nu, element)
    return d, m, F, np.full(d.n_active, E0)


def test_euler_column_2d_converges() -> None:
    """b = 4, L = 160, t = 1, E = 1000, ν = 0: slenderness 40, so the shear correction is below 0.1 %."""
    ratios = []
    for nb in (2, 4, 8):
        d, m, F, E = _column((nb, 40 * nb), 4.0 / nb)
        U = m.solve_numpy(E, F)
        mu, modes = LinearBuckling(m).modes(E, E, U[:, 0], 4)
        pe = np.pi ** 2 * 1000.0 * 4.0 ** 3 / 12 / (4 * 160.0 ** 2)
        ratios.append(1.0 / mu[0] / pe)
        assert modes.shape == (d.ndof, 4) and np.all(np.diff(mu) <= 1e-12)
    print(f"\n  Euler column 2-D, P_cr/P_Euler for 2, 4, 8 elements across: {np.round(ratios, 4)}; "
          f"mode ratios at the finest mesh {np.round(mu[0] / mu[1:3], 2)} (clamped-free: 9, 25)")
    err = np.abs(np.array(ratios) - 1)
    assert err[2] < err[1] < err[0] and err[2] < 0.01      # measured 0.123, 0.030, 0.006
    assert mu[0] / mu[1] == pytest.approx(9.0, rel=0.02)


def test_euler_column_3d_h8im() -> None:
    """4 x 4 x 64 column of side 2 (h = 0.5), H8IM: within 2 % of Euler (H8 locks: several % stiffer)."""
    out = {}
    for element in ("H8IM", "H8"):
        d, m, F, E = _column((4, 4, 64), 0.5, element)
        U = m.solve_numpy(E, F)
        mu, _ = LinearBuckling(m).modes(E, E, U[:, 0], 3)
        pe = np.pi ** 2 * 1000.0 * (2.0 ** 4 / 12) / (4 * 32.0 ** 2)
        out[element] = 1.0 / mu[0] / pe
        assert mu[0] == pytest.approx(mu[1], rel=1e-6)      # square section: the critical load is double
    print(f"\n  Euler column 3-D, P_cr/P_Euler: {out}")
    assert abs(out["H8IM"] - 1) < 0.02 and out["H8"] > out["H8IM"]


def test_stress_stiffness_of_rigid_motion_is_zero_and_linear_in_load() -> None:
    d, m, F, E = _column((3, 12), 1.0)
    lb = LinearBuckling(m)
    X = d.node_coords()
    rigid = np.zeros(d.ndof)
    rigid[0::2] = 0.3 - 0.01 * X[:, 1]                       # translation + small rotation: no strain
    rigid[1::2] = -0.2 + 0.01 * X[:, 0]
    assert abs(lb.geometric_matrix(E, rigid)).max() < 1e-12
    U = m.solve_numpy(E, F)[:, 0]
    K1, K3 = lb.geometric_matrix(E, U), lb.geometric_matrix(E, 3 * U)
    assert abs(K3 - 3 * K1).max() < 1e-12 * abs(K1).max() and abs(K1 - K1.T).max() == 0
    mu1, _ = lb.modes(E, E, U, 2)
    mu3, _ = lb.modes(E, E, 3 * U, 2)
    assert np.allclose(mu3, 3 * mu1, rtol=1e-8)             # λ halves when the load doubles
    with pytest.raises(ValueError):
        lb.modes(E, E, U, 0)
    with pytest.raises(TypeError):
        LinearBuckling(object())                            # type: ignore[arg-type]


def _study(dim: int, response) -> tuple[T.Study, T.Domain]:
    if dim == 2:
        d = T.Domain((6, 18), 1.0, thickness=0.8)
    else:
        d = T.Domain((3, 3, 10), 0.5)
    X = d.node_coords()
    ax = d.dim - 1
    fixed = d.dofs(np.flatnonzero(X[:, ax] == 0))
    top = np.flatnonzero(np.isclose(X[:, ax], X[:, ax].max()))
    F = d.load_vector()
    F[d.dim * top + ax, 0] = -1.0 / len(top)
    F[d.dim * top[:1], 0] = 0.02                             # a small side load: no symmetry, simple eigenvalues
    st = T.Study(d, T.Elasticity(d, fixed, 0.3), F, T.standard_chain(d, 1.5, beta=2.0), T.SIMP(d, E0=50.0),
                 T.Compliance(), [T.Constraint(T.Volume(), 0.6), T.Constraint(response, 1.0, scale=1.0)], T.MMA())
    return st, d


def _fd4(check, h: float = 1e-3) -> float:
    r1, r2 = check(h), check(2 * h)
    err = 0.0
    for a, b in zip(r1["directions"], r2["directions"], strict=True):
        fd = (4 * a["fd"] - b["fd"]) / 3
        err = max(err, abs(fd - a["ad"]) / max(abs(fd), abs(a["ad"]), 1e-300))
    return err


@pytest.mark.parametrize("dim", [2, 3])
def test_gradient_buckling_ks(dim: int) -> None:
    resp = BucklingKS(n_modes=4, rho=20.0, scale=1.0)
    study, d = _study(dim, resp)
    rng = np.random.default_rng(11)
    x0 = 0.45 + 0.4 * rng.random(d.n_design)
    err = _fd4(lambda h: study_gradcheck(study, x0, 1, n_dirs=4, h=h, seed=2), h=2e-4)
    print(f"\n  buckling KS gradient, {dim}-D: {err:.2e}; load factors {np.round(resp.last_load_factors, 3)}")
    assert err <= TOL, err


def test_gradient_single_load_factor_and_the_displacement_term() -> None:
    """dλ_1/dx by finite differences; and the same check with the dependence on u cut (detached) must FAIL: that is
    the term a hand-written sensitivity is reported to get wrong when it is dropped."""
    resp = BucklingLoadFactor(1, n_modes=3)
    study, d = _study(2, resp)
    x0 = 0.45 + 0.4 * np.random.default_rng(5).random(d.n_design)
    err = _fd4(lambda h: study_gradcheck(study, x0, 1, n_dirs=4, h=h, seed=3), h=2e-4)
    assert err <= TOL, err

    def cut(xt: torch.Tensor) -> torch.Tensor:
        ctx = study.context(xt, True)
        ctx.U = ctx.U.detach()
        return 1.0 / inverse_load_factors(ctx, 3)[0]

    xt = torch.as_tensor(x0).requires_grad_(True)
    (g_cut,) = torch.autograd.grad(cut(xt), xt)
    full = study.evaluate(x0).dg[1]
    rel = np.linalg.norm(g_cut.numpy() - full) / np.linalg.norm(full)
    print(f"\n  without the term through u the gradient of lambda_1 is off by {rel:.1%} (2-norm)")
    assert rel > 0.05


def test_cache_errors_and_report() -> None:
    resp = BucklingKS(n_modes=3)
    study, d = _study(2, resp)
    x0 = np.full(d.n_design, 0.6)
    ctx = study.context(torch.as_tensor(x0), True)
    a = inverse_load_factors(ctx, 3)
    assert inverse_load_factors(ctx, 3) is a                 # one eigen solve per context and setting
    v = resp(ctx)
    lam = resp.last_load_factors
    assert float(v) >= float(a.max()) and lam[0] == pytest.approx(1 / float(a[0])) and resp.scale is not None
    resp.reset_scale()
    assert resp.scale is None
    for bad in (lambda: BucklingKS(n_modes=0), lambda: BucklingLoadFactor(4, n_modes=3),
                lambda: inverse_load_factors(ctx, 3, case=5)):
        with pytest.raises(ValueError):
            bad()
    ctx.physics = object()
    ctx.extra.clear()
    with pytest.raises(TypeError, match="plain Elasticity"):
        inverse_load_factors(ctx, 3)


def test_constraint_raises_the_load_factor() -> None:
    """Column-like domain under compression with a volume bound: with the buckling constraint the critical load
    factor of the result is higher than without it, at a compliance penalty (both printed)."""
    def run(with_buckling: bool):
        d = T.Domain((16, 48), 1.0)
        X = d.node_coords()
        fixed = d.dofs(np.flatnonzero(X[:, 1] == 0))
        top = np.flatnonzero((X[:, 1] == 48) & (np.abs(X[:, 0] - 8) <= 2))
        F = d.load_vector()
        F[2 * top + 1, 0] = -1.0 / len(top)
        mon = BucklingKS(n_modes=6, rho=30.0)
        cons = [T.Constraint(T.Volume(), 0.35)]
        if with_buckling:
            cons.append(T.Constraint(mon, 1.0 / target, scale=target))
        st = T.Study(d, T.Elasticity(d, fixed, 0.3), F, T.standard_chain(d, 2.0, projection=False),
                     T.SIMP(d, E0=1.0, emin_ratio=1e-6), T.Compliance(), cons, T.MMA(move=0.1))
        res = st.run(max_iter=60, tol=1e-4)
        ctx = st.context(torch.as_tensor(res.x), True)
        mon(ctx)
        return res, float(mon.last_load_factors[0])

    target = 1.0
    base, lam0 = run(False)
    target = 1.6 * lam0
    con, lam1 = run(True)
    print(f"\n  compliance only: c = {base.objective:.2f}, lambda_1 = {lam0:.4f}; with lambda >= {target:.4f}: "
          f"c = {con.objective:.2f}, lambda_1 = {lam1:.4f}, KS value {con.constraint_values['buckling']:.4f} "
          f"(bound {1 / target:.4f})")
    assert lam1 > 1.3 * lam0
    assert con.objective >= base.objective * (1 - 1e-3)
