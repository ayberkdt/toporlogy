"""Geometric multigrid solver (phase A2): kernels, accuracy against the direct solver, Study regression, gradients,
fallback. Reference = :class:`toporlogy.Elasticity` with :class:`DirectSolver` (PARDISO, else SuperLU).

Thresholds: solutions ≤ 1e-8 relative to the direct solve at tol 1e-10; Study compliance history ≤ 1e-6 (measured
1.5e-12 at tol 1e-10, 2.5e-7 at tol 1e-5 on the 24×12×8 cantilever, docs/olcumler/A2.md); gradients ≤ 1e-6 with the
five-point Richardson rule of ``test_gradients.py``. GPU cases are skipped without CUDA.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import (OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, EvalContext,
                          Response, Study, Volume, standard_chain, study_gradcheck)
from toporlogy.fem.elastic_gmg import ElasticityGMG
from toporlogy.fem.elements import element_matrix
from toporlogy.solvers import gmg_kernels as gk
from toporlogy.solvers.gmg import GMGSolver
from toporlogy.solvers.gmg_deflation import IslandDeflation
from toporlogy.solvers.gmg_transfer import BlackBoxTransfer

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
KE = element_matrix(3, "H8", 0.3)


def _dense_op(L: gk.Level) -> torch.Tensor:
    n = L.ndof
    cols = []
    for i in range(n):
        e = torch.zeros(n, dtype=torch.float64)
        e[i] = 1.0
        cols.append(L.apply(e.view(L.free.shape)).reshape(-1))
    return torch.stack(cols, 1)


def _random_level(shape, seed=0, fix_odd=True) -> gk.Level:
    rng = np.random.default_rng(seed)
    E = torch.as_tensor(1e-6 + rng.uniform(0, 1, shape) ** 3).reshape(-1)
    free = torch.ones(tuple(s + 1 for s in shape) + (3,), dtype=torch.float64)
    free[0] = 0.0
    if fix_odd:
        free[1, 1, 1, 2] = 0.0                     # a fixed DOF that is not on a coarse node
    return gk.Level(shape, free, s=E, KE=torch.as_tensor(KE))


@pytest.mark.parametrize("transfer", ["trilinear", "blackbox"])
def test_element_galerkin_equals_global(transfer):
    """K_c from element-level Galerkin == P̃ᵀ (M K M) P̃ assembled globally (masked prolongation)."""
    shape = (4, 4, 2)
    L = _random_level(shape)
    nc = tuple(s // 2 + 1 for s in shape) + (3,)
    Nc = int(np.prod(nc))
    if transfer == "trilinear":
        prol = gk.prolong
        Kc = gk.galerkin_from_scale(L.s, L.KE, shape, L.free, torch.as_tensor(gk.local_P()))
    else:
        T = BlackBoxTransfer(L)
        prol = T.prolong
        Kc = T.galerkin(L)
    P = torch.stack([prol(torch.eye(Nc, dtype=torch.float64)[j].view(nc)).reshape(-1) for j in range(Nc)], 1)
    Pt = L.free.reshape(-1, 1) * P
    K = _dense_op(L) - torch.diag(L.fixed.reshape(-1))
    ref = Pt.T @ K @ Pt
    Lc = gk.Level(tuple(s // 2 for s in shape), torch.ones(nc, dtype=torch.float64), Ke=Kc)
    got = _dense_op(Lc)                              # free = 1: apply = raw
    assert float((got - ref).abs().max() / ref.abs().max()) < 1e-13


@pytest.mark.parametrize("cycle", ["V", "W"])
def test_cycle_and_deflation_symmetric(cycle):
    """<M a, b> = <a, M b> for the V/W cycle with the deflation correction (CPU float64)."""
    shape = (8, 4, 4)
    rng = np.random.default_rng(2)
    E = np.full(shape, 1e-6)
    E[:3] = 1.0
    E[5:7, 1:3, 1:3] = 1.0                         # floating stiff island
    fixed = np.zeros((9, 5, 5, 3), bool)
    fixed[0] = True
    s = GMGSolver(shape, KE, 1.0, device="cpu", cycle=cycle, n_geometric=2)
    s.update(E, fixed)
    assert s.deflation is not None and s.deflation.n == 1
    free = s.A64.free
    a = torch.as_tensor(rng.standard_normal(free.shape)) * free
    b = torch.as_tensor(rng.standard_normal(free.shape)) * free
    lhs, rhs = float((s.vcycle(a) * b).sum()), float((a * s.vcycle(b)).sum())
    assert abs(lhs - rhs) <= 1e-10 * abs(rhs)          # contrast 1e6: roundoff ~1e-12


def test_deflation_gram_matches_dense():
    """Zᵀ A Z assembled element-wise equals the dense product."""
    shape = (8, 4, 4)
    E = np.full(shape, 1e-6)
    E[:2] = 1.0
    E[4:6, 1:3, 1:3] = 1.0
    E[7:8, 3:4, 0:2] = 0.5
    free = torch.ones((9, 5, 5, 3), dtype=torch.float64)
    free[0] = 0
    L = gk.Level(shape, free, s=torch.as_tensor(E).reshape(-1), KE=torch.as_tensor(KE))
    D = IslandDeflation(L, free < 0.5, tau=1e-3)
    assert D.n == 2
    Z = torch.stack([D.Z(torch.eye(6 * D.n, dtype=torch.float64)[k], free).reshape(-1) for k in range(6 * D.n)], 1)
    G = Z.T @ (_dense_op(L) - torch.diag(L.fixed.reshape(-1))) @ Z
    Gi = torch.linalg.pinv(G)
    assert float((D.Ginv - Gi).abs().max() / Gi.abs().max()) < 1e-8
    zt = D.Zt(torch.arange(free.numel(), dtype=torch.float64).view(free.shape))
    assert torch.allclose(zt, Z.T @ torch.arange(free.numel(), dtype=torch.float64))


def _masked_domain(shape=(12, 8, 6), h=0.5) -> Domain:
    mask = np.ones(shape, bool)
    mask[8:, 5:, :] = False                       # L shape
    mask[3:6, 2:5, 2:4] = False                   # cavity
    rng = np.random.default_rng(7)
    frac = np.where(mask, 0.6 + 0.4 * rng.random(shape), 0.0)
    return Domain(shape, h, frac=frac)


def _cantilever_bc(d: Domain, n_cases: int = 1):
    eps = 1e-6 * d.h
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: abs(X[:, 0] - d.shape[0] * d.h) < eps)
    F = d.load_vector(n_cases)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    if n_cases > 1:
        F[d.dofs(tip, 1), 1] = 0.7 / len(tip)
    return fixed, F


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("element", ["H8", "H8IM"])
def test_gmg_matches_direct_masked_domain(device, element):
    """Masked (non-box) domain, odd sizes (padding), orphan nodes, two load cases, high-contrast moduli."""
    d = _masked_domain((13, 7, 6))
    fixed, F = _cantilever_bc(d, 2)
    rng = np.random.default_rng(3)
    E = (1e-6 + rng.uniform(0, 1, d.n_active) ** 3) * d.frac.ravel()[d.active_elements]
    ref = Elasticity(d, fixed, element=element).solve_numpy(E, F)
    g = ElasticityGMG(d, fixed, element=element, tol=1e-10, direct_below=0, device=device)
    U = g.solve_numpy(E, F)
    assert g.n_fallbacks == 0
    assert g.shape_pad != d.shape
    rel = np.linalg.norm(U - ref) / np.linalg.norm(ref)
    print(f"\n  {device} {element}: rel diff {rel:.2e}, iters {[r['iters'] for r in g.solve_log]}")
    assert rel <= 1e-8
    inactive = np.setdiff1d(np.arange(d.ndof), d.active_dofs)
    assert np.all(U[inactive] == 0) and np.all(U[fixed] == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_island_deflation_reduces_iterations(device):
    shape = (16, 8, 8)
    E = np.full(shape, 1e-6)
    E[:2] = 1.0
    E[:, :, :2] = 1.0
    E[8:12, 3:6, 4:7] = 1.0                       # floating island
    fixed = np.zeros((17, 9, 9, 3), bool)
    fixed[0] = True
    F = np.zeros((17, 9, 9, 3))
    F[16, :, 0, 2] = -1.0 / 9
    out = {}
    for tau in (None, 1e-3):
        s = GMGSolver(shape, KE, 1.0, device=device, deflate_tau=tau, tol=1e-10, maxit=1000)
        out[tau] = (s.solve(E, F, fixed), s.last["iters"])
    assert out[1e-3][1] < out[None][1]
    assert np.linalg.norm(out[1e-3][0] - out[None][0]) <= 1e-8 * np.linalg.norm(out[None][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graph_reused_across_updates():
    shape = (16, 8, 8)
    fixed = np.zeros((17, 9, 9, 3), bool)
    fixed[0] = True
    F = np.zeros((17, 9, 9, 3))
    F[16, :, 0, 2] = -1.0
    s = GMGSolver(shape, KE, 1.0, device="cuda", tol=1e-10)
    rng = np.random.default_rng(0)
    for _ in range(3):
        E = 1e-3 + rng.uniform(0, 1, shape) ** 3
        u = s.solve(E, F, fixed)
        assert s.last["converged"]
    assert s.n_graph_captures == 1 and s.last_setup["graph_reused"]
    ref = GMGSolver(shape, KE, 1.0, device="cuda", graphs=False, tol=1e-10).solve(E, F, fixed)
    assert np.linalg.norm(u - ref) <= 1e-8 * np.linalg.norm(ref)


def _study(physics_kind: str, shape=(24, 12, 8), **kw) -> Study:
    d = Domain(shape, 1e-3)
    eps = 1e-9
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - shape[0] * 1e-3) < eps) & (X[:, 2] < eps))
    F = d.load_vector(1)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    phys = Elasticity(d, fixed) if physics_kind == "direct" else ElasticityGMG(d, fixed, direct_below=0, **kw)
    chain = standard_chain(d, 1.5, projection=True, beta=1.0)
    sched = BetaContinuation(chain, 1.0, 16.0, every=8, tol=0.0)
    return Study(d, phys, F, chain, SIMP(d, emin_ratio=1e-6), Compliance(),
                 [Constraint(Volume(), 0.3, scale=1.0)], OC(dv_scale=d.n_active), sched)


def test_study_regression_against_direct():
    """25 OC iterations with projection + β continuation: compliance history within 1e-6 of PARDISO."""
    hist = {}
    for kind in ("direct", "gmg"):
        st = _study(kind, tol=1e-10)
        hist[kind] = np.array(st.run(max_iter=25, tol=0.0).history["f0"])
        if kind == "gmg":
            assert st.physics.n_fallbacks == 0 and st.physics.n_adjoint_solves == 0
    rel = float(np.max(np.abs(hist["gmg"] / hist["direct"] - 1)))
    print(f"\n  compliance history max rel diff {rel:.2e}")
    assert rel <= 1e-6


class _TipDisplacement(Response):
    name = "tip_u"
    needs = frozenset({"u"})

    def __init__(self, dof: int) -> None:
        self.dof = dof

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return -ctx.U[self.dof, 0]


def _fd4(check, h=1e-3) -> float:
    r1, r2 = check(h), check(2 * h)
    err = 0.0
    for a, b in zip(r1["directions"], r2["directions"], strict=True):
        fd = (4 * a["fd"] - b["fd"]) / 3
        err = max(err, abs(fd - a["ad"]) / max(abs(fd), abs(a["ad"]), 1e-300))
    return err


@pytest.mark.parametrize("which", ["compliance", "tip"])
def test_gradients(which):
    """Self-adjoint shortcut (compliance) and adjoint GMG solve (unloaded tip DOF), 5-point FD, tol 1e-12."""
    st = _study("gmg", shape=(12, 6, 4), tol=1e-12)
    if which == "tip":
        d = st.domain
        st.objective = _TipDisplacement(int(d.dofs(d.nodes_where(lambda X: (X[:, 0] > 11.5e-3) & (X[:, 2] > 3.5e-3)
                                                                   & (X[:, 1] < 1e-9)), 2)[0]))
    x = 0.2 + 0.6 * np.random.default_rng(8).random(st.domain.n_design)
    err = _fd4(lambda h: study_gradcheck(st, x, n_dirs=4, h=h))
    print(f"\n  gradcheck GMG {which}: {err:.2e}, adjoint solves {st.physics.n_adjoint_solves}")
    assert err <= 1e-6
    assert (st.physics.n_adjoint_solves > 0) == (which == "tip")


def test_fallback_is_logged():
    d = Domain((8, 4, 4), 1.0)
    fixed, F = _cantilever_bc(d)
    E = np.full(d.n_active, 0.5)
    ref = Elasticity(d, fixed).solve_numpy(E, F)
    small = ElasticityGMG(d, fixed)                                   # 128 elements < direct_below
    assert np.allclose(small.solve_numpy(E, F), ref, rtol=1e-10, atol=0)
    assert small.n_fallbacks == 1 and "small problem" in small.fallback_log[0]["reason"]
    capped = ElasticityGMG(d, fixed, direct_below=0, maxit=1, tol=1e-12)
    U = capped.solve_numpy(E, F)
    assert capped.n_fallbacks == 1 and "not converged" in capped.fallback_log[0]["reason"]
    assert np.linalg.norm(U - ref) <= 1e-10 * np.linalg.norm(ref)


@pytest.mark.parametrize("device", DEVICES)
def test_tolerance_contract_is_exact(device):
    """R0-2: converged ⇔ recomputed ‖F − Ku‖/‖F‖ ≤ requested tol (no hidden factor); values are reported."""
    shape = (16, 8, 8)
    rng = np.random.default_rng(5)
    E = 1e-6 + rng.uniform(0, 1, shape) ** 3
    fixed = np.zeros((17, 9, 9, 3), bool)
    fixed[0] = True
    F = np.zeros((17, 9, 9, 3))
    F[16, :, 0, 2] = -1.0 / 9
    s = GMGSolver(shape, KE, 1.0, device=device)
    for tol, maxit in ((1e-10, 500), (1e-10, 2), (1e-3, 500)):
        u = s.solve(E, F, fixed, tol=tol, maxit=maxit)
        last = s.last
        assert last["requested_tol"] == tol
        # independent recomputation of the true relative residual
        b = s.A64.free * torch.as_tensor(F, device=s.device)
        res = float((b - s.operator(u)).norm() / b.norm())
        assert abs(res - last["achieved_true_rel_residual"]) <= 1e-6 * res + 1e-15
        assert last["converged"] == (last["achieved_true_rel_residual"] <= tol)
        if maxit == 2:
            assert not last["converged"] and last["achieved_true_rel_residual"] > tol
        else:
            assert last["converged"]


def test_fail_closed_without_fallback():
    """R0-3: no direct fallback allowed + CG misses tol → GMGConvergenceError by default; allow_inexact=True
    returns the result, warns and counts it."""
    from toporlogy.solvers.gmg import GMGConvergenceError
    d = Domain((8, 4, 4), 1.0)
    fixed, F = _cantilever_bc(d)
    E = np.full(d.n_active, 0.5)
    strict = ElasticityGMG(d, fixed, direct_below=0, fallback_max_elements=0, maxit=1, tol=1e-12)
    assert strict.strict
    with pytest.raises(GMGConvergenceError, match="requested tol"):
        strict.solve_numpy(E, F)
    assert strict.n_fallbacks == 0 and strict.n_inexact == 0
    loose = ElasticityGMG(d, fixed, direct_below=0, fallback_max_elements=0, maxit=1, tol=1e-12,
                          allow_inexact=True)
    with pytest.warns(RuntimeWarning, match="allow_inexact"):
        U = loose.solve_numpy(E, F)
    assert np.all(np.isfinite(U))
    assert loose.n_inexact == 1 and loose.fallback_log[-1]["inexact"]
    assert loose.fallback_log[-1]["achieved_true_rel_residual"] > 1e-12
    assert loose.stats()["n_inexact"] == 1


def test_deflation_of_an_anchored_island_counted_as_floating_is_harmless():
    """B-19 (measured 17 Nov 2026): IslandDeflation gives each node the largest label of its elements, so a stiff
    block anchored only through nodes it shares with a higher-labelled block is counted as floating and deflated.
    Deflation only changes the preconditioner: the solution and the iteration count must not get worse. Measured at
    tol 1e-10 on this setup: 18 vs 19 iterations, solutions within 6e-11 (both stall near 7e-10, conditioning)."""
    from scipy.ndimage import label

    from toporlogy.solvers import gmg_kernels as gk
    shape = (16, 8, 8)
    E = np.full(shape, 1e-6)
    E[2:6, 2:6, 2:6] = 1.0                        # block A (label 1)
    E[6:10, 6:8, 2:6] = 1.0                       # block B (label 2), shares only an edge with A
    fixed = np.zeros((17, 9, 9, 3), bool)
    fixed[6, 6, 2:7] = True                       # pinned on the shared edge only
    fixed[16] = True                              # a soft far face removes the rotation about that edge
    lab, n = label(E >= 1e-3)
    node_lab = gk.node_max(torch.as_tensor(lab.reshape(-1), dtype=torch.float64), shape).long().numpy()
    assert n == 2 and set(node_lab[6, 6, 2:7].tolist()) == {2}          # A is not seen as anchored
    F = np.zeros((17, 9, 9, 3))
    F[2, 2:7, 2:7, 1] = 1.0 / 25
    F[10, 6:9, 2:7, 2] = -1.0 / 15
    out = {}
    for tau in (None, 1e-3):
        s = GMGSolver(shape, KE, 1.0, device="cpu", deflate_tau=tau, tol=1e-8, maxit=1000)
        out[tau] = (s.solve(E, F, fixed), s.last)
    assert out[None][1]["converged"] and out[1e-3][1]["converged"]
    assert out[1e-3][1]["iters"] <= out[None][1]["iters"]
    assert np.linalg.norm(out[1e-3][0] - out[None][0]) <= 1e-7 * np.linalg.norm(out[None][0])
