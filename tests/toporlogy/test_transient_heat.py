"""Transient heat conduction (docs/olcumler/gecici_isi_onkayit.md, TH1-TH6, pre-registered before the code)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.heat import assemble_conduction
from toporlogy.fem.transient_heat import TransientHeat

L = 1.0


def _strip(n: int) -> tuple[Domain, np.ndarray, np.ndarray]:
    d = Domain((n, 1), h=L / n, thickness=0.1)
    X = d.node_coords()
    ends = np.flatnonzero((X[:, 0] == 0) | (np.isclose(X[:, 0], L)))
    return d, X, ends


def _final(d, X, ends, theta, dt, t_end):
    m = TransientHeat(d, ends, 0.0, theta)
    T0 = torch.as_tensor(np.sin(np.pi * X[:, 0] / L))
    one = torch.ones(d.n_active, dtype=torch.float64)
    return m.run(one, one, T0, dt, int(round(t_end / dt)))[-1].numpy()


@pytest.mark.parametrize(("theta", "band"), [(1.0, (0.9, 1.1)), (0.5, (1.9, 2.1))])
def test_th1_time_order(theta: float, band: tuple[float, float]) -> None:
    d, X, ends = _strip(40)
    t_end = 0.1
    dts = [1 / 10, 1 / 20, 1 / 40, 1 / 80]
    dts = [t_end * f for f in dts]
    ref = _final(d, X, ends, theta, dts[-1] / 64, t_end)
    err = [np.abs(_final(d, X, ends, theta, dt, t_end) - ref).max() for dt in dts]
    orders = [float(np.log2(err[i] / err[i + 1])) for i in range(len(err) - 1)]
    print(f"\n  TH1 theta {theta}: errors {err}, orders {orders}")
    assert all(band[0] <= o <= band[1] for o in orders)


def test_th2_long_time_limit_is_the_steady_state() -> None:
    d, X, ends = _strip(16)
    k = 0.5 + np.random.default_rng(0).random(d.n_active)
    q = np.zeros(d.nn)
    q[np.flatnonzero(np.isclose(X[:, 0], 0.5))] = 1.0
    m = TransientHeat(d, ends, 0.0, theta=1.0)
    T = m.run(torch.as_tensor(k), torch.ones(d.n_active, dtype=torch.float64), torch.zeros(d.nn, dtype=torch.float64),
              dt=5.0, n_steps=60, q=torch.as_tensor(q))[-1].numpy()
    K = assemble_conduction(d, k).tocsr()
    free = np.setdiff1d(np.arange(d.nn), ends)
    Ts = np.zeros(d.nn)
    import scipy.sparse.linalg as spla
    Ts[free] = spla.spsolve(K[free][:, free].tocsc(), q[free])
    assert np.abs(T - Ts).max() <= 1e-10 * np.abs(Ts).max()


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_th3_insulated_body_conserves_heat(theta: float) -> None:
    d = Domain((6, 4), h=0.3)
    rng = np.random.default_rng(1)
    k = torch.as_tensor(0.5 + rng.random(d.n_active))
    c = torch.as_tensor(0.5 + rng.random(d.n_active))
    m = TransientHeat(d, theta=theta)
    T = m.run(k, c, torch.as_tensor(rng.random(d.nn)), dt=0.05, n_steps=40)
    H = m.total_heat(c, T).numpy()
    assert np.abs(H - H[0]).max() <= 1e-12 * abs(H[0])


def test_th4_design_gradient_through_time() -> None:
    d, X, ends = _strip(12)
    m = TransientHeat(d, ends[:1], 0.0, theta=0.5)
    probe = int(np.flatnonzero(np.isclose(X[:, 0], L) & (X[:, 1] == 0))[0])
    q = torch.zeros(d.nn, dtype=torch.float64)
    q[probe] = 1.0
    x0 = torch.as_tensor(0.3 + 0.6 * np.random.default_rng(2).random(d.n_active))

    def J(x: torch.Tensor) -> torch.Tensor:
        T = m.run(0.1 + x ** 3, 0.5 + x, torch.zeros(d.nn, dtype=torch.float64), dt=0.02, n_steps=25, q=q)
        return T[:, probe].sum()
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(J(x), x)
    v = torch.as_tensor(np.random.default_rng(3).normal(size=d.n_active))
    t = 1e-6
    fd = (J(x0 + t * v) - J(x0 - t * v)).item() / (2 * t)
    assert abs(float(g @ v) - fd) <= 1e-6 * abs(fd)


def test_th5_space_time_convergence_against_the_exact_solution() -> None:
    t_end = 0.05
    errs = []
    for n in (8, 16, 32):
        d, X, ends = _strip(n)
        T = _final(d, X, ends, 0.5, t_end / (n // 2), t_end)
        exact = np.exp(-np.pi ** 2 * t_end / L ** 2) * np.sin(np.pi * X[:, 0] / L)
        errs.append(np.abs(T - exact).max())
    orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    print(f"\n  TH5 errors {errs}, orders {orders}")
    assert all(1.8 <= o <= 2.2 for o in orders)


def test_th6_refusals() -> None:
    d = Domain((3, 2))
    with pytest.raises(ValueError, match="theta"):
        TransientHeat(d, theta=1.5)
    m = TransientHeat(d)
    one = torch.ones(d.n_active, dtype=torch.float64)
    with pytest.raises(ValueError, match="dt"):
        m.run(one, one, torch.zeros(d.nn, dtype=torch.float64), dt=0.0, n_steps=1)
    with pytest.raises(ValueError, match="T0"):
        m.run(one, one, torch.zeros(d.nn - 1, dtype=torch.float64), dt=0.1, n_steps=1)
