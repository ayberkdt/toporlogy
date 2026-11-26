"""Newmark structural dynamics (docs/olcumler/newmark_onkayit.md, ND1-ND6, pre-registered before the code)."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.dynamics_newmark import Newmark


def _cantilever(n: int = 12, rayleigh: tuple[float, float] = (0.0, 0.0)) -> tuple[Domain, Newmark, np.ndarray]:
    d = Domain((n, 2), h=1.0 / 2)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    return d, Newmark(d, fixed, 0.3, rayleigh=rayleigh), X


def _first_mode(d: Domain, m: Newmark) -> tuple[float, np.ndarray]:
    one = np.ones(d.n_active)
    K = m.effective.matrix(np.stack([one, np.zeros_like(one)])).toarray()
    M = m.mass.matrix(one).toarray()
    free = m.effective.free
    w2, P = sla.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], subset_by_index=[0, 0])
    phi = np.zeros(d.ndof)
    phi[free] = P[:, 0]
    return float(np.sqrt(w2[0])), phi


def _ones(d: Domain) -> torch.Tensor:
    return torch.ones(d.n_active, dtype=torch.float64)


def _modal_orders(periods: float) -> tuple[list[float], list[float]]:
    d, m, _ = _cantilever()
    w, phi = _first_mode(d, m)
    T = 2 * np.pi / w
    errs = []
    for k in (20, 40, 80, 160):
        dt = T / k
        r = m.run(_ones(d), _ones(d), torch.as_tensor(phi), torch.zeros(d.ndof, dtype=torch.float64), dt,
                  int(round(periods * k)))
        exact = phi * np.cos(w * periods * T)
        errs.append(np.abs(r.u[-1].numpy() - exact).max() / np.abs(phi).max())
    return errs, [float(np.log2(errs[i] / errs[i + 1])) for i in range(3)]


@pytest.mark.xfail(strict=True, reason="pre-registered ND1 order band [1.9, 2.1] FAILED on the first run: orders 3.97, "
                   "3.99, 4.00. The check measured at t = 2 T, where sin(w t) = 0, so the O(dt^2) phase error delta "
                   "enters the displacement only as delta^2 / 2 (order 4); the method's second order shows at "
                   "t = 2.25 T (post hoc test below)")
def test_nd1_second_order_against_the_modal_solution() -> None:
    errs, orders = _modal_orders(2.0)
    print(f"\n  ND1 errors {errs}, orders {orders}")
    assert all(1.9 <= o <= 2.1 for o in orders)


def test_nd1_posthoc_second_order_where_the_phase_error_shows() -> None:
    """Post hoc (after ND1 failed): the same comparison at t = 2.25 T, where sin(w t) = 1."""
    errs, orders = _modal_orders(2.25)
    print(f"\n  ND1 post hoc errors {errs}, orders {orders}")
    assert all(1.9 <= o <= 2.1 for o in orders)


def test_nd2_energy_is_conserved_without_damping_and_load() -> None:
    d, m, _ = _cantilever()
    rng = np.random.default_rng(0)
    free = m.effective.free
    u0, v0 = np.zeros(d.ndof), np.zeros(d.ndof)
    u0[free], v0[free] = 1e-3 * rng.normal(size=len(free)), 1e-3 * rng.normal(size=len(free))
    E, rho = _ones(d), _ones(d)
    r = m.run(E, rho, torch.as_tensor(u0), torch.as_tensor(v0), dt=0.01, n_steps=200)
    H = np.array([m.energy(E, rho, r.u[i], r.v[i]).item() for i in range(201)])
    assert np.abs(H - H[0]).max() <= 1e-12 * H[0]


def test_nd3_rayleigh_damping_decay() -> None:
    d, m0, _ = _cantilever()
    w, phi = _first_mode(d, m0)
    zeta = 0.02
    a, b = zeta * w, zeta / w                          # each part gives zeta / 2 at w
    _, m, _ = _cantilever(rayleigh=(a, b))
    T = 2 * np.pi / w
    k = 200
    r = m.run(_ones(d), _ones(d), torch.as_tensor(phi), torch.zeros(d.ndof, dtype=torch.float64), T / k, 10 * k)
    tip = int(np.argmax(np.abs(phi)))
    amp = r.u[-1, tip].item() / phi[tip]
    wd = w * np.sqrt(1 - zeta ** 2)
    exact = np.exp(-zeta * w * 10 * T) * (np.cos(wd * 10 * T) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * 10 * T))
    print(f"\n  ND3 amplitude {amp:.6f} vs {exact:.6f}")
    assert abs(amp - exact) <= 1e-3 * abs(exact)


def test_nd4_design_gradient_through_time() -> None:
    d, m, X = _cantilever(8)
    tip = 2 * int(np.flatnonzero((X[:, 0] == X[:, 0].max()) & (X[:, 1] == 0))[0]) + 1
    F = torch.zeros(d.ndof, dtype=torch.float64)
    F[tip] = -1.0
    x0 = torch.as_tensor(0.4 + 0.5 * np.random.default_rng(1).random(d.n_active))

    def J(x: torch.Tensor) -> torch.Tensor:
        r = m.run(0.1 + x ** 3, 0.2 + x, torch.zeros(d.ndof, dtype=torch.float64),
                  torch.zeros(d.ndof, dtype=torch.float64), 0.05, 30, force=lambda t: F if t <= 0.1 else 0 * F)
        return (r.u[:, tip] ** 2).sum()
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(J(x), x)
    v = torch.as_tensor(np.random.default_rng(2).normal(size=d.n_active))
    t = 1e-6
    fd = (J(x0 + t * v) - J(x0 - t * v)).item() / (2 * t)
    assert abs(float(g @ v) - fd) <= 1e-6 * abs(fd)


def test_nd5_one_factorisation_for_constant_coefficients() -> None:
    d, m, _ = _cantilever()
    m.run(_ones(d), _ones(d), torch.zeros(d.ndof, dtype=torch.float64), torch.zeros(d.ndof, dtype=torch.float64),
          0.01, 50)
    assert m.effective.n_factorizations == 1 and m.mass.n_factorizations == 1


def test_nd6_refusals() -> None:
    d, m, _ = _cantilever(4)
    z = torch.zeros(d.ndof, dtype=torch.float64)
    with pytest.raises(ValueError, match="dt"):
        m.run(_ones(d), _ones(d), z, z, 0.0, 1)
    with pytest.raises(ValueError, match="entries"):
        m.run(_ones(d), _ones(d), z[:-1], z, 0.1, 1)
