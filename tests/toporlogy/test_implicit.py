"""General implicit differentiation (docs/olcumler/ortuk_turev_onkayit.md, I1-I6, pre-registered before the code)."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from toporlogy.optimization.implicit import ImplicitState, NonFiniteResidual, NotConverged

N = 6
RNG = np.random.default_rng(0)
A0 = RNG.normal(size=(N, N)) + 4 * np.eye(N)              # not symmetric
A1 = RNG.normal(size=(N, N))
B0, B1, C = RNG.normal(size=N), RNG.normal(size=N), RNG.normal(size=N)


def _A(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(A0) + x[0] * torch.as_tensor(A1)


def _b(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(B0) + x[1] * torch.as_tensor(B1)


def _linear(transpose: bool = True) -> ImplicitState:
    return ImplicitState(lambda y, x: _A(x) @ y - _b(x), transpose=transpose)


def _analytic(x: np.ndarray) -> np.ndarray:
    A, b = A0 + x[0] * A1, B0 + x[1] * B1
    y = np.linalg.solve(A, b)
    lam = np.linalg.solve(A.T, C)                          # J = c . y
    return np.array([-lam @ (A1 @ y), lam @ B1])


def _grad(state: ImplicitState, x0: np.ndarray) -> np.ndarray:
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    y = state.solve(x, torch.zeros(N, dtype=torch.float64))
    (g,) = torch.autograd.grad(torch.as_tensor(C) @ y, x)
    return g.numpy()


def test_i1_linear_non_symmetric_against_the_analytic_gradient() -> None:
    x0 = np.array([0.3, -0.7])
    g, ref = _grad(_linear(), x0), _analytic(x0)
    assert np.abs(g - ref).max() <= 1e-12 * np.abs(ref).max()


def test_i2_negative_control_without_the_transpose() -> None:
    x0 = np.array([0.3, -0.7])
    g, ref = _grad(_linear(transpose=False), x0), _analytic(x0)
    assert np.abs(g - ref).max() > 1e-3 * np.abs(ref).max()


Bn = RNG.normal(size=(N, N)) * 0.3


def _nonlinear(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return y + 0.5 * y ** 3 + torch.as_tensor(Bn) @ y - x


def _orders(J, x0, v, gv, steps) -> list[float]:
    j0 = J(x0).item()
    rem = [abs(J(x0 + e * v).item() - j0 - e * gv) for e in steps]
    return [float(np.log2(rem[i] / rem[i + 1])) for i in range(len(rem) - 1)]


def test_i3_nonlinear_residual_central_difference() -> None:
    J, x0, v, gv = _i3_first_run()
    t = 1e-6
    fd = (J(x0 + t * v) - J(x0 - t * v)).item() / (2 * t)
    assert abs(gv - fd) <= 1e-7 * abs(fd)


@pytest.mark.xfail(strict=True, reason="pre-registered I3 Taylor order in [1.8, 2.2] FAILED on the first run: orders "
                   "2.91, 3.02, 3.14 for steps 0.04 ... 0.005. Diagnosis (not a criterion): along that direction "
                   "v'Hv = 0.006 while the third directional derivative is -16.4, so the cubic term dominates the "
                   "remainder at these steps; the gradient itself passes the central-difference check")
def test_i3_taylor_remainder_order() -> None:
    J, x0, v, gv = _i3_first_run()
    orders = _orders(J, x0, v, gv, (4e-2, 2e-2, 1e-2, 5e-3))
    print(f"\n  I3 orders {orders}")
    assert all(1.8 <= o <= 2.2 for o in orders)


def _i3_first_run():
    """The direction of the first run (the module RNG after the matrices), kept for the xfail above."""
    s = ImplicitState(_nonlinear)
    rng = np.random.default_rng(0)
    for shape in ((N, N), (N, N), (N,), (N,), (N,), (N, N)):        # A0, A1, B0, B1, C, Bn
        rng.normal(size=shape)
    x0, v = torch.as_tensor(rng.normal(size=N)), torch.as_tensor(rng.normal(size=N))
    y0 = torch.zeros(N, dtype=torch.float64)

    def J(x: torch.Tensor) -> torch.Tensor:
        y = s.solve(x, y0)
        return (y ** 2).sum() + torch.as_tensor(C) @ y
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(J(x), x)
    return J, x0, v, float(g @ v)


def test_i3_posthoc_taylor_order_where_the_quadratic_term_dominates() -> None:
    """Post hoc (after I3 failed): the same direction with steps 1e-4 ... 1.25e-5, where the quadratic term
    (v'Hv / 2 e^2) is larger than the cubic one."""
    J, x0, v, gv = _i3_first_run()
    orders = _orders(J, x0, v, gv, (1e-4, 5e-5, 2.5e-5, 1.25e-5))
    print(f"\n  I3 post hoc orders {orders}")
    assert all(1.8 <= o <= 2.2 for o in orders)


def test_i4_two_responses_share_one_adjoint_factorisation() -> None:
    s = _linear()
    x = torch.tensor([0.3, -0.7], dtype=torch.float64, requires_grad=True)
    y = s.solve(x, torch.zeros(N, dtype=torch.float64))
    g1 = torch.autograd.grad(torch.as_tensor(C) @ y, x, retain_graph=True)[0]
    g2 = torch.autograd.grad((y ** 2).sum(), x)[0]
    assert s.n_adjoint_factorizations == 1 and s.n_adjoint_solves == 2
    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()


def test_i5_fails_closed() -> None:
    no_root = ImplicitState(lambda y, x: y ** 2 + 1.0 + 0 * x, max_iter=20)
    with pytest.raises(NotConverged, match="history"):
        no_root.solve(torch.zeros(1, dtype=torch.float64), torch.full((1,), 0.5, dtype=torch.float64))
    bad = ImplicitState(lambda y, x: y - x * torch.inf)
    with pytest.raises(NonFiniteResidual):
        bad.solve(torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))


def test_i6_user_sparse_jacobian_gives_the_same_gradient() -> None:
    def jac(y: np.ndarray, x: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix(A0 + x[0] * A1)
    x0 = np.array([0.3, -0.7])
    g_ad = _grad(_linear(), x0)
    g_sp = _grad(ImplicitState(lambda y, x: _A(x) @ y - _b(x), jacobian=jac), x0)
    assert np.abs(g_sp - g_ad).max() <= 1e-12 * np.abs(g_ad).max()
