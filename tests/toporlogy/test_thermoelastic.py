"""One-way thermoelasticity (docs/olcumler/termoelastisite_onkayit.md, TE1-TE9, pre-registered before the code)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.core.material import Orthotropic, plane_stress, rotation_about_z
from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.thermoelastic import ThermoElastic

NU, ALPHA, DT = 0.3, 1.2e-3, 40.0


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _determinate(d: Domain) -> np.ndarray:
    """Statically determinate support: the origin node fixed, the node at (x_max, 0[, 0]) held in y (and z), and in
    3-D one more node held in z to stop the rotation about x."""
    X = d.node_coords()
    o = int(np.flatnonzero(np.all(X == 0, axis=1))[0])
    a = int(np.flatnonzero((X[:, 0] == X[:, 0].max()) & np.all(X[:, 1:] == 0, axis=1))[0])
    fixed = [d.dim * o + c for c in range(d.dim)] + [d.dim * a + c for c in range(1, d.dim)]
    if d.dim == 3:
        b = int(np.flatnonzero((X[:, 1] == X[:, 1].max()) & (X[:, 0] == 0) & (X[:, 2] == 0))[0])
        fixed.append(d.dim * b + 2)
    return np.array(fixed)


@pytest.mark.parametrize("shape", [(5, 3), (3, 2, 2)])
def test_te1_free_expansion(shape: tuple[int, ...]) -> None:
    d = Domain(shape, h=0.5, thickness=0.2)
    m = ThermoElastic(d, _determinate(d), ALPHA, NU)
    E = torch.full((d.n_active,), 3.0, dtype=torch.float64)
    T = torch.full((d.nn,), DT, dtype=torch.float64)
    u, _ = m.displacement(E, T)
    want = ALPHA * DT * d.node_coords()
    assert _rel(u.numpy().reshape(-1, d.dim), want) <= 1e-12
    s = m.element_stress(E.numpy(), u.numpy(), T.numpy())
    assert np.abs(s).max() <= 1e-12 * 3.0 * ALPHA * DT


def test_te2_constrained_strip_plane_stress() -> None:
    d = Domain((8, 2), h=0.5, thickness=0.3)
    X = d.node_coords()
    ends = np.flatnonzero((X[:, 0] == 0) | (X[:, 0] == X[:, 0].max()))
    o = int(np.flatnonzero(np.all(X == 0, axis=1))[0])
    fixed = np.concatenate([2 * ends, [2 * o + 1]])
    m = ThermoElastic(d, fixed, ALPHA, NU)
    E0 = 2.0
    E = torch.full((d.n_active,), E0, dtype=torch.float64)
    T = torch.full((d.nn,), DT, dtype=torch.float64)
    u, F = m.displacement(E, T)
    s = m.element_stress(E.numpy(), u.numpy(), T.numpy())
    assert _rel(s[:, 0], np.full(len(s), -E0 * ALPHA * DT)) <= 1e-12
    assert np.abs(s[:, 1:]).max() <= 1e-12 * E0 * ALPHA * DT
    K = m.mech.matrix(E.numpy())
    R = K @ u.numpy() - F.numpy()                                   # reactions at the fixed DOFs
    left = 2 * np.flatnonzero(X[:, 0] == 0)
    H = X[:, 1].max()
    assert _rel(R[left].sum(), E0 * ALPHA * DT * H * d.thickness) <= 1e-12


def test_te3_fully_constrained_plane_strain() -> None:
    d = Domain((4, 3))
    X = d.node_coords()
    edge = np.flatnonzero((X[:, 0] == 0) | (X[:, 0] == X[:, 0].max()) | (X[:, 1] == 0) | (X[:, 1] == X[:, 1].max()))
    fixed = np.concatenate([2 * edge, 2 * edge + 1])
    m = ThermoElastic(d, fixed, ALPHA, NU, plane="strain")
    E = torch.ones(d.n_active, dtype=torch.float64)
    T = torch.full((d.nn,), DT, dtype=torch.float64)
    u, _ = m.displacement(E, T)
    s = m.element_stress(E.numpy(), u.numpy(), T.numpy())
    want = -ALPHA * DT / (1 - 2 * NU)
    assert _rel(s[:, :2], np.full((len(s), 2), want)) <= 1e-12
    assert np.abs(s[:, 2]).max() <= 1e-12 * abs(want)


@pytest.mark.parametrize("dim", [2, 3])
def test_te4_rotated_orthotropic_expansion(dim: int) -> None:
    d = Domain((4, 3) if dim == 2 else (3, 2, 2))
    R = rotation_about_z(35.0)
    A = R @ np.diag([2e-3, 5e-4, 9e-4]) @ R.T
    mat = Orthotropic(10, 2, 2, 1.0, 1.0, 0.8, 0.25, 0.25, 0.3)
    C3 = mat.stiffness(R)
    D = plane_stress(C3) if dim == 2 else C3
    m = ThermoElastic(d, _determinate(d), A, D=D)
    E = torch.ones(d.n_active, dtype=torch.float64)
    T = torch.full((d.nn,), DT, dtype=torch.float64)
    u, _ = m.displacement(E, T)
    u = u.numpy().reshape(-1, dim)
    X = d.node_coords()
    eps = (A * DT)[:dim, :dim]
    # free expansion is determined up to a rigid rotation: compare the symmetric gradient by least squares
    G, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], u, rcond=None)
    grad = G[:dim].T
    assert _rel(0.5 * (grad + grad.T), eps) <= 1e-12
    assert np.abs(u - (np.c_[X, np.ones(len(X))] @ G)).max() <= 1e-12 * np.abs(u).max()


def test_te5_uniform_load_equals_boundary_traction() -> None:
    d = Domain((5, 4), h=0.4, thickness=0.7)
    m = ThermoElastic(d, [], ALPHA, NU)
    F = m.thermal_load(torch.ones(d.n_active, dtype=torch.float64),
                       torch.full((d.nn,), DT, dtype=torch.float64)).numpy().reshape(-1, 2)
    X = d.node_coords()
    sig = D_plane_stress(NU) @ np.array([ALPHA, ALPHA, 0.0]) * DT                 # β ΔT (xx, yy, xy)
    S = np.array([[sig[0], sig[2]], [sig[2], sig[1]]])
    want = np.zeros_like(F)
    L, H, h, t = X[:, 0].max(), X[:, 1].max(), d.h, d.thickness
    for normal, on in (((-1, 0), X[:, 0] == 0), ((1, 0), X[:, 0] == L), ((0, -1), X[:, 1] == 0),
                       ((0, 1), X[:, 1] == H)):
        trac = S @ np.array(normal, float)
        along = 1 if normal[0] != 0 else 0
        nodes = np.flatnonzero(on)
        coord = X[nodes, along]
        w = np.where((coord == 0) | (coord == coord.max()), 0.5, 1.0) * h * t      # consistent: trapezoid
        want[nodes] += w[:, None] * trac[None, :]
    assert _rel(F, want) <= 1e-12


def test_te6_two_layer_conduction() -> None:
    d = Domain((8, 2))
    X = d.node_coords()
    left, right = np.flatnonzero(X[:, 0] == 0), np.flatnonzero(X[:, 0] == X[:, 0].max())
    m = ThermoElastic(d, [], ALPHA, NU, heat_fixed=np.r_[left, right],
                      heat_values=np.r_[np.zeros(len(left)), np.full(len(right), 10.0)])
    cx = d.element_centers()[:, 0]
    k = np.where(cx < 3.0, 1.0, 4.0)
    T = m.temperature(torch.as_tensor(k)).numpy()
    q = 10.0 / (3.0 / 1.0 + 5.0 / 4.0)                               # series: flux per unit area
    want = np.where(X[:, 0] <= 3.0, q * X[:, 0] / 1.0, q * 3.0 + q * (X[:, 0] - 3.0) / 4.0)
    assert _rel(T, want) <= 1e-12


def _coupled(cut: str | None = None) -> tuple:
    d = Domain((6, 4))
    X = d.node_coords()
    left = np.flatnonzero(X[:, 0] == 0)
    fixed = np.r_[2 * left, 2 * left + 1]
    hot = np.flatnonzero(X[:, 0] == X[:, 0].max())
    m = ThermoElastic(d, fixed, ALPHA, NU, heat_fixed=np.r_[left, hot],
                      heat_values=np.r_[np.zeros(len(left)), np.full(len(hot), DT)])
    Fm = np.zeros(d.ndof)
    Fm[2 * int(np.flatnonzero((X[:, 0] == X[:, 0].max()) & (X[:, 1] == 0))[0]) + 1] = -1e-3

    def J(x: torch.Tensor) -> torch.Tensor:
        E = 1e-3 + x ** 3
        kk = 0.1 + x ** 2
        T = m.temperature(kk)
        if cut == "T":
            T = T.detach()
        F_th = m.thermal_load(E.detach() if cut == "E" else E, T)
        u = m.mech.solve(E, F_th + torch.as_tensor(Fm))
        return (F_th + torch.as_tensor(Fm)) @ u
    x0 = 0.3 + 0.6 * torch.rand(d.n_active, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    return J, x0


def _directional(J, x0) -> tuple[float, float]:
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(J(x), x)
    v = torch.randn(x0.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    t = 1e-6
    fd = (J(x0 + t * v) - J(x0 - t * v)).item() / (2 * t)
    return float(g @ v), fd


def test_te7_design_gradient_through_load_and_temperature() -> None:
    J, x0 = _coupled()
    ad, fd = _directional(J, x0)
    print(f"\n  TE7: {ad:.10e} vs {fd:.10e}")
    assert abs(ad - fd) <= 1e-6 * abs(fd)


@pytest.mark.parametrize("cut", ["E", "T"])
def test_te8_negative_control_cut_chain(cut: str) -> None:
    J, x0 = _coupled()
    Jc, _ = _coupled(cut)
    x = x0.clone().requires_grad_(True)
    (gc,) = torch.autograd.grad(Jc(x), x)
    v = torch.randn(x0.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    t = 1e-6
    fd = (J(x0 + t * v) - J(x0 - t * v)).item() / (2 * t)
    print(f"\n  TE8 {cut}: {float(gc @ v):.6e} vs {fd:.6e}")
    assert abs(float(gc @ v) - fd) > 1e-2 * abs(fd)


def test_te9_refusals() -> None:
    d = Domain((3, 3))
    with pytest.raises(ValueError, match="Q4 and H8"):
        ThermoElastic(d, [], ALPHA, NU, element="Q4IM")
    m = ThermoElastic(d, [], ALPHA, NU)
    with pytest.raises(ValueError, match="entries"):
        m.thermal_load(torch.ones(d.n_active, dtype=torch.float64), torch.ones(d.nn - 1, dtype=torch.float64))
