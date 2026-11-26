"""Mindlin-Reissner plate element (docs/olcumler/plaka_onkayit.md, PL1-PL7)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.plate import MindlinPlate, hard_simply_supported, navier_deflection, plate_element_matrix

NU = 0.3


def test_pl1_rigid_modes() -> None:
    h, t = 0.7, 0.05
    K = plate_element_matrix(h, t, NU)
    c = np.array([[0, 0], [h, 0], [h, h], [0, h]], float)
    modes = []
    for w, bx, by in [(np.ones(4), 0 * c[:, 0], 0 * c[:, 0]), (c[:, 0], np.ones(4), 0 * c[:, 0]),
                      (c[:, 1], 0 * c[:, 0], np.ones(4))]:
        r = np.zeros(12)
        r[0::3], r[1::3], r[2::3] = w, bx, by
        modes.append(r)
    for r in modes:
        assert np.abs(K @ r).max() <= 1e-12 * np.abs(K).max()


def test_pl2_constant_curvature_patch() -> None:
    h, t = 0.5, 0.1
    K = plate_element_matrix(h, t, NU)
    c = np.array([[0, 0], [h, 0], [h, h], [0, h]], float) + np.array([1.3, -0.4])
    u = np.zeros(12)
    u[0::3], u[1::3] = c[:, 0] ** 2 / 2, c[:, 0]
    D11 = t ** 3 / (12 * (1 - NU ** 2))
    energy = 0.5 * u @ K @ u
    assert abs(energy - D11 * h * h / 2) <= 1e-12 * D11 * h * h / 2


def _centre(n: int, ratio: float, shear: str = "reduced") -> tuple[float, float]:
    a = 1.0
    t = a / ratio
    d = Domain((n, n), h=a / n)
    p = MindlinPlate(d, t, NU, hard_simply_supported(d), shear=shear)
    u = p.solve(torch.ones(d.n_active), p.pressure_load(1.0)).numpy()
    X = d.node_coords()
    c = int(np.argmin(np.abs(X[:, 0] - a / 2) + np.abs(X[:, 1] - a / 2)))
    return float(u[3 * c]), t


@pytest.mark.parametrize("ratio", [10.0, 100.0])
def test_pl3_navier_convergence(ratio) -> None:
    ref = navier_deflection(1.0, 1.0, 1.0 / ratio, 1.0, NU, 1.0, 0.5, 0.5)
    err = {n: abs(_centre(n, ratio)[0] - ref) / ref for n in (8, 16, 32)}
    order = np.log2(err[16] / err[32])
    print(f"PL3 a/t={ratio:g}: errors {err}, order 16->32 {order:.2f}, ref {ref:.6e}")
    assert err[32] <= 0.01
    assert order >= 1.5


def test_pl4_full_shear_integration_locks() -> None:
    ref = navier_deflection(1.0, 1.0, 1e-3, 1.0, NU, 1.0, 0.5, 0.5)
    w, _ = _centre(16, 1000.0, shear="full")
    print(f"PL4: full / Navier = {w / ref:.4f}")
    assert w <= 0.5 * ref


def test_pl5_thin_limit_kirchhoff() -> None:
    ref = navier_deflection(1.0, 1.0, 1e-3, 1.0, NU, 1.0, 0.5, 0.5, kirchhoff=True)
    w, _ = _centre(32, 1000.0)
    print(f"PL5: error {abs(w - ref) / ref:.3e}")
    assert abs(w - ref) <= 0.02 * ref


def test_pl6_design_gradient() -> None:
    d = Domain((10, 10), h=0.1)
    p = MindlinPlate(d, 0.02, NU, hard_simply_supported(d))
    F = p.pressure_load(1.0)
    x = 0.2 + 0.7 * np.random.default_rng(3).random(d.n_active)
    v = np.random.default_rng(4).normal(size=d.n_active)

    def comp(xx: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(F) @ p.solve(1e-6 + xx ** 3, F)
    xt = torch.tensor(x, requires_grad=True)
    (g,) = torch.autograd.grad(comp(xt), xt)
    s = 1e-6
    fd = (float(comp(torch.as_tensor(x + s * v))) - float(comp(torch.as_tensor(x - s * v)))) / (2 * s)
    print(f"PL6: rel {abs(float(g @ torch.as_tensor(v)) - fd) / abs(fd):.2e}")
    assert abs(float(g @ torch.as_tensor(v)) - fd) <= 1e-6 * abs(fd)


def test_pl7_refusals() -> None:
    with pytest.raises(ValueError, match="2-D"):
        MindlinPlate(Domain((3, 3, 3)), 0.1)
    with pytest.raises(ValueError, match="thickness"):
        MindlinPlate(Domain((3, 3)), 0.0)
    with pytest.raises(ValueError, match="Poisson"):
        MindlinPlate(Domain((3, 3)), 0.1, nu=0.5)


def test_navier_series_truncation_is_small() -> None:
    """Record helper (not pre-registered as a threshold): change of the series from 399 to 799 terms."""
    for ratio in (10.0, 100.0, 1000.0):
        a = navier_deflection(1.0, 1.0, 1 / ratio, 1.0, NU, 1.0, 0.5, 0.5, terms=399)
        b = navier_deflection(1.0, 1.0, 1 / ratio, 1.0, NU, 1.0, 0.5, 0.5, terms=799)
        print(f"truncation a/t={ratio:g}: {abs(a - b) / abs(b):.2e}")
        assert abs(a - b) <= 1e-3 * abs(b)
