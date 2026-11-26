"""Mass properties (docs/olcumler/kutle_ozellikleri_onkayit.md, MP1-MP6, pre-registered before the code)."""
from __future__ import annotations

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.core.mass_properties import mass_properties


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _box(m: float, a: float, b: float, c: float) -> np.ndarray:
    return m / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def test_mp1_full_prism_and_plate() -> None:
    d = Domain((5, 3, 4), h=0.5, origin=(1.0, -2.0, 0.5))
    p = mass_properties(d, np.ones(d.shape), rho0=7.0)
    a, b, c = 2.5, 1.5, 2.0
    m = 7.0 * a * b * c
    assert _rel(p.mass, m) <= 1e-12
    assert _rel(p.centre, [1.0 + a / 2, -2.0 + b / 2, 0.5 + c / 2]) <= 1e-12
    assert _rel(p.inertia, _box(m, a, b, c)) <= 1e-12
    d2 = Domain((6, 2), h=0.25, thickness=0.1)
    p2 = mass_properties(d2, np.ones(d2.shape), rho0=3.0)
    m2 = 3.0 * 1.5 * 0.5 * 0.1
    assert _rel(p2.mass, m2) <= 1e-12
    assert _rel(p2.inertia, _box(m2, 1.5, 0.5, 0.1)) <= 1e-12


def test_mp2_parallel_axis_theorem() -> None:
    d = Domain((6, 5, 4), h=0.3)
    x = np.random.default_rng(0).random(d.shape)
    pc = mass_properties(d, x)
    o = np.array([0.7, -1.1, 2.3])
    po = mass_properties(d, x, about=o)
    r = pc.centre - o
    assert _rel(po.inertia, pc.inertia + pc.mass * (r @ r * np.eye(3) - np.outer(r, r))) <= 1e-12


def test_mp3_translation_invariance() -> None:
    x = np.random.default_rng(1).random((5, 4))
    a = mass_properties(Domain((5, 4)), x)
    b = mass_properties(Domain((5, 4), origin=(10.0, -3.0)), x)
    assert _rel(a.mass, b.mass) <= 1e-12
    assert _rel(b.centre - a.centre, [10.0, -3.0, 0.0]) <= 1e-12
    assert _rel(b.inertia, a.inertia) <= 1e-12


def test_mp4_hole_by_superposition() -> None:
    d = Domain((8, 6, 4), h=0.5)
    x = np.ones(d.shape)
    x[2:5, 1:4, 1:3] = 0.0
    p = mass_properties(d, x, about=np.zeros(3))
    full = mass_properties(d, np.ones(d.shape), about=np.zeros(3))
    hole_d = Domain((3, 3, 2), h=0.5, origin=(1.0, 0.5, 0.5))
    hole = mass_properties(hole_d, np.ones(hole_d.shape), about=np.zeros(3))
    assert _rel(p.mass, full.mass - hole.mass) <= 1e-12
    assert _rel(p.inertia, full.inertia - hole.inertia) <= 1e-12


def test_mp5_partial_volume_and_void() -> None:
    frac = np.ones((4, 3))
    frac[0, 0] = 0.25
    void = np.zeros((4, 3), bool)
    void[3, 2] = True
    d = Domain((4, 3), frac=frac, void=void)
    p = mass_properties(d, np.ones(d.shape))
    assert _rel(p.mass, 12 - 0.75 - 1.0) <= 1e-12


def test_mp6_design_derivatives() -> None:
    d = Domain((5, 4, 3), h=0.4)
    x0 = torch.rand(d.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 0.8 + 0.1
    v = torch.randn(d.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    def q(x: torch.Tensor) -> torch.Tensor:
        p = mass_properties(d, x, rho0=2.0, density=lambda t: t ** 2)
        return torch.stack([p.mass, p.centre[0], p.inertia[2, 2]])
    x = x0.clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(q, x).reshape(3, -1)
    ad = J @ v.reshape(-1)
    t = 1e-6
    fd = (q(x0 + t * v) - q(x0 - t * v)) / (2 * t)
    assert float((ad - fd).abs().max() / fd.abs().max()) <= 1e-8
