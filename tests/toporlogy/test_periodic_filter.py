"""Periodic density filter (docs/olcumler/periyodik_filtre_onkayit.md, PF1-PF4; PF5 is the baseline run)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import ndimage

from toporlogy.core.domain import Domain
from toporlogy.design.chain import DensityFilter

CASES = [((9, 7), 2.5), ((6, 5, 4), 1.8)]


def _reference(x: np.ndarray, rmin: float) -> np.ndarray:
    r = math.ceil(rmin) - 1
    ax = np.arange(-r, r + 1)
    g = np.meshgrid(*([ax] * x.ndim), indexing="ij")
    w = np.maximum(0.0, rmin - np.sqrt(sum(a.astype(float) ** 2 for a in g)))
    return ndimage.convolve(x, w / w.sum(), mode="wrap")


@pytest.mark.parametrize("shape,rmin", CASES)
@pytest.mark.parametrize("backend", ["sparse", "conv"])
def test_pf1_pf2_matches_wrap_convolution_and_is_shift_equivariant(shape, rmin, backend) -> None:
    f = DensityFilter(Domain(shape), rmin, backend=backend, periodic=True)
    x = np.random.default_rng(3).random(shape)
    y = f(torch.as_tensor(x)).numpy()
    assert np.max(np.abs(y - _reference(x, rmin))) <= 1e-14                                    # PF1
    s = tuple(range(1, len(shape) + 1))
    ys = f(torch.as_tensor(np.roll(x, s, axis=tuple(range(len(shape)))))).numpy()
    assert np.max(np.abs(ys - np.roll(y, s, axis=tuple(range(len(shape)))))) <= 1e-14          # PF2


@pytest.mark.parametrize("shape,rmin", CASES)
@pytest.mark.parametrize("backend", ["sparse", "conv"])
def test_pf3_adjoint(shape, rmin, backend) -> None:
    f = DensityFilter(Domain(shape), rmin, backend=backend, periodic=True)
    rng = np.random.default_rng(4)
    x = torch.tensor(rng.random(shape), dtype=torch.float64, requires_grad=True)
    y = torch.as_tensor(rng.normal(size=shape))
    fx = f(x)
    (g,) = torch.autograd.grad(fx, x, y)
    lhs, rhs = float((fx * y).sum()), float((x * g).sum())
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_pf4_periodic_needs_the_full_box() -> None:
    void = np.zeros((8, 6), bool)
    void[2:4, 2:4] = True
    with pytest.raises(ValueError, match="periodic"):
        DensityFilter(Domain((8, 6), void=void), 1.5, periodic=True)
    frac = np.ones((8, 6))
    frac[0, 0] = 0.0
    with pytest.raises(ValueError, match="periodic"):
        DensityFilter(Domain((8, 6), frac=frac), 1.5, periodic=True)
