"""Helmholtz filter (design/pde_filter.py): the stated properties, the measured kernel moments, gradients."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import DensityFilter, Domain, gradcheck
from toporlogy.design.pde_filter import HelmholtzFilter, helmholtz_matrix


def _part() -> Domain:
    shape = (12, 9)
    mask = np.ones(shape, bool)
    mask[7:, 5:] = False
    mask[2:4, 2:5] = False
    return Domain(shape, frac=mask.astype(float))


def test_matrix_is_symmetric_m_matrix_with_unit_column_sums() -> None:
    d = _part()
    A, el = helmholtz_matrix(d.active, 1.3)
    assert (A != A.T).nnz == 0 and len(el) == d.n_active
    assert np.allclose(np.asarray(A.sum(axis=0)).ravel(), 1.0)
    off = A - __import__("scipy.sparse").sparse.diags(A.diagonal())
    assert off.max() <= 0 and A.diagonal().min() >= 1.0


@pytest.mark.parametrize("dom", [_part(), Domain((6, 5, 4))], ids=["2d-part", "3d-box"])
def test_volume_preserving_bounded_and_zero_outside(dom: Domain) -> None:
    f = HelmholtzFilter(dom, 3.0)
    rng = np.random.default_rng(0)
    x = rng.random(dom.shape) * dom.active
    y = f(torch.as_tensor(x)).numpy()
    assert abs(y.sum() - x.sum()) < 1e-11 * x.sum()
    assert y.min() >= -1e-14 and y.max() <= 1 + 1e-14
    assert np.all(y[~dom.active] == 0)
    one = f(torch.as_tensor(dom.active.astype(float))).numpy()
    assert np.allclose(one[dom.active], 1.0, atol=1e-12)          # constants are reproduced, also at walls and holes


@pytest.mark.parametrize("dim,shape,cone_factor", [(2, (81, 81), 3 / 20), (3, (29, 29, 29), 2 / 15)])
def test_kernel_second_moment(dim: int, shape: tuple, cone_factor: float) -> None:
    """Variance per axis: PDE 2 r² = rmin²/6 (exact for the discrete operator too, away from walls); cone filter
    cone_factor · rmin² in the continuum (discrete kernel within 20 % at rmin = 4 to 6)."""
    d = Domain(shape)
    rmin = 6.0 if dim == 2 else 4.0
    x = np.zeros(shape)
    c = tuple(s // 2 for s in shape)
    x[c] = 1.0
    r2 = sum((np.indices(shape)[k] - c[k]) ** 2 for k in range(dim))
    hp = HelmholtzFilter(d, rmin).apply_numpy(x).reshape(shape)
    hc = DensityFilter(d, rmin)(torch.as_tensor(x)).numpy()
    assert (hp * r2).sum() / dim == pytest.approx(rmin ** 2 / 6, rel=1e-3)
    assert (hc * r2).sum() / dim == pytest.approx(cone_factor * rmin ** 2, rel=0.2)
    assert hp.max() > 2 * hc.max()                                 # peaked kernel


def test_gradient_and_errors() -> None:
    d = _part()
    f = HelmholtzFilter(d, 2.5)
    w = torch.as_tensor(np.random.default_rng(1).random(d.shape))
    x0 = 0.2 + 0.6 * np.random.default_rng(2).random(d.shape)
    r = gradcheck(lambda x: (w * f(x) ** 3).sum(), x0, n_dirs=5)
    assert r["max_rel_error"] <= 1e-6, r
    with pytest.raises(ValueError):
        HelmholtzFilter(d, 0.0)
