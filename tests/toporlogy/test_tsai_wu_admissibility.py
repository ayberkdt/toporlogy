"""Tsai-Wu admissibility (docs/olcumler/tsai_wu_kabul_onkayit.md, TW1-TW4, pre-registered before the fix)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.material import OrthotropicStrength
from toporlogy.optimization.strength import make_criterion

ST = dict(Xt=1500.0, Xc=1200.0, Yt=50.0, Yc=200.0, S12=70.0)


def _crit(f12: float):
    return make_criterion("tsai_wu", OrthotropicStrength(**ST, f12=f12))


def _probe(f12: float) -> torch.Tensor:
    a = OrthotropicStrength(**ST, f12=f12).allowables()
    Fd = np.array([1 / (a["Xt"] * a["Xc"]), 1 / (a["Yt"] * a["Yc"]), 1 / (a["Zt"] * a["Zc"])])
    return torch.tensor([list(-100 / np.sqrt(Fd)) + [0.0, 0.0, 0.0]], dtype=torch.float64)


@pytest.mark.parametrize("f12", [-0.5, -0.2, 0.0, 0.5, 0.9])
def test_tw1_index_is_non_negative_where_admissible(f12: float) -> None:
    s = torch.as_tensor(np.random.default_rng(0).normal(scale=300.0, size=(10_000, 6)))
    assert bool((_crit(f12).index(s) >= 0).all())


def test_tw2_inadmissible_3d_stress_raises() -> None:
    with pytest.raises(ValueError, match="f12"):
        _crit(-0.8).index(_probe(-0.8))


def test_tw3_plane_stress_is_admissible_for_any_f12_in_range() -> None:
    s = np.random.default_rng(1).normal(scale=300.0, size=(10_000, 6))
    s[:, [2, 3, 4]] = 0.0
    assert bool((_crit(-0.8).index(torch.as_tensor(s)) >= 0).all())
