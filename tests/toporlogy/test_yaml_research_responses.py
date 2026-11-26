"""YAML responses of the research modules: scenario_compliance (optimization.uq) and compliance_case
(optimization.pareto). Values equal the Python classes; gradients by central differences (≤ 1e-6, the library's
gradient bound of tests/toporlogy/test_gradients.py)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError


def _study(objective, cases: int = 3):
    loads = [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                         "force": [0.2 * (k - 1), -1.0]}]} for k in range(cases)]
    cfg = {"domain": {"type": "box", "shape": [12, 6], "h": 1.0},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": loads, "chain": [{"type": "density_filter", "rmin": 1.5}],
           "objective": objective, "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma"}
    return build_setup(cfg).study


@pytest.mark.parametrize("objective", [{"type": "scenario_compliance", "measure": "mean"},
                                       {"type": "scenario_compliance", "measure": "mean_std", "k": 1.0},
                                       {"type": "scenario_compliance", "measure": "cvar", "alpha": 0.5},
                                       {"type": "compliance_case", "case": 2}])
def test_value_and_gradient(objective: dict) -> None:
    st = _study(objective)
    rng = np.random.default_rng(0)
    x = 0.3 + 0.5 * rng.random(st.n_var)
    v = rng.normal(size=st.n_var)
    ev = st.evaluate(x)
    t = 1e-6
    fd = (st.evaluate(x + t * v, False).f0 - st.evaluate(x - t * v, False).f0) / (2 * t)
    assert abs(ev.df0 @ v - fd) <= 1e-6 * abs(fd)
    if objective["type"] == "compliance_case":
        ref = float((np.asarray(st.F)[:, 2] * ev.U[:, 2]).sum())
        assert abs(ev.f0 - ref) <= 1e-12 * abs(ref)


def test_case_out_of_range_is_refused() -> None:
    with pytest.raises(ConfigError, match="load case"):
        _study({"type": "compliance_case", "case": 3})
