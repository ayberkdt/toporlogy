"""YAML start design ``run.start`` (docs/olcumler/baslangic_tasarimi_onkayit.md, ST2-ST5; ST1 is the baseline run
and the resolved-config snapshot)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError

BEAM = {"domain": {"type": "box", "shape": [12, 6], "h": 1.0},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                   "force": [0, -1]}]}],
        "chain": [{"type": "density_filter", "rmin": 1.5}], "objective": "compliance",
        "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma"}


def _study(cfg=BEAM):
    return build_setup(copy.deepcopy(cfg)).study


def test_st2_cosine() -> None:
    st = _study()
    d = st.domain
    x0 = st.perturbed_start({"type": "cosine", "amplitude": 0.2, "periods": 2})
    xu = st.default_start()
    ijk = np.stack(np.unravel_index(np.asarray(d.design_elements), d.shape), axis=1)
    p = 0.2 * np.prod(np.cos(2 * np.pi * 2 * ijk / np.asarray(d.shape)), axis=1)
    inside = (xu + p >= 0) & (xu + p <= 1)
    assert np.max(np.abs((x0 - xu)[inside] - p[inside])) <= 1e-15
    assert x0.min() >= 0.0 and x0.max() <= 1.0


def test_st3_random_seeded() -> None:
    st = _study()
    a = st.perturbed_start({"type": "random", "amplitude": 0.1}, seed=3)
    b = st.perturbed_start({"type": "random", "amplitude": 0.1}, seed=3)
    c = st.perturbed_start({"type": "random", "amplitude": 0.1}, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.abs(a - st.default_start()).max() <= 0.1


def test_st4_yaml_inverse_homogenisation_moves_from_a_cosine_start(tmp_path) -> None:
    from toporlogy.execution.runner import run_setup
    cfg = {"domain": {"type": "box", "shape": [10, 10], "h": 1.0}, "material": {"E0": 1.0, "nu": 0.3},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "max"]},
                                      "force": [0, -1]}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5, "periodic": True}],
           "objective": {"type": "effective_poisson", "convention": "stiffness_ratio"},
           "constraints": [{"response": "volume", "upper": 0.6},
                           {"response": {"type": "effective_stiffness", "negate": True}, "upper": -0.02}],
           "optimizer": "mma", "run": {"max_iter": 40, "start": {"type": "cosine", "amplitude": 0.2}},
           "output": {"folder": str(tmp_path)}}
    setup = build_setup(copy.deepcopy(cfg))
    st = setup.study
    nu0 = st.evaluate(st.perturbed_start({"type": "cosine", "amplitude": 0.2}), False).f0
    out = run_setup(setup, folder=tmp_path)
    nu1 = float(out["objective"])
    print(f"ST4: nu12 {nu0:.4f} -> {nu1:.4f}")
    assert nu1 <= nu0 - 0.1


@pytest.mark.parametrize("spec", [{"type": "noise", "amplitude": 0.1}, {"type": "cosine", "amplitude": 0.0},
                                  {"type": "cosine", "amplitude": 0.6}, {"type": "cosine", "amplitude": 0.1,
                                                                         "periods": 0},
                                  {"type": "random"}, {"type": "random", "amplitude": 0.1, "periods": 2}])
def test_st5_rejects(spec) -> None:
    with pytest.raises((ValueError, ConfigError)):
        _study().perturbed_start(spec)


def test_st5_rejects_phases_and_angle_block() -> None:
    from pathlib import Path

    import yaml
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "examples/research/fiber_angle_cantilever2d.yaml").read_text(encoding="utf-8"))
    st = build_setup(cfg).study
    with pytest.raises(ValueError, match="phases or an angle block"):
        st.perturbed_start({"type": "cosine", "amplitude": 0.1})
