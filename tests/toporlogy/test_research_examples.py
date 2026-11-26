"""The research examples (examples/research): every one builds from YAML, runs a few iterations through the CLI
runner and writes a consistent result. Their numerics are checked in the module tests (gradients, closed forms);
this test guards the configuration layer and the registered components."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from toporlogy.configuration.build import build_setup
from toporlogy.execution.runner import run_setup

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "research").glob("*.yaml"))


def test_examples_are_found() -> None:
    assert [p.stem for p in EXAMPLES] == ["buckling_column2d", "fiber_angle_cantilever2d", "inverter2d", "mbb_simpl2d",
                                          "stress_al_lbracket2d", "stress_lbracket2d"]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_builds_and_runs(path: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    setup = build_setup(raw, overrides=["run.max_iter=4"])
    study = setup.study
    x0 = study.default_start()
    ev = study.evaluate(x0)
    assert np.isfinite(ev.f0) and np.all(np.isfinite(ev.df0)) and np.all(np.isfinite(ev.dg))
    assert np.abs(ev.df0).max() > 0
    run_setup(setup, tmp_path / "run")
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    result = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    assert result["iterations"] == 4 and np.isfinite(summary["objective"])
    if raw.get("post"):
        post = summary["post"]["discrete_check"]
        assert post["threshold_rule"] and isinstance(post["passed"], bool)


def test_inverter_inverts() -> None:
    """After 40 iterations the output moves against the input (negative x displacement), and less than the input."""
    raw = yaml.safe_load(next(p for p in EXAMPLES if p.stem == "inverter2d").read_text(encoding="utf-8"))
    study = build_setup(raw).study
    res = study.run(max_iter=40)
    assert res.objective < -0.3 and res.feasible
    d = study.domain
    x_in = float(np.mean([res.U[2 * n, 0] for n in d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] <= 2))]))
    assert x_in > 0
