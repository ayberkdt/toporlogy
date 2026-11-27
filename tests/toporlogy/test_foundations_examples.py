"""The Foundations II examples (examples/foundations) run (smoke test; their numbers are verified elsewhere)."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EX = Path(__file__).resolve().parents[2] / "examples" / "foundations"


def test_material_point_histories_runs(capsys) -> None:
    runpy.run_path(str(EX / "material_point_histories.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "Norton creep" in out and "generalized Maxwell" in out


@pytest.mark.slow
def test_viscoelastic_minmax_runs(capsys) -> None:
    runpy.run_path(str(EX / "viscoelastic_minmax.py"), run_name="__main__")
    assert "min-max" in capsys.readouterr().out
