"""torch threads set once by the CLI (docs/olcumler/hizlandirma_onkayit.md, TT1-TT2, pre-registered before the code;
TT2 measured inside the run, see the record)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from toporlogy.execution.cli import main
from toporlogy.optimization.study import Study

CFG = {"domain": {"type": "box", "shape": [12, 6], "h": 1.0},
       "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
       "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                  "force": [0, -1]}]}],
       "chain": [{"type": "density_filter", "rmin": 1.5}],
       "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma", "run": {"max_iter": 3}}


@pytest.mark.parametrize(("env", "want"), [(None, 1), ("3", 3)])
def test_tt2_threads_inside_the_run_and_restored(env: str | None, want: int, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    if env is None:
        monkeypatch.delenv("TOPORLOGY_TORCH_THREADS", raising=False)
    else:
        monkeypatch.setenv("TOPORLOGY_TORCH_THREADS", env)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(CFG), encoding="utf-8")
    seen = []
    real = Study.evaluate

    def spy(self, x, gradients=True):
        seen.append(torch.get_num_threads())
        return real(self, x, gradients)
    monkeypatch.setattr(Study, "evaluate", spy)
    before = torch.get_num_threads()
    main(["run", str(p), "--out", str(tmp_path / "r"), "-q"])
    assert seen and set(seen) == {want}
    assert torch.get_num_threads() == before
