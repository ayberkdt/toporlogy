"""Run folder: settings, git revision, environment, per-iteration history, final design (docs/mimari_plan.md §2)."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Recorder", "git_revision"]


def git_revision(path: str | Path = ".") -> str | None:
    """``git rev-parse HEAD`` (+ "-dirty" with uncommitted changes) or None outside a repository."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,  # noqa: S607
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=path,  # noqa: S607
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - not a repo, git missing, no commit yet
        return None


class Recorder:
    """Writes ``setup.json``, ``history.jsonl`` and ``result.npz`` + ``result.json`` into ``folder``.

    ``snapshot_every`` > 0 also stores x̄ every k iterations (float16) in ``snapshots.npz``.
    """

    def __init__(self, folder: str | Path, snapshot_every: int = 0, meta: dict[str, Any] | None = None) -> None:
        self.folder = Path(folder)
        self.snapshot_every = int(snapshot_every)
        self.meta = dict(meta or {})
        self._snaps: dict[str, np.ndarray] = {}
        self._t0 = 0.0

    def start(self, study: Any) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        import torch
        setup = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_revision(Path(__file__).parent),
                 "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                 "machine": platform.platform(), "settings": study.settings(), "meta": self.meta}
        (self.folder / "setup.json").write_text(json.dumps(setup, indent=1, default=str), encoding="utf-8")
        (self.folder / "history.jsonl").write_text("", encoding="utf-8")
        self._study = study
        self._t0 = time.perf_counter()

    def log(self, it: int, ev: Any, change: float, state: dict[str, float]) -> None:
        row = {"it": it, "t": round(time.perf_counter() - self._t0, 4), "f0": ev.f0, "change": change,
               **ev.values, **state}
        with open(self.folder / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if self.snapshot_every and it % self.snapshot_every == 0:
            self._snaps[f"it{it:05d}"] = ev.xphys.astype(np.float16)

    def finish(self, res: Any) -> None:
        np.savez_compressed(self.folder / "result.npz", x=res.x, xphys=res.xphys)
        if self._snaps:
            # numpy stub: **kwds could also bind the keyword-only allow_pickle: bool
            np.savez_compressed(self.folder / "snapshots.npz", **self._snaps)  # type: ignore[arg-type]
        out = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
               "objective": res.objective if (res.final is not None or res.history["f0"]) else None,
               "constraints": res.constraint_values, "termination": getattr(res, "termination", ""),
               "feasible": getattr(res, "feasible", None), "max_violation": getattr(res, "max_violation", None),
               "feasibility_tol": getattr(res, "feasibility_tol", None),
               "kkt_residual": getattr(res, "kkt_residual", None), "settings": res.settings}
        (self.folder / "result.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
