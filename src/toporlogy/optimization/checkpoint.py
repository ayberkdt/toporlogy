"""Checkpoint and restart of a :class:`~toporlogy.optimization.study.Study` run (docs/olcumler/checkpoint_onkayit.md).

A checkpoint holds the *plain state* of the objects a run changes, not the objects: numbers, strings, NumPy arrays and
torch tensors found in the attributes of the chain links (β of a projection, ...), the schedule (its counters), the
optimizer (asymptotes, previous iterates, objective scale) and the responses (scales frozen at a first evaluation,
recursively through wrapped responses), plus the design vector, the iteration, the history, the torch and NumPy
random states and a fingerprint of the problem. Restoring writes those values back into the objects of a study built
from the same configuration, so references between them (a schedule and its projection link) stay intact and
nothing that cannot be serialised (factorisations, solvers, physics caches) is touched. The file is written to a
temporary name and moved into place (``os.replace``), so a reader never sees half a file.
"""
from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["capture", "restore", "fingerprint", "save_checkpoint", "load_checkpoint", "CHECKPOINT_VERSION"]

CHECKPOINT_VERSION = 1
_PLAIN = (bool, int, float, str, type(None), np.ndarray, np.generic, torch.Tensor)
_WRAPPED = ("response", "objective", "numerator", "denominator", "field")


def _plain(v: Any) -> bool:
    if isinstance(v, _PLAIN):
        return True
    if isinstance(v, (list, tuple)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _plain(x) for k, x in v.items())
    return False


def _attrs(obj: Any) -> dict[str, Any]:
    d = vars(obj) if hasattr(obj, "__dict__") else {}
    return {k: copy.deepcopy(v) for k, v in d.items() if not k.startswith("__") and _plain(v)}


def capture(obj: Any, _seen: set[int] | None = None) -> dict[str, Any]:
    """Plain attributes of ``obj`` and, for responses, of the responses and scale objects it wraps."""
    seen = set() if _seen is None else _seen
    if obj is None or id(obj) in seen:
        return {}
    seen.add(id(obj))
    out: dict[str, Any] = {"attrs": _attrs(obj), "children": {}}
    for name in (*_WRAPPED, "scale"):
        child = getattr(obj, name, None)
        if child is not None and not _plain(child) and hasattr(child, "__dict__") and not isinstance(
                child, torch.nn.Module):
            out["children"][name] = capture(child, seen)
    subs = getattr(obj, "responses", None)
    if isinstance(subs, (list, tuple)):
        out["children"]["responses"] = [capture(r, seen) for r in subs]
    return out


def restore(obj: Any, state: dict[str, Any]) -> None:
    """Write a :func:`capture` back into ``obj`` (a study object of the same configuration)."""
    if obj is None or not state:
        return
    for k, v in state["attrs"].items():
        setattr(obj, k, copy.deepcopy(v))
    for name, sub in state["children"].items():
        if name == "responses":
            for r, s in zip(obj.responses, sub, strict=True):
                restore(r, s)
        else:
            restore(getattr(obj, name, None), sub)


def fingerprint(study: Any) -> dict[str, Any]:
    """What must match between the saving and the resuming study."""
    s = study.settings()
    return {"n_var": int(study.n_var), "optimizer": s.get("optimizer"), "objective": s.get("objective"),
            "constraints": [list(c) for c in s.get("constraints", [])], "links": s.get("links"),
            "interp": s.get("interp"), "domain": s.get("domain")}


def save_checkpoint(path: str | Path, study: Any, payload: dict[str, Any]) -> None:
    """Atomic write of the run state (``payload``: x, iteration, history, ...) and the captured objects."""
    state = {"version": CHECKPOINT_VERSION, "fingerprint": fingerprint(study), **payload,
             "links": [capture(lk) for lk in study.chain.links], "schedule": capture(study.schedule),
             "optimizer": capture(study.optimizer), "objective": capture(study.objective),
             "constraints": [capture(c.response) for c in study.constraints],
             "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}}
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def load_checkpoint(path: str | Path, study: Any) -> dict[str, Any]:
    """Restore the captured objects into ``study`` and return the run payload; refuses another problem."""
    with open(path, "rb") as f:
        state = pickle.load(f)  # noqa: S301 - a file this module wrote; do not load untrusted checkpoints
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {state.get('version')} is not {CHECKPOINT_VERSION}")
    want = fingerprint(study)
    if state["fingerprint"] != want:
        diff = sorted(k for k in want if state["fingerprint"].get(k) != want[k])
        raise ValueError(f"checkpoint {path} belongs to another problem (different {', '.join(diff)})")
    for lk, s in zip(study.chain.links, state["links"], strict=True):
        restore(lk, s)
    restore(study.schedule, state["schedule"])
    restore(study.optimizer, state["optimizer"])
    restore(study.objective, state["objective"])
    for c, s in zip(study.constraints, state["constraints"], strict=True):
        restore(c.response, s)
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    return state
