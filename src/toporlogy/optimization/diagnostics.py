"""Checks every new link and response must pass (docs/mimari_plan.md §3.1): finite-difference gradients, grey level."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

__all__ = ["gradcheck", "study_gradcheck", "grey_fraction"]


def gradcheck(fn: Callable[[torch.Tensor], torch.Tensor], x: np.ndarray, n_dirs: int = 4, h: float = 1e-6,
              seed: int = 0) -> dict[str, Any]:
    """Directional derivatives of the scalar ``fn`` (tensor → scalar tensor) by central differences vs autograd.

    Returns ``max_rel_error`` = max over random unit directions d of |FD − ∇f·d| / max(|∇f·d|, |FD|, tiny),
    plus the per-direction values. float64 throughout.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x, float)
    xt = torch.as_tensor(x0, dtype=torch.float64).clone().requires_grad_(True)
    f = fn(xt)
    (g,) = torch.autograd.grad(f, xt)
    gn = g.detach().numpy().ravel()
    rows = []
    for _ in range(n_dirs):
        d = rng.standard_normal(x0.size)
        d /= np.linalg.norm(d)
        with torch.no_grad():
            fp = float(fn(torch.as_tensor(x0 + h * d.reshape(x0.shape), dtype=torch.float64)))
            fm = float(fn(torch.as_tensor(x0 - h * d.reshape(x0.shape), dtype=torch.float64)))
        fd = (fp - fm) / (2 * h)
        ad = float(gn @ d)
        rows.append({"fd": fd, "ad": ad, "rel": abs(fd - ad) / max(abs(ad), abs(fd), 1e-300)})
    return {"max_rel_error": max(r["rel"] for r in rows), "directions": rows, "f": float(f)}


def study_gradcheck(study: Any, x: np.ndarray, which: str | int = "objective", **kw: Any) -> dict[str, Any]:
    """:func:`gradcheck` of the study's objective (``"objective"``) or constraint ``i`` (int, the scaled g)."""
    def fn(xt: torch.Tensor) -> torch.Tensor:
        rs = [study.objective] + [c.response for c in study.constraints]
        ctx = study.context(xt, "u" in study._needs(rs))
        if which == "objective":
            return study.objective(ctx)
        c = study.constraints[int(which)]
        return c.g(c.response(ctx))
    return gradcheck(fn, x, **kw)


def grey_fraction(xphys: np.ndarray, mask: np.ndarray | None = None, lo: float = 0.1, hi: float = 0.9) -> float:
    """Share of (masked) elements with lo < x̄ < hi."""
    v = np.asarray(xphys, float)
    v = v[np.asarray(mask, bool)] if mask is not None else v.ravel()
    return float(np.mean((v > lo) & (v < hi))) if v.size else 0.0
