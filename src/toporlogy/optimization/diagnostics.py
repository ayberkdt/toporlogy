"""Checks every new link and response must pass (docs/mimari_plan.md §2): finite-difference gradients, grey level;
mirror-symmetry error of element fields (docs/yol_haritasi_2026-11-18.md §2.3); an optimizer-independent first-order
optimality measure of a returned design (docs/olcumler/dogrulama_turu.md §4)."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

__all__ = ["gradcheck", "study_gradcheck", "grey_fraction", "field_symmetry", "kkt_estimate"]


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


def field_symmetry(domain: Any, fields: Mapping[str, np.ndarray], axis: Any, coordinate: float,
                   threshold: float = 0.5) -> dict[str, Any]:
    """Mirror-symmetry error of element fields about the plane ``axis = coordinate``.

    For a field f on the box (``domain.shape``), R(f)[e] = f[p(e)] with p the mirror partner of
    :func:`toporlogy.design.manufacturing.mirror_partner` (the cell holding the mirrored centre, the relation
    ``SymmetryTying`` ties); R(f)[e] = 0 where the mirror image leaves the box (the field is void outside). Per field:

    * ``e_sym`` = ‖f − R(f)‖₂ / ‖f‖₂ over the whole box (0 for f ≡ 0); 0 for a field that is mirror symmetric on
      the grid, about √2 for a field and its mirror image with disjoint supports;
    * ``mismatch_cells`` = number of cells where (f > threshold) ≠ (R(f) > threshold); for a 0/1 mask
      e_sym = √(mismatch_cells / cells in the mask).

    Plane: ``grid_shift_max`` = largest distance along ``axis`` between a mirrored centre and its partner's centre
    (0 when the plane lies on a grid line or a cell centre, < h/2 otherwise: the grid itself is then not mirror
    symmetric and part of e_sym is discretisation, not design); ``cells_outside_box`` = active cells whose mirror
    image leaves the box.
    """
    from toporlogy.design.manufacturing import _axis_index, mirror_partner
    p = mirror_partner(domain, axis, coordinate)
    k = _axis_index(axis, domain.dim)
    X = domain.element_centers()
    ok = p >= 0
    shift = np.abs(2.0 * float(coordinate) - X[ok, k] - X[p[ok], k])
    out: dict[str, Any] = {"axis": "xyz"[k], "coordinate": float(coordinate),
                           "grid_shift_max": float(shift.max()) if shift.size else 0.0,
                           "cells_outside_box": int((~ok & np.asarray(domain.active).ravel()).sum()), "fields": {}}
    for name, f in fields.items():
        v = np.asarray(f, float).ravel()
        if v.size != p.size:
            raise ValueError(f"field {name!r} has {v.size} values, the box has {p.size} elements")
        rv = np.where(ok, v[np.maximum(p, 0)], 0.0)
        nf = float(np.linalg.norm(v))
        out["fields"][name] = {"e_sym": float(np.linalg.norm(v - rv)) / nf if nf > 0 else 0.0,
                               "mismatch_cells": int(((v > threshold) != (rv > threshold)).sum())}
    return out


def kkt_estimate(x: np.ndarray, df0: np.ndarray, g: np.ndarray, dg: np.ndarray, active_tol: float = 1e-3,
                 bound_tol: float = 1e-9) -> dict[str, Any]:
    """First-order optimality of ``x`` for min f0 s.t. g ≤ 0, 0 ≤ x ≤ 1, with multipliers found here.

    Constraints with g_j ≥ −``active_tol`` are treated as active and get λ_j ≥ 0 minimising
    ‖(∇f0 + Σ_j λ_j ∇g_j)_I‖₂ over the variables I strictly inside the box (non-negative least squares); the others
    get λ_j = 0. The residual is the larger of the projected stationarity max|r| / max|∇f0| (r = ∇f0 + Σ λ_j ∇g_j,
    with only its infeasible sign kept at a bound: r_i < 0 at x_i = 0 and r_i > 0 at x_i = 1 count) and the
    constraint violation max(g, 0). Unlike ``Result.kkt_residual`` it needs no optimizer multipliers and belongs to
    ``x`` itself; it is a lower estimate of the KKT residual with the optimizer's own multipliers.

    Returns {"residual", "stationarity", "violation", "multipliers" (m,), "active" (m,) bool}.
    """
    from scipy.optimize import nnls
    x = np.asarray(x, float).reshape(-1)
    df0 = np.asarray(df0, float).reshape(-1)
    g = np.atleast_1d(np.asarray(g, float)).reshape(-1)
    dg = np.asarray(dg, float).reshape(len(g), -1)
    if dg.shape[1] != x.size or df0.size != x.size:
        raise ValueError("kkt_estimate: x, df0 and the rows of dg must have the same length")
    lo, hi = x <= bound_tol, x >= 1.0 - bound_tol
    inner = ~(lo | hi)
    active = g >= -float(active_tol)
    lam = np.zeros(len(g))
    if active.any() and inner.any():
        lam[active], _ = nnls(dg[active][:, inner].T, -df0[inner])
    r = df0 + lam @ dg
    r = np.where(lo, np.minimum(r, 0.0), np.where(hi, np.maximum(r, 0.0), r))
    scale = max(float(np.max(np.abs(df0))), 1e-300) if df0.size else 1.0
    stat = float(np.max(np.abs(r))) / scale if r.size else 0.0
    viol = float(max(np.max(g), 0.0)) if g.size else 0.0
    return {"residual": max(stat, viol), "stationarity": stat, "violation": viol, "multipliers": lam,
            "active": active}
