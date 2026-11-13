"""Shape regularisation on the signed distance field without changing the topology.

* :func:`remove_islands` — components of the solid that are neither the largest nor touch a Keep body are removed
  (they carry no load); enclosed voids are filled or reported (:func:`fill_voids`).
* :func:`adaptive_opening` — opening (erode, then dilate by r) removes spikes and knife edges and rounds convex
  edges to ≥ r, but it also deletes every member thinner than 2r. The radius therefore starts at ``r_start`` and is
  reduced (× ``factor`` per trial) until the opened part (with the exact interfaces re-imposed) has the same number
  of bodies, cavities and Euler characteristic as before and loses at most ``max_loss`` of its volume. A member
  that ends on a Keep body would, if deleted, disconnect that body, so the body count catches it; a member in a
  closed load path would, if deleted, open a tunnel, so χ catches it.
* :func:`adaptive_closing` — closing (dilate, then erode by r) fillets every concave corner with ≥ r but may close
  holes narrower than 2r or bridge two members; same reduction rule with ``max_gain``.

Every trial is logged (radius, volume change, topology, accepted). A volume-loss rule alone is not enough: a
fixed opening radius can delete load-carrying members while removing only a few per cent of the volume.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from aerokit.topo.post import sdf as S
from aerokit.topo.post.topology import enclosed_voids, grid_topology, label_bodies, same_topology

__all__ = ["remove_islands", "fill_voids", "adaptive_opening", "adaptive_closing", "radii"]


def radii(r_start: float, factor: float, n: int, r_min: float) -> list[float]:
    """Trial radii r_start, r_start·factor, … (n values, none below r_min), then 0."""
    out = []
    r = float(r_start)
    for _ in range(int(n)):
        if r < r_min:
            break
        out.append(r)
        r *= factor
    return out + [0.0]


def remove_islands(phi_free: np.ndarray, combine: Callable[[np.ndarray], np.ndarray], keep_masks: list[np.ndarray],
                   h: float, mode: str = "interfaces") -> tuple[np.ndarray, dict[str, Any]]:
    """Remove bodies of combine(φ_free) that are not kept. ``mode``: "largest" | "interfaces" (largest + every body
    touching a Keep mask) | "all" (nothing removed)."""
    solid = combine(phi_free) < 0
    lab, n = label_bodies(solid)
    info: dict[str, Any] = {"bodies_before": int(n), "removed": 0, "removed_volume": 0.0, "mode": mode}
    if n <= 1 or mode == "all":
        return phi_free, info
    sizes = np.bincount(lab.ravel())[1:]
    keep = {1 + int(np.argmax(sizes))}
    if mode == "interfaces":
        for m in keep_masks:
            keep |= set(np.unique(lab[m & solid]).tolist()) - {0}
    elif mode != "largest":
        raise ValueError("keep_bodies: largest | interfaces | all")
    drop = (lab > 0) & ~np.isin(lab, list(keep))
    if not drop.any():
        return phi_free, info
    out = S.redistance(np.where(drop, np.maximum(phi_free, h), phi_free), h)
    info.update({"removed": int(n - len(keep)), "removed_volume": float(drop.sum()) * h ** 3})
    return out, info


def fill_voids(phi: np.ndarray, h: float, fill: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Enclosed voids (powder traps) of φ < 0: filled when ``fill`` (φ set inside), else only counted."""
    from scipy import ndimage
    cav = enclosed_voids(phi < 0)
    _, n = ndimage.label(cav)
    info = {"enclosed_voids": int(n), "void_volume": float(cav.sum()) * h ** 3, "filled": bool(fill and n > 0)}
    if fill and n:
        phi = np.where(cav, -0.5 * h, phi)
        phi = S.redistance(phi, h)
    return phi, info


def _trial(phi_new: np.ndarray, combine: Callable, topo_ref: dict[str, Any], v_ref: float, h: float, r: float,
           sign: int, bound: float) -> dict[str, Any]:
    comp = combine(phi_new)
    topo = grid_topology(comp < 0)
    v = S.volume(comp, h)
    change = (v - v_ref) / v_ref
    ok = same_topology(topo, topo_ref) and sign * change <= bound + 1e-12
    return {"r": r, "volume": v, "change": change, "topology": topo, "accepted": bool(ok)}


def adaptive_opening(phi: np.ndarray, combine: Callable[[np.ndarray], np.ndarray], h: float, r_start: float,
                     max_loss: float = 0.02, factor: float = 0.75, n_trials: int = 6
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Largest opening radius from the trial list that keeps topology and loses ≤ ``max_loss`` volume."""
    comp0 = combine(phi)
    topo0 = grid_topology(comp0 < 0)
    v0 = S.volume(comp0, h)
    log = []
    for r in radii(r_start, factor, n_trials, 0.5 * h):
        if r == 0.0:
            break
        op = S.opening(phi, r, h)
        t = _trial(op, combine, topo0, v0, h, r, -1, max_loss)
        log.append(t)
        if t["accepted"]:
            return op, {"r_used": r, "volume_change": t["change"], "trials": log,
                                          "topology_before": topo0, "volume_before": v0}
    return phi, {"r_used": 0.0, "volume_change": 0.0, "trials": log, "topology_before": topo0, "volume_before": v0}


def adaptive_closing(phi: np.ndarray, combine: Callable[[np.ndarray], np.ndarray], h: float, r_start: float,
                     max_gain: float = 0.05, factor: float = 0.75, n_trials: int = 6
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Largest closing radius from the trial list that keeps topology and adds ≤ ``max_gain`` volume."""
    comp0 = combine(phi)
    topo0 = grid_topology(comp0 < 0)
    v0 = S.volume(comp0, h)
    log = []
    for r in radii(r_start, factor, n_trials, 0.5 * h):
        if r == 0.0:
            break
        cl = S.closing(phi, r, h)
        t = _trial(cl, combine, topo0, v0, h, r, +1, max_gain)
        log.append(t)
        if t["accepted"]:
            return cl, {"r_used": r, "volume_change": t["change"], "trials": log, "topology_before": topo0,
                        "volume_before": v0}
    return phi, {"r_used": 0.0, "volume_change": 0.0, "trials": log, "topology_before": topo0, "volume_before": v0}
