"""The L-bracket stress experiment of docs/olcumler/gerilme_onkayit.md §5, used by the slow test and by
``docs/olcumler/gerilme.py``.

**History.** Before 26 Nov 2026 the unchanged examples stopped in this environment (torch 2.5.1): the qp relaxation
x̄^q had an infinite derivative where x̄ = 0 and torch turned the aggregate's zero incoming gradient times it into NaN.
The library now sets that derivative to 0 (``optimization/stress.py``, gerilme_onkayit.md §6); the experiment-only
subclass that did the same is gone.

Everything is measured on the design the study returns (``Result.xphys``) and on its threshold at 0.5, never on the
optimizer's last accepted values. ``raw`` is the unrelaxed solid von Mises stress over σ_lim, ``relaxed`` the qp form
x̄^0.5 · raw; the two are kept apart.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy import ndimage

import _independent_fe as ind
from toporlogy.configuration.build import build_setup
from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.mechanisms import InvalidModuliError
from toporlogy.optimization.benchmark import binarize
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.stress import VonMises
from toporlogy.optimization.stress_recovery import StressRecovery
from toporlogy.optimization.stress_singularity import assess_peak

REPO = Path(__file__).resolve().parents[2]
SIGMA_LIM, NU, Q, EMIN = 2.5, 0.3, 0.5, 1e-6
METHODS = {"aggregated": "stress_lbracket2d", "augmented_lagrangian": "stress_al_lbracket2d"}
HISTORICAL = {"aggregated": {"grey": 1.131, "binary": 1.715},
              "augmented_lagrangian": {"grey": 1.103, "binary": 1.025}}       # yontemler.md §3
HISTORICAL_SOLID = 0.49
EXCLUDE_LO, EXCLUDE_HI = (34.0, 12.0), (40.0, 16.0)                            # the YAML's exclude box
LOADS = [(float(x), 16.0) for x in (37, 38, 39, 40)]                            # -0.25 each in y
LEVELS = (1, 2, 4, 8)


def attempt_unchanged(method: str) -> dict[str, Any]:
    """The pre-registered run, as it ends: where it stops and with what (or its numbers, where it can run)."""
    raw = yaml.safe_load((REPO / "examples" / "research" / f"{METHODS[method]}.yaml").read_text(encoding="utf-8"))
    st = build_setup(raw).study
    first_bad: dict[str, int] = {}
    evaluate = st.evaluate

    def watched(x: np.ndarray, gradients: bool = True) -> Any:
        ev = evaluate(x, gradients)
        bad = ~np.isfinite(ev.dg).all(axis=0) | ~np.isfinite(ev.df0)
        if gradients and bad.any() and not first_bad:
            first_bad.update(evaluation=int(st.n_evaluations), design_variables=int(bad.sum()),
                             elements_with_zero_density=int((ev.xphys[st.domain.active] == 0.0).sum()))
        return ev

    st.evaluate = watched
    done: list[int] = []
    out: dict[str, Any] = {"completed": True, "error": None}
    try:
        st.run(max_iter=raw["run"]["max_iter"], tol=raw["run"]["tol"], callback=lambda it, ev, ch: done.append(it))
    except InvalidModuliError as e:
        out = {"completed": False, "error": f"{type(e).__name__}: {e}"}
    return {**out, "iterations_completed": len(done), "state_solves": int(st.n_evaluations),
            "first_non_finite_gradient": first_bad or None}


def _excluded(d: Domain) -> np.ndarray:
    c = d.element_centers()
    return ((c >= EXCLUDE_LO) & (c <= EXCLUDE_HI)).all(1)


def _peak(values: np.ndarray, points: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    i = np.flatnonzero(mask)
    top = i[int(np.argmax(values[i]))]
    return {"ratio": float(values[top] / SIGMA_LIM), "at": [float(v) for v in points[top]]}


def recovery_peaks(physics: Any, U: np.ndarray, solid: np.ndarray) -> dict[str, Any]:
    """Largest solid von Mises ratio of every output type: recovery over the solid elements only, entries of
    non-excluded elements; ``excluded_centroid``: what the exclusion box hides."""
    d = physics.domain
    el = np.flatnonzero(solid)
    rec = StressRecovery(physics, U, 1.0, elements=el)
    keep = ~_excluded(d)[el]
    cen, corner = rec.centroids(), rec.extrapolated()
    top, avg, spr = rec.ip_maximum(), rec.nodal_averaged(), rec.patch_recovered()
    out = {"centroid": _peak(cen.von_mises()[:, 0], cen.points, keep),
           "integration_point_max": _peak(top.value[:, 0], top.points[:, 0], keep),
           "extrapolated": _peak(corner.von_mises()[..., 0].reshape(-1), corner.points.reshape(-1, 2),
                                 np.repeat(keep, 4))}
    for f in (avg, spr):
        used = np.zeros(len(f.node), bool)
        used[f.corner_entry[keep].ravel()] = True
        out[f.kind] = _peak(f.von_mises()[:, 0], f.points, used)
    out["patch_recovered"]["fallback_entries"] = int(spr.fallback.sum())
    out["excluded_centroid"] = _peak(cen.von_mises()[:, 0], cen.points, ~keep) if (~keep).any() else None
    return out


def load_path(study: Any, x: np.ndarray, U: np.ndarray) -> dict[str, Any]:
    """Post-hoc: does the design carry its load through solid material? Compliance, the densities of the elements
    touching a loaded node, and whether one of them (x̄ > 0.5) is edge-connected to an element on the clamped edge."""
    d = study.domain
    F = np.asarray(study.F)
    loaded = np.unique(np.flatnonzero(np.abs(F).sum(axis=1)) // d.dim)
    nodes = d.edof[:, ::d.dim] // d.dim
    touching = np.isin(nodes, loaded).any(1).reshape(d.shape) & d.active
    labels, _ = ndimage.label((x > 0.5) & d.active)                    # 4-neighbour connectivity
    top = labels[:, -1]                                                # elements under the clamped top edge
    held = np.isin(labels, top[top > 0]) & (labels > 0)
    return {"compliance": float((F * U).sum()), "max_density_at_load": float(x[touching].max()),
            "elements_at_load": int(touching.sum()), "load_attached_to_supported_solid": bool((held & touching).any())}


@cache
def run(method: str) -> dict[str, Any]:
    """Run the YAML example unchanged and measure the returned design."""
    raw = yaml.safe_load((REPO / "examples" / "research" / f"{METHODS[method]}.yaml").read_text(encoding="utf-8"))
    st = build_setup(raw).study
    res = st.run(max_iter=raw["run"]["max_iter"], tol=raw["run"]["tol"])
    d = st.domain
    out: dict[str, Any] = {
        "example": METHODS[method], "iterations": int(res.iterations),
        "state_solves": int(st.n_evaluations),
        "termination": res.termination, "feasible_flag": bool(res.feasible), "objective": float(res.objective),
        "optimizer_constraint_values": {k: float(v) for k, v in res.constraint_values.items()},
        "m_nd": float(np.mean(4 * res.xphys[d.design] * (1 - res.xphys[d.design])))}
    el = d.active_elements                                  # every array below is over the active elements
    included = ~_excluded(d)[el]
    arrays: dict[str, np.ndarray] = {}
    for variant, x in (("grey", np.asarray(res.xphys)), ("binary", binarize(res.xphys, d, None, 0.5))):
        E = st.interp(torch.as_tensor(x)).numpy()
        Ea = E.ravel()[el]
        U = st.physics.solve_numpy(Ea, st.F)
        xa = x.ravel()[el]
        solid = xa > 0.5
        cen = StressRecovery(st.physics, U, 1.0).centroids()
        ratio = cen.von_mises()[:, 0] / SIGMA_LIM
        # the same number through the constraint module (historical definition)
        vm = VonMises(d, SIGMA_LIM, NU, 1.0, q=0.0, exclude=_excluded(d).reshape(d.shape))
        ctx = EvalContext(d, torch.as_tensor(x), torch.as_tensor(E), torch.as_tensor(U), st.F, interp=st.interp)
        via_constraint = float(vm(ctx).numpy()[:, 0][x.ravel()[vm.elements] > 0.5].max())
        relaxed = np.clip(xa, 0.0, None) ** Q * ratio
        box_solid = np.zeros(d.ne, bool)
        box_solid[el] = solid
        row = {
            "volume": float(x[d.active].mean()),
            "raw_solid_max": _peak(ratio * SIGMA_LIM, cen.points, solid & included),
            "raw_solid_max_constraint_module": via_constraint,
            "raw_all_elements_max": _peak(ratio * SIGMA_LIM, cen.points, included),
            "relaxed_qp_max": _peak(relaxed * SIGMA_LIM, cen.points, included),
            "interpolated_material_max": _peak(ratio * SIGMA_LIM * Ea, cen.points, included),
            "output_types": recovery_peaks(st.physics, U, box_solid),
            "load_path": load_path(st, x, U)}
        if method == "augmented_lagrangian":
            lam = ratio - 1.0
            row["relaxed_polynomial_g_max"] = float((xa ** 3 * lam * (lam * lam + 1.0))[included].max())
        out[variant] = row
        arrays[f"x_{variant}"] = x
        box_ratio = np.zeros(d.ne)
        box_ratio[el] = ratio
        arrays[f"ratio_{variant}"] = box_ratio.reshape(d.shape)
        if variant == "binary":
            Ds = np.broadcast_to(ind.plane_stress_isotropic(1.0, NU), (d.ne, 3, 3))
            want = ind.centroid_stresses(d.shape, d.h, Ds, np.ones(d.ne), U[:, 0])[el]
            out["oracle_relative_difference"] = float(np.abs(cen.stress[:, :, 0] - want).max() / np.abs(want).max())
    out["arrays"] = arrays
    return out


def refined(design: np.ndarray, k: int) -> dict[str, Any]:
    """The same 0/1 geometry (40 × 40 element array) with every element split into k × k, same supports, same four
    point loads, same exclusion box: peak solid von Mises ratio outside the box."""
    x = np.kron(np.asarray(design, float), np.ones((k, k)))
    mask = np.kron(_base_domain().active.astype(float), np.ones((k, k)))
    d = Domain((40 * k, 40 * k), 1.0 / k, frac=mask)
    X = d.node_coords()
    fixed = d.dofs(d.nodes_where(lambda X: np.isclose(X[:, 1], 40.0)))
    F = d.load_vector()
    for p in LOADS:
        hit = np.flatnonzero(np.isclose(X, p).all(1))
        F[2 * hit[0] + 1, 0] = -0.25
    ph = Elasticity(d, fixed, NU)
    xa = x.ravel()[d.active_elements]
    U = ph.solve_numpy(np.where(xa > 0.5, 1.0, EMIN), F)
    solid = (x.ravel() > 0.5) & d.active.ravel()
    rec = StressRecovery(ph, U, 1.0, elements=np.flatnonzero(solid))
    keep = ~_excluded(d)[rec.elements]
    cen, top = rec.centroids(), rec.ip_maximum()
    vm = cen.von_mises()[:, 0]
    return {"k": k, "h": 1.0 / k, "elements_active": int(d.n_active), "solid_elements": int(solid.sum()),
            "peak_centroid": _peak(vm, cen.points, keep),
            "peak_integration_point": _peak(top.value[:, 0], top.points[:, 0], keep),
            "mean_ratio": float(vm[keep].mean() / SIGMA_LIM),
            "compliance": float((F * U).sum())}


@cache
def _base_domain() -> Domain:
    raw = yaml.safe_load((REPO / "examples" / "research" / "stress_lbracket2d.yaml").read_text(encoding="utf-8"))
    return build_setup(raw).study.domain


@cache
def refinement(which: str) -> dict[str, Any]:
    """``which``: "solid" (the full bracket) or a method name (the 0/1 design of its run)."""
    design = np.ones((40, 40)) if which == "solid" else run(which)["arrays"]["x_binary"]
    levels = [refined(design * _base_domain().active, k) for k in LEVELS]
    a = assess_peak([r["h"] for r in levels], [r["peak_centroid"]["ratio"] for r in levels])
    return {"levels": levels, "status": a.status, "slopes": list(a.slopes), "increment_ratio": a.increment_ratio}
