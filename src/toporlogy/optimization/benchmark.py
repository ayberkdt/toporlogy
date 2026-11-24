"""Fair comparison of optimisation results: evaluate the thresholded (0/1) design, count every solve.

A density result with grey elements is not a structure. Its objective profits from intermediate stiffness that no
manufactured part has, and two methods that leave different amounts of grey cannot be compared on that number.
The recommendation attributed to Sigmund (2022) — compare designs after converting them to 0/1, at equal resolution
and with the full computational cost reported — is what :func:`discrete_report` and :func:`compare_runs` implement. Only
the record of that paper and statements of citing papers were seen in this work (``docs/literatur_2026-11-21.md``).

* :func:`binarize` — threshold the physical density, either at a fixed level or at the level that meets a volume
  fraction exactly (the largest densities are kept; ties broken by element number, so the result is deterministic).
* :func:`discrete_report` — objective and constraints of the binarised design re-evaluated by the study
  (:meth:`~toporlogy.optimization.study.Study.evaluate_density`: same mesh, same physics, same interpolation at
  x̄ ∈ {0, 1}), next to the values of the grey design, the grey measure M_nd and the number of evaluations.
* :func:`compare_runs` — the same for several (study, result) pairs, as rows and as a text table.

What this does not do: it does not smooth or re-mesh. The binarised design is the staircase voxel model; the gap to a
body-fitted model of the smoothed part is a separate question (``post.verify``, and the open problem R-01 of the
roadmap). Responses that need the design chain (robust realisations) cannot be evaluated on a density and are
reported as not evaluated, never as passed.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from toporlogy.core.verdict import BINARY_DESIGN, combine, constraint_verdict
from toporlogy.optimization.responses import Volume
from toporlogy.references import cite

__all__ = ["binarize", "discrete_report", "compare_runs", "format_table"]


def _m_nd(x: np.ndarray, mask: np.ndarray) -> float:
    v = np.asarray(x, float)[mask]
    return float(np.mean(4.0 * v * (1.0 - v))) if v.size else 0.0


def binarize(xphys: np.ndarray, domain: Any, volume: float | None = None, threshold: float = 0.5) -> np.ndarray:
    """0/1 density on ``domain.shape``.

    ``volume`` None: ``x̄ > threshold``. ``volume`` v: the ``round(v · n_active)`` elements of the part with the
    largest x̄ become solid (count rule of :class:`~toporlogy.optimization.responses.Volume`, unweighted); keep
    regions are solid whatever their rank needs, void regions empty. Elements outside the part are 0.
    """
    x = np.asarray(xphys, float)
    if x.shape != tuple(domain.shape):
        raise ValueError(f"xphys has shape {x.shape}, the domain {tuple(domain.shape)}")
    out = np.zeros(domain.shape)
    if volume is None:
        out[(x > threshold) & domain.active] = 1.0
    else:
        if not 0 < volume <= 1:
            raise ValueError("0 < volume <= 1 is required")
        n_solid = int(round(volume * domain.n_active))
        free = np.flatnonzero((domain.active & ~domain.keep & ~domain.void).ravel())
        n_free = max(n_solid - int(domain.keep.sum()), 0)
        order = free[np.argsort(-x.ravel()[free], kind="stable")]
        out.ravel()[order[:n_free]] = 1.0
    out[domain.keep] = 1.0
    out[domain.void] = 0.0
    return out


@cite("sigmund_2022", note="compare thresholded designs; record seen, text not read")
def discrete_report(study: Any, result: Any, volume: float | str | None = "constraint",
                    threshold: float = 0.5) -> dict[str, Any]:
    """Grey versus binarised design of ``result`` (module notes).

    ``volume``: a volume fraction for :func:`binarize`; ``"constraint"`` = the bound of the study's first plain
    volume constraint (falls back to ``threshold`` when there is none); None = fixed ``threshold``.

    Returns a dict with ``objective`` (grey), ``objective_binary``, ``ratio`` (binary / grey), ``m_nd`` (grey design),
    ``volume_binary`` (element count), ``constraints`` ({name: {grey, binary, upper, verdict}}: verdict of the 0/1
    value with tolerance 1e-9, UNVERIFIED when it could not be evaluated, :mod:`toporlogy.core.verdict`),
    ``not_evaluated`` ({name: reason}), ``evaluations`` (study total so far),
    ``inner_evaluations`` (inside optimizer steps, when reported), ``iterations`` and ``threshold_rule``.
    """
    d = study.domain
    vol: float | None
    if volume == "constraint":
        vol = next((float(c.upper) for c in study.constraints
                    if type(c.response) is Volume and not c.response.weighted), None)
    elif volume is None or isinstance(volume, (int, float)):
        vol = None if volume is None else float(volume)
    else:
        raise ValueError('volume: a number, "constraint" or None')
    xb = binarize(result.xphys, d, vol, threshold)
    n_before = study.n_evaluations
    ev = study.evaluate_density(xb)
    study.n_evaluations = n_before                          # the check is not part of the optimisation cost
    name = study.objective.name
    grey = float(result.objective)
    binary = ev.values.get(name)
    cons: dict[str, Any] = {}
    for c in study.constraints:
        v = ev.values.get(c.key)
        cons[c.key] = {"grey": result.constraint_values.get(c.key), "binary": v, "upper": float(c.upper),
                       "verdict": constraint_verdict(v, float(c.upper), 1e-9, BINARY_DESIGN)}
    return {"objective": grey, "objective_binary": binary,
            "ratio": None if binary is None or grey == 0 else float(binary) / grey,
            "m_nd": _m_nd(result.xphys, d.design), "volume_binary": float(xb[d.active].sum() / d.n_active),
            "constraints": cons, "not_evaluated": dict(ev.skipped),
            "evaluations": int(study.n_evaluations), "iterations": int(result.iterations),
            "inner_evaluations": result.settings.get("inner_evaluations"),
            "threshold_rule": f"volume {vol:g}" if vol is not None else f"threshold {threshold:g}"}


def compare_runs(runs: Mapping[str, tuple[Any, Any]], volume: float | str | None = "constraint",
            threshold: float = 0.5) -> list[dict[str, Any]]:
    """:func:`discrete_report` of every ``name -> (study, result)``, one row each (key ``name`` added), in order."""
    return [{"name": k, **discrete_report(s, r, volume, threshold)} for k, (s, r) in runs.items()]


def format_table(rows: list[dict[str, Any]]) -> str:
    """Text table of :func:`compare_runs` rows; the reference of ``rel`` is the best binary objective."""
    vals = [r["objective_binary"] for r in rows if r["objective_binary"] is not None]
    best = min(vals) if vals else None
    head = f"{'name':<18}{'grey obj':>12}{'binary obj':>12}{'rel':>8}{'M_nd':>8}{'vol(bin)':>10}{'evals':>7}  verdict"
    lines = [head, "-" * len(head)]
    for r in rows:
        b = r["objective_binary"]
        feas = combine(c["verdict"] for c in r["constraints"].values())
        lines.append(f"{r['name']:<18}{r['objective']:>12.5g}"
                     + (f"{b:>12.5g}{b / best:>8.3f}" if b is not None and best else f"{'-':>12}{'-':>8}")
                     + f"{r['m_nd']:>8.3f}{r['volume_binary']:>10.4f}{r['evaluations']:>7d}  {feas}")
    return "\n".join(lines)
