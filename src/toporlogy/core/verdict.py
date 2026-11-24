"""Result contract (docs/yol_haritasi_2026-11-18.md §3, PR-08): verdict words, value sources, the constraint rule.

Every judged constraint value carries a ``verdict`` and the ``source`` of the value. Only a value evaluated on the
design being judged can PASS or FAIL: the returned design of a run, a repaired design, or a thresholded 0/1 design.
A value from any other source (the last iterate, an old file that cannot tell which), or no value, is UNVERIFIED.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["PASS", "FAIL", "UNVERIFIED", "RETURNED_DESIGN", "REPAIRED_DESIGN", "BINARY_DESIGN", "LAST_ITERATE",
           "UNKNOWN", "MISSING", "EVALUATED", "constraint_verdict", "constraint_entry", "combine"]

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"
RETURNED_DESIGN = "returned design"   # evaluated at the returned design of a run (result.npz x)
REPAIRED_DESIGN = "repaired design"   # evaluated at a repaired density (connectivity_gate bridge repair)
BINARY_DESIGN = "0/1 design"          # evaluated at the thresholded design (optimization.benchmark)
LAST_ITERATE = "last iterate"         # history: the design analysed before the last optimizer update
UNKNOWN = "unknown"                   # a summary.json written before 18 Nov 2026 (no "flags"): either of the two
MISSING = "missing"                   # no value recorded or the response could not be evaluated
EVALUATED = (RETURNED_DESIGN, REPAIRED_DESIGN, BINARY_DESIGN)


def constraint_verdict(value: float | None, upper: float, tol: float, source: str) -> str:
    """PASS when ``value <= upper + tol * max(|upper|, 1e-12)`` (the rule of ``Study.run``) and the value was
    evaluated on the judged design (``source`` in :data:`EVALUATED`), FAIL when it was and exceeds it, else
    UNVERIFIED."""
    if value is None or source not in EVALUATED:
        return UNVERIFIED
    return PASS if value <= upper + tol * max(abs(upper), 1e-12) else FAIL


def constraint_entry(name: str, value: float | None, upper: float, tol: float, source: str) -> dict[str, Any]:
    """``{name, value, upper, verdict, source}`` with :func:`constraint_verdict`."""
    return {"name": name, "value": value, "upper": upper,
            "verdict": constraint_verdict(value, upper, tol, source), "source": source}


def combine(verdicts: Iterable[str]) -> str:
    """FAIL if any verdict FAILs, else UNVERIFIED if any is UNVERIFIED, else PASS (also for none)."""
    v = list(verdicts)
    return FAIL if FAIL in v else UNVERIFIED if UNVERIFIED in v else PASS
