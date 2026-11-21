"""Smooth aggregates of many values into one: the common building block of global stress, buckling, eigenvalue and
length-scale constraints, and the first place to look when deriving a new one.

Every function takes a tensor ``g`` (any shape, flattened) and returns a scalar tensor that approximates ``max(g)``.
What differs is on which side of the maximum the value lies and how it behaves when the number of entries N grows
(mesh refinement). The bounds below are elementary and are checked for random inputs in
``tests/toporlogy/test_aggregates.py``; they hold for any real ``g`` unless a sign condition is stated.

=====================  =====================================================  ==========================================
name                   definition                                             bound (m = max g)
=====================  =====================================================  ==========================================
``ks``                 m + (1/ρ) ln Σ exp(ρ (g_i − m))                        m ≤ KS ≤ m + ln(N)/ρ
``ks_lower``           KS − ln(N)/ρ                                           m − ln(N)/ρ ≤ KS_l ≤ m
``pnorm``   (g ≥ 0)    (Σ g_i^p)^(1/p)                                        m ≤ PN ≤ N^(1/p) m
``pmean``   (g ≥ 0)    ((1/N) Σ g_i^p)^(1/p)                                  N^(−1/p) m ≤ PM ≤ m
``induced_exponential``  Σ g_i exp(ρ g_i) / Σ exp(ρ g_i)                      mean(g) ≤ IE ≤ m
``induced_power`` (g ≥ 0)  Σ g_i^(p+1) / Σ g_i^p                              PM-type lower bound, ≤ m
=====================  =====================================================  ==========================================

Weighted forms: ``weights`` w_i ≥ 0 replace Σ by Σ w_i (and N by Σ w_i), e.g. element volumes, so that the value
approximates a domain integral and does not drift when the mesh is refined.

The upper-bound forms (``ks``, ``pnorm``) are conservative but their gap grows with N; the lower-bound forms never
exceed the maximum and need a correction to be used as a constraint (:class:`AdaptiveScale`). The induced forms are
the ones Kennedy & Hicken (2015) introduce; their abstract states that the classical aggregates do not converge under
mesh refinement while the induced ones do. Only the record and abstract of that paper were seen in this work
(``docs/literatur_2026-11-21.md``); the definitions here are the standard ones, written from first principles.

All functions are plain torch expressions: derivatives come from autograd. ``ks`` and ``induced_exponential`` shift by
the maximum, so large ρ does not overflow.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from toporlogy.references import cite

__all__ = ["ks", "ks_lower", "pnorm", "pmean", "induced_exponential", "induced_power", "AGGREGATES", "aggregate",
           "is_upper_bound", "AdaptiveScale"]


def _flat(g: torch.Tensor, weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    g = g.reshape(-1)
    if g.numel() == 0:
        raise ValueError("aggregate of an empty set")
    if weights is None:
        return g, None
    w = torch.as_tensor(weights, dtype=g.dtype, device=g.device).reshape(-1)
    if w.numel() != g.numel():
        raise ValueError(f"{w.numel()} weights for {g.numel()} values")
    if bool((w < 0).any()) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return g, w


@cite("kreisselmeier_steinhauser_1979", note="KS envelope function; shifted by the maximum against overflow")
def ks(g: torch.Tensor, rho: float = 50.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Kreisselmeier–Steinhauser upper bound of max(g): ``m + ln(Σ w_i exp(ρ (g_i − m))) / ρ``."""
    if not rho > 0:
        raise ValueError("rho > 0 is required")
    g, w = _flat(g, weights)
    m = g.detach().max()
    e = torch.exp(rho * (g - m))
    s = e.sum() if w is None else (w * e).sum()
    return m + torch.log(s) / rho


def ks_lower(g: torch.Tensor, rho: float = 50.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Lower-bound KS: :func:`ks` minus ``ln(N)/ρ`` (``ln(Σ w)/ρ`` with weights); equals g for constant g."""
    g, w = _flat(g, weights)
    n = float(g.numel()) if w is None else float(w.sum())
    return ks(g, rho, w) - torch.log(torch.as_tensor(n, dtype=g.dtype)) / rho


def _nonneg(g: torch.Tensor, what: str) -> torch.Tensor:
    if bool((g.detach() < 0).any()):
        raise ValueError(f"{what} needs non-negative values (min {float(g.detach().min()):.3g}); shift or use ks")
    return g


def pnorm(g: torch.Tensor, p: float = 8.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """P-norm ``(Σ w_i g_i^p)^(1/p)`` of non-negative values: upper bound of max(g) (for weights ≥ 1 on the
    maximum). Scaled by the maximum internally so that large p does not overflow."""
    if not p >= 1:
        raise ValueError("p >= 1 is required")
    g, w = _flat(g, weights)
    _nonneg(g, "pnorm")
    m = g.detach().max()
    if float(m) == 0.0:
        return g.sum() * 0.0
    r = (g / m) ** p
    s = r.sum() if w is None else (w * r).sum()
    return m * s ** (1.0 / p)


def pmean(g: torch.Tensor, p: float = 8.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """P-mean ``(Σ w_i g_i^p / Σ w_i)^(1/p)`` of non-negative values: lower bound of max(g), exact for constant g."""
    g, w = _flat(g, weights)
    n = float(g.numel()) if w is None else float(w.sum())
    return pnorm(g, p, w) / n ** (1.0 / p)


@cite("kennedy_hicken_2015", note="induced exponential aggregate; record and abstract seen, text not read")
def induced_exponential(g: torch.Tensor, rho: float = 50.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """``Σ w g exp(ρ g) / Σ w exp(ρ g)``: a weighted mean of g that tends to max(g) from below as ρ → ∞."""
    if not rho > 0:
        raise ValueError("rho > 0 is required")
    g, w = _flat(g, weights)
    e = torch.exp(rho * (g - g.detach().max()))
    if w is not None:
        e = w * e
    return (g * e).sum() / e.sum()


@cite("kennedy_hicken_2015", note="induced power aggregate; record and abstract seen, text not read")
def induced_power(g: torch.Tensor, p: float = 8.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """``Σ w g^(p+1) / Σ w g^p`` of non-negative values: tends to max(g) from below as p → ∞."""
    if not p >= 1:
        raise ValueError("p >= 1 is required")
    g, w = _flat(g, weights)
    _nonneg(g, "induced_power")
    m = g.detach().max()
    if float(m) == 0.0:
        return g.sum() * 0.0
    r = (g / m) ** p
    if w is not None:
        r = w * r
    return (g * r).sum() / r.sum()


AGGREGATES: dict[str, Callable[..., torch.Tensor]] = {
    "ks": ks, "ks_lower": ks_lower, "pnorm": pnorm, "pmean": pmean,
    "induced_exponential": induced_exponential, "induced_power": induced_power,
}
_UPPER = frozenset({"ks", "pnorm"})


def is_upper_bound(kind: str) -> bool:
    """True for the aggregates that never fall below max(g) (unweighted)."""
    return kind in _UPPER


def aggregate(g: torch.Tensor, kind: str = "ks", parameter: float | None = None,
              weights: torch.Tensor | None = None) -> torch.Tensor:
    """Aggregate ``g`` with the function called ``kind`` (:data:`AGGREGATES`) and its parameter (ρ or p; None = the
    function's default). ``"max"``: the exact maximum (gradient of the largest entry only)."""
    if kind == "max":
        return g.reshape(-1).max()
    try:
        fn = AGGREGATES[kind]
    except KeyError:
        raise ValueError(f"aggregate: one of {['max', *AGGREGATES]}, got {kind!r}") from None
    return fn(g, weights=weights) if parameter is None else fn(g, float(parameter), weights=weights)


@cite("le_2010", note="adaptive constraint scaling c_k = a (max/aggregate)_k + (1 − a) c_(k−1); record seen, "
                      "text not read")
class AdaptiveScale:
    """Correction factor that ties an aggregate to the true maximum (Le et al. 2010 "adaptive constraint scaling").

    ``c ← a · (max / aggregate) + (1 − a) · c`` at every update, starting from c = max/aggregate at the first one;
    the constraint then uses ``c · aggregate``. The factor is a constant in the derivative, so the constraint is not
    differentiable across updates: with a = 1 the scaled value equals the true maximum at the update point and the
    optimiser sees its value jump; a < 1 damps the jump (0.5 is the usual choice). ``frozen`` stops the updates
    (finite-difference checks).
    """

    def __init__(self, damping: float = 0.5) -> None:
        if not 0 < damping <= 1:
            raise ValueError("0 < damping <= 1 is required")
        self.damping = float(damping)
        self.value: float | None = None
        self.frozen = False

    def reset(self) -> None:
        self.value = None

    def update(self, true_max: float, aggregated: float) -> float:
        """New factor from the current maximum and aggregate; unchanged when frozen or the aggregate is not positive."""
        if self.frozen and self.value is not None:
            return self.value
        if not aggregated > 0 or not true_max > 0:
            self.value = 1.0 if self.value is None else self.value
            return self.value
        ratio = true_max / aggregated
        self.value = ratio if self.value is None else self.damping * ratio + (1.0 - self.damping) * self.value
        return self.value
