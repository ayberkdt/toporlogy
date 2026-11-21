"""Building new responses from existing ones — the short path from an idea to an optimisation run.

A response only writes its forward computation; autograd and the adjoint of the linear solve supply the derivative.
The classes here make the common cases one line:

* :class:`Functional` — a response from a function ``fn(ctx) -> scalar tensor``;
* :class:`WeightedSum`, :class:`Ratio`, :class:`Shifted` — algebra on responses (multi-objective scalarisation,
  normalised or specific quantities such as stiffness per mass);
* :class:`Displacement` — ``Lᵀ u`` of one load case: displacement at a point or the mean over a set, the output of a
  compliant mechanism, a deflection constraint. Its adjoint load is L (one extra solve unless L ∝ F).

``needs`` of a composite is the union of its parts, so the study solves only when a part needs the state. Every
composite must still pass :func:`toporlogy.optimization.diagnostics.gradcheck` before it is used in a study
(``tests/toporlogy/test_compose.py`` shows the pattern); the rule of ``docs/mimari_plan.md`` §2 applies to derived
responses exactly as to built-in ones.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.optimization.responses import EvalContext, Response

__all__ = ["Functional", "WeightedSum", "Ratio", "Shifted", "Displacement"]


class Functional(Response):
    """A response from ``fn(ctx) -> scalar tensor``; ``needs`` as for :class:`Response` (``{"u"}`` = state)."""

    def __init__(self, fn: Callable[[EvalContext], torch.Tensor], name: str = "functional",
                 needs: frozenset[str] | set[str] | Sequence[str] = frozenset()) -> None:
        self.fn, self.name, self.needs = fn, name, frozenset(needs)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        v = self.fn(ctx)
        if v.dim() != 0:
            raise ValueError(f"{self.name}: the function must return a scalar tensor, got shape {tuple(v.shape)}")
        return v


def _union(responses: Sequence[Response]) -> frozenset[str]:
    out: frozenset[str] = frozenset()
    for r in responses:
        out |= frozenset(r.needs)
    return out


class WeightedSum(Response):
    """``Σ_i w_i · r_i / ref_i``.

    ``references``: None (no normalisation), numbers, or ``"first"`` — each response's absolute value at the first
    evaluation after :meth:`reset_scale` (the usual way to combine quantities of different units; the result then
    depends on the start design, which is part of the method and must be reported).
    """

    def __init__(self, responses: Sequence[Response], weights: Sequence[float] | None = None,
                 references: Sequence[float] | str | None = None, name: str | None = None) -> None:
        if not responses:
            raise ValueError("WeightedSum needs at least one response")
        self.responses = list(responses)
        self.weights = [1.0] * len(self.responses) if weights is None else [float(w) for w in weights]
        if len(self.weights) != len(self.responses):
            raise ValueError(f"{len(self.weights)} weights for {len(self.responses)} responses")
        if not (references is None or references == "first" or len(references) == len(self.responses)):
            raise ValueError("references: None, 'first' or one number per response")
        self._given = references
        self.needs = _union(self.responses)
        self.name = name or "sum_" + "_".join(r.name for r in self.responses)
        self.reset_scale()

    def reset_scale(self) -> None:
        g = self._given
        self.references: list[float] | None = ([1.0] * len(self.responses) if g is None else
                                               None if g == "first" else [float(v) for v in g])

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        vals = [r(ctx) for r in self.responses]
        if self.references is None:
            self.references = [max(abs(float(v.detach())), 1e-300) for v in vals]
        total = vals[0] * (self.weights[0] / self.references[0])
        for v, w, s in zip(vals[1:], self.weights[1:], self.references[1:], strict=True):
            total = total + v * (w / s)
        return total


class Ratio(Response):
    """``numerator / denominator`` (e.g. compliance × volume, or a specific stiffness); the denominator must not
    change sign."""

    def __init__(self, numerator: Response, denominator: Response, name: str | None = None) -> None:
        self.numerator, self.denominator = numerator, denominator
        self.needs = _union([numerator, denominator])
        self.name = name or f"{numerator.name}_per_{denominator.name}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        den = self.denominator(ctx)
        if abs(float(den.detach())) < 1e-300:
            raise ValueError(f"{self.name}: the denominator is zero")
        return self.numerator(ctx) / den


class Shifted(Response):
    """``scale · response + offset``."""

    def __init__(self, response: Response, scale: float = 1.0, offset: float = 0.0, name: str | None = None) -> None:
        self.response, self.scale, self.offset = response, float(scale), float(offset)
        self.needs = frozenset(response.needs)
        self.name = name or f"{response.name}_shifted"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return self.scale * self.response(ctx) + self.offset


class Displacement(Response):
    """``Lᵀ u_c``: a weighted displacement of load case ``case``.

    Parameters
    ----------
    dofs : sequence of int
        Box DOF numbers.
    weights : sequence of float, optional
        One per DOF; default 1/len(dofs) (the mean displacement of the set).
    case : int
        Load case (column of U).

    As an objective this is the output displacement of a compliant mechanism (minimise it with the output pointing in
    the negative direction, or wrap it in ``maximize``); input and output stiffness are springs on those DOFs
    (``boundary`` support ``elastic``). As a constraint it bounds a deflection.
    """

    needs = frozenset({"u"})

    def __init__(self, dofs: Any, weights: Any = None, case: int = 0, name: str = "displacement") -> None:
        self.dofs = np.asarray(dofs, np.int64).reshape(-1)
        if self.dofs.size == 0:
            raise ValueError("Displacement needs at least one DOF")
        self.weights = (np.full(self.dofs.size, 1.0 / self.dofs.size) if weights is None
                        else np.asarray(weights, float).reshape(-1))
        if self.weights.size != self.dofs.size:
            raise ValueError(f"{self.weights.size} weights for {self.dofs.size} DOFs")
        self.case, self.name = int(case), name

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        if ctx.U is None:
            raise ValueError("Displacement needs U")
        if not 0 <= self.case < ctx.U.shape[1]:
            raise ValueError(f"load case {self.case} of {ctx.U.shape[1]}")
        if self.dofs.min() < 0 or self.dofs.max() >= ctx.U.shape[0]:
            raise ValueError("Displacement: DOF out of range")
        idx = torch.as_tensor(self.dofs, dtype=torch.long, device=ctx.U.device)
        w = torch.as_tensor(self.weights, dtype=ctx.U.dtype, device=ctx.U.device)
        return (w * ctx.U[idx, self.case]).sum()
