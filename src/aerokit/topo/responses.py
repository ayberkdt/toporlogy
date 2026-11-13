"""Responses (objectives and constraints) — docs/mimari_plan.md §3.1.

A response is called with an :class:`EvalContext` and returns a scalar tensor. It writes only the forward
computation; derivatives come from autograd (the linear solve has its own adjoint). ``needs`` tells the study which
state to compute: ``"u"`` = displacements of every load case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from aerokit.topo.domain import Domain

__all__ = ["EvalContext", "Response", "Compliance", "Volume"]


@dataclass
class EvalContext:
    """State of one evaluation: physical density and moduli on the element grid, displacements, loads."""

    domain: Domain
    xphys: torch.Tensor                     # (shape) physical density
    E: torch.Tensor                         # (shape) element modulus incl. frac
    U: torch.Tensor | None = None           # (ndof, n_cases)
    F: np.ndarray | None = None             # (ndof, n_cases)
    extra: dict[str, Any] = field(default_factory=dict)
    x: torch.Tensor | None = None           # design vector (leaf of the autograd graph)
    fields: list[torch.Tensor] = field(default_factory=list)   # output of every chain link, in order
    physics: Any = None                     # the study's physics (responses may run extra solves with it)
    interp: Any = None                      # the study's interpolation x̄ → E
    chain: Any = None                       # the study's chain (links are reachable for robust realisations)

    def field_after(self, kind: type) -> torch.Tensor:
        """Output of the first chain link of type ``kind`` (e.g. the filtered field x̃ after ``DensityFilter``)."""
        if self.chain is None:
            raise ValueError("context has no chain")
        for link, f in zip(self.chain.links, self.fields, strict=True):
            if isinstance(link, kind):
                return f
        raise KeyError(f"no {kind.__name__} link in the chain")


class Response:
    """Base class: ``name``, ``needs`` and ``__call__(ctx) -> scalar tensor``."""

    name = "response"
    needs: frozenset[str] = frozenset()

    def __call__(self, ctx: EvalContext) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class Compliance(Response):
    """``Σ_c w_c F_cᵀ U_c``; weights default 1 (normalise with ``weights = 1 / c0`` for several cases)."""

    name = "compliance"
    needs = frozenset({"u"})

    def __init__(self, weights: Any = None) -> None:
        self.weights = None if weights is None else np.asarray(weights, float).reshape(-1)

    def per_case(self, ctx: EvalContext) -> torch.Tensor:
        if ctx.U is None or ctx.F is None:
            raise ValueError("Compliance needs U and F")
        F = torch.as_tensor(ctx.F, dtype=ctx.U.dtype, device=ctx.U.device)
        return (F * ctx.U).sum(dim=0)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        c = self.per_case(ctx)
        if self.weights is None:
            return c.sum()
        w = torch.as_tensor(self.weights, dtype=c.dtype, device=c.device)
        if w.numel() != c.numel():
            raise ValueError(f"{w.numel()} weights for {c.numel()} load cases")
        return (w * c).sum()


class Volume(Response):
    """Mean physical density over the part.

    ``weighted=False`` (default): element count, keep elements included — the rule of
    the legacy voxel driver (``Σ x̄ ≤ V* n``). ``weighted=True``: ``Σ frac x̄ / Σ frac`` (true solid
    volume fraction for partial voxels).
    """

    name = "volume"

    def __init__(self, weighted: bool = False) -> None:
        self.weighted = bool(weighted)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        d = ctx.domain
        if self.weighted:
            fr = torch.as_tensor(d.frac, dtype=ctx.xphys.dtype, device=ctx.xphys.device)
            return (fr * ctx.xphys).sum() / float(d.frac.sum())
        act = torch.as_tensor(d.active, device=ctx.xphys.device)
        return ctx.xphys[act].sum() / float(d.n_active)
