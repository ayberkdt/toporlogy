"""Responses (objectives and constraints) — docs/mimari_plan.md §2.

A response is called with an :class:`EvalContext` and returns a scalar tensor. It writes only the forward
computation; derivatives come from autograd (the linear solve has its own adjoint). ``needs`` tells the study which
state to compute: ``"u"`` = displacements of every load case.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain

__all__ = ["EvalContext", "Response", "Compliance", "Volume", "PhaseVolume", "Mass", "Cost"]


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
    # Set by the config layer on some built responses and read there with getattr(..., None): annotations only, so
    # an instance has them only when they were set.
    config_upper: float                 # default constraint bound
    config_scale: float                 # default constraint scale
    settings: dict[str, Any]            # resolved settings of the response

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
        if ctx.xphys.dim() != d.dim:
            raise ValueError("Volume needs one design field; with several phases use PhaseVolume (solid or per "
                             "phase), Mass or Cost")
        if self.weighted:
            fr = torch.as_tensor(d.frac, dtype=ctx.xphys.dtype, device=ctx.xphys.device)
            return (fr * ctx.xphys).sum() / float(d.frac.sum())
        act = torch.as_tensor(d.active, device=ctx.xphys.device)
        return ctx.xphys[act].sum() / float(d.n_active)


# ---------------------------------------------------------------------------------------------- several phases
def _fractions(ctx: EvalContext) -> tuple[Any, torch.Tensor]:
    """(interpolation, volume fractions [void, phase 1 … m]) of a multi-phase context."""
    interp = ctx.interp
    if interp is None or not hasattr(interp, "fractions"):
        raise ValueError("phase responses need a multi-phase interpolation (MultiPhaseSIMP) in the context")
    return interp, interp.fractions(ctx.xphys)


def _element_volume(ctx: EvalContext, dtype: torch.dtype) -> torch.Tensor:
    d = ctx.domain
    ve = d.frac * d.h ** d.dim * (d.thickness if d.dim == 2 else 1.0)
    return torch.as_tensor(ve, dtype=dtype, device=ctx.xphys.device)


class PhaseVolume(Response):
    """Volume fraction of some phases of a multi-phase design (:mod:`toporlogy.design.phases`).

    ``phases``: None = every solid phase (solid fraction 1 − η_void), a phase name, or a list of names. The mean is
    taken as in :class:`Volume`: over the active elements by count (keep included), or ``weighted`` by frac.
    Default name ``solid_volume``, ``volume_<phase>`` or ``volume_<a>+<b>``.
    """

    def __init__(self, phases: str | Sequence[str] | None = None, weighted: bool = False) -> None:
        self.phases = None if phases is None else ([phases] if isinstance(phases, str) else [str(p) for p in phases])
        self.weighted = bool(weighted)
        self.name = "solid_volume" if self.phases is None else "volume_" + "+".join(self.phases)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        interp, eta = _fractions(ctx)
        names = [p.name for p in interp.phases]
        if self.phases is None:
            sel = 1.0 - eta[0]
        else:
            missing = [p for p in self.phases if p not in names]
            if missing:
                raise ValueError(f"unknown phase(s) {missing}; phases are {names}")
            sel = sum(eta[1 + names.index(p)] for p in self.phases)
        d = ctx.domain
        if self.weighted:
            fr = torch.as_tensor(d.frac, dtype=sel.dtype, device=sel.device)
            return (fr * sel).sum() / float(d.frac.sum())
        act = torch.as_tensor(d.active, device=sel.device)
        return sel[act].sum() / float(d.n_active)


class Mass(Response):
    """Total mass Σ_e v_e Σ_i ρ_i η_ie (v_e = h^dim · frac_e, times the thickness in 2-D; ρ_i the phase
    densities; unpenalised fractions)."""

    name = "mass"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        interp, eta = _fractions(ctx)
        rho = torch.as_tensor([p.density for p in interp.phases], dtype=eta.dtype, device=eta.device)
        dens = torch.tensordot(rho, eta[1:], dims=1)
        return (_element_volume(ctx, eta.dtype) * dens).sum()


class Cost(Response):
    """Total cost Σ_e v_e Σ_i c_i η_ie (c_i cost per unit volume of fully dense phase i)."""

    name = "cost"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        interp, eta = _fractions(ctx)
        c = torch.as_tensor([p.cost for p in interp.phases], dtype=eta.dtype, device=eta.device)
        return (_element_volume(ctx, eta.dtype) * torch.tensordot(c, eta[1:], dims=1)).sum()
