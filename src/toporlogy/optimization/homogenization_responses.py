"""Homogenised properties of the design as a periodic cell, as Study responses (07 §1: effective-property targets).

The whole box domain is the cell (:mod:`toporlogy.fem.homogenization`); E is the study's interpolated modulus of
x̄. The effective stiffness is computed in NumPy and enters torch through an autograd function whose backward uses
the element sensitivities dC_ij/dE_e (energy form, self-adjoint). No state solve of the study is needed. A lower
bound on a stiffness is a constraint on ``negate: true`` (−C_ij ≤ −c). Pre-registered checks:
``docs/olcumler/homojenlestirme_onkayit.md`` (H15-H17).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toporlogy.fem.homogenization import homogenize_elasticity
from toporlogy.optimization.responses import EvalContext, Response

__all__ = ["EffectiveStiffness", "EffectivePoisson", "effective_stiffness_tensor"]


class _Homogenize(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E: torch.Tensor, domain: Any, nu: float, D: Any) -> torch.Tensor:
        h = homogenize_elasticity(domain, E.detach().cpu().double().numpy(), nu, D=D, sensitivities=True)
        ctx.dC = h.dC
        ctx.dtype, ctx.device = E.dtype, E.device
        return torch.as_tensor(h.C, dtype=E.dtype, device=E.device)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, None, None, None]:
        G = g.detach().cpu().double().numpy()
        dE = np.einsum("eij,ij->e", ctx.dC, G)
        return torch.as_tensor(dE, dtype=ctx.dtype, device=ctx.device), None, None, None


def effective_stiffness_tensor(ctx: EvalContext, nu: float, D: Any = None) -> torch.Tensor:
    """C_eff (Voigt) of the design in ``ctx``, differentiable in its moduli."""
    d = ctx.domain
    act = torch.as_tensor(d.active_elements, dtype=torch.long, device=ctx.E.device)
    E = ctx.E.reshape(-1)[act]
    out: torch.Tensor = _Homogenize.apply(E, d, float(nu), D)
    return out


class EffectiveStiffness(Response):
    """C_ij of the homogenised cell (Voigt, engineering shear); ``negate`` returns −C_ij for a lower bound."""

    name = "effective_stiffness"

    def __init__(self, i: int = 0, j: int = 0, nu: float = 0.3, negate: bool = False, D: Any = None) -> None:
        self.i, self.j, self.nu, self.sign, self.D = int(i), int(j), float(nu), (-1.0 if negate else 1.0), D

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        C = effective_stiffness_tensor(ctx, self.nu, self.D)
        return self.sign * C[self.i, self.j]


class EffectivePoisson(Response):
    """Effective in-plane Poisson ratio ν12 = C12 / C11 of a 2-D cell (minimise it for an auxetic cell)."""

    name = "effective_poisson"

    def __init__(self, nu: float = 0.3, D: Any = None) -> None:
        self.nu, self.D = float(nu), D

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        if ctx.domain.dim != 2:
            raise ValueError("effective_poisson is defined for a 2-D cell (nu12 = C12 / C11)")
        C = effective_stiffness_tensor(ctx, self.nu, self.D)
        return C[0, 1] / C[0, 0]
