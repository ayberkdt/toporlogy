"""End compliance of geometrically non-linear physics (:class:`toporlogy.fem.nonlinear.NonlinearElasticity`).

End compliance is the work of the dead load F on the displacement u(F) of the converged equilibrium at the FULL load,
``C = Σ_c w_c F_cᵀ u_c`` — the objective of stiffness design under large displacements (Buhl, Pedersen & Sigmund
2000 [H: not read]). Unlike linear compliance it is not the strain energy, and its gradient needs one adjoint solve
with the converged tangent, K_T λ = F, per load case (``NonlinearElasticity.solve`` does this in its backward
pass). The study, chain and optimizers are used unchanged: the physics has the ``solve(E, F)`` interface the study
expects.

The plain ``compliance`` response computes the same number from the same U; :class:`EndCompliance` differs only by
refusing linear physics, so a configuration says which quantity it optimises.

:class:`EndStrainEnergy` is the stored energy at the converged equilibrium, W(u*(E), E) = Σ_e E_e [ψ_NL(γu) +
(1 − γ²) ψ_L(u)] (energy interpolation included). Its gradient has an explicit design part and an implicit one:
∂W/∂u = f_int = F at equilibrium, so the implicit part needs the adjoint K_T λ = f_int (the solve's backward), and
the explicit part is ∂W_e/∂E_e at fixed u (:meth:`NonlinearElasticity.element_energies`). For a hyperelastic model
under a dead load W ≠ ½ Fᵀu in general (they coincide for linear elasticity). Valid for the cutoff of the solve:
across an adaptive cutoff change the response is discontinuous.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toporlogy.optimization.responses import Compliance, EvalContext, Response

__all__ = ["EndCompliance", "EndStrainEnergy"]


class EndCompliance(Compliance):
    """``Σ_c w_c F_cᵀ u_c`` at the converged non-linear equilibrium under the full load (weights default 1).

    Raises ``ValueError`` when the study's physics is not :class:`~toporlogy.fem.nonlinear.NonlinearElasticity`.
    """

    name = "end_compliance"

    def __init__(self, weights: Any = None) -> None:
        super().__init__(weights)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        from toporlogy.fem.nonlinear import NonlinearElasticity
        if not isinstance(ctx.physics, NonlinearElasticity):
            raise ValueError("end_compliance needs NonlinearElasticity physics (physics nonlinear_elasticity); for "
                             "linear physics use compliance")
        return super().__call__(ctx)


class EndStrainEnergy(Response):
    """``Σ_c w_c W(u_c, E)``: the stored energy of the converged non-linear equilibrium under the full load (module
    notes; weights default 1). Hyperelastic models only."""

    name = "end_strain_energy"
    needs = frozenset({"u"})

    def __init__(self, weights: Any = None) -> None:
        self.weights = None if weights is None else np.asarray(weights, float).reshape(-1)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        from toporlogy.fem.nonlinear import NonlinearElasticity
        phys = ctx.physics
        if not isinstance(phys, NonlinearElasticity) or phys.stateful:
            raise ValueError("end_strain_energy needs NonlinearElasticity physics with a hyperelastic material")
        if ctx.U is None:
            raise ValueError("end_strain_energy needs U")
        act = torch.as_tensor(ctx.domain.active_elements, dtype=torch.long, device=ctx.E.device)
        E = ctx.E.reshape(-1)[act]
        n = ctx.U.shape[1]
        w = np.ones(n) if self.weights is None else self.weights
        if len(w) != n:
            raise ValueError(f"{len(w)} weights for {n} load cases")
        return _StrainEnergy.apply(E, ctx.U, phys, w, phys.cutoffs())


class _StrainEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E: torch.Tensor, U: torch.Tensor, phys: Any, w: np.ndarray,
                cutoffs: list[float | None]) -> torch.Tensor:
        En = E.detach().cpu().double().numpy()
        Un = U.detach().cpu().double().numpy()
        total, gU, gE = 0.0, np.zeros_like(Un), np.zeros_like(En)
        for c in range(Un.shape[1]):
            cut = cutoffs[c] if c < len(cutoffs) else None
            We, dW = phys.element_energies(Un[:, c], En, cut)
            total += w[c] * float(We.sum())
            gU[:, c] = w[c] * phys.internal_force(Un[:, c], En, cut)
            gE += w[c] * dW
        ctx.gU, ctx.gE, ctx.dev, ctx.dtype = gU, gE, E.device, E.dtype
        return torch.tensor(total, dtype=torch.float64, device=E.device)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, Any, None, None, None]:
        s = float(g)
        return (torch.from_numpy(s * ctx.gE).to(device=ctx.dev, dtype=ctx.dtype),
                torch.from_numpy(s * ctx.gU).to(device=ctx.dev, dtype=torch.float64), None, None, None)
