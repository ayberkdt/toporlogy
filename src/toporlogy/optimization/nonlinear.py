"""End compliance of geometrically non-linear physics (:class:`toporlogy.fem.nonlinear.NonlinearElasticity`).

End compliance is the work of the dead load F on the displacement u(F) of the converged equilibrium at the FULL load,
``C = Σ_c w_c F_cᵀ u_c`` — the objective of stiffness design under large displacements (Buhl, Pedersen & Sigmund
2000 [H: not read]). Unlike linear compliance it is not the strain energy, and its gradient needs one adjoint solve
with the converged tangent, K_T λ = F, per load case (``NonlinearElasticity.solve`` does this in its backward
pass). The study, chain and optimizers are used unchanged: the physics has the ``solve(E, F)`` interface the study
expects.

The plain ``compliance`` response computes the same number from the same U; :class:`EndCompliance` differs only by
refusing linear physics, so a configuration says which quantity it optimises.
"""
from __future__ import annotations

from typing import Any

import torch

from toporlogy.optimization.responses import Compliance, EvalContext

__all__ = ["EndCompliance"]


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
