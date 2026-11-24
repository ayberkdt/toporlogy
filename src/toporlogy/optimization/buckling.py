"""Linear buckling as a response: differentiable buckling load factors without a hand-written adjoint.

With the modes of :class:`toporlogy.fem.buckling.LinearBuckling` (K-normalised, computed numerically and held fixed)
the inverse load factors are Rayleigh quotients,

    μ_i(x) = − φ_iᵀ K_G(E_G(x̄), u(x)) φ_i  /  φ_iᵀ K(E(x̄)) φ_i ,          λ_i = 1/μ_i ,

and the derivative of a Rayleigh quotient at its eigenvector, with the vector held fixed, IS the derivative of the
(simple) eigenvalue. Both quadratic forms are written as torch expressions in E, E_G and u, so autograd gives

    dμ_i/dx = ∂μ_i/∂x̄ · dx̄/dx  +  (∂μ_i/∂u) · du/dx ,

and the second term — the dependence of the stress stiffness on the displacements, which costs an adjoint solve and
whose omission is reported to give wrong sensitivities (Ferrari & Sigmund 2019) — comes from the adjoint of the
linear solve the library already has (:class:`toporlogy.fem.elastic.Elasticity`). One adjoint right-hand side per
response evaluation, whatever the number of modes, because the aggregate is one scalar.

Repeated eigenvalues: a single μ_i is then not differentiable, but a symmetric function of a whole cluster is, and
its derivative is the sum of the fixed-vector Rayleigh-quotient derivatives over any basis of the cluster. The KS
aggregate over all computed modes is symmetric, so it is safe as long as the cluster is not cut by ``n_modes``
(compute a few more modes than are expected to matter; the open question of how many is discussed in the
literature notes).

Interpolation: stiffness from the study's interpolation; stress modulus ``E_G = E0 · x̄^penal_g · frac`` without a
lower bound, so that void elements have no stress stiffness and cannot produce spurious modes (module notes of
:mod:`toporlogy.fem.buckling`).

Aggregate: ``J = s · KS(μ / s)`` with a fixed scale s (given, or the largest μ of the first evaluation after
:meth:`BucklingKS.reset_scale`), an upper bound of μ_max = 1/λ_1 with gap ≤ s · ln(n_modes)/ρ. The constraint
``λ_1 ≥ λ_req`` is ``J ≤ 1/λ_req``; it is conservative by at most that gap.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toporlogy.fem.buckling import LinearBuckling
from toporlogy.fem.elastic import Elasticity
from toporlogy.optimization.aggregates import ks
from toporlogy.optimization.responses import EvalContext, Response
from toporlogy.references import cite

__all__ = ["inverse_load_factors", "BucklingKS", "BucklingLoadFactor"]


def _analysis(physics: Any) -> LinearBuckling:
    lb = getattr(physics, "_linear_buckling", None)
    if lb is None:
        if not isinstance(physics, Elasticity):
            raise TypeError("buckling responses need plain Elasticity physics (direct solver, axis-aligned fixed "
                            f"DOFs, one support set); got {type(physics).__name__}")
        lb = LinearBuckling(physics)
        physics._linear_buckling = lb          # type: ignore[attr-defined]
    return lb


def inverse_load_factors(ctx: EvalContext, n_modes: int = 6, case: int = 0, penal_g: float = 3.0,
                         E0: float | None = None) -> torch.Tensor:
    """μ_1 ≥ … ≥ μ_n (differentiable, module notes) of load case ``case``; cached in the context per setting."""
    if ctx.U is None:
        raise ValueError("buckling needs U")
    key = f"buckling_mu:{int(n_modes)}:{int(case)}:{float(penal_g)!r}:{E0!r}"
    hit = ctx.extra.get(key)
    if hit is not None:
        return hit
    lb = _analysis(ctx.physics)
    m = lb.model
    if not 0 <= case < ctx.U.shape[1]:
        raise ValueError(f"load case {case} of {ctx.U.shape[1]}")
    e0 = float(E0) if E0 is not None else getattr(ctx.interp, "E0", None)
    if e0 is None:
        raise ValueError("buckling: give E0 (the interpolation has no attribute E0)")
    act = torch.as_tensor(m.elements, dtype=torch.long)
    E = ctx.E.reshape(-1)[act]
    frac = torch.as_tensor(m.domain.frac.ravel()[m.elements], dtype=ctx.xphys.dtype)
    EG = float(e0) * ctx.xphys.reshape(-1)[act].clamp_min(0.0) ** float(penal_g) * frac
    u = ctx.U[:, case]
    _, modes = lb.modes(E.detach().numpy(), EG.detach().numpy(), u.detach().numpy(), n_modes)
    C, kE = lb.mode_forms(modes)
    ue = u[torch.as_tensor(m.edof, dtype=torch.long)]                              # (na, nloc)
    num = torch.einsum("e,ekl,el->k", EG, torch.as_tensor(C), ue)                  # φᵀ K_G φ
    den = torch.einsum("e,ek->k", E, torch.as_tensor(kE))                          # φᵀ K φ (= 1 at this design)
    mu = -num / den
    ctx.extra[key] = mu
    return mu


@cite("ferrari_sigmund_2019", "ferrari_sigmund_guest_2021",
      note="KS aggregate of μ = 1/λ and stress interpolation without E_min as reported (records seen, texts not "
           "read); the sensitivity is obtained by autograd through the Rayleigh quotient and the solve's adjoint")
class BucklingKS(Response):
    """Smooth upper bound of ``1/λ_1`` (module notes); constrain it by ``1/λ_required`` or minimise it.

    Parameters
    ----------
    n_modes : int
        Modes aggregated.
    rho : float
        KS parameter.
    case : int
        Load case whose load is the reference load.
    penal_g : float
        Exponent of the stress-modulus interpolation.
    scale : float, optional
        Fixed KS scale s; None = the largest μ of the first evaluation.

    Attributes
    ----------
    last_load_factors : numpy.ndarray
        λ_i = 1/μ_i of the last evaluation (inf for μ_i ≤ 0), ascending in λ for the positive ones.
    """

    needs = frozenset({"u"})

    def __init__(self, n_modes: int = 6, rho: float = 50.0, case: int = 0, penal_g: float = 3.0,
                 scale: float | None = None, E0: float | None = None, name: str = "buckling") -> None:
        if n_modes < 1 or not rho > 0 or penal_g <= 0:
            raise ValueError("need n_modes >= 1, rho > 0 and penal_g > 0")
        self.n_modes, self.rho, self.case, self.penal_g = int(n_modes), float(rho), int(case), float(penal_g)
        self.E0, self.name = E0, name
        self.fixed_scale = scale
        self.scale: float | None = scale
        self.last_load_factors: np.ndarray = np.zeros(0)

    def reset_scale(self) -> None:
        self.scale = self.fixed_scale

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        mu = inverse_load_factors(ctx, self.n_modes, self.case, self.penal_g, self.E0)
        md = mu.detach().numpy()
        with np.errstate(divide="ignore"):
            self.last_load_factors = np.where(md > 0, 1.0 / np.where(md > 0, md, 1.0), np.inf)
        if self.scale is None:
            self.scale = max(float(np.abs(md).max()), 1e-300)
        s = self.scale
        return s * ks(mu / s, self.rho)


class BucklingLoadFactor(Response):
    """λ_k = 1/μ_k (k = 1 is the critical one). A monitor: differentiable only while λ_k is simple, and undefined
    when the mode does not buckle (μ_k ≤ 0 raises)."""

    needs = frozenset({"u"})

    def __init__(self, k: int = 1, n_modes: int = 6, case: int = 0, penal_g: float = 3.0,
                 E0: float | None = None) -> None:
        if not 1 <= k <= n_modes:
            raise ValueError("need 1 <= k <= n_modes")
        self.k, self.n_modes, self.case, self.penal_g, self.E0 = int(k), int(n_modes), int(case), float(penal_g), E0
        self.name = f"buckling_load_factor_{self.k}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        mu = inverse_load_factors(ctx, self.n_modes, self.case, self.penal_g, self.E0)[self.k - 1]
        if float(mu.detach()) <= 0:
            raise ValueError(f"mode {self.k} does not buckle under this load (mu = {float(mu.detach()):.3g} <= 0)")
        return 1.0 / mu
