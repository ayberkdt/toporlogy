"""Eigenfrequency topology optimisation: interpolations for dynamics and eigenvalue responses.

Physics: :class:`toporlogy.fem.modal.Modal` (consistent mass, shift-invert Lanczos, differentiable λ). The study's
``physics`` is the :class:`Modal` object; its ``mass`` module maps x̄ → ρ, the study's ``interp`` maps x̄ → E.
Responses fetch the eigenvalues through :func:`eigenvalues` (computed once per evaluation and cached in
``ctx.extra``). Pass ``loads = domain.load_vector()`` (zeros) to the :class:`~toporlogy.study.Study`; the
responses declare ``needs = {"modes"}``, so no static solve is run.

Interpolations
--------------
With E ∝ x̄^p and ρ ∝ x̄ the local ratio E/ρ ∝ x̄^(p−1) → 0 in low-density regions, so plain SIMP produces
spurious localised modes there whose eigenvalues fall below the structural ones. Two remedies are offered, both
two-range laws, continuous (C0, not C1) at the threshold ``x_t``:

* :class:`TwoRangeSIMP` (stiffness): ``x̄^p`` above ``x_t``, ``x_t^(p−q) x̄^q`` below (q = 1: E/ρ stays at
  x_t^(p−1) E0/ρ0 in the void). This is the "modified SIMP with a lower power below a threshold" usually attributed
  to Du & Olhoff (2007) — primary source NOT read in this session.
* :class:`TwoRangeMass` (mass): ``x̄`` above ``x_t``, ``x̄^q / x_t^(q−1)`` below (q = 6). Attributed to Pedersen
  (2000), primary source NOT read in this session; the formula with x_t = 0.1, q = 6 was read in the pymoto
  documentation (example "Eigenfrequency maximization", which credits Du & Olhoff 2007).

Do not evaluate gradients exactly at x̄ = x_t (one-sided derivatives differ there).

Aggregation
-----------
``MinEigenvalueKS`` is the Kreisselmeier–Steinhauser lower bound of min λ_i over the lowest ``n_modes`` (KS
source NOT read in this session):

    KS = −(s/ρ) · log Σ_i exp(−ρ λ_i / s)  ≤  min λ_i,     min λ_i − s·log(n)/ρ ≤ KS ,

with a fixed scale ``s`` (default λ_1 of the first evaluation, frozen so the response is a function of x only).
It is a smooth symmetric function of the eigenvalues, so it stays differentiable when eigenvalues coalesce,
*provided every mode of a cluster is included* (the last mode must not be cut out of a cluster; the eigen backward
warns). Limits: KS is a lower bound with error ≤ s·log(n)/ρ; large ρ approaches the non-smooth min (ill-conditioned
for the optimizer), small ρ under-estimates λ_1; it treats repeated eigenvalues only through smoothing, not through
the exact sub-gradient / bound formulation for multiple eigenvalues. ``kind = "pnorm"`` gives
``(Σ (λ_i/s)^(−P))^(−1/P) · s`` with P = ρ (also a lower bound of the min).
"""
from __future__ import annotations

import math
from typing import Any

import torch

from toporlogy.domain import Domain
from toporlogy.fem.modal import Modal
from toporlogy.responses import EvalContext, Response
from toporlogy.study import Constraint

__all__ = ["LinearMass", "TwoRangeMass", "TwoRangeSIMP", "eigenvalues", "Eigenvalue", "Frequency",
           "MinEigenvalueKS", "EigenvalueGap", "FrequencyMargin", "min_frequency_constraint", "Negated",
           "omega_squared_factor"]


# ============================================================================================ interpolations
class LinearMass(torch.nn.Module):
    """``ρ_e = ρ0 (ρmin + (1 − ρmin) x̄_e) · frac_e`` with ``ρmin = rhomin_ratio``."""

    def __init__(self, domain: Domain, rho0: float = 1.0, rhomin_ratio: float = 0.0) -> None:
        super().__init__()
        self.rho0, self.rhomin_ratio = float(rho0), float(rhomin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        r = self.rhomin_ratio
        return self.rho0 * (r + (1.0 - r) * xphys) * self.frac.to(xphys.dtype)


class TwoRangeMass(torch.nn.Module):
    """``ρ = ρ0 · frac · m(x̄)``, ``m = x̄`` for x̄ ≥ x_t, ``x̄^q / x_t^(q−1)`` below (C0 at x_t; module notes)."""

    def __init__(self, domain: Domain, rho0: float = 1.0, threshold: float = 0.1, q: float = 6.0) -> None:
        super().__init__()
        if not 0 < threshold < 1:
            raise ValueError("0 < threshold < 1 is required")
        self.rho0, self.threshold, self.q = float(rho0), float(threshold), float(q)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        t, q = self.threshold, self.q
        low = xphys.clamp(min=0.0) ** q / t ** (q - 1.0)
        m = torch.where(xphys >= t, xphys, low)
        return self.rho0 * m * self.frac.to(xphys.dtype)


class TwoRangeSIMP(torch.nn.Module):
    """``E = (Emin + s(x̄)(E0 − Emin)) · frac``, ``s = x̄^p`` for x̄ ≥ x_t, ``x_t^(p−q) x̄^q`` below (module notes).

    ``penal`` may be scheduled like :class:`toporlogy.interp.SIMP`.
    """

    def __init__(self, domain: Domain, E0: float = 1.0, penal: float = 3.0, threshold: float = 0.1,
                 q: float = 1.0, emin_ratio: float = 1e-9) -> None:
        super().__init__()
        if not 0 < threshold < 1:
            raise ValueError("0 < threshold < 1 is required")
        self.E0, self.penal, self.threshold, self.q = float(E0), float(penal), float(threshold), float(q)
        self.emin_ratio = float(emin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        t, p, q = self.threshold, self.penal, self.q
        xc = xphys.clamp(min=0.0)
        s = torch.where(xphys >= t, xc ** p, t ** (p - q) * xc ** q)
        Emin = self.emin_ratio * self.E0
        return (Emin + s * (self.E0 - Emin)) * self.frac.to(xphys.dtype)


# ============================================================================================ eigenvalue access
def eigenvalues(ctx: EvalContext, n: int) -> torch.Tensor:
    """Lowest ``n`` eigenvalues of the evaluation (differentiable), computed once per context and cached.

    ``ctx.physics`` must be a :class:`Modal`. At least ``physics.n_modes`` modes are computed.
    """
    model = ctx.physics
    if not isinstance(model, Modal):
        raise TypeError("eigenvalue responses need a Modal physics object")
    cached = ctx.extra.get("eigenvalues")
    if cached is not None and cached.numel() >= n:
        return cached[:n]
    k = max(int(n), model.n_modes)
    act = torch.as_tensor(model.elements, dtype=torch.long, device=ctx.E.device)
    rho = model.density(ctx.xphys)
    lam = model.eigenvalues(ctx.E.reshape(-1)[act], rho.reshape(-1)[act], k)
    ctx.extra["eigenvalues"], ctx.extra["rho"] = lam, rho
    return lam[:n]


def omega_squared_factor(E_unit: float = 1.0, rho_unit: float = 1.0, length_unit: float = 1.0) -> float:
    """Factor c with ω² [rad²/s²] = c · λ, when the model's E, ρ and lengths are in the given SI multiples."""
    return float(E_unit) / (float(rho_unit) * float(length_unit) ** 2)


class _ModalResponse(Response):
    needs = frozenset({"modes"})


class Eigenvalue(_ModalResponse):
    """λ_k (k = 1 is the fundamental), in model units. Only differentiable while λ_k is simple."""

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError("k ≥ 1 (1-based mode number)")
        self.k = int(k)
        self.name = f"eigenvalue_{self.k}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return eigenvalues(ctx, self.k)[self.k - 1]


class Frequency(_ModalResponse):
    """f_k = √(c λ_k) / (2π) in Hz, c = :func:`omega_squared_factor` (E, ρ, length units of the model in SI)."""

    def __init__(self, k: int = 1, E_unit: float = 1.0, rho_unit: float = 1.0, length_unit: float = 1.0) -> None:
        if k < 1:
            raise ValueError("k ≥ 1 (1-based mode number)")
        self.k = int(k)
        self.c = omega_squared_factor(E_unit, rho_unit, length_unit)
        self.name = f"frequency_{self.k}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        lam = eigenvalues(ctx, self.k)[self.k - 1]
        return torch.sqrt(self.c * lam) / (2.0 * math.pi)


class MinEigenvalueKS(_ModalResponse):
    """Smooth lower bound of min(λ_1 … λ_n) (KS or p-norm, module notes), in model units.

    Parameters
    ----------
    n_modes : int
        Number of lowest eigenvalues aggregated (include every mode of a cluster).
    rho : float
        Aggregation parameter (KS ρ, or P for ``kind = "pnorm"``).
    scale : float, optional
        Fixed eigenvalue scale s; default λ_1 at the first evaluation (then frozen; :meth:`reset_scale`).
    kind : {"ks", "pnorm"}
    """

    def __init__(self, n_modes: int = 3, rho: float = 50.0, scale: float | None = None, kind: str = "ks") -> None:
        if n_modes < 1:
            raise ValueError("n_modes ≥ 1")
        if kind not in ("ks", "pnorm"):
            raise ValueError("kind: ks | pnorm")
        if not rho > 0:
            raise ValueError("rho > 0")
        self.n_modes, self.rho, self.scale, self.kind = int(n_modes), float(rho), scale, kind
        self.name = f"min_eigenvalue_{kind}"

    def reset_scale(self) -> None:
        self.scale = None

    def aggregate(self, lam: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            self.scale = float(lam.detach()[0])
        s = self.scale
        mu = lam / s
        if self.kind == "ks":
            return -(s / self.rho) * torch.logsumexp(-self.rho * mu, dim=0)
        return s * (mu ** (-self.rho)).sum() ** (-1.0 / self.rho)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return self.aggregate(eigenvalues(ctx, self.n_modes))


class EigenvalueGap(_ModalResponse):
    """(λ_{k+1} − λ_k) / scale (scale default 1). Maximise it with :class:`Negated`; needs simple λ_k, λ_{k+1}."""

    def __init__(self, k: int = 1, scale: float = 1.0) -> None:
        if k < 1:
            raise ValueError("k ≥ 1")
        self.k, self.scale = int(k), float(scale)
        self.name = f"eigenvalue_gap_{self.k}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        lam = eigenvalues(ctx, self.k + 1)
        return (lam[self.k] - lam[self.k - 1]) / self.scale


class FrequencyMargin(_ModalResponse):
    """``1 − λ_low / λ_min`` with λ_min = (2π f_min)² / c; ≤ 0 ⇔ f_low ≥ f_min.

    λ_low is λ_1 for ``n_modes = 1`` and the KS lower bound of the lowest ``n_modes`` otherwise (scale λ_min, so the
    bound is conservative: KS ≥ λ_min ⇒ every λ_i ≥ λ_min).
    """

    def __init__(self, f_min: float, n_modes: int = 1, rho: float = 50.0, E_unit: float = 1.0,
                 rho_unit: float = 1.0, length_unit: float = 1.0) -> None:
        if not f_min > 0:
            raise ValueError("f_min > 0")
        self.f_min = float(f_min)
        self.lam_min = (2.0 * math.pi * self.f_min) ** 2 / omega_squared_factor(E_unit, rho_unit, length_unit)
        self.n_modes = int(n_modes)
        self._ks = MinEigenvalueKS(self.n_modes, rho, scale=self.lam_min) if self.n_modes > 1 else None
        self.name = "frequency_margin"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        lam = eigenvalues(ctx, self.n_modes)
        low = lam[0] if self._ks is None else self._ks.aggregate(lam)
        return 1.0 - low / self.lam_min


def min_frequency_constraint(f_min: float, n_modes: int = 1, rho: float = 50.0, E_unit: float = 1.0,
                             rho_unit: float = 1.0, length_unit: float = 1.0) -> Constraint:
    """``f_1 ≥ f_min`` as a study constraint ``g = 1 − λ_low/λ_min ≤ 0`` (:class:`FrequencyMargin`)."""
    return Constraint(FrequencyMargin(f_min, n_modes, rho, E_unit, rho_unit, length_unit), upper=0.0, scale=1.0)


class Negated(Response):
    """``−response`` (to maximise with a minimising study); keeps the wrapped response's ``needs``."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.needs = frozenset(getattr(response, "needs", frozenset()))
        self.name = f"neg_{getattr(response, 'name', 'response')}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return -self.response(ctx)
