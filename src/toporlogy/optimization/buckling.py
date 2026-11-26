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

Preload (``preload_case``): that case's load stays fixed and only ``case`` is scaled,
``μ_i = − φ_iᵀ K_G(u1) φ_i / φ_iᵀ (K + K_G(u0)) φ_i``; both displacement fields enter through autograd.

Diagnostics of every evaluation (attributes of the responses): the low-density strain-energy share of each mode
(spurious-mode flag) and whether the last aggregated mode is cut from a cluster (one extra mode is computed for
this; :class:`~toporlogy.fem.buckling.BucklingClusterWarning`).

Reliability (:class:`BucklingReliability`, :func:`reliability`; ``docs/olcumler/burkulma_geometri_onkayit.md``).
``reliable`` is False when the aggregate or λ_1 cannot be trusted on this evaluation:

* ``cluster_cut`` — the last aggregated mode and the next one form a cluster, so the aggregate is not a symmetric
  function of the cluster and its derivative is that of one arbitrary basis vector (the one-sided derivatives differ;
  measured in ``docs/olcumler/burkulma_geometri.md``);
* ``spurious_low_density`` — mode 1 holds at least ``SPURIOUS_SHARE`` (0.5) of its strain energy in elements with
  x̄ < 0.5: a mode of grey or void material, not of the structure. On a design with no element at x̄ ≥ 0.5 (a uniform
  start x = 0.35, say) the share is 1 by definition and the flag says only that there is no solid structure yet
  (the buckling-constrained column started at x = 0.35 is flagged on its first 9 of 61 evaluations,
  ``docs/olcumler/burkulma_geometri.md`` RL4);
* ``not_buckling`` — μ_1 ≤ 0 (no positive load factor in the computed span).

``localized`` (mode 1 holds ≥ ``LOCALIZED_SHARE`` = 0.5 of its energy in 1 % of the elements) is reported but does
not lower ``reliable``: one mesh cannot tell a stress singularity from a genuinely local mode; refine and look at the
trend. ``reliable = True`` means only that no flag was raised — nothing in this module verifies a load factor. The two
thresholds are choices, not measured bounds.

Mode tracking: :class:`BucklingKS` pairs the modes of successive evaluations with
:func:`toporlogy.optimization.mode_tracking.track_modes` (K-weighted MAC on the free DOFs, clusters compared as
subspaces) and keeps the result in ``last_tracking``; a mode switch (``order`` not the identity) means the aggregated
set changed character between the two evaluations. :func:`first_failure_load_factor` gives the load
factor at which a strength criterion is first reached, to compare with λ_1 (which of the two governs).

Aggregate: ``J = s · KS(μ / s)`` with a fixed scale s (given, or the largest μ of the first evaluation after
:meth:`BucklingKS.reset_scale`), an upper bound of μ_max = 1/λ_1 with gap ≤ s · ln(n_modes)/ρ. The constraint
``λ_1 ≥ λ_req`` is ``J ≤ 1/λ_req``; it is conservative by at most that gap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import warnings

import numpy as np
import scipy.sparse as sp
import torch

from toporlogy.fem.buckling import BucklingClusterWarning, LinearBuckling
from toporlogy.fem.modal import find_clusters
from toporlogy.fem.elastic import Elasticity
from toporlogy.optimization.aggregates import ks
from toporlogy.optimization.mode_tracking import Tracking, track_modes
from toporlogy.optimization.responses import EvalContext, Response
from toporlogy.references import cite

__all__ = ["inverse_load_factors", "BucklingKS", "BucklingLoadFactor", "first_failure_load_factor",
           "BucklingReliability", "reliability", "SPURIOUS_SHARE", "LOCALIZED_SHARE"]

SPURIOUS_SHARE = 0.5         # mode-1 strain-energy share in x̄ < 0.5 that flags a spurious mode (a choice)
LOCALIZED_SHARE = 0.5        # mode-1 energy share in the 1 % most loaded elements that flags localisation (a choice)


@dataclass(frozen=True)
class BucklingReliability:
    """Flags of one buckling evaluation (module notes). ``reliable`` is False if any of ``cluster_cut``,
    ``spurious_low_density`` or ``not_buckling`` is in ``flags``; ``localized`` is reported only."""

    reliable: bool
    flags: tuple[str, ...]
    load_factor_1: float
    low_density_share_1: float
    stress_energy_share_1: float
    localization_1: float
    mu_next: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reliability(mu: np.ndarray, diag: dict[str, Any], spurious_share: float = SPURIOUS_SHARE,
                localized_share: float = LOCALIZED_SHARE) -> BucklingReliability:
    """Reliability record (module notes) from μ (descending) and the diagnostics of :func:`inverse_load_factors`."""
    mu = np.asarray(mu, float)
    lds, ses, loc = (float(np.asarray(diag[k])[0]) for k in ("low_density_share", "stress_energy_share",
                                                              "localization"))
    flags = []
    if diag["cluster_cut"]:
        flags.append("cluster_cut")
    if lds >= spurious_share:
        flags.append("spurious_low_density")
    if mu[0] <= 0:
        flags.append("not_buckling")
    if loc >= localized_share:
        flags.append("localized")
    hard = {"cluster_cut", "spurious_low_density", "not_buckling"}
    return BucklingReliability(not hard.intersection(flags), tuple(flags),
                               float(1.0 / mu[0]) if mu[0] > 0 else float("inf"), lds, ses, loc,
                               float(diag["mu_next"]))


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
                         E0: float | None = None, preload_case: int | None = None,
                         cluster_tol: float = 1e-3, density_threshold: float = 0.5,
                         exclude: np.ndarray | None = None) -> torch.Tensor:
    """μ_1 ≥ … ≥ μ_n (differentiable, module notes) of load case ``case`` (with ``preload_case`` held fixed);
    cached in the context per setting. Diagnostics go to ``ctx.extra[<key> + ":diag"]``. ``exclude``: bool mask
    (domain shape) of elements without stress stiffness (singular points; see :mod:`toporlogy.fem.buckling`)."""
    if ctx.U is None:
        raise ValueError("buckling needs U")
    key = (f"buckling_mu:{int(n_modes)}:{int(case)}:{float(penal_g)!r}:{E0!r}:{preload_case!r}:"
           f"{None if exclude is None else id(exclude)}")
    hit = ctx.extra.get(key)
    if hit is not None:
        return hit
    lb = _analysis(ctx.physics)
    m = lb.model
    if not 0 <= case < ctx.U.shape[1]:
        raise ValueError(f"load case {case} of {ctx.U.shape[1]}")
    if preload_case is not None and (not 0 <= preload_case < ctx.U.shape[1] or preload_case == case):
        raise ValueError(f"preload_case {preload_case}: another load case of {ctx.U.shape[1]}")
    e0 = float(E0) if E0 is not None else getattr(ctx.interp, "E0", None)
    if e0 is None:
        raise ValueError("buckling: give E0 (the interpolation has no attribute E0)")
    act = torch.as_tensor(m.elements, dtype=torch.long)
    E = ctx.E.reshape(-1)[act]
    frac = torch.as_tensor(m.domain.frac.ravel()[m.elements], dtype=ctx.xphys.dtype)
    EG = float(e0) * ctx.xphys.reshape(-1)[act].clamp_min(0.0) ** float(penal_g) * frac
    if exclude is not None:
        ex = np.asarray(exclude, bool)
        if ex.shape != tuple(m.domain.shape):
            raise ValueError(f"exclude has shape {ex.shape}, the domain {tuple(m.domain.shape)}")
        EG = EG * torch.as_tensor(~ex.ravel()[m.elements], dtype=EG.dtype)
    u = ctx.U[:, case]
    En, EGn = E.detach().numpy(), EG.detach().numpy()
    KG0 = None
    if preload_case is not None:
        KG0 = lb.geometric_matrix(EGn, ctx.U[:, preload_case].detach().numpy())
    k_extra = n_modes + 1 if n_modes + 2 < len(m.free) else n_modes             # one more mode: cluster check
    mu_all, modes_all = lb.modes(En, EGn, u.detach().numpy(), k_extra, KG0=KG0)
    modes = modes_all[:, :n_modes]
    cut = False
    if k_extra > n_modes and mu_all[n_modes - 1] > 0:
        cut = any(n_modes - 1 in c and n_modes in c for c in find_clusters(-mu_all, cluster_tol))  # ascending
        if cut:
            warnings.warn(f"buckling: modes {n_modes} and {n_modes + 1} form a cluster (mu {mu_all[n_modes - 1]:.6g}, "
                          f"{mu_all[n_modes]:.6g}); aggregate more modes (n_modes)", BucklingClusterWarning,
                          stacklevel=3)
    xa = ctx.xphys.reshape(-1)[act].detach().numpy()
    ctx.extra[key + ":diag"] = {"cluster_cut": cut, "mu_next": float(mu_all[-1]),
                                "low_density_share": lb.low_density_share(En, modes, xa, density_threshold),
                                "stress_energy_share": lb.stress_energy_share(EGn, u.detach().numpy(), modes, xa,
                                                                              density_threshold),
                                "localization": lb.localization(En, modes), "modes": modes}
    C, kE = lb.mode_forms(modes)
    edof = torch.as_tensor(m.edof, dtype=torch.long)
    ue = u[edof]                                                                   # (na, nloc)
    num = torch.einsum("e,ekl,el->k", EG, torch.as_tensor(C), ue)                  # φᵀ K_G φ
    den = torch.einsum("e,ek->k", E, torch.as_tensor(kE))                          # φᵀ K φ
    if preload_case is not None:
        den = den + torch.einsum("e,ekl,el->k", EG, torch.as_tensor(C), ctx.U[:, preload_case][edof])
    mu = -num / den
    ctx.extra[key] = mu
    return mu


def first_failure_load_factor(ctx: EvalContext, field: Any, case: int = 0, preload_case: int | None = None,
                              threshold: float = 0.5) -> float:
    """Smallest load factor λ_f at which a strength criterion is reached in a solid element (x̄ > ``threshold``).

    ``field`` is a :class:`~toporlogy.optimization.stress.FailureIndex` (or ``VonMises``) whose ``solid_ratio``
    gives the unrelaxed index per element and case. Without preload the index is homogeneous of degree 1 in the
    load, so λ_f = 1 / max I. With ``preload_case`` (fixed) the stress is σ0 + λ σ1 and λ_f is the smallest λ ≥ 0
    with max I(σ0 + λ σ1) = 1, found by bisection (60 halvings; the index is convex along the ray for the criteria
    of :mod:`toporlogy.optimization.strength`); 0 if the preload alone fails, inf if nothing fails. Compare with
    λ_1 of the buckling responses: the smaller governs. No gradient (a monitor).
    """
    if ctx.U is None:
        raise ValueError("first_failure_load_factor needs U")
    solid = ctx.xphys.reshape(-1).detach().numpy()[field.elements] > threshold
    if not solid.any():
        return float("inf")
    saved = field.cases

    def worst(Uc: torch.Tensor) -> float:
        sub = EvalContext(domain=ctx.domain, xphys=ctx.xphys, E=ctx.E, U=Uc[:, None], interp=ctx.interp,
                          physics=ctx.physics)
        return float(field.solid_ratio(sub)[:, 0].numpy()[solid].max())

    try:
        field.cases = None
        with torch.no_grad():
            U1 = ctx.U[:, case]
            if preload_case is None:
                mx = worst(U1)
                return float("inf") if mx <= 0 else 1.0 / mx
            U0 = ctx.U[:, preload_case]
            if worst(U0) >= 1.0:
                return 0.0
            hi = 1.0
            while worst(U0 + hi * U1) < 1.0:
                hi *= 2.0
                if hi > 1e12:
                    return float("inf")
            lo = 0.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if worst(U0 + mid * U1) >= 1.0:
                    hi = mid
                else:
                    lo = mid
            return hi
    finally:
        field.cases = saved


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
    preload_case : int, optional
        Load case held fixed while ``case`` is scaled (module notes); None = no preload.
    exclude : bool array of the domain shape, optional
        Elements without stress stiffness (around supports ends, re-entrant corners, point loads; module notes of
        :mod:`toporlogy.fem.buckling`: this moves a singular localisation, it does not remove it).

    Attributes
    ----------
    last_load_factors : numpy.ndarray
        λ_i = 1/μ_i of the last evaluation (inf for μ_i ≤ 0), ascending in λ for the positive ones.
    last_low_density_share : numpy.ndarray
        Share of each mode's strain energy in elements with x̄ < 0.5 (near 1: spurious mode in grey material).
    last_cluster_cut : bool
        The last aggregated mode and the next one form a cluster (a BucklingClusterWarning was issued).
    last_localization : numpy.ndarray
        Share of each mode's strain energy in the 1 % most loaded elements (rising with mesh refinement: the mode is
        driven by a stress singularity, λ is then a mesh property).
    last_reliability : BucklingReliability or None
        Flags of the last evaluation (module notes); ``reliable`` is False after a cluster cut or for a spurious mode.
    last_tracking : Tracking or None
        Pairing of the last evaluation's modes with the previous evaluation's (K-weighted MAC); None on the first
        evaluation, after :meth:`reset_tracking` or when the number of free DOFs changed.
    """

    needs = frozenset({"u"})

    def __init__(self, n_modes: int = 6, rho: float = 50.0, case: int = 0, penal_g: float = 3.0,
                 scale: float | None = None, E0: float | None = None, name: str = "buckling",
                 preload_case: int | None = None, exclude: np.ndarray | None = None) -> None:
        if n_modes < 1 or not rho > 0 or penal_g <= 0:
            raise ValueError("need n_modes >= 1, rho > 0 and penal_g > 0")
        self.n_modes, self.rho, self.case, self.penal_g = int(n_modes), float(rho), int(case), float(penal_g)
        self.E0, self.name = E0, name
        self.preload_case = None if preload_case is None else int(preload_case)
        self.exclude = None if exclude is None else np.asarray(exclude, bool)
        self.fixed_scale = scale
        self.scale: float | None = scale
        self.last_load_factors: np.ndarray = np.zeros(0)
        self.last_low_density_share: np.ndarray = np.zeros(0)
        self.last_cluster_cut = False
        self.last_localization: np.ndarray = np.zeros(0)
        self.last_reliability: BucklingReliability | None = None
        self.last_tracking: Tracking | None = None
        self._previous: tuple[np.ndarray, np.ndarray] | None = None

    def reset_scale(self) -> None:
        self.scale = self.fixed_scale

    def reset_tracking(self) -> None:
        self._previous = None
        self.last_tracking = None

    def _track(self, ctx: EvalContext, modes: np.ndarray, mu: np.ndarray) -> None:
        """Pair ``modes`` with the previous evaluation's (K of this evaluation on the free DOFs; module notes)."""
        m = _analysis(ctx.physics).model
        A = m._A
        assert A is not None                                              # factored by inverse_load_factors
        K = (A + A.T - sp.diags(A.diagonal())).tocsr()
        phi = np.asarray(modes, float)[m.free]
        if self._previous is not None and self._previous[0].shape == phi.shape:
            prev, prev_mu = self._previous
            self.last_tracking = track_modes(prev, phi, M=K, lam_old=-prev_mu, cluster_tol=1e-3)
        else:
            self.last_tracking = None
        self._previous = (phi, np.asarray(mu, float).copy())

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        mu = inverse_load_factors(ctx, self.n_modes, self.case, self.penal_g, self.E0, self.preload_case,
                                  exclude=self.exclude)
        diag = ctx.extra[f"buckling_mu:{self.n_modes}:{self.case}:{self.penal_g!r}:{self.E0!r}:"
                         f"{self.preload_case!r}:{None if self.exclude is None else id(self.exclude)}:diag"]
        self.last_low_density_share = np.asarray(diag["low_density_share"])
        self.last_cluster_cut = bool(diag["cluster_cut"])
        self.last_localization = np.asarray(diag["localization"])
        md = mu.detach().numpy()
        self.last_reliability = reliability(md, diag)
        self._track(ctx, diag["modes"], md)
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
                 E0: float | None = None, preload_case: int | None = None) -> None:
        if not 1 <= k <= n_modes:
            raise ValueError("need 1 <= k <= n_modes")
        self.k, self.n_modes, self.case, self.penal_g, self.E0 = int(k), int(n_modes), int(case), float(penal_g), E0
        self.preload_case = None if preload_case is None else int(preload_case)
        self.name = f"buckling_load_factor_{self.k}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        mu = inverse_load_factors(ctx, self.n_modes, self.case, self.penal_g, self.E0,
                                  self.preload_case)[self.k - 1]
        if float(mu.detach()) <= 0:
            raise ValueError(f"mode {self.k} does not buckle under this load (mu = {float(mu.detach()):.3g} <= 0)")
        return 1.0 / mu
