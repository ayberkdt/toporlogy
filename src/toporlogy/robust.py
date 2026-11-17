"""Robust (eroded / intermediate / dilated) formulation: minimum solid AND void size in 2-D and 3-D.

Realisations. From the filtered field x̃ (output of the chain's :class:`~toporlogy.chain.DensityFilter`, taken from
:meth:`EvalContext.field_after`) three projected fields are built with the tanh projection
(Ferrari & Sigmund 2020 Eq. 1; Qian & Sigmund 2013 Eq. 25)::

    x̂_k = (tanh(βη_k) + tanh(β(x̃ − η_k))) / (tanh(βη_k) + tanh(β(1 − η_k))),   k ∈ {eroded, intermediate, dilated}

with η_eroded = η_e > η_i > η_dilated = η_d (default η_e = 0.75, η_i = 0.5, η_d = 1 − η_e), then the passive rule
(keep = 1, void / outside = 0), the study's interpolation and one linear solve each (differentiable through the
autograd-aware solve). The intermediate (blueprint) realisation is the design to manufacture.

Objective = worst case over the realisations, either aggregated (:class:`Robust` / :class:`RobustCompliance`:
KS, p-norm or max) or exactly by the bound formulation min z s.t. J_k/J_ref ≤ z (:func:`bound_formulation`,
needs MMA). The plain max over eroded and dilated is the form of Lazarov, Wang & Sigmund (2016) Eq. (37), as used by
the validated 2-D driver :func:`aerokit.optimization.projection.topopt_projected` (registry note: DTU postprint
read; not re-read in this session). The KS function J = (s/ρ) log Σ_k exp(ρ J_k / s) is the standard
Kreisselmeier-Steinhauser aggregate (original 1979 paper not read; not in the registry).

Volume. The constraint is put on the dilated realisation (:class:`DilatedVolume`; Qian & Sigmund p. 8 as recorded in
:mod:`aerokit.optimization.projection`). Its bound is updated every few iterations as V*_d = V* · V_d / V_i so the
blueprint keeps the target volume V* (:class:`RobustContinuation`; Wang, Lazarov & Sigmund 2011, SMO 43:767-784 —
not read in this session, not in the registry).

Minimum size (Qian & Sigmund 2013 Eq. 29-30, ported unchanged from :mod:`aerokit.optimization.projection`, whose notes
say the author preprint was read on 2026-11-03; NOT re-read in this session). For a 1-D hat filter of radius R and
η_i = ½, the blueprint has minimum solid width b with

    b/R = 2√(η_e − ½)            for η_e ∈ [½, ¾],
    b/R = 2 − 2√(1 − η_e)        for η_e ∈ [¾, 1]         (η_e = ¾ → b = R),

and, by the symmetry x → 1 − x of the projection, minimum void width b_v with the same formula in 1 − η_d. Derivation
(reproduced numerically by :func:`blueprint_size` and checked in the tests): the thinnest eroded member is a bar whose
filtered peak just reaches η_e; b is the width of {x̃ ≥ ½} of that bar. In 2-D / 3-D the thinnest feature is a straight
bar / plate, whose cross-section sees the filter kernel summed over the other axes — not the 1-D hat. The default of
:func:`robust_chain_settings` therefore repeats the same derivation numerically with that summed profile of the actual
discrete cone kernel of :class:`DensityFilter` (``kernel="discrete"``); ``kernel="hat"`` uses Eq. 29-30.

Corners: the blueprint is (approximately) the eroded design dilated by a ball of radius b/2, so its convex corners are
rounded with radius ≥ b/2, and its re-entrant corners (convex corners of the void) with radius ≥ b_v/2. This is what
:func:`toporlogy.lengthscale.measure_lengthscale` measures: ``solid_min`` ≥ b and ``void_min`` ≥ b_v cover members,
gaps and corners together.

Caveats (well known, also seen in the tests): the guarantee needs the eroded member to exist, i.e. it holds for members
that carry load; at box faces the filter normalisation acts like a symmetry plane, so members along a free face may be
thinner than b; it is a nominal size, ± one element on the grid. Measured (half MBB 90 × 30, b = 4, β 32, 300 OC
iterations; ``measure_lengthscale(..., smooth=0.5, margin=b)``): solid_min 4.64, void_min 2.69 — the void minimum sits
in a few re-entrant corners where members join (37 of 18 818 fine void cells below b − 1), i.e. those corners are
rounded with radius ≈ 1.35 h instead of b/2 = 2 h. Verify a design with ``smooth ≈ 0.5`` (removes staircase teeth)
and a ``margin`` at free faces.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.references import cite
from toporlogy.design.chain import Chain, DensityFilter, HeavisideProjection, Passive
from toporlogy.optim.schedule import Schedule
from toporlogy.responses import Compliance, EvalContext, Response, Volume
from toporlogy.study import Constraint

__all__ = ["project", "min_size_ratio", "eta_for_size", "filter_profile", "blueprint_size", "robust_chain_settings",
           "RobustSet", "Robust", "RobustCompliance", "RealisationVolume", "DilatedVolume", "RobustContinuation",
           "BoundSplit", "BoundObjective", "RealisationBound", "bound_formulation"]

REALISATIONS = ("eroded", "intermediate", "dilated")


@cite("ferrari_sigmund_2020", "qian_sigmund_2013", note="Eq. (1) / Eq. (25)")
def project(xt: torch.Tensor, beta: float, eta: float) -> torch.Tensor:
    """Tanh projection of the filtered field (same formula as :class:`~toporlogy.chain.HeavisideProjection`)."""
    a, d = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return (a + torch.tanh(beta * (xt - eta))) / (a + d)


# ============================================================================================ sizes and thresholds
@cite("qian_sigmund_2013", note="Eq. (29), 1-D hat filter (ported from aerokit.optimization.projection)")
def min_size_ratio(eta_e: float) -> float:
    """Blueprint minimum member width over filter radius, b/R, for the eroded threshold η_e ∈ [0.5, 1] (1-D hat)."""
    if not 0.5 <= eta_e <= 1.0:
        raise ValueError("eroded threshold must lie in [0.5, 1]")
    return 2.0 * math.sqrt(eta_e - 0.5) if eta_e <= 0.75 else 2.0 - 2.0 * math.sqrt(1.0 - eta_e)


@cite("qian_sigmund_2013", note="Eq. (30) (ported from aerokit.optimization.projection)")
def eta_for_size(b_over_r: float) -> float:
    """Eroded threshold η_e that gives a blueprint minimum width b = (b/R)·R (1-D hat)."""
    if b_over_r < 0:
        raise ValueError("b/R must be non-negative")
    if b_over_r <= 1.0:
        return 0.25 * b_over_r ** 2 + 0.5
    if b_over_r <= 2.0:
        return -0.25 * b_over_r ** 2 + b_over_r
    return 1.0


@cite("bourdin_2001", note="cone weights max(0, r_min − dist) of DensityFilter")
def filter_profile(rmin: float, dim: int) -> np.ndarray:
    """Normalised cross-section profile of the :class:`DensityFilter` cone kernel for a straight bar (2-D) / plate
    (3-D): the kernel summed over the other ``dim − 1`` axes, on offsets −r…r (r = ceil(rmin) − 1). ``dim = 1`` is the
    discrete hat."""
    r = math.ceil(rmin) - 1
    ax = np.arange(-r, r + 1)
    grids = np.meshgrid(*([ax] * dim), indexing="ij")
    w = np.maximum(0.0, rmin - np.sqrt(sum(g.astype(float) ** 2 for g in grids)))
    prof = w.reshape(w.shape[0], -1).sum(axis=1)
    return prof / prof.sum()


def _bar_filtered(prof: np.ndarray, w: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Filtered values on element centres −n…n of a bar of width ``w`` (elements) centred on element 0."""
    i = np.arange(-n, n + 1, dtype=float)
    cov = np.clip(np.minimum(i + 0.5, w / 2) - np.maximum(i - 0.5, -w / 2), 0.0, 1.0)
    return i, np.convolve(cov, prof, mode="same")


def _width_above(i: np.ndarray, v: np.ndarray, level: float) -> float:
    """Length of {v ≥ level} with linear interpolation between samples (single interval around the centre)."""
    c = len(v) // 2
    if v[c] < level:
        return 0.0
    edges = []
    for direction in (1, -1):
        k = c
        while 0 <= k + direction < len(v) and v[k + direction] >= level:
            k += direction
        if not 0 <= k + direction < len(v):
            edges.append(i[k])
            continue
        a, b = v[k], v[k + direction]
        edges.append(i[k] + direction * (a - level) / (a - b))
    return float(abs(edges[0] - edges[1]))


def blueprint_size(eta_e: float, rmin: float, dim: int = 2, eta_i: float = 0.5) -> float:
    """Blueprint minimum width (elements) for eroded threshold ``eta_e`` and the cone filter of radius ``rmin``.

    The thinnest eroded member is a bar of width w with filtered peak η_e (bisection on w); the result is the width of
    {x̃ ≥ η_i} of that bar, linearly interpolated between element centres. ``dim = 1`` (hat) reproduces Qian & Sigmund
    Eq. 29 as rmin → ∞ (test). For the void size call it with 1 − η_d.
    """
    if not eta_i <= eta_e < 1.0:
        raise ValueError("need eta_i ≤ eta_e < 1")
    prof = filter_profile(rmin, dim)
    n = len(prof) + 2
    lo, hi = 0.0, float(len(prof) + 1)
    for _ in range(80):
        w = 0.5 * (lo + hi)
        i, v = _bar_filtered(prof, w, n)
        if v[n] < eta_e:
            lo = w
        else:
            hi = w
    i, v = _bar_filtered(prof, hi, n)
    return _width_above(i, v, eta_i)


def _solve_eta(target: float, rmin: float, dim: int, kernel: str, eta_hi: float) -> float:
    if kernel == "hat":
        return min(eta_for_size(target / rmin), eta_hi)
    lo, hi = 0.5, eta_hi
    if blueprint_size(hi, rmin, dim) <= target:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if blueprint_size(mid, rmin, dim) < target:
            lo = mid
        else:
            hi = mid
    return hi


def robust_chain_settings(min_solid: float, min_void: float | None = None, h: float = 1.0, dim: int = 2,
                          eta_max: float = 0.75, kernel: str = "discrete") -> dict[str, float]:
    """Filter radius and thresholds of the robust formulation for minimum solid / void sizes in physical units.

    The larger of the two sizes sets the radius with threshold ``eta_max`` (default ¾, where the 1-D rule gives
    b = R); the smaller one gets its own threshold for the same radius:

    * ``kernel="hat"`` (Qian & Sigmund Eq. 29-30): R = b_max / (b/R)(η_max), η_e = eta_for_size(b_s/R),
      η_d = 1 − eta_for_size(b_v/R);
    * ``kernel="discrete"`` (default): the same rule with :func:`blueprint_size` on the actual cone kernel of
      :class:`DensityFilter` in ``dim`` dimensions (bar in 2-D, plate in 3-D); rmin found by bisection.

    Returns ``rmin`` (elements, for :class:`DensityFilter`), ``R`` (= rmin·h), ``eta_e``, ``eta_d``, ``eta_i`` (½),
    and the nominal sizes ``solid_size`` / ``void_size`` (unit of h) that the chosen values give.
    """
    if kernel not in ("discrete", "hat"):
        raise ValueError("kernel: discrete | hat")
    if not 0.5 < eta_max < 1.0:
        raise ValueError("eta_max must lie in (0.5, 1)")
    bs = float(min_solid) / h
    bv = float(min_solid if min_void is None else min_void) / h
    if not (bs > 0 and bv > 0):
        raise ValueError("sizes must be positive")
    big = max(bs, bv)
    if kernel == "hat":
        rmin = big / min_size_ratio(eta_max)
    else:
        lo, hi = 1.0 + 1e-9, 4.0 * big + 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if blueprint_size(eta_max, mid, dim) < big:
                lo = mid
            else:
                hi = mid
        rmin = hi
    eta_e = _solve_eta(bs, rmin, dim, kernel, eta_max)
    eta_d = 1.0 - _solve_eta(bv, rmin, dim, kernel, eta_max)
    size = (lambda e: min_size_ratio(e) * rmin) if kernel == "hat" else (lambda e: blueprint_size(e, rmin, dim))
    return {"rmin": rmin, "R": rmin * h, "eta_e": eta_e, "eta_d": eta_d, "eta_i": 0.5,
            "solid_size": size(eta_e) * h, "void_size": size(1.0 - eta_d) * h}


# ============================================================================================ realisations
class RobustSet:
    """Thresholds and β of the realisations; builds them from an :class:`EvalContext` (cached per evaluation).

    Parameters
    ----------
    eta_e, eta_d, eta_i : float
        Eroded, dilated and intermediate thresholds (η_d default 1 − η_e). ``eta_i=None`` takes η of the chain's
        :class:`HeavisideProjection` (so the intermediate realisation is the study's x̄), else ½.
    beta : float or None
        None (default) = use β of the chain's :class:`HeavisideProjection`, so one
        :class:`~toporlogy.optim.BetaContinuation` (or :class:`RobustContinuation`) drives all realisations.
    """

    def __init__(self, eta_e: float = 0.75, eta_d: float | None = None, eta_i: float | None = None,
                 beta: float | None = None) -> None:
        self.eta_e = float(eta_e)
        self.eta_d = 1.0 - self.eta_e if eta_d is None else float(eta_d)
        self.eta_i = None if eta_i is None else float(eta_i)
        if not 0.0 < self.eta_d < self.eta_e < 1.0:
            raise ValueError("need 0 < eta_d < eta_e < 1")
        if self.eta_i is not None and not self.eta_d <= self.eta_i <= self.eta_e:
            raise ValueError("need eta_d ≤ eta_i ≤ eta_e")
        self.beta = None if beta is None else float(beta)

    @classmethod
    def from_settings(cls, s: dict[str, float], beta: float | None = None) -> RobustSet:
        return cls(s["eta_e"], s["eta_d"], s.get("eta_i", 0.5), beta)

    def _proj(self, ctx: EvalContext) -> HeavisideProjection | None:
        link = ctx.chain.find(HeavisideProjection) if ctx.chain is not None else None
        return link if isinstance(link, HeavisideProjection) else None

    def beta_of(self, ctx: EvalContext) -> float:
        if self.beta is not None:
            return self.beta
        p = self._proj(ctx)
        if p is None:
            raise ValueError("RobustSet(beta=None) needs a HeavisideProjection link in the chain")
        return p.beta

    def thresholds(self, ctx: EvalContext | None = None) -> dict[str, float]:
        ei = self.eta_i
        if ei is None:
            p = self._proj(ctx) if ctx is not None else None
            ei = p.eta if p is not None else 0.5
        return {"eroded": self.eta_e, "intermediate": ei, "dilated": self.eta_d}

    @staticmethod
    def _projection_then_passive(ctx: EvalContext, p: HeavisideProjection) -> bool:
        """True when the chain ends with exactly ``p`` → Passive (then x̄ of the study = intermediate realisation)."""
        links = list(ctx.chain.links)
        return len(links) >= 2 and links[-2] is p and isinstance(links[-1], Passive)

    def realisation(self, ctx: EvalContext, name: str, need_u: bool = False) -> dict[str, Any]:
        """``{"xphys", "E", "U"}`` of one realisation (U only when ``need_u``); cached in ``ctx.extra``."""
        if name not in REALISATIONS:
            raise ValueError(f"realisation: one of {REALISATIONS}")
        cache = ctx.extra.setdefault(("robust", id(self)), {})
        st = cache.get(name)
        if st is None:
            beta, eta = self.beta_of(ctx), self.thresholds(ctx)[name]
            p = self._proj(ctx)
            if name == "intermediate" and p is not None and p.beta == beta and p.eta == eta and \
                    self._projection_then_passive(ctx, p):
                xr, E, U = ctx.xphys, ctx.E, ctx.U                 # the study's x̄ is this realisation
            else:
                d = ctx.domain
                xt = ctx.field_after(DensityFilter)
                xr = project(xt, beta, eta)
                keep = torch.as_tensor(d.keep, device=xr.device)
                off = torch.as_tensor(d.void | ~d.active, device=xr.device)
                xr = torch.where(keep, torch.ones_like(xr), torch.where(off, torch.zeros_like(xr), xr))
                E, U = ctx.interp(xr), None
            st = {"xphys": xr, "E": E, "U": U}
            cache[name] = st
        if need_u and st["U"] is None:
            if ctx.physics is None or ctx.F is None:
                raise ValueError("context has no physics / loads for a realisation solve")
            act = torch.as_tensor(ctx.domain.active_elements, dtype=torch.long, device=st["E"].device)
            st["U"] = ctx.physics.solve(st["E"].reshape(-1)[act], ctx.F)
        return st

    def context(self, ctx: EvalContext, name: str, need_u: bool = False) -> EvalContext:
        """An :class:`EvalContext` whose x̄, E (and U) are those of realisation ``name``."""
        st = self.realisation(ctx, name, need_u)
        return EvalContext(ctx.domain, st["xphys"], st["E"], st["U"], ctx.F, extra=ctx.extra, x=ctx.x,
                           fields=ctx.fields, physics=ctx.physics, interp=ctx.interp, chain=ctx.chain)

    def field(self, name: str = "dilated") -> Any:
        """Callable ``ctx -> x̄ of realisation name`` (e.g. as ``source`` of a LocalVolume)."""
        return lambda ctx: self.realisation(ctx, name)["xphys"]


# ============================================================================================ robust responses
class Robust(Response):
    """Worst case of ``response`` over the realisations, aggregated.

    Parameters
    ----------
    response : Response
        Evaluated on each realisation's context (its own solve when it needs ``"u"``).
    robust : RobustSet
    realisations : sequence of str
        Default all three; ``("eroded", "dilated")`` is the Lazarov et al. Eq. (37) form (two solves).
    aggregate : {"ks", "pnorm", "max"}
        "ks": (s/ρ)·log Σ exp(ρ J_k / s) with a fixed scale s (``scale``, or the largest J_k of the first evaluation
        after :meth:`reset_scale`); upper bound of max J_k, gap ≤ s·log(n)/ρ. "pnorm": (Σ J_k^p)^(1/p) (J_k > 0,
        scale-free, gap ≤ max·(n^(1/p) − 1)). "max": exact max (gradient of the worst realisation only; the form of
        the validated 2-D driver).
    rho, p : float
        KS parameter (default 50) and p-norm exponent (default 16).

    ``needs`` is empty: the study does no solve of its own for this response (the realisation solves are separate).
    ``last_values`` holds the per-realisation values of the last gradient-enabled evaluation.
    """

    name = "robust"
    needs: frozenset[str] = frozenset()

    def __init__(self, response: Response, robust: RobustSet, realisations: Sequence[str] = REALISATIONS,
                 aggregate: str = "ks", rho: float = 50.0, p: float = 16.0, scale: float | None = None,
                 name: str | None = None) -> None:
        if aggregate not in ("ks", "pnorm", "max"):
            raise ValueError("aggregate: ks | pnorm | max")
        bad = [r for r in realisations if r not in REALISATIONS]
        if bad or not realisations:
            raise ValueError(f"realisations must be a non-empty subset of {REALISATIONS}")
        self.response, self.robust = response, robust
        self.realisations = tuple(realisations)
        self.aggregate, self.rho, self.p = aggregate, float(rho), float(p)
        self.fixed_scale = scale
        self.scale: float | None = scale
        self.last_values: dict[str, float] = {}
        self.name = name or f"robust_{response.name}"

    def reset_scale(self) -> None:
        """Forget the automatic KS scale (taken again at the next evaluation)."""
        self.scale = self.fixed_scale

    def values(self, ctx: EvalContext) -> torch.Tensor:
        need_u = "u" in self.response.needs
        return torch.stack([self.response(self.robust.context(ctx, r, need_u)) for r in self.realisations])

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        v = self.values(ctx)
        if torch.is_grad_enabled():
            self.last_values = {r: float(x) for r, x in zip(self.realisations, v, strict=True)}
        if self.aggregate == "max":
            return v.max()
        if self.aggregate == "pnorm":
            return (v.clamp_min(0.0) ** self.p).sum() ** (1.0 / self.p)
        if self.scale is None:
            self.scale = max(float(v.detach().abs().max()), 1e-300)
        s = self.scale
        return s / self.rho * torch.logsumexp(self.rho * v / s, dim=0)


class RobustCompliance(Robust):
    """:class:`Robust` of :class:`~toporlogy.responses.Compliance` (one solve per realisation)."""

    def __init__(self, robust: RobustSet, weights: Any = None, **kw: Any) -> None:
        kw.setdefault("name", "robust_compliance")
        super().__init__(Compliance(weights), robust, **kw)


class RealisationVolume(Volume):
    """:class:`~toporlogy.responses.Volume` of one realisation (default the dilated one).

    Records the volumes of this realisation and of the intermediate one at every gradient-enabled evaluation
    (``last``), which :class:`RobustContinuation` uses to update the bound. Being a :class:`Volume`, the study's
    default start uses its bound.
    """

    def __init__(self, robust: RobustSet, realisation: str = "dilated", weighted: bool = False,
                 name: str | None = None) -> None:
        super().__init__(weighted)
        if realisation not in REALISATIONS:
            raise ValueError(f"realisation: one of {REALISATIONS}")
        self.robust, self.realisation = robust, realisation
        self.name = name or f"volume_{realisation}"
        self.last: dict[str, float] = {}

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        v = super().__call__(self.robust.context(ctx, self.realisation))
        if torch.is_grad_enabled():
            with torch.no_grad():
                vi = super().__call__(self.robust.context(ctx, "intermediate"))
            self.last = {self.realisation: float(v), "intermediate": float(vi)}
        return v


class DilatedVolume(RealisationVolume):
    """Volume of the dilated realisation (the robust formulation's volume constraint)."""

    def __init__(self, robust: RobustSet, weighted: bool = False) -> None:
        super().__init__(robust, "dilated", weighted, name="volume_dilated")


# ============================================================================================ continuation
class RobustContinuation(Schedule):
    """β continuation of all realisations plus the periodic update of the dilated volume bound.

    β ← min(2β, β_max) every ``every`` iterations or when the change falls to ``tol`` (the rule of
    :class:`~toporlogy.optim.BetaContinuation`), applied to the chain's :class:`HeavisideProjection` and to
    ``robust.beta`` when that is set. On a β step the automatic KS scales of ``rescale`` responses are reset.

    Volume: when ``volume`` (the :class:`Constraint` holding a :class:`RealisationVolume`) and ``target`` (V* of the
    blueprint) are given, every ``volume_every`` iterations ``volume.upper = target · V_d / V_i`` from the last
    evaluation (Wang, Lazarov & Sigmund 2011 rule, not read in this session). This does not count as a problem
    change (no optimizer reset).
    """

    def __init__(self, chain: Chain, robust: RobustSet | None = None, beta0: float = 1.0, beta_max: float = 32.0,
                 every: int = 50, tol: float = 0.01, volume: Constraint | None = None, target: float | None = None,
                 volume_every: int = 20, rescale: Sequence[Robust] = ()) -> None:
        link = chain.find(HeavisideProjection)
        self.proj = link if isinstance(link, HeavisideProjection) else None
        self.robust = robust
        if self.proj is None and (robust is None or robust.beta is None):
            raise ValueError("no β to drive: the chain has no HeavisideProjection and robust.beta is None")
        self.beta_max, self.every, self.tol = float(beta_max), int(every), float(tol)
        self._set_beta(float(beta0))
        self.volume, self.target, self.volume_every = volume, target, int(volume_every)
        if volume is not None and target is None:
            raise ValueError("target volume required with a volume constraint")
        self.rescale = list(rescale)
        self._beta0 = float(beta0)
        self._upper0 = None if volume is None else float(volume.upper)
        self._count = 0

    def reset(self) -> None:
        """β back to β0, counter 0, the volume bound back to its value at construction, automatic KS scales of
        the ``rescale`` responses forgotten (Study.run calls it before the loop)."""
        self._set_beta(self._beta0)
        self._count = 0
        if self.volume is not None and self._upper0 is not None:
            self.volume.upper = self._upper0
        for r in self.rescale:
            r.reset_scale()

    def _set_beta(self, b: float) -> None:
        if self.proj is not None:
            self.proj.beta = b
        if self.robust is not None and self.robust.beta is not None:
            self.robust.beta = b

    @property
    def beta(self) -> float:
        if self.proj is not None:
            return self.proj.beta
        return float(self.robust.beta)  # type: ignore[union-attr,arg-type]

    @property
    def finished(self) -> bool:  # type: ignore[override]
        return self.beta >= self.beta_max

    def update(self, iteration: int, change: float) -> bool:
        if self.volume is not None and iteration % self.volume_every == 0:
            last = getattr(self.volume.response, "last", {})
            name = getattr(self.volume.response, "realisation", "dilated")
            if last.get("intermediate", 0.0) > 0:
                self.volume.upper = float(self.target) * last[name] / last["intermediate"]  # type: ignore[arg-type]
        self._count += 1
        if self.beta < self.beta_max and (self._count >= self.every or change <= self.tol):
            self._set_beta(min(2.0 * self.beta, self.beta_max))
            self._count = 0
            for r in self.rescale:
                r.reset_scale()
            return True
        return False

    def state(self) -> dict[str, float]:
        s = {"beta": self.beta}
        if self.volume is not None:
            s["volume_bound"] = float(self.volume.upper)
        return s


# ============================================================================================ bound formulation
class BoundSplit(torch.nn.Module):
    """First chain link for the bound formulation: variables (x_design, z) → x_design; z is the last entry.

    Declares ``n_in`` and ``reduce`` (the hooks :class:`~toporlogy.study.Study` reads from the first link).
    """

    def __init__(self, n_design: int, z0: float = 1.0) -> None:
        super().__init__()
        self.n_in = int(n_design) + 1
        self.z0 = float(z0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:-1]

    def reduce(self, x_design: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(x_design, float), [self.z0]])


class BoundObjective(Response):
    """Objective z (the last variable)."""

    name = "bound"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        if ctx.x is None:
            raise ValueError("context has no design vector")
        return ctx.x[-1]


class _Ref:
    def __init__(self, value: float | None, factor: float) -> None:
        self.value, self.factor = value, factor


class RealisationBound(Response):
    """Constraint J_k / J_ref − z ≤ 0 for one realisation (use with ``Constraint(..., upper=0.0, scale=1.0)``).

    J_ref is shared by the constraints made in one :func:`bound_formulation` call: given, or ``factor`` × the first
    value evaluated (so z stays inside MMA's [0, 1] box).
    """

    needs: frozenset[str] = frozenset()

    def __init__(self, response: Response, robust: RobustSet, realisation: str, ref: _Ref) -> None:
        self.response, self.robust, self.realisation, self.ref = response, robust, realisation, ref
        self.name = f"bound_{realisation}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        v = self.response(self.robust.context(ctx, self.realisation, "u" in self.response.needs))
        if self.ref.value is None:
            self.ref.value = self.ref.factor * max(float(v.detach()), 1e-300)
        return v / self.ref.value - ctx.x[-1]  # type: ignore[index]


def bound_formulation(chain: Chain, robust: RobustSet, n_design: int, response: Response | None = None,
                      realisations: Sequence[str] = REALISATIONS, reference: float | None = None,
                      factor: float = 2.0, z0: float = 1.0) -> tuple[Chain, Response, list[Constraint]]:
    """Exact worst case by the bound formulation: min z s.t. J_k(x)/J_ref ≤ z for every realisation k.

    Returns ``(chain', objective, constraints)``: ``chain'`` = [:class:`BoundSplit`] + the links of ``chain`` (the
    design vector gets one extra entry z, start value ``z0``), ``objective`` = :class:`BoundObjective`, and one
    :class:`Constraint` per realisation. Add the volume constraint yourself and use an optimizer with several
    constraints (:class:`toporlogy.optim.mma.MMA`). ``response`` default :class:`Compliance`.
    """
    resp = response or Compliance()
    ref = _Ref(reference, factor)
    new = Chain([BoundSplit(n_design, z0)] + list(chain.links))
    cons = [Constraint(RealisationBound(resp, robust, r, ref), 0.0, scale=1.0) for r in realisations]
    return new, BoundObjective(), cons
