"""Irreversible (viscoplastic, metal-type) creep of an isotropic small-strain solid: Norton and Norton–Bailey laws,
Arrhenius temperature dependence and an adaptive driver for long holds (WP13;
``docs/foundations/CREEP_FORMULATIONS.md``).

Scope: the **irreversible** J2 creep strain ε_cr of ε = ε_e + ε_cr, with isotropic elasticity (physical K, G in Pa).
Recoverable (viscoelastic) creep is a different mechanism and lives elsewhere; the two are never added by default.
Units: stress and moduli Pa, time s, temperature K (absolute), activation energy J/mol, A and rates 1/s.
Conventions as in :mod:`toporlogy.core.constitutive`: tension positive, Voigt order [xx, yy, zz, yz, xz, xy], stress
with the physical shear component, strain with the engineering shear. Parameters are synthetic in all tests; nothing
here is material data.

Laws (q = √(3/2 s:s), s = dev σ, N = 3s/(2q) = ∂q/∂σ, ε̇_cr = ṗ N, ε_cr deviatoric, p ≥ 0 accumulated):

* :class:`NortonCreep`: ṗ = A f(T) (q/σ_ref)^n, f the Arrhenius factor (:class:`ArrheniusFactor`).
* :class:`NortonBaileyCreep`: constant-stress law ε_cr = B (q/σ_ref)^n (t_load/t_ref)^m. ``hardening="time"``:
  Δp = B (q_{n+1}/σ_ref)^n [Φ(t_load + Δt) − Φ(t_load)], Φ(t) = (t/t_ref)^m (the exact increment of the time function,
  no regularisation at t = 0); ``hardening="strain"``: the same constant-stress law written as a function of the
  accumulated strain; with w = p^{1/m} it reads ẇ = c(q)^{1/m}/t_ref (c = B (q/σ_ref)^n), regular at p = 0.

Backward Euler, radial: with the trial stress s_tr = 2G (dev ε − ε_cr,n) and q_tr, the update is
s = (1 − 3GΔp/q_tr) s_tr, ε_cr ← ε_cr + Δp N_tr, and Δp is the root of the scalar residual

    g(Δp) = Δp − h(q_tr − 3GΔp) = 0 ,

where h(q) ≥ 0 is the creep increment of the step at end stress q (nondecreasing in q, h(0) = 0). g' = 1 + 3G h' ≥ 1,
g(0) ≤ 0 and g(cap) > 0 with cap = min(q_tr/3G, h(q_tr)): the root is unique and bracketed. It is found by a safeguarded
Newton iteration (bisection whenever the Newton point leaves the bracket or is not finite) until
|g| ≤ ``local_tol`` · cap, which bounds the error of Δp by |g| (g' ≥ 1). q_tr = 0 or Δt = 0: no evolution.
Algorithmic tangent (derived in the doc):

    C = K 1⊗1 + 2G θ I_dev − 4G² (Δp' − Δp/q_tr) N⊗N ,  θ = q/q_tr,  Δp' = h'/(1 + 3G h'),

the last term contains the change of direction of N; the frozen-direction tangent K 1⊗1 + 2G I_dev − 4G² Δp' N⊗N is
not consistent. Dissipation increment q Δp = σ:Δε_cr ≥ 0 (there is no stored creep energy); energy is the elastic
strain energy ½ σ:(ε − ε_cr).

Long holds: :func:`integrate_creep` (step doubling on the accumulated creep strain, events honoured exactly, rejected
steps rolled back through :class:`StateStore`, log-spaced outputs, bounded number of steps).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (ConstitutiveFailure, HistoryUpdate, Kinematics, MaterialRequest,
                                         MaterialResponse, ReturnMappingFailure, StateStore, TimeStep,
                                         drive_material_point, require_step, small_strain_history_evaluate)

__all__ = ["R_GAS", "NOT_IMPLEMENTED", "ArrheniusFactor", "NortonCreep", "NortonBaileyCreep", "LoadSegment",
           "CreepRun", "CreepStepLimit", "integrate_creep"]

R_GAS = 8.314462618                       # J/(mol K), CODATA 2018 (exact value of the 2019 SI)

NOT_IMPLEMENTED = ("tertiary creep and creep damage", "creep rupture and life prediction (Larson-Miller etc.)",
                   "Garofalo / sinh and diffusion-mechanism creep laws", "temperature-dependent K, G (constant here)",
                   "pressure-sensitive creep (J2 creep has no volumetric part)")

_DEFAULT_REQUEST = MaterialRequest()
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
_I6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = _I6 - np.outer(_ONE6, _ONE6) / 3.0
_GLX, _GLW = np.polynomial.legendre.leggauss(8)
_GL8 = (0.5 * (_GLX + 1.0), 0.5 * _GLW)                    # 8-point Gauss-Legendre nodes and weights on [0, 1]
_EXP_MAX = 700.0                           # exp overflows at 709.78


def _pos(v: Any, what: str, strict: bool = True) -> float:
    x = float(v)
    if not (math.isfinite(x) and (x > 0.0 if strict else x >= 0.0)):
        raise ValueError(f"{what} must be finite and {'> 0' if strict else '>= 0'}, got {v!r}")
    return x


# ----------------------------------------------------------------------------------------------- Arrhenius
class ArrheniusFactor:
    """f(T) = exp[−Q/R (1/T − 1/T_ref)], Q in J/mol, T and T_ref absolute (K), R = 8.314462618 J/(mol K); f(T_ref) = 1.
    Evaluated in the log domain (:meth:`log_factor`); :meth:`factor` refuses an exponent above 700 (float64
    overflow) instead of returning inf. No unit conversion is done: Q is J/mol, T is kelvin, nothing else.

    :meth:`step_average` is the mean of f over a step in which T varies linearly in time,
    (1/Δt) ∫ f(T0 + (T1 − T0) τ/Δt) dτ = ∫_0^1 f(T0 + (T1 − T0) s) ds, by composite 8-point Gauss–Legendre with the
    number of equal panels doubled (1, 2, 4, …, ``max_panels``) until two successive estimates agree to ``rtol``
    (relative) for every point; the estimate of the finer panel set is returned. For the model it is exact for a step
    at constant stress; at varying stress it multiplies the end-of-step stress law (first-order, like backward
    Euler), so the driver step must resolve a simultaneous change of q and T."""

    def __init__(self, Q: float, T_ref: float) -> None:
        self.Q = _pos(Q, "Arrhenius Q [J/mol]", strict=False)
        self.T_ref = _pos(T_ref, "Arrhenius T_ref [K]")

    @staticmethod
    def _temp(T_: Any, what: str = "T") -> np.ndarray:
        a = np.asarray(T_, float)
        if not (np.all(np.isfinite(a)) and np.all(a > 0.0)):
            raise ValueError(f"{what} is an absolute temperature in kelvin and must be finite and > 0")
        return a

    def log_factor(self, T_: Any) -> np.ndarray:
        return -(self.Q / R_GAS) * (1.0 / self._temp(T_) - 1.0 / self.T_ref)

    def factor(self, T_: Any) -> np.ndarray:
        lf = self.log_factor(T_)
        if np.any(lf > _EXP_MAX):
            raise ValueError("Arrhenius factor overflows float64 (log factor > 700): use log_factor or lower Q/T")
        return np.exp(lf)

    def dfactor(self, T_: Any) -> np.ndarray:
        """df/dT = f Q/(R T²) [1/K]."""
        a = self._temp(T_)
        return self.factor(a) * (self.Q / R_GAS) / a ** 2

    def step_average(self, T0: Any, T1: Any, rtol: float = 1e-12, max_panels: int = 1024) -> np.ndarray:
        a0, a1 = np.broadcast_arrays(self._temp(T0, "T0"), self._temp(T1, "T1"))
        a0, a1 = np.array(a0, float), np.array(a1, float)
        s, w = _GL8

        def estimate(k: int) -> np.ndarray:
            sk = (np.arange(k)[:, None] + s[None, :]).reshape(-1) / k          # nodes of k panels on [0, 1]
            wk = np.tile(w, k) / k
            Tn = a0[..., None] + (a1 - a0)[..., None] * sk
            return np.sum(self.factor(Tn) * wk, axis=-1)

        k = 1
        prev = estimate(k)
        while True:
            k *= 2
            cur = estimate(k)
            if np.all(np.abs(cur - prev) <= rtol * np.abs(cur)):
                break
            if k >= max_panels:
                raise ConstitutiveFailure(f"Arrhenius step average not converged to {rtol:g} with {k} panels")
            prev = cur
        return np.where(a0 == a1, self.factor(a0), cur)


# ----------------------------------------------------------------------------------------------- scalar root
def _power_increment(base: np.ndarray, delta: np.ndarray, m: float) -> np.ndarray:
    """(base + delta)^m − base^m for base ≥ 0, delta ≥ 0, without cancellation and exact for base = 0."""
    with np.errstate(all="ignore"):
        inc = base ** m * np.expm1(m * np.log1p(delta / base))
        return np.where(base > 0.0, inc, delta ** m)


def _solve_increment(q_tr: np.ndarray, G: float, rate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                     tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root Δp of g(Δp) = Δp − h(q_tr − 3GΔp) on [0, cap] (module notes). ``rate(q)`` returns (h, h') at the end
    stresses q (h' may be 0, inf or nan where undefined; those points are bisected). Returns Δp, the relative
    residual |g|/cap and the number of iterations. Raises ReturnMappingFailure if not converged."""
    cap = q_tr / (3.0 * G)
    with np.errstate(all="ignore"):
        h_tr, _ = rate(q_tr)
    hi0 = np.minimum(cap, np.where(np.isnan(h_tr), cap, h_tr))
    lo, hi, x = np.zeros_like(q_tr), hi0.copy(), hi0.copy()
    done = hi0 <= 0.0
    scale = np.where(hi0 > 0.0, hi0, 1.0)
    its = np.zeros(q_tr.shape, int)
    res = np.zeros(q_tr.shape)
    for k in range(max_iter + 1):
        q = np.maximum(q_tr - 3.0 * G * x, 0.0)
        with np.errstate(all="ignore"):
            hv, hp = rate(q)
        g = x - hv
        res = np.where(done, 0.0, np.abs(g) / scale)
        active = res > tol
        if not active.any():
            break
        if k == max_iter:
            raise ReturnMappingFailure(f"creep update: {int(active.sum())} points not converged in {max_iter} local "
                                       f"iterations (max relative residual {res.max():.3g})")
        its += active
        pos = g > 0.0
        hi = np.where(active & pos, x, hi)
        lo = np.where(active & ~pos, x, lo)
        with np.errstate(all="ignore"):
            xn = x - g / (1.0 + 3.0 * G * hp)
        ok = np.isfinite(xn) & (xn > lo) & (xn < hi)
        xn = np.where(ok, xn, 0.5 * (lo + hi))
        x = np.where(active, xn, x)
    return x, res, its


# ----------------------------------------------------------------------------------------------- models
class _IsotropicCreep:
    """Shared part of the J2 creep models: isotropic elasticity (K, G), state ``creep_strain`` (n, 3, 3) and ``p_cr``
    (n,), backward-Euler radial update and algorithmic tangent (module notes). Subclasses give the increment h(q)."""

    kinematics = "small"
    conservative = False
    physical = True
    name = "creep"
    state_fields: tuple[str, ...] = ("creep_strain", "p_cr")
    _needs_temperature = False

    def __init__(self, K: float, G: float, local_tol: float, local_max_iter: int) -> None:
        self.K, self.G = _pos(K, "K [Pa]"), _pos(G, "G [Pa]")
        self.local_tol, self.local_max_iter = _pos(local_tol, "local_tol"), int(local_max_iter)
        if self.local_max_iter < 1:
            raise ValueError("local_max_iter must be >= 1")
        self.Ce6 = self.K * np.outer(_ONE6, _ONE6) + 2.0 * self.G * _IDEV6

    # ---- to be provided by the subclasses
    def _rate_factory(self, committed: Mapping[str, np.ndarray], step: TimeStep, n: int
                      ) -> Callable[[np.ndarray], Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        raise NotImplementedError

    def _next_extra_state(self, committed: Mapping[str, np.ndarray], step: TimeStep) -> dict[str, np.ndarray]:
        return {}

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"creep_strain": np.zeros((n, 3, 3)), "p_cr": np.zeros(n)}

    # ---- update
    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        step = require_step(self, request, temperature=self._needs_temperature)
        if request.tangent_kind != "algorithmic":
            raise ValueError(f"{self.name}: only the algorithmic tangent is defined (tangent_kind='algorithmic')")
        eps = np.asarray(eps, float)
        n = len(eps)
        K, G = self.K, self.G
        ecr_n, p_n = np.asarray(committed["creep_strain"]), np.asarray(committed["p_cr"])
        eye = np.eye(3)
        tr = np.trace(eps, axis1=1, axis2=2)
        s_tr = 2.0 * G * (eps - (tr / 3.0)[:, None, None] * eye - ecr_n)
        q_tr = math.sqrt(1.5) * np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
        factory = self._rate_factory(committed, step, n)
        active = (q_tr > 0.0) & (step.dt > 0.0)
        idx = np.flatnonzero(active)
        dp = np.zeros(n)
        resid = np.zeros(n)
        its = np.zeros(n)
        q_new = q_tr.copy()
        theta = np.ones(n)
        N = np.zeros((n, 3, 3))
        dpp = np.zeros(n)
        if idx.size:
            rate = factory(idx)
            x, r, it = _solve_increment(q_tr[idx], G, rate, self.local_tol, self.local_max_iter)
            qn = np.maximum(q_tr[idx] - 3.0 * G * x, 0.0)
            with np.errstate(all="ignore"):
                _, hp = rate(qn)
                dpp_a = 1.0 / (3.0 * G + 1.0 / hp)              # h'/(1 + 3G h'); h' = 0 -> 0, h' = inf -> 1/(3G)
            dp[idx], resid[idx], its[idx], q_new[idx] = x, r, it, qn
            theta[idx] = qn / q_tr[idx]
            N[idx] = 1.5 * s_tr[idx] / q_tr[idx][:, None, None]
            dpp[idx] = dpp_a
        dec = dp[:, None, None] * N
        sig = theta[:, None, None] * s_tr + (K * tr)[:, None, None] * eye
        ecr = ecr_n + dec
        C = None
        if request.tangent:
            C = np.broadcast_to(self.Ce6, (n, 6, 6)).copy()
            if idx.size:
                nv = N[idx][:, _VA, _VB]                                      # tensor components, not doubled
                coef = -4.0 * G ** 2 * (dpp[idx] - dp[idx] / q_tr[idx])
                C[idx] = (K * np.outer(_ONE6, _ONE6)[None] + 2.0 * G * theta[idx][:, None, None] * _IDEV6[None]
                          + coef[:, None, None] * nv[:, :, None] * nv[:, None, :])
        psi = 0.5 * np.einsum("nij,nij->n", sig, eps - ecr) if request.energy else None
        trial = {"creep_strain": ecr, "p_cr": p_n + dp, **self._next_extra_state(committed, step)}
        diag = {"delta_p": dp, "q": q_new, "creep_residual": resid, "local_iterations": its,
                "creep_active": active.astype(float)}
        return HistoryUpdate(sig, C, trial, diag, psi, q_new * dp)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)


def _power_rate(coef: np.ndarray, sigma_ref: float, n_exp: float
                ) -> Callable[[np.ndarray], Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """Factory of h(q) = coef (q/σ_ref)^n and h' for the points ``idx`` (coef is (n,))."""

    def factory(idx: np.ndarray) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        c = coef[idx]

        def rate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = q / sigma_ref
            return c * u ** n_exp, c * n_exp * u ** (n_exp - 1.0) / sigma_ref

        return rate

    return factory


class NortonCreep(_IsotropicCreep):
    """J2 power-law creep ṗ = A f(T) (q/σ_ref)^n with isotropic elasticity (K, G) (module notes).

    ``A`` [1/s] ≥ 0 is the rate at q = σ_ref and T = T_ref, ``sigma_ref`` [Pa] > 0, ``n`` > 0 (n < 1 is allowed; the
    rate is then not differentiable at q = 0 and the tangent at a fully relaxed point uses the limit 1/(3G)).
    ``arrhenius`` (:class:`ArrheniusFactor`) makes the model need the step temperatures: the factor is the
    mean over the step of the linearly varying T (``TimeStep.temperature_prev`` → ``temperature``). Without it the
    temperature is ignored. State: ``creep_strain`` (n, 3, 3), ``p_cr`` (n,). Diagnostics: ``delta_p``, ``q`` (end of
    step), ``creep_residual`` (|g|/cap of the root), ``local_iterations``, ``creep_active``. Dissipation q Δp [J/m³]."""

    name = "norton_creep"

    def __init__(self, K: float, G: float, A: float, sigma_ref: float, n: float,
                 arrhenius: ArrheniusFactor | None = None, local_tol: float = 1e-13,
                 local_max_iter: int = 200) -> None:
        super().__init__(K, G, local_tol, local_max_iter)
        self.A = _pos(A, "A [1/s]", strict=False)
        self.sigma_ref = _pos(sigma_ref, "sigma_ref [Pa]")
        self.n = _pos(n, "n")
        if arrhenius is not None and not isinstance(arrhenius, ArrheniusFactor):
            raise ValueError("arrhenius must be an ArrheniusFactor or None")
        self.arrhenius = arrhenius
        self._needs_temperature = arrhenius is not None
        self._avg_cache: tuple[tuple[bytes, bytes], np.ndarray] | None = None     # memo of the last step average

    def _rate_factory(self, committed: Mapping[str, np.ndarray], step: TimeStep, n: int):  # noqa: ANN202
        F: Any = 1.0
        if self.arrhenius is not None:
            key = (np.asarray(step.temperature_prev, float).tobytes(), np.asarray(step.temperature, float).tobytes())
            if self._avg_cache is None or self._avg_cache[0] != key:        # the Newton loops of one step repeat it
                self._avg_cache = (key, self.arrhenius.step_average(step.temperature_prev, step.temperature))
            F = self._avg_cache[1]
        coef = np.broadcast_to(step.dt * self.A * np.asarray(F, float), (n,)).copy()
        return _power_rate(coef, self.sigma_ref, self.n)


class NortonBaileyCreep(_IsotropicCreep):
    """Norton–Bailey primary + secondary creep ε_cr = B (q/σ_ref)^n (t_load/t_ref)^m at constant stress, isotropic
    elasticity (K, G), isothermal (the temperature of the step is ignored; B is the value at the temperature of the
    test). ``B`` ≥ 0 (strain), ``sigma_ref`` [Pa], ``n`` > 0, ``m`` > 0 (m < 1 primary, m = 1 steady), ``t_ref`` [s].

    ``hardening="time"``: ṗ = B (q/σ_ref)^n Φ'(t_load), exact increment ΔΦ = Φ(t_load + Δt) − Φ(t_load) of
    Φ = (t/t_ref)^m (t_load = 0 gives (Δt/t_ref)^m: no singular epsilon). ``hardening="strain"``: the same constant-
    stress law with the strain as hardening variable, ṗ = (m/t_ref) c^{1/m} p^{1 − 1/m}, c = B (q/σ_ref)^n,
    integrated in w = p^{1/m} (ẇ = c^{1/m}/t_ref). At constant stress both give the same curve; after a stress change
    they differ (time hardening keeps the clock, strain hardening keeps the accumulated strain).

    ``t_load`` is a state field: the time since the load was applied, advanced by Δt of every committed step (whether
    or not a load acts: it is a clock, not a load detector) and starting at 0 at the first step. To restart the
    clock at a new load application call :meth:`reset_load_time` on the committed state. It is not the absolute time
    of the :class:`TimeStep`. In the strain-hardening variant the clock is carried but unused."""

    name = "norton_bailey_creep"
    state_fields = ("creep_strain", "p_cr", "t_load")

    def __init__(self, K: float, G: float, B: float, sigma_ref: float, n: float, m: float, t_ref: float,
                 hardening: str = "time", local_tol: float = 1e-13, local_max_iter: int = 200) -> None:
        super().__init__(K, G, local_tol, local_max_iter)
        self.B = _pos(B, "B", strict=False)
        self.sigma_ref = _pos(sigma_ref, "sigma_ref [Pa]")
        self.n, self.m, self.t_ref = _pos(n, "n"), _pos(m, "m"), _pos(t_ref, "t_ref [s]")
        if hardening not in ("time", "strain"):
            raise ValueError("hardening: 'time' or 'strain'")
        self.hardening = hardening

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {**super().initial_state(n), "t_load": np.zeros(n)}

    def reset_load_time(self, committed: Mapping[str, np.ndarray], t: float = 0.0,
                        mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Copy of the committed state with ``t_load`` set to ``t`` (≥ 0) at the points of ``mask`` (default all)."""
        t = _pos(t, "t_load", strict=False)
        out = {k: np.array(v, copy=True) for k, v in committed.items()}
        sel = np.ones(len(out["t_load"]), bool) if mask is None else np.asarray(mask, bool)
        out["t_load"][sel] = t
        return out

    def _next_extra_state(self, committed: Mapping[str, np.ndarray], step: TimeStep) -> dict[str, np.ndarray]:
        return {"t_load": np.asarray(committed["t_load"]) + step.dt}

    def _rate_factory(self, committed: Mapping[str, np.ndarray], step: TimeStep, n: int):  # noqa: ANN202
        m, tref, dt = self.m, self.t_ref, step.dt
        if self.hardening == "time":
            dphi = _power_increment(np.asarray(committed["t_load"], float) / tref, np.full(n, dt / tref), m)
            return _power_rate(self.B * dphi, self.sigma_ref, self.n)
        w_n = np.asarray(committed["p_cr"], float) ** (1.0 / m)
        B, sref, nexp = self.B, self.sigma_ref, self.n

        def factory(idx: np.ndarray) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
            w0 = w_n[idx]

            def rate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                delta = (dt / tref) * (B * (q / sref) ** nexp) ** (1.0 / m)
                W = w0 + delta
                hp = np.where(q > 0.0, nexp * W ** (m - 1.0) * delta / np.where(q > 0.0, q, 1.0), np.inf)
                return _power_increment(w0, delta, m), hp

            return rate

        return factory


# ----------------------------------------------------------------------------------------------- adaptive driver
class CreepStepLimit(ConstitutiveFailure):
    """The adaptive driver needed more attempts than ``max_steps`` (the work cap) or a step below the resolvable
    size."""


@dataclass(frozen=True)
class LoadSegment:
    """Targets held on (t_prev_event, t_end]: ``strain`` (6,) for the strain-controlled Voigt components and
    ``stress`` (6,) for the others (the ``control`` mask of the run; unused entries are ignored). ``temperature``
    [K] at ``t_end`` (T varies linearly from the temperature at the segment start); None keeps the previous value
    (isothermal runs: None everywhere)."""

    t_end: float
    strain: Sequence[float] = (0.0,) * 6
    stress: Sequence[float] = (0.0,) * 6
    temperature: float | None = None


@dataclass
class CreepRun:
    """Accepted grid of :func:`integrate_creep`: per grid point (the end of each accepted half step) the time, strain
    and stress (k, 6), creep strain (k, 3, 3), ``p_cr`` (k,), dissipation increment (k,), the targets and temperature
    that drove the step ending there; ``output_index`` the grid indices of the requested output times; counters;
    restart data: ``store`` (committed state), ``strain_final`` (6,), ``t_final``, ``dt_next`` (the step size the
    controller would try next)."""

    times: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    creep_strain: np.ndarray
    p_cr: np.ndarray
    dissipation: np.ndarray
    strain_target: np.ndarray
    stress_target: np.ndarray
    temperature: np.ndarray | None
    output_index: np.ndarray
    n_accepted: int
    n_rejected: int
    store: StateStore
    strain_final: np.ndarray
    t_final: float
    dt_next: float
    extra: dict[str, np.ndarray] = field(default_factory=dict)


def integrate_creep(model: Any, control: Sequence[bool], segments: Sequence[LoadSegment],
                    output_times: Sequence[float] = (), *, t0: float = 0.0, temperature0: float | None = None,
                    rtol: float = 1e-6, atol: float = 1e-12, dt0: float | None = None, dt_next: float | None = None,
                    dt_max: float = math.inf, max_steps: int = 10000, stress_tol: float | None = None,
                    safety: float = 0.9, store: StateStore | None = None,
                    strain0: Sequence[float] | None = None) -> CreepRun:
    """Integrate a creep material point through piecewise-constant mixed targets with an adaptive physical time step.

    Per attempt from t to t + Δt: one full step (on a scratch copy of the committed state) and two half steps (on
    the real store), both through :func:`drive_material_point`. The error estimate is the step-doubling difference of
    the accumulated creep strain, e = ‖ε_cr(Δt) − ε_cr(Δt/2, Δt/2)‖_F / (rtol ‖ε_cr‖_F + atol); the two-half-steps
    result is kept when e ≤ 1 (backward Euler is first order: the retained result has about the error e). New step
    Δt·clip(``safety``·e^(−1/2), 0.2, 5) (×5 when e = 0). A rejected step (e > 1, or a failed local/mixed-control
    solve) restores the committed state of the store from its checkpoint (``store.restore`` + ``rollback``). Segment
    ends and ``output_times`` are events: a step never crosses one and ends on it exactly (t is then set to the event
    time itself). A step that reaches an event is not allowed to shrink the controller's proposal below its previous
    value unless the error says so. At most ``max_steps`` attempts (accepted + rejected), then :class:`CreepStepLimit`.

    ``dt0`` first proposal (default 1e-3 × the time to the first event); ``dt_next`` (a previous run's
    ``dt_next``) overrides it for a restart together with ``store`` (committed state), ``strain0`` (6,), ``t0`` and
    ``temperature0``; the segments of a restart are those after ``t0``. ``stress_tol`` [Pa] is the tolerance of the
    mixed-control Newton (default 1e-11 × (max |stress target| + G max |strain target|)). The grid ``times`` contains
    both half-step ends of every accepted attempt, so it can be replayed exactly with one
    :func:`drive_material_point` call using ``strain_target`` / ``stress_target`` / ``temperature``."""
    ctrl = np.asarray(control, bool).reshape(6)
    segs = list(segments)
    if not segs:
        raise ValueError("integrate_creep: at least one segment")
    t0 = float(t0)
    ends = np.array([float(s.t_end) for s in segs])
    if not (np.all(np.isfinite(ends)) and ends[0] > t0 and np.all(np.diff(ends) > 0.0)):
        raise ValueError("segment end times must be finite, increasing and > t0")
    t_final = float(ends[-1])
    outs = np.asarray(output_times, float).reshape(-1)
    if outs.size and not (np.all(np.isfinite(outs)) and np.all(outs > t0) and np.all(outs <= t_final)):
        raise ValueError("output_times must lie in (t0, t_final]")
    isothermal = temperature0 is None
    if isothermal and any(s.temperature is not None for s in segs):
        raise ValueError("segment temperatures given without temperature0")
    if rtol <= 0 or atol < 0 or not (0.0 < safety < 1.0) or max_steps < 1:
        raise ValueError("rtol > 0, atol >= 0, 0 < safety < 1, max_steps >= 1 are required")
    events = np.unique(np.concatenate([ends, outs]))
    seg_t = [np.asarray(s.strain, float).reshape(6) for s in segs]
    seg_s = [np.asarray(s.stress, float).reshape(6) for s in segs]
    if stress_tol is None:
        Gm = float(getattr(model, "G", 1.0))
        scale = max(np.abs(np.array(seg_s)).max(), Gm * np.abs(np.array(seg_t)).max())
        stress_tol = max(1e-11 * scale, 1e-30)
    store = StateStore(model, 1) if store is None else store
    e = np.zeros(6) if strain0 is None else np.asarray(strain0, float).reshape(6).copy()
    # temperature bookkeeping: T at the start and end of each segment
    T_start, T_end = [], []
    Tc: float | None = None if isothermal else float(temperature0)  # type: ignore[arg-type]
    for s in segs:
        T_start.append(Tc)
        Tc = Tc if (s.temperature is None or Tc is None) else float(s.temperature)
        T_end.append(Tc)
    t_start = [t0] + [float(x) for x in ends[:-1]]
    if dt_next is not None:
        dtp = float(dt_next)
    elif dt0 is not None:
        dtp = float(dt0)
    else:
        dtp = 1e-3 * (float(events[events > t0][0]) - t0)
    if not (math.isfinite(dtp) and dtp > 0.0):
        raise ValueError("the first step proposal must be finite and > 0")
    t = t0
    Tcur: float | None = None if isothermal else float(temperature0)  # type: ignore[arg-type]
    rec: dict[str, list] = {k: [] for k in ("t", "e", "s", "ecr", "p", "d", "et", "st", "T")}
    n_acc = n_rej = attempts = 0
    eps_f = np.finfo(float).eps

    def T_at(i: int, tau: float) -> float | None:
        if isothermal:
            return None
        a, b = T_start[i], T_end[i]
        if tau >= float(ends[i]):
            return b
        assert a is not None and b is not None
        return a + (b - a) * (tau - t_start[i]) / (float(ends[i]) - t_start[i])

    while t < t_final:
        attempts += 1
        if attempts > max_steps:
            raise CreepStepLimit(f"integrate_creep: {max_steps} attempts used at t = {t:g} s of {t_final:g} s")
        i = int(np.searchsorted(ends, t, side="right"))             # segment holding (t, ...]
        nxt = float(events[np.searchsorted(events, t, side="right")])
        dt = min(dtp, dt_max)
        hit = t + dt >= nxt
        t_new = nxt if hit else t + dt
        dt = t_new - t
        if dt <= 64.0 * eps_f * abs(t) or dt <= 0.0:
            raise CreepStepLimit(f"integrate_creep: step {dt:g} s is not resolvable at t = {t:g} s")
        t_mid = t + 0.5 * dt
        T_mid, T_new = T_at(i, t_mid), T_at(i, t_new)
        tgt_e, tgt_s = seg_t[i][None], seg_s[i][None]
        cp, e0 = store.checkpoint(), e.copy()

        def run(st: StateStore, e_in: np.ndarray, ta: float, Ta: float | None, tb: float, Tb: float | None,
                tgt_e: np.ndarray = tgt_e, tgt_s: np.ndarray = tgt_s):
            return drive_material_point(model, tgt_e, tgt_s, ctrl, tol=stress_tol, times=[tb],
                                        temperatures=None if Tb is None else [Tb], t0=ta, temperature0=Ta,
                                        store=st, strain0=e_in)

        try:
            scratch = StateStore(model, 1)
            scratch.restore(cp)
            hA = run(scratch, e0, t, Tcur, t_new, T_new)
            h1 = run(store, e0, t, Tcur, t_mid, T_mid)
            h2 = run(store, h1.strain[-1], t_mid, T_mid, t_new, T_new)
        except ConstitutiveFailure:
            store.restore(cp)
            store.rollback()
            n_rej += 1
            dtp = 0.25 * dt
            continue
        ecA = hA.states[-1]["creep_strain"][0]
        ecB = h2.states[-1]["creep_strain"][0]
        err = float(np.linalg.norm(ecA - ecB)) / (rtol * float(np.linalg.norm(ecB)) + atol) if (
            rtol * float(np.linalg.norm(ecB)) + atol) > 0.0 else (0.0 if np.array_equal(ecA, ecB) else math.inf)
        if err > 1.0:
            store.restore(cp)
            store.rollback()
            n_rej += 1
            dtp = dt * max(0.2, safety * err ** -0.5)
            continue
        n_acc += 1
        f = 5.0 if err == 0.0 else float(np.clip(safety * err ** -0.5, 0.2, 5.0))
        dtp = max(dt * f, dtp) if (hit and f >= 1.0) else dt * f
        for h, tt, TT in ((h1, t_mid, T_mid), (h2, t_new, T_new)):
            rec["t"].append(tt)
            rec["e"].append(h.strain[-1])
            rec["s"].append(h.stress[-1])
            rec["ecr"].append(h.states[-1]["creep_strain"][0])
            rec["p"].append(float(h.states[-1]["p_cr"][0]))
            rec["d"].append(float(h.dissipation[-1]))
            rec["et"].append(tgt_e[0])
            rec["st"].append(tgt_s[0])
            rec["T"].append(np.nan if TT is None else TT)
        e, t, Tcur = h2.strain[-1].copy(), t_new, T_new
    times = np.array(rec["t"])
    oi = np.searchsorted(times, outs) if outs.size else np.zeros(0, int)
    if outs.size and not np.array_equal(times[oi], outs):
        raise ConstitutiveFailure("integrate_creep: an output time is not in the accepted grid (internal error)")
    out = CreepRun(times, np.array(rec["e"]), np.array(rec["s"]), np.array(rec["ecr"]), np.array(rec["p"]),
                   np.array(rec["d"]), np.array(rec["et"]), np.array(rec["st"]),
                   None if isothermal else np.array(rec["T"]), oi, n_acc, n_rej, store, e, t, dtp)
    if "t_load" in store.committed:
        out.extra["t_load"] = np.array(store.committed["t_load"])
    return out
