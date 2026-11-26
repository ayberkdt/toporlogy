"""Uncertainty quantification and robust design: parameter metadata, seeded sampling, propagation without dropping
failed samples, estimates with confidence intervals, Sobol indices, and risk measures usable as Study objectives.

Parameters and sampling
-----------------------
A :class:`Parameter` carries its distribution (uniform, truncated normal, truncated lognormal, fixed), its nominal
value, the physically admissible open interval, unit and ``source`` ("synthetic" unless the numbers come from
somewhere). A distribution whose support leaves the admissible interval is refused (a normal modulus must be
truncated above zero). A :class:`ParameterSpace` may add a Gaussian-copula correlation (the correlation of the
underlying normal scores, not the Pearson correlation of the marginals) and a joint admissibility check; samples that
fail the joint check raise, they are not removed.

:meth:`ParameterSpace.sample` draws with ``"mc"`` (NumPy PCG64), ``"lhs"`` or ``"sobol"``
(:mod:`scipy.stats.qmc`, scrambled; Sobol needs a power of two). The seed and the NumPy / SciPy versions are kept in
the :class:`SampleSet`; a sequence is reproducible for the same versions only.

Failures
--------
:func:`propagate` records a model that raises or returns a non-finite value. No statistic is ever computed on the
successful subset: with failures present the default policy ``"raise"`` refuses, and ``"worst"`` counts each failure
as +∞ (for quantities where larger is worse), so means and tail averages become infinite and a quantile stays finite
only when the failed fraction is smaller than the tail.

Estimates
---------
:func:`mean_estimate` (t interval) and :func:`quantile_estimate` (distribution-free order-statistic interval) assume
independent samples, i.e. ``"mc"``. For ``"lhs"`` / ``"sobol"`` those intervals are not valid; use
:func:`replicated_mean` (independent randomisations). Tail quantities are flagged ``reliable = False`` when fewer
than :data:`MIN_TAIL_SAMPLES` samples lie in the tail: a few runs do not give P99.9.

Robust design
-------------
:class:`ScenarioCompliance` is a Study response over load cases that are samples of an uncertain load. The sample is
drawn once and kept (common random numbers), so the optimisation problem is deterministic and its gradient is that
of the sampled estimator. :func:`risk_measure` gives mean, mean + k·std or CVaR (Rockafellar-Uryasev form with the
sample quantile; piecewise smooth, the gradient is one subgradient where two scenarios tie). This is aleatory
parameter uncertainty; the erosion / dilation formulation of :mod:`toporlogy.optimization.robust` addresses a
different thing (manufacturing perturbations of the geometry).

Verification: ``tests/toporlogy/test_uq.py`` (records U1-U10 of ``docs/olcumler/laminat_uq_onkayit.md``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
import torch
from scipy import stats
from scipy.stats import qmc

from toporlogy.optimization.responses import EvalContext, Response

__all__ = ["Parameter", "ParameterSpace", "SampleSet", "InadmissibleSampleError", "FailedSamplesError",
           "Propagation", "propagate", "Estimate", "mean_estimate", "variance_estimate", "quantile_estimate",
           "cvar_estimate", "exceedance_probability", "replicated_mean", "sobol_indices", "SobolIndices",
           "risk_measure", "ScenarioCompliance", "load_scenarios", "parameter_table", "SAMPLING_METHODS",
           "MIN_TAIL_SAMPLES", "RISK_MEASURES"]

SAMPLING_METHODS = ("mc", "lhs", "sobol")
RISK_MEASURES = ("mean", "mean_std", "cvar")
#: fewer samples than this beyond a quantile / threshold: the tail estimate is flagged unreliable
MIN_TAIL_SAMPLES = 10


class InadmissibleSampleError(ValueError):
    """Samples that violate the joint admissibility check of a :class:`ParameterSpace`."""


class FailedSamplesError(RuntimeError):
    """A statistic was asked for while some model evaluations failed."""


# ================================================================================================ parameters
@dataclass(frozen=True)
class Parameter:
    """One uncertain parameter. Build with :meth:`uniform`, :meth:`normal`, :meth:`lognormal` or :meth:`fixed`.

    ``lower`` / ``upper``: support after truncation. ``admissible``: open interval of physically meaningful values
    (e.g. (0, inf) for a modulus or density, (−1, 0.5) for an isotropic Poisson ratio)."""

    name: str
    kind: str
    nominal: float
    loc: float
    scale: float
    lower: float
    upper: float
    admissible: tuple[float, float] = (-math.inf, math.inf)
    unit: str = ""
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "normal", "lognormal", "fixed"):
            raise ValueError("kind: uniform | normal | lognormal | fixed")
        if not self.name:
            raise ValueError("a parameter needs a name")
        lo, hi = self.admissible
        if not lo < hi:
            raise ValueError(f"{self.name}: admissible = (low, high) with low < high")
        if self.kind == "fixed":
            if not (self.lower == self.upper == self.nominal):
                raise ValueError(f"{self.name}: a fixed parameter has lower = upper = nominal")
        elif not (self.scale > 0 and self.lower < self.upper):
            raise ValueError(f"{self.name}: the distribution needs a positive spread and lower < upper")
        inside = lambda v: lo < v < hi                                      # noqa: E731
        if not inside(self.nominal):
            raise ValueError(f"{self.name}: nominal {self.nominal:g} is not admissible {self.admissible}")
        # a lognormal support (0, ∞) is open at 0: lower = 0 is admissible for a positive quantity
        low_ok = self.lower > lo or (math.isinf(lo) and math.isinf(self.lower)) or \
            (self.kind == "lognormal" and self.lower >= lo)
        if not low_ok or not (self.upper < hi or (math.isinf(hi) and math.isinf(self.upper))):
            raise ValueError(f"{self.name}: the support [{self.lower:g}, {self.upper:g}] leaves the admissible "
                             f"interval {self.admissible}; truncate the distribution")
        if not self.lower <= self.nominal <= self.upper:
            raise ValueError(f"{self.name}: nominal outside the support")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, name: str, low: float, high: float, nominal: float | None = None, **meta: Any) -> Parameter:
        nom = 0.5 * (low + high) if nominal is None else float(nominal)
        return cls(name, "uniform", nom, float(low), float(high - low), float(low), float(high), **meta)

    @classmethod
    def normal(cls, name: str, mean: float, std: float, truncate: tuple[float, float] | None = None,
               nominal: float | None = None, **meta: Any) -> Parameter:
        lo, hi = (-math.inf, math.inf) if truncate is None else (float(truncate[0]), float(truncate[1]))
        return cls(name, "normal", float(mean if nominal is None else nominal), float(mean), float(std), lo, hi,
                   **meta)

    @classmethod
    def lognormal(cls, name: str, median: float, sigma_log: float, truncate: tuple[float, float] | None = None,
                  nominal: float | None = None, **meta: Any) -> Parameter:
        if not median > 0:
            raise ValueError("a lognormal median is positive")
        lo, hi = (0.0, math.inf) if truncate is None else (float(truncate[0]), float(truncate[1]))
        if lo < 0:
            raise ValueError("a lognormal support starts at 0 or above")
        return cls(name, "lognormal", float(median if nominal is None else nominal), math.log(median),
                   float(sigma_log), lo, hi, **meta)

    @classmethod
    def fixed(cls, name: str, value: float, **meta: Any) -> Parameter:
        v = float(value)
        return cls(name, "fixed", v, v, 0.0, v, v, **meta)

    # ------------------------------------------------------------------ distribution
    def ppf(self, u: Any) -> np.ndarray:
        """Quantile function at u ∈ (0, 1)."""
        q = np.asarray(u, float)
        if self.kind == "fixed":
            return np.full(q.shape, self.nominal)
        if self.kind == "uniform":
            return self.loc + self.scale * q
        if self.kind == "normal":
            a, b = (self.lower - self.loc) / self.scale, (self.upper - self.loc) / self.scale
            return np.asarray(stats.truncnorm.ppf(q, a, b, loc=self.loc, scale=self.scale))
        base = stats.lognorm(s=self.scale, scale=math.exp(self.loc))
        c0, c1 = float(base.cdf(self.lower)), float(base.cdf(self.upper))
        return np.asarray(base.ppf(c0 + q * (c1 - c0)))

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "distribution": self.kind, "nominal": self.nominal, "loc": self.loc,
                "scale": self.scale, "support": [self.lower, self.upper], "admissible": list(self.admissible),
                "unit": self.unit, "source": self.source}


@dataclass(frozen=True)
class SampleSet:
    """``values`` (n, d) in the order of ``names``; ``unit`` (n, d) the points of the unit cube they came from."""

    names: tuple[str, ...]
    values: np.ndarray
    unit: np.ndarray
    method: str
    seed: int
    versions: dict[str, str]

    def __len__(self) -> int:
        return len(self.values)

    def row(self, i: int) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.names, self.values[i], strict=True)}


def _unit_points(d: int, n: int, method: str, seed: int) -> np.ndarray:
    if method not in SAMPLING_METHODS:
        raise ValueError(f"method: one of {SAMPLING_METHODS}")
    if n < 1:
        raise ValueError("n >= 1 is required")
    if method == "mc":
        u = np.random.default_rng(seed).random((n, d))
    elif method == "lhs":
        u = qmc.LatinHypercube(d, seed=seed).random(n)
    else:
        if n & (n - 1):
            raise ValueError(f"a Sobol sample needs a power of two, got {n}")
        u = qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(math.log2(n)))
    return np.clip(u, 2.0 ** -53, 1.0 - 2.0 ** -53)


@dataclass(frozen=True)
class ParameterSpace:
    """Parameters with optional Gaussian-copula correlation and joint admissibility check (module notes)."""

    parameters: tuple[Parameter, ...]
    correlation: np.ndarray | None = None
    joint_check: Callable[[dict[str, float]], bool] | None = None

    def __post_init__(self) -> None:
        pars = tuple(self.parameters)
        if not pars:
            raise ValueError("at least one parameter")
        names = [p.name for p in pars]
        if len(set(names)) != len(names):
            raise ValueError(f"parameter names must be unique: {names}")
        object.__setattr__(self, "parameters", pars)
        if self.correlation is not None:
            C = np.asarray(self.correlation, float)
            d = len(pars)
            if C.shape != (d, d) or not np.allclose(C, C.T, atol=1e-12) or not np.allclose(np.diag(C), 1.0):
                raise ValueError("correlation: a symmetric (d, d) matrix with unit diagonal")
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                raise ValueError("correlation matrix is not positive definite") from None
            object.__setattr__(self, "correlation", C)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def nominal(self) -> dict[str, float]:
        return {p.name: p.nominal for p in self.parameters}

    def transform(self, unit: np.ndarray) -> np.ndarray:
        """Unit-cube points (n, d) → parameter values (copula first, then the marginals)."""
        u = np.asarray(unit, float)
        if u.ndim != 2 or u.shape[1] != self.dim:
            raise ValueError(f"unit points must be (n, {self.dim})")
        if self.correlation is not None:
            z = stats.norm.ppf(u) @ np.linalg.cholesky(self.correlation).T
            u = np.clip(stats.norm.cdf(z), 2.0 ** -53, 1.0 - 2.0 ** -53)
        return np.column_stack([p.ppf(u[:, j]) for j, p in enumerate(self.parameters)])

    def check(self, values: np.ndarray) -> None:
        if self.joint_check is None:
            return
        bad = [i for i, row in enumerate(values)
               if not self.joint_check({k: float(v) for k, v in zip(self.names, row, strict=True)})]
        if bad:
            raise InadmissibleSampleError(f"{len(bad)} of {len(values)} samples fail the joint admissibility check "
                                          f"(first: sample {bad[0]}); change the distributions, do not drop samples")

    def sample(self, n: int, method: str = "mc", seed: int = 0) -> SampleSet:
        u = _unit_points(self.dim, n, method, seed)
        v = self.transform(u)
        self.check(v)
        return SampleSet(self.names, v, u, method, int(seed),
                         {"numpy": np.__version__, "scipy": scipy.__version__})


# ================================================================================================ propagation
@dataclass(frozen=True)
class Propagation:
    """Model values over a sample. ``values`` (n,) with NaN where the model failed; ``failed`` (n,) bool;
    ``errors`` {sample index: message}."""

    values: np.ndarray
    failed: np.ndarray
    errors: dict[int, str]

    @property
    def n_failed(self) -> int:
        return int(self.failed.sum())

    def sample_values(self, policy: str = "raise") -> np.ndarray:
        """Values for statistics (module notes): ``"raise"`` refuses with failures, ``"worst"`` sets them to +inf."""
        if policy not in ("raise", "worst"):
            raise ValueError("policy: 'raise' or 'worst' (failed samples are never dropped)")
        if not self.n_failed:
            return self.values.copy()
        if policy == "raise":
            first = min(self.errors)
            raise FailedSamplesError(f"{self.n_failed} of {len(self.values)} evaluations failed (sample {first}: "
                                     f"{self.errors[first]}); choose policy='worst' or repair the model")
        return np.where(self.failed, math.inf, self.values)


def propagate(model: Callable[[dict[str, float]], float], samples: SampleSet,
              map_fn: Callable[..., Iterable[Any]] = map) -> Propagation:
    """Evaluate ``model(parameters)`` on every sample. ``map_fn`` may be an executor's ``map`` (order preserved; the
    sample is fixed beforehand, so parallel execution gives the same values)."""

    def one(i: int) -> tuple[float, str]:
        try:
            y = float(model(samples.row(i)))
        except Exception as e:  # noqa: BLE001  (every failure is recorded, none is swallowed silently)
            return math.nan, f"{type(e).__name__}: {e}"
        return (y, "") if math.isfinite(y) else (math.nan, f"non-finite value {y}")

    out = list(map_fn(one, range(len(samples))))
    values = np.array([v for v, _ in out], float)
    errors = {i: msg for i, (_, msg) in enumerate(out) if msg}
    failed = np.zeros(len(samples), bool)
    failed[list(errors)] = True
    return Propagation(values, failed, errors)


# ================================================================================================ estimates
@dataclass(frozen=True)
class Estimate:
    """A sample estimate with a confidence interval. ``reliable`` False: the sample does not support the quantity."""

    value: float
    ci_low: float
    ci_high: float
    n: int
    level: float
    method: str
    standard_error: float = math.nan
    reliable: bool = True


def _sample(y: Any, least: int = 2) -> np.ndarray:
    a = np.asarray(y, float).reshape(-1)
    if a.size < least:
        raise ValueError(f"at least {least} values are required")
    if np.isnan(a).any():
        raise FailedSamplesError("NaN in the sample: failed evaluations are not dropped (see Propagation)")
    return a


def mean_estimate(y: Any, level: float = 0.95) -> Estimate:
    """Mean with a Student t interval (independent samples)."""
    a = _sample(y)
    if np.isinf(a).any():
        return Estimate(math.inf, math.inf, math.inf, a.size, level, "iid-t", math.inf, False)
    m, se = float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size))
    h = float(stats.t.ppf(0.5 + level / 2, a.size - 1)) * se
    return Estimate(m, m - h, m + h, a.size, level, "iid-t", se)


def variance_estimate(y: Any, level: float = 0.95) -> Estimate:
    """Unbiased variance with the large-sample standard error √((m4 − (n−3)/(n−1) s⁴) / n) (independent samples)."""
    a = _sample(y, 4)
    n = a.size
    if np.isinf(a).any():
        return Estimate(math.inf, math.inf, math.inf, n, level, "iid-normal", math.inf, False)
    s2 = float(a.var(ddof=1))
    m4 = float(((a - a.mean()) ** 4).mean())
    se = math.sqrt(max(m4 - (n - 3) / (n - 1) * s2 ** 2, 0.0) / n)
    h = float(stats.norm.ppf(0.5 + level / 2)) * se
    return Estimate(s2, s2 - h, s2 + h, n, level, "iid-normal", se)


def _order_quantile(sorted_a: np.ndarray, alpha: float) -> float:
    """Smallest order statistic y_(k) with k/n ≥ α (inverse of the empirical distribution function)."""
    k = max(1, math.ceil(alpha * sorted_a.size - 1e-9))
    return float(sorted_a[k - 1])


def quantile_estimate(y: Any, alpha: float, level: float = 0.95) -> Estimate:
    """α quantile with a distribution-free interval from two order statistics (binomial counts, independent samples).
    ``reliable`` needs :data:`MIN_TAIL_SAMPLES` samples on each side of the quantile."""
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1")
    a = np.sort(_sample(y))
    n = a.size
    lo = int(stats.binom.ppf((1 - level) / 2, n, alpha))
    hi = int(stats.binom.ppf((1 + level) / 2, n, alpha)) + 1
    low = float(a[lo - 1]) if lo >= 1 else -math.inf
    high = float(a[hi - 1]) if hi <= n else math.inf
    reliable = n * min(alpha, 1 - alpha) >= MIN_TAIL_SAMPLES
    return Estimate(_order_quantile(a, alpha), low, high, n, level, "order-statistic", reliable=reliable)


def cvar_estimate(y: Any, alpha: float, level: float = 0.95) -> Estimate:
    """Conditional value at risk (mean of the upper 1 − α tail), q + E[(y − q)₊] / (1 − α) with the sample quantile q;
    interval from the large-sample variance of (y − q)₊ / (1 − α) (independent samples)."""
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1")
    a = np.sort(_sample(y))
    n = a.size
    reliable = n * (1 - alpha) >= MIN_TAIL_SAMPLES
    if np.isinf(a).any():
        return Estimate(math.inf, math.inf, math.inf, n, level, "rockafellar-uryasev", math.inf, False)
    q = _order_quantile(a, alpha)
    g = np.maximum(a - q, 0.0) / (1 - alpha)
    v = q + float(g.mean())
    se = float(g.std(ddof=1) / math.sqrt(n))
    h = float(stats.norm.ppf(0.5 + level / 2)) * se
    return Estimate(v, v - h, v + h, n, level, "rockafellar-uryasev", se, reliable)


def exceedance_probability(y: Any, threshold: float, level: float = 0.95) -> Estimate:
    """P(y > threshold) with the Clopper-Pearson interval (independent samples). ``reliable`` needs at least
    :data:`MIN_TAIL_SAMPLES` samples on each side of the threshold."""
    a = _sample(y)
    n = a.size
    k = int((a > threshold).sum())
    low = 0.0 if k == 0 else float(stats.beta.ppf((1 - level) / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf((1 + level) / 2, k + 1, n - k))
    p = k / n
    return Estimate(p, low, high, n, level, "clopper-pearson", math.sqrt(p * (1 - p) / n),
                    min(k, n - k) >= MIN_TAIL_SAMPLES)


def replicated_mean(model: Callable[[dict[str, float]], float], space: ParameterSpace, n: int, method: str,
                    seeds: Sequence[int], level: float = 0.95) -> Estimate:
    """Mean of the means of independent randomisations (one per seed), with a t interval over the replicates: the
    valid interval for LHS and scrambled Sobol. Any failed evaluation raises."""
    if len(set(seeds)) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct seeds")
    means = [float(propagate(model, space.sample(n, method, s)).sample_values().mean()) for s in seeds]
    e = mean_estimate(means, level)
    return Estimate(e.value, e.ci_low, e.ci_high, n * len(seeds), level, f"replicated-{method}", e.standard_error)


# ================================================================================================ Sobol indices
@dataclass(frozen=True)
class SobolIndices:
    names: tuple[str, ...]
    first: np.ndarray
    total: np.ndarray
    variance: float
    n_evaluations: int


def sobol_indices(model: Callable[[dict[str, float]], float], space: ParameterSpace, n_base: int,
                  seed: int = 0) -> SobolIndices:
    """First-order (Saltelli 2010 estimator) and total-effect (Jansen) indices for **independent** parameters, from
    a scrambled Sobol sample of dimension 2d: n_base (d + 2) evaluations. Estimators written from their standard
    form; the primary papers were not read."""
    if space.correlation is not None:
        raise ValueError("Sobol indices are defined here for independent parameters only")
    d = space.dim
    u = _unit_points(2 * d, n_base, "sobol", seed)
    ua, ub = u[:, :d], u[:, d:]

    def run(unit: np.ndarray) -> np.ndarray:
        v = space.transform(unit)
        space.check(v)
        s = SampleSet(space.names, v, unit, "sobol", seed, {})
        return propagate(model, s).sample_values()

    fa, fb = run(ua), run(ub)
    var = float(np.concatenate([fa, fb]).var())
    if not var > 0:
        raise ValueError("the model is constant over the sample: indices are undefined")
    first, total = np.zeros(d), np.zeros(d)
    for i in range(d):
        uab = ua.copy()
        uab[:, i] = ub[:, i]
        fab = run(uab)
        first[i] = float(np.mean(fb * (fab - fa))) / var
        total[i] = 0.5 * float(np.mean((fa - fab) ** 2)) / var
    return SobolIndices(space.names, first, total, var, n_base * (d + 2))


# ================================================================================================ robust design
def risk_measure(values: torch.Tensor, measure: str = "mean", k: float = 0.0, alpha: float = 0.95) -> torch.Tensor:
    """Mean, mean + k·std (sample std) or CVaR_α of equally weighted scenario values (n,), differentiable."""
    if measure not in RISK_MEASURES:
        raise ValueError(f"measure: one of {RISK_MEASURES}")
    v = values.reshape(-1)
    if measure == "mean":
        return v.mean()
    if v.numel() < 2:
        raise ValueError(f"{measure} needs at least two scenarios")
    if measure == "mean_std":
        m = v.mean()
        floor = (1e-12 * m.detach().abs()) ** 2
        return m + k * torch.sqrt(((v - m) ** 2).sum() / (v.numel() - 1) + floor)
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1")
    idx = torch.argsort(v.detach())[max(1, math.ceil(alpha * v.numel() - 1e-9)) - 1]
    q = v[idx]
    return q + torch.clamp(v - q, min=0.0).sum() / ((1 - alpha) * v.numel())


def load_scenarios(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Loads (ndof, n_scenarios) = basis (ndof, m) · coefficients (n_scenarios, m)ᵀ."""
    B = np.asarray(basis, float)
    C = np.asarray(coefficients, float)
    if B.ndim != 2 or C.ndim != 2 or B.shape[1] != C.shape[1]:
        raise ValueError(f"basis (ndof, m) and coefficients (n, m) must share m, got {B.shape} and {C.shape}")
    return B @ C.T


class ScenarioCompliance(Response):
    """:func:`risk_measure` of the compliances of all load cases, each case one scenario of an uncertain load
    (module notes). Every scenario shares one factorisation of K."""

    name = "scenario_compliance"
    needs = frozenset({"u"})

    def __init__(self, measure: str = "mean", k: float = 0.0, alpha: float = 0.95,
                 scale: float = 1.0) -> None:
        if measure not in RISK_MEASURES:
            raise ValueError(f"measure: one of {RISK_MEASURES}")
        if not scale > 0:
            raise ValueError("scale > 0")
        self.measure, self.k, self.alpha, self.scale = measure, float(k), float(alpha), float(scale)

    def scenario_values(self, ctx: EvalContext) -> torch.Tensor:
        if ctx.U is None or ctx.F is None:
            raise ValueError("ScenarioCompliance needs U and F")
        from toporlogy.optimization.responses import Compliance
        return Compliance().per_case(ctx)                  # F_cᵀ U_c, with the physics' total load if it has one

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return self.scale * risk_measure(self.scenario_values(ctx), self.measure, self.k, self.alpha)


def parameter_table(space: ParameterSpace) -> list[Mapping[str, Any]]:
    """Metadata of every parameter (for records)."""
    return [p.metadata() for p in space.parameters]
