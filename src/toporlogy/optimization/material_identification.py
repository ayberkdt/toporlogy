"""Identification of rheology / creep parameters from (synthetic) test data, with identifiability and data provenance
(WP20 D03/D05; ``docs/foundations/MATERIAL_IDENTIFICATION.md``, ``docs/foundations/prereg/D_identification.md``).

Built on :mod:`toporlogy.optimization.identification` (``Observations``, ``identify``, ``weighted_rms``); nothing of it
is copied. Units: time s, stress / modulus Pa, temperature K (absolute), rate 1/s, activation energy J/mol. Every
number used with this module in this repository is **synthetic**; no real material data is shipped or implied.

* Relaxation: M(t) = M∞ + Σ A_i e^{−t/τ_i}. :func:`fit_prony_fixed_times`: τ fixed on a (log-spaced) grid, amplitudes by
  non-negative least squares on y/σ (``scipy.optimize.nnls``), σ_i = ``rel_sigma``·y_i. :func:`fit_prony_free_times`:
  θ = (M∞, A_i, ln τ_i), analytic Jacobian, ``identify`` from each of a stated list of start sets (the amplitudes of a
  start are the NNLS solution at its τ), best cost wins, times sorted.
* Norton: ṗ = A f(T) (σ/σ_ref)^n, f = exp[−Q/R (1/T − 1/T_ref)]; ln ṗ = ln A + n ln(σ/σ_ref) + Q' (1 − T_ref/T) with
  Q' = Q/(R T_ref) is linear in (ln A, n, Q'). :func:`fit_norton_loglinear` is the reference (ordinary least squares in
  ln ṗ, exact for noise-free data); :func:`fit_norton_nonlinear` fits ṗ itself with ``identify``
  (σ_i = ``rel_sigma``·ṗ_i).
* :func:`identifiability_report`: SVD of the weighted Jacobian in scaled coordinates (columns × scale / σ), condition,
  rank (``RANK_RTOL``), the weakest parameter combination (right singular vector of the smallest singular value) and two
  linearised parameter standard deviations: ``std_nominal`` (the stated σ are right) and ``std_residual`` (scaled by the
  residual standard deviation s = sqrt(RSS_w/(m − p)), None for m ≤ p). Both are None when the Jacobian is rank
  deficient. Local and linearised: no claim beyond the model and the noise level assumed; a good fit proves nothing
  about model error.
* :class:`MaterialDataRecord`: arrays with one unit string each, source ``synthetic`` / ``measured``, origin text,
  license (required for measured), a sha256 checksum of the arrays and an explicit :class:`ValidityBox`; the fits take
  SI records only and every prediction outside the box raises :class:`ValidityBoxError`.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import nnls

from toporlogy.core.creep import R_GAS
from toporlogy.core.viscoelastic import PronySeries
from toporlogy.optimization.identification import (DATA_KINDS, RANK_RTOL, Identification, Observations, identify,
                                                    weighted_rms)

__all__ = ["SI_UNITS", "UNIT_FACTORS", "COND_LIMIT", "ValidityBoxError", "ValidityBox", "MaterialDataRecord",
           "convert_to_si", "IdentifiabilityReport", "identifiability_report", "PronyFit", "log_spaced_taus",
           "fit_prony_fixed_times", "fit_prony_free_times", "NortonFit", "fit_norton_loglinear",
           "fit_norton_nonlinear"]

SI_UNITS = {"time": "s", "stress": "Pa", "temperature": "K", "rate": "1/s"}
UNIT_FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0},
    "stress": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9},
    "temperature": {"K": 1.0},                       # degC / degF are refused: an offset is not a factor
    "rate": {"1/s": 1.0, "1/min": 1.0 / 60.0, "1/h": 1.0 / 3600.0},
}
COND_LIMIT = 1e4                                     # condition number above which a design is flagged

# quantity -> {array name: unit kind}; the box coordinates of each quantity
_QUANTITIES: dict[str, dict[str, str]] = {
    "relaxation_modulus": {"time": "time", "modulus": "stress"},
    "creep_rate": {"stress": "stress", "temperature": "temperature", "rate": "rate"},
}
_BOX_KEYS: dict[str, tuple[str, ...]] = {"relaxation_modulus": ("time",), "creep_rate": ("stress", "temperature")}


class ValidityBoxError(ValueError):
    """A prediction was requested outside the validity box of the data it was calibrated on."""


@dataclass(frozen=True)
class ValidityBox:
    """Closed intervals ``ranges[name] = (lo, hi)`` in SI units inside which a calibrated model may be evaluated."""

    ranges: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        r: dict[str, tuple[float, float]] = {}
        for k, (lo, hi) in dict(self.ranges).items():
            lo, hi = float(lo), float(hi)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"validity box '{k}': need finite lo <= hi, got ({lo}, {hi})")
            r[str(k)] = (lo, hi)
        object.__setattr__(self, "ranges", r)

    def check(self, **coords: Any) -> None:
        """Raise :class:`ValidityBoxError` unless every given coordinate (array) lies inside its closed interval."""
        for k, v in coords.items():
            if k not in self.ranges:
                raise ValueError(f"'{k}' is not a coordinate of this validity box {tuple(self.ranges)}")
            a = np.asarray(v, float)
            lo, hi = self.ranges[k]
            if not np.all(np.isfinite(a)):
                raise ValidityBoxError(f"{k}: non-finite value in the prediction request")
            if a.size and (a.min() < lo or a.max() > hi):
                raise ValidityBoxError(f"{k} in [{a.min():.6g}, {a.max():.6g}] is outside the validity box "
                                       f"[{lo:.6g}, {hi:.6g}] of the calibration data (no extrapolation)")


def _checksum(arrays: Mapping[str, np.ndarray]) -> str:
    """sha256 over the arrays, names sorted: name UTF-8, 0x00, '<f8', repr(shape), 0x00, C-order little-endian bytes."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name], dtype="<f8")
        h.update(name.encode("utf-8") + b"\x00" + b"<f8" + repr(a.shape).encode() + b"\x00" + a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class MaterialDataRecord:
    """Test data with units, provenance, checksum and validity box (module notes).

    ``quantity``: ``"relaxation_modulus"`` (arrays ``time``, ``modulus``) or ``"creep_rate"`` (arrays ``stress``,
    ``temperature``, ``rate``, one row per experiment). ``units`` has one entry per array (known units:
    :data:`UNIT_FACTORS`). ``source`` is ``"synthetic"`` or ``"measured"``; ``origin`` says how the data came about
    (generator and seed, or specimen / test report); ``license`` is required for measured data. All arrays are finite,
    strictly positive and of equal length. ``box`` defaults to the range of the data (time; stress and temperature);
    an explicit box must contain the data. ``checksum``: given → must match the arrays. Invalid input raises
    ValueError."""

    quantity: str
    arrays: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    source: str
    origin: str
    license: str | None = None
    box: ValidityBox | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        if self.quantity not in _QUANTITIES:
            raise ValueError(f"quantity must be one of {tuple(_QUANTITIES)}, got {self.quantity!r}")
        spec = _QUANTITIES[self.quantity]
        if self.source not in DATA_KINDS:
            raise ValueError(f"provenance: source must be one of {DATA_KINDS}, got {self.source!r}")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("provenance: 'origin' (how the data came about) is required")
        if self.source == "measured" and not (isinstance(self.license, str) and self.license.strip()):
            raise ValueError("provenance: measured data needs a license statement")
        if set(self.arrays) != set(spec):
            raise ValueError(f"{self.quantity}: arrays must be exactly {tuple(spec)}, got {tuple(self.arrays)}")
        arr: dict[str, np.ndarray] = {}
        for name, kind in spec.items():
            unit = self.units.get(name)
            if unit is None:
                raise ValueError(f"unit missing for '{name}'")
            if unit not in UNIT_FACTORS[kind]:
                raise ValueError(f"unit '{unit}' of '{name}' is not a {kind} unit {tuple(UNIT_FACTORS[kind])}")
            a = np.array(self.arrays[name], float)
            if a.ndim != 1 or a.size == 0:
                raise ValueError(f"'{name}' must be a non-empty 1-D array")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"'{name}' contains non-finite values")
            if not np.all(a > 0.0):
                raise ValueError(f"'{name}' must be strictly positive")
            a.setflags(write=False)
            arr[name] = a
        if len({a.size for a in arr.values()}) != 1:
            raise ValueError(f"arrays differ in length: {({k: a.size for k, a in arr.items()})}")
        box = self.box
        if box is None:
            box = ValidityBox({k: (float(arr[k].min()), float(arr[k].max())) for k in _BOX_KEYS[self.quantity]})
        else:
            if not isinstance(box, ValidityBox) or set(box.ranges) != set(_BOX_KEYS[self.quantity]):
                raise ValueError(f"box must be a ValidityBox over {_BOX_KEYS[self.quantity]}")
            for k in box.ranges:
                lo, hi = box.ranges[k]
                if arr[k].min() < lo or arr[k].max() > hi:
                    raise ValueError(f"the validity box of '{k}' does not contain the data")
        digest = _checksum(arr)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("checksum does not match the arrays")
        object.__setattr__(self, "arrays", arr)
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "checksum", digest)

    @property
    def validity_box(self) -> ValidityBox:
        """The resolved box (the given one, else the range of the data)."""
        assert self.box is not None
        return self.box

    @property
    def digest(self) -> str:
        """The sha256 checksum of the arrays."""
        assert self.checksum is not None
        return self.checksum

    @property
    def is_si(self) -> bool:
        return all(self.units[n] == SI_UNITS[k] for n, k in _QUANTITIES[self.quantity].items())

    def ranges(self) -> dict[str, tuple[float, float]]:
        """(min, max) of every array in its own unit."""
        return {k: (float(a.min()), float(a.max())) for k, a in self.arrays.items()}

    def verify(self) -> None:
        """Recompute the checksum; ValueError when the arrays no longer match."""
        if _checksum(self.arrays) != self.checksum:
            raise ValueError("checksum does not match the arrays")


def convert_to_si(record: MaterialDataRecord) -> MaterialDataRecord:
    """The same data in SI units (s, Pa, K, 1/s) with the box converted and a new checksum; provenance is kept and the
    origin says it was converted. The only place a unit factor is applied."""
    spec = _QUANTITIES[record.quantity]
    f = {n: UNIT_FACTORS[k][record.units[n]] for n, k in spec.items()}
    arrays = {n: record.arrays[n] * f[n] for n in spec}
    box = ValidityBox({k: (lo * f[k], hi * f[k]) for k, (lo, hi) in record.validity_box.ranges.items()})
    units = {n: SI_UNITS[k] for n, k in spec.items()}
    return MaterialDataRecord(record.quantity, arrays, units, record.source, record.origin + " [converted to SI]",
                              record.license, box)


def _require(record: MaterialDataRecord, quantity: str) -> None:
    if not isinstance(record, MaterialDataRecord):
        raise ValueError("a MaterialDataRecord is required")
    if record.quantity != quantity:
        raise ValueError(f"this fit needs a '{quantity}' record, got '{record.quantity}'")
    record.verify()
    if not record.is_si:
        raise ValueError(f"units {record.units} are not SI {SI_UNITS}: convert explicitly with convert_to_si")


# ----------------------------------------------------------------------------------------------- identifiability
@dataclass(frozen=True)
class IdentifiabilityReport:
    """SVD of the weighted Jacobian in scaled coordinates (module notes). ``weakest``: unit vector (scaled
    coordinates, sign: largest component positive) of the smallest singular value; ``std_*`` in parameter units."""

    names: tuple[str, ...]
    scale: np.ndarray
    singular_values: np.ndarray
    condition: float
    rank: int
    weakest: np.ndarray
    residual_dof: int
    residual_std: float | None
    std_nominal: np.ndarray | None
    std_residual: np.ndarray | None
    cond_limit: float

    @property
    def practically_unidentifiable(self) -> bool:
        return self.condition > self.cond_limit

    @property
    def identifiable(self) -> bool:
        return self.rank == len(self.names) and not self.practically_unidentifiable

    def weakest_combination(self) -> dict[str, float]:
        return {n: float(c) for n, c in zip(self.names, self.weakest, strict=True)}

    def describe(self, threshold: float = 0.05) -> str:
        """One line naming the weakest parameter combination (components with |c| ≥ ``threshold``)."""
        pairs = zip(self.names, self.weakest, strict=True)
        comps = sorted(((abs(c), n, c) for n, c in pairs if abs(c) >= threshold), reverse=True)
        terms = " ".join(f"{c:+.3f}*{n}" for _, n, c in comps)
        state = "NOT identifiable" if not self.identifiable else "identifiable"
        return (f"{state}: condition {self.condition:.3g}, rank {self.rank}/{len(self.names)}; weakest combination "
                f"(scaled coordinates, s_min {self.singular_values[-1]:.3g}): {terms}")


def identifiability_report(jac: Any, names: Sequence[str], scale: Any = None, weighted_residuals: Any = None,
                           rank_rtol: float = RANK_RTOL, cond_limit: float = COND_LIMIT) -> IdentifiabilityReport:
    """Report for the weighted Jacobian ``jac`` (m, p) = ∂(model/σ)/∂θ in parameter units, columns ``names``;
    ``scale`` (p,) typical magnitudes (default 1); ``weighted_residuals`` (m,) = (model − data)/σ at the estimate, used
    for ``std_residual`` when m > p."""
    J = np.asarray(jac, float)
    if J.ndim != 2 or J.shape[1] != len(names) or not np.all(np.isfinite(J)):
        raise ValueError("jac must be a finite (m, p) array with one column per name")
    m, p = J.shape
    s = np.ones(p) if scale is None else np.broadcast_to(np.asarray(scale, float), (p,)).copy()
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError("scales must be finite and positive")
    Jq = J * s
    _, sv, Vt = np.linalg.svd(Jq, full_matrices=True)
    sv_full = np.concatenate([sv, np.zeros(p - sv.size)])
    smax = float(sv_full.max(initial=0.0))
    rank = int((sv_full > rank_rtol * smax).sum()) if smax > 0 else 0
    cond = float(smax / sv_full.min()) if sv_full.min() > 0 else math.inf
    v = Vt[p - 1].copy()
    v *= 1.0 if v[int(np.argmax(np.abs(v)))] >= 0 else -1.0
    dof = m - p
    s_res: float | None = None
    std_nom = std_res = None
    if rank == p and m >= p:
        pinv = np.linalg.pinv(Jq)
        std_nom = np.sqrt(np.diag(pinv @ pinv.T)) * s
        if weighted_residuals is not None and dof > 0:
            r = np.asarray(weighted_residuals, float).reshape(-1)
            if r.shape != (m,):
                raise ValueError("weighted_residuals must have one entry per row of jac")
            s_res = float(math.sqrt(float(r @ r) / dof))
            std_res = std_nom * s_res
    return IdentifiabilityReport(tuple(names), s, sv_full, cond, rank, v, dof, s_res, std_nom, std_res, cond_limit)


# ----------------------------------------------------------------------------------------------- Prony series
def log_spaced_taus(lo: float, hi: float, n: int) -> np.ndarray:
    """n relaxation times log-spaced on [lo, hi] s (the usual fixed grid for the NNLS spectrum)."""
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi and int(n) >= 1):
        raise ValueError("need 0 < lo <= hi and n >= 1")
    return np.logspace(math.log10(lo), math.log10(hi), int(n))


@dataclass(frozen=True, eq=False)
class PronyFit:
    """A fitted relaxation series with its report, provenance and validity box; :meth:`predict` refuses times outside
    the box. ``n_starts`` / ``n_agree``: start sets run / agreeing with the best in ln τ to 1e-4 (0 for fixed times)."""

    series: PronySeries
    names: tuple[str, ...]
    free_times: bool
    report: IdentifiabilityReport
    weighted_rms: float
    box: ValidityBox
    checksum: str
    source: str
    n_starts: int = 0
    n_agree: int = 0

    @property
    def identifiable(self) -> bool:
        return self.report.identifiable

    def predict(self, t: Any) -> np.ndarray:
        self.box.check(time=t)
        return self.series.relaxation(t)


def _prony_names(taus: np.ndarray, free: bool) -> tuple[str, ...]:
    k = taus.size
    out = ("equilibrium",) + tuple(f"A_{tau:g}" for tau in taus)
    return out + tuple(f"ln_tau_{i + 1}" for i in range(k)) if free else out


def _prony_data(record: MaterialDataRecord, rel_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    _require(record, "relaxation_modulus")
    if not (math.isfinite(rel_sigma) and rel_sigma > 0):
        raise ValueError("rel_sigma must be finite and positive")
    t, y = record.arrays["time"], record.arrays["modulus"]
    return t, y, rel_sigma * y, float(y.max())


def _prony_design(t: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(t), np.exp(-t[:, None] / taus[None, :])])


def _nnls_amplitudes(t: np.ndarray, y: np.ndarray, sig: np.ndarray, taus: np.ndarray, M: float) -> np.ndarray:
    X = _prony_design(t, taus)
    x, _ = nnls(X / sig[:, None] * M, y / sig, maxiter=50 * X.shape[1])   # unknowns in units of M: well scaled
    return x * M


def fit_prony_fixed_times(record: MaterialDataRecord, taus: Any, rel_sigma: float = 0.01) -> PronyFit:
    """M∞ ≥ 0 and amplitudes ≥ 0 at the fixed times ``taus`` by NNLS on (X x − y)/σ, σ_i = ``rel_sigma``·y_i. The report
    uses the full design (all columns), so a degenerate design is flagged whatever NNLS returns."""
    t, y, sig, M = _prony_data(record, rel_sigma)
    tau = np.sort(np.asarray(taus, float).reshape(-1))
    if tau.size == 0 or not (np.all(np.isfinite(tau)) and np.all(tau > 0)):
        raise ValueError("taus must be finite and positive")
    x = _nnls_amplitudes(t, y, sig, tau, M)
    series = PronySeries(float(x[0]), x[1:], tau)
    names = _prony_names(tau, False)
    X = _prony_design(t, tau)
    r = (X @ x - y) / sig
    rep = identifiability_report(X / sig[:, None], names, np.full(x.size, M), r)
    return PronyFit(series, names, False, rep, float(math.sqrt(np.mean(r ** 2))), record.validity_box, record.digest,
                    record.source)


def _free_model(t: np.ndarray, k: int):  # noqa: ANN202
    def model(theta: np.ndarray) -> np.ndarray:
        return theta[0] + np.exp(-t[:, None] / np.exp(theta[k + 1:])[None, :]) @ theta[1:k + 1]

    def jac(theta: np.ndarray) -> np.ndarray:
        E = np.exp(-t[:, None] / np.exp(theta[k + 1:])[None, :])
        return np.column_stack([np.ones_like(t), E, E * (t[:, None] / np.exp(theta[k + 1:])[None, :]) * theta[1:k + 1]])

    return model, jac


def fit_prony_free_times(record: MaterialDataRecord, starts: Sequence[Any], rel_sigma: float = 0.01,
                         tau_bounds: tuple[float, float] | None = None) -> PronyFit:
    """θ = (M∞, A_1..A_k, ln τ_1..ln τ_k) fitted with :func:`identify` from every start set in ``starts`` (each a
    sequence of k times; the amplitudes of a start are the NNLS solution at its times); the lowest cost wins, times
    are returned sorted. ``tau_bounds`` default (t_min/10, 10 t_max). A start outside the bounds raises."""
    t, y, sig, M = _prony_data(record, rel_sigma)
    S = [np.sort(np.asarray(s, float).reshape(-1)) for s in starts]
    if not S or any(s.size != S[0].size or s.size == 0 for s in S):
        raise ValueError("starts: a non-empty list of equally long, non-empty time sets")
    k = S[0].size
    lo, hi = tau_bounds if tau_bounds is not None else (float(t.min()) / 10.0, float(t.max()) * 10.0)
    if not (0 < lo < hi):
        raise ValueError("tau_bounds need 0 < lo < hi")
    if any(s.min() < lo or s.max() > hi for s in S):
        raise ValueError("every start time must lie inside tau_bounds")
    obs = Observations(y, sig, record.source, f"relaxation modulus, record {record.digest[:12]}")
    model, jac = _free_model(t, k)
    bounds = (np.r_[0.0, np.zeros(k), np.full(k, math.log(lo))],
              np.r_[10 * M, np.full(k, 10 * M), np.full(k, math.log(hi))])
    scale = np.r_[np.full(k + 1, M), np.ones(k)]
    names = _prony_names(np.ones(k), True)
    runs: list[Identification] = []
    for s in S:
        x0 = np.minimum(_nnls_amplitudes(t, y, sig, s, M), 10 * M)
        runs.append(identify(model, obs, np.r_[x0, np.log(s)], bounds, names, jacobian=jac, scale=scale))
    runs.sort(key=lambda r: r.cost)
    best = runs[0]
    order = np.argsort(best.theta[k + 1:])
    amps = best.theta[1:k + 1][order]
    lt = best.theta[k + 1:][order]

    def lt_sorted(r: Identification) -> np.ndarray:
        return np.sort(r.theta[k + 1:])

    n_agree = sum(bool(np.max(np.abs(lt_sorted(r) - lt)) <= 1e-4) for r in runs)
    theta = np.r_[best.theta[0], amps, lt]
    series = PronySeries(float(theta[0]), amps, np.exp(lt))
    names = _prony_names(series.taus, True)
    r = (model(theta) - y) / sig
    rep = identifiability_report(jac(theta) / sig[:, None], names, scale, r)
    return PronyFit(series, names, True, rep, weighted_rms(model, obs, theta), record.validity_box, record.digest,
                    record.source, len(runs), n_agree)


# ----------------------------------------------------------------------------------------------- Norton creep
def _norton_log_rate(ln_A: Any, n: Any, Q: Any, stress: Any, T: Any, sigma_ref: float, T_ref: float) -> np.ndarray:
    """ln ṗ = ln A + n ln(σ/σ_ref) − (Q/R)(1/T − 1/T_ref)."""
    return (ln_A + n * np.log(np.asarray(stress, float) / sigma_ref)
            - (Q / R_GAS) * (1.0 / np.asarray(T, float) - 1.0 / T_ref))


def _norton_columns(stress: np.ndarray, T: np.ndarray, sigma_ref: float, T_ref: float) -> np.ndarray:
    """∂ ln ṗ/∂(ln A, n, Q) [1, 1, J⁻¹ mol] (exact: the log law is linear in these)."""
    return np.column_stack([np.ones_like(stress), np.log(stress / sigma_ref), -(1.0 / T - 1.0 / T_ref) / R_GAS])


_NORTON_NAMES = ("ln_A", "n", "Q")


@dataclass(frozen=True, eq=False)
class NortonFit:
    """A, n, Q (rate A at σ_ref, T_ref; Q in J/mol) with report, provenance and validity box; :meth:`predict_rate`
    refuses stresses / temperatures outside the box. ``identification`` is the :func:`identify` result of the nonlinear
    fit (None for the log-linear one)."""

    A: float
    n: float
    Q: float
    sigma_ref: float
    T_ref: float
    method: str
    report: IdentifiabilityReport
    weighted_rms: float
    box: ValidityBox
    checksum: str
    source: str
    identification: Identification | None = None

    @property
    def ln_A(self) -> float:
        return math.log(self.A)

    @property
    def identifiable(self) -> bool:
        return self.report.identifiable

    def predict_rate(self, stress: Any, temperature: Any) -> np.ndarray:
        """ṗ [1/s] at the given stresses [Pa] and temperatures [K] (broadcast), inside the box only."""
        self.box.check(stress=stress, temperature=temperature)
        return np.exp(_norton_log_rate(self.ln_A, self.n, self.Q, stress, temperature, self.sigma_ref, self.T_ref))


def _norton_inputs(record: MaterialDataRecord, sigma_ref: float, T_ref: float, rel_sigma: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(record, "creep_rate")
    for nm, v in (("sigma_ref", sigma_ref), ("T_ref", T_ref), ("rel_sigma", rel_sigma)):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{nm} must be finite and positive")
    s, T, r = record.arrays["stress"], record.arrays["temperature"], record.arrays["rate"]
    if r.size < 3:
        raise ValueError("Norton A, n, Q needs at least 3 experiments")
    return s, T, r


def _norton_report(X: np.ndarray, rate_pred: np.ndarray, rate_obs: np.ndarray, rel_sigma: float, nonlinear: bool,
                   T_ref: float) -> IdentifiabilityReport:
    sig = rel_sigma * rate_obs if nonlinear else np.full(rate_obs.size, rel_sigma)
    J = (rate_pred[:, None] * X if nonlinear else X) / sig[:, None]            # ∂(model/σ)/∂(ln A, n, Q)
    r = ((rate_pred - rate_obs) if nonlinear else np.log(rate_pred / rate_obs)) / sig
    return identifiability_report(J, _NORTON_NAMES, np.array([1.0, 1.0, R_GAS * T_ref]), r)


def fit_norton_loglinear(record: MaterialDataRecord, sigma_ref: float, T_ref: float,
                         rel_sigma: float = 0.01) -> NortonFit:
    """Reference fit: ordinary least squares in ln ṗ (unit weights, nominal σ_ln = ``rel_sigma``). Exact for noise-free
    data. A rank-deficient design (one temperature, one stress) returns the minimum-norm solution with a report that
    says so: no parameter recovery is claimed then."""
    s, T, r = _norton_inputs(record, sigma_ref, T_ref, rel_sigma)
    X = _norton_columns(s, T, sigma_ref, T_ref)
    beta = np.linalg.lstsq(X, np.log(r), rcond=None)[0]
    pred = np.exp(X @ beta)
    rep = _norton_report(X, pred, r, rel_sigma, False, T_ref)
    rms = float(math.sqrt(np.mean((np.log(pred / r) / rel_sigma) ** 2)))
    return NortonFit(float(math.exp(beta[0])), float(beta[1]), float(beta[2]), sigma_ref, T_ref, "loglinear", rep, rms,
                     record.validity_box, record.digest, record.source)


def fit_norton_nonlinear(record: MaterialDataRecord, sigma_ref: float, T_ref: float, theta0: Any = None,
                         rel_sigma: float = 0.01, bounds: tuple[Any, Any] | None = None) -> NortonFit:
    """Fit ṗ itself with :func:`identify` (σ_i = ``rel_sigma``·ṗ_i, analytic Jacobian). ``theta0`` = (A, n, Q), default
    the log-linear solution; ``bounds`` default A ∈ [1e-6, 1e6]·A_loglinear, n ∈ [1e-3, 50], Q ∈ [0, 2e6] J/mol."""
    s, T, r = _norton_inputs(record, sigma_ref, T_ref, rel_sigma)
    ll = fit_norton_loglinear(record, sigma_ref, T_ref, rel_sigma)
    t0 = np.array([ll.A, ll.n, max(ll.Q, 0.0)] if theta0 is None else np.asarray(theta0, float).reshape(3))
    if bounds is None:
        bounds = (np.array([1e-6 * ll.A, 1e-3, 0.0]), np.array([1e6 * ll.A, 50.0, 2e6]))
    X = _norton_columns(s, T, sigma_ref, T_ref)

    def model(th: np.ndarray) -> np.ndarray:
        return np.exp(_norton_log_rate(math.log(th[0]), th[1], th[2], s, T, sigma_ref, T_ref))

    def jac(th: np.ndarray) -> np.ndarray:
        return model(th)[:, None] * X * np.array([1.0 / th[0], 1.0, 1.0])

    obs = Observations(r, rel_sigma * r, record.source, f"creep rate, record {record.digest[:12]}")
    ident = identify(model, obs, t0, bounds, _NORTON_NAMES, jacobian=jac,
                     scale=np.array([abs(t0[0]), max(abs(t0[1]), 1.0), max(t0[2], R_GAS * T_ref)]))
    A, n, Q = (float(v) for v in ident.theta)
    pred = model(ident.theta)
    rep = _norton_report(X, pred, r, rel_sigma, True, T_ref)
    return NortonFit(A, n, Q, sigma_ref, T_ref, "nonlinear", rep, ident.weighted_rms, record.validity_box,
                     record.digest, record.source, ident)
