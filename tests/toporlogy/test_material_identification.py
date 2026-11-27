"""Inverse identification of Prony and Norton parameters from SYNTHETIC data, identifiability and data provenance
(prereg ``docs/foundations/prereg/D_identification.md``, checks D4-D7; D8 is NOT VERIFIED: no licensed data).

All numbers are synthetic; nothing here is material data. References are written here with plain numpy (closed forms,
dense lstsq, hashlib), not with the production fit functions."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from toporlogy.optimization.identification import Observations, verify_model_jacobian
from toporlogy.optimization.material_identification import (MaterialDataRecord, ValidityBox, ValidityBoxError,
                                                            _free_model, convert_to_si, fit_norton_loglinear,
                                                            fit_norton_nonlinear, fit_prony_fixed_times,
                                                            fit_prony_free_times, identifiability_report)

R = 8.314462618
# ---- synthetic truth (written here, never read from the production code)
M_INF = 2.0e8
AMPS = np.array([1.5e9, 8.0e8, 4.0e8])
TAUS = np.array([0.1, 10.0, 1000.0])
T_GRID = np.logspace(-3, 5, 81)
A_TRUE, N_TRUE, Q_TRUE, S_REF, T_REF = 2.0e-9, 4.5, 2.5e5, 75e6, 600.0
STRESSES = np.array([50e6, 75e6, 100e6])
TEMPS = np.array([560.0, 600.0, 640.0])
# held-out times: geometric midpoints of the training grid (set-up fix: the preregistered logspace(-2.7, 4.7, 37)
# has 3 points that coincide with grid points, so it is not a held-out set)
T_HELD = np.sqrt(T_GRID[:-1] * T_GRID[1:])


def relax(t, inf=M_INF, amps=AMPS, taus=TAUS):
    t = np.asarray(t, float)
    return inf + (np.asarray(amps) * np.exp(-t[..., None] / np.asarray(taus))).sum(-1)


def norton_rate(s, T, A=A_TRUE, n=N_TRUE, Q=Q_TRUE):
    return A * np.exp(-(Q / R) * (1.0 / np.asarray(T, float) - 1.0 / T_REF)) * (np.asarray(s, float) / S_REF) ** n


def relax_record(t=T_GRID, y=None, **kw):
    t = np.asarray(t, float)
    y = relax(t) if y is None else y
    return MaterialDataRecord("relaxation_modulus", {"time": t, "modulus": y}, {"time": "s", "modulus": "Pa"},
                              "synthetic", "test generator: closed-form 3-branch Prony series, noise-free", **kw)


def grid():
    s, T = np.meshgrid(STRESSES, TEMPS, indexing="ij")
    return s.ravel(), T.ravel()


def creep_record(s=None, T=None, rate=None, units=None, **kw):
    if s is None:
        s, T = grid()
    rate = norton_rate(s, T) if rate is None else rate
    return MaterialDataRecord("creep_rate", {"stress": s, "temperature": T, "rate": rate},
                              units or {"stress": "Pa", "temperature": "K", "rate": "1/s"}, "synthetic",
                              "test generator: closed-form Norton law with Arrhenius factor", **kw)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1.0)))


# ------------------------------------------------------------------------------------------------ D4 Prony
def test_D4a_fixed_times_nnls_exact_recovery():
    fit = fit_prony_fixed_times(relax_record(), TAUS)
    assert rel(np.r_[fit.series.inf, fit.series.amplitudes], np.r_[M_INF, AMPS]) <= 1e-8
    assert np.array_equal(fit.series.taus, TAUS)
    assert fit.source == "synthetic"


def test_D4b_held_out_prediction():
    fit = fit_prony_fixed_times(relax_record(), TAUS)
    t_new = T_HELD
    assert not np.any(np.isin(t_new, T_GRID))
    assert rel(fit.predict(t_new), relax(t_new)) <= 1e-8


def test_D4c_free_times_from_stated_starts():
    starts = [(0.03, 3.0, 300.0), (0.3, 30.0, 3000.0), (0.03, 30.0, 300.0)]
    fit = fit_prony_free_times(relax_record(), starts)
    print("D4c: n_starts", fit.n_starts, "n_agree", fit.n_agree, "tau", fit.series.taus, "rms", fit.weighted_rms)
    assert fit.free_times and fit.n_starts == 3
    assert rel(fit.series.taus, TAUS) <= 1e-5
    assert rel(fit.series.amplitudes, AMPS) <= 1e-5
    assert abs(fit.series.inf / M_INF - 1.0) <= 1e-5
    t_new = T_HELD
    assert rel(fit.predict(t_new), relax(t_new)) <= 1e-6


def test_D4d_well_spaced_design_is_well_conditioned():
    rep = fit_prony_fixed_times(relax_record(), TAUS).report
    print("D4d: condition", rep.condition, rep.describe())
    assert rep.rank == 4 and rep.condition < 1e3 and not rep.practically_unidentifiable and rep.identifiable


def test_D4e_free_time_jacobian_matches_central_differences():
    model, jac = _free_model(T_GRID, 3)
    theta = np.r_[M_INF, AMPS, np.log(TAUS)]
    scale = np.r_[np.full(4, 1e9), np.ones(3)]
    assert verify_model_jacobian(model, jac, theta, scale) <= 1e-6


# ------------------------------------------------------------------------------------------------ D5 rank deficiency
def test_D5a_times_far_below_tau_are_flagged_not_recovered():
    t = np.logspace(-3, -1, 31)
    rep = fit_prony_fixed_times(relax_record(t), TAUS).report
    w = rep.weakest
    print("D5a: condition", rep.condition, "weakest", rep.weakest_combination(), "|", rep.describe())
    assert rep.names == ("equilibrium", "A_0.1", "A_10", "A_1000")
    assert rep.condition > 1e4 and rep.practically_unidentifiable and not rep.identifiable
    assert w[0] ** 2 + w[2] ** 2 + w[3] ** 2 >= 0.99
    assert abs(w[3]) >= 0.5 and abs(w[0]) >= 0.4
    assert "A_1000" in rep.describe() and "equilibrium" in rep.describe()


@pytest.mark.xfail(strict=True, reason="measured: condition 8.0e8, rank 3 of 4 at RANK_RTOL 1e-8, so std_nominal is None "
                   "(the preregistered condition estimate was only a lower bound ~1e5; the covariance is undefined at that rank, not > 1)")
def test_D5a_std_of_the_weak_amplitude_exceeds_the_amplitude():
    rep = fit_prony_fixed_times(relax_record(np.logspace(-3, -1, 31)), TAUS).report
    assert rep.std_nominal is not None
    assert rep.std_nominal[3] / AMPS[2] > 1.0


def test_D5a_post_hoc_std_with_a_stricter_rank_tolerance():
    """POST HOC (not preregistered): same design, covariance forced with rank_rtol 1e-12 from a Jacobian written here."""
    t = np.logspace(-3, -1, 31)
    X = np.column_stack([np.ones_like(t), np.exp(-t[:, None] / TAUS[None, :])])
    sig = 0.01 * relax(t)
    rep = identifiability_report(X / sig[:, None], ("eq", "a1", "a2", "a3"), np.full(4, 2.885e9), rank_rtol=1e-12)
    ratio = rep.std_nominal[3] / AMPS[2]
    print("D5a post hoc: std_nominal(A_1000)/A_1000 =", ratio, "condition", rep.condition)
    assert ratio > 1.0


def test_D5b_single_temperature_makes_Q_ambiguous():
    s, T = np.array([50e6, 75e6, 100e6]), np.full(3, 640.0)
    rep = fit_norton_loglinear(creep_record(s, T), S_REF, T_REF).report
    g = 1.0 - 600.0 / 640.0
    v = np.array([g, 0.0, -1.0]) / np.sqrt(1.0 + g * g)           # (ln A, n, Q') null vector derived in the prereg
    print("D5b: condition", rep.condition, "weakest", rep.weakest, rep.describe())
    assert rep.rank == 2 and (rep.condition > 1e12)
    assert min(np.max(np.abs(rep.weakest - v)), np.max(np.abs(rep.weakest + v))) <= 1e-8
    assert rep.std_nominal is None and rep.std_residual is None and not rep.identifiable
    # the same stresses at two temperatures are identifiable
    s2, T2 = np.repeat(STRESSES, 2), np.tile([600.0, 640.0], 3)
    rep2 = fit_norton_loglinear(creep_record(s2, T2), S_REF, T_REF).report
    assert rep2.rank == 3 and rep2.condition < 1e4 and rep2.identifiable


# ------------------------------------------------------------------------------------------------ D6 Norton
def test_D6a_loglinear_noise_free_exact():
    fit = fit_norton_loglinear(creep_record(), S_REF, T_REF)
    assert rel(fit.A, A_TRUE) <= 1e-8 and rel(fit.n, N_TRUE) <= 1e-8 and rel(fit.Q, Q_TRUE) <= 1e-8
    assert fit.identifiable


def test_D6b_nonlinear_noise_free_exact_from_offset_start():
    fit = fit_norton_nonlinear(creep_record(), S_REF, T_REF, theta0=(3 * A_TRUE, N_TRUE + 0.7, 0.6 * Q_TRUE))
    print("D6b: nfev", fit.identification.n_evaluations, "rms", fit.weighted_rms)
    assert rel(fit.A, A_TRUE) <= 1e-8 and rel(fit.n, N_TRUE) <= 1e-8 and rel(fit.Q, Q_TRUE) <= 1e-8


def _noisy():
    s, T = grid()
    z = np.random.default_rng(7).standard_normal(9)
    return creep_record(s, T, norton_rate(s, T) * (1.0 + 0.01 * z)), s, T


def test_D6c_noisy_loglinear_coverage_covariance_and_holdout():
    rec, s, T = _noisy()
    fit = fit_norton_loglinear(rec, S_REF, T_REF)
    # independent dense reference written here
    X = np.column_stack([np.ones(9), np.log(s / S_REF), -(1.0 / T - 1.0 / T_REF) / R])
    y = np.log(rec.arrays["rate"])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    cov = (res @ res / 6.0) * np.linalg.inv(X.T @ X)
    std_ref = np.sqrt(np.diag(cov))
    std = fit.report.std_residual
    truth = np.array([np.log(A_TRUE), N_TRUE, Q_TRUE])
    est = np.array([np.log(fit.A), fit.n, fit.Q])
    print("D6c: estimate", est, "truth", truth, "std", std, "z", (est - truth) / std)
    assert np.max(np.abs(std / std_ref - 1.0)) <= 1e-8
    assert np.max(np.abs(est - beta) / std_ref) <= 1e-8
    assert np.all(np.abs(est - truth) <= 3.0 * std)
    hs, hT = np.array([62.5e6, 90e6]), np.array([580.0, 620.0])
    err = fit.predict_rate(hs, hT) / norton_rate(hs, hT) - 1.0
    print("D6c: held-out relative prediction error", err)
    assert np.all(np.abs(err) <= 0.05)


def test_D6d_nonlinear_noisy_agrees_with_loglinear_to_a_fraction_of_a_std():
    rec, _, _ = _noisy()
    ll = fit_norton_loglinear(rec, S_REF, T_REF)
    nl = fit_norton_nonlinear(rec, S_REF, T_REF)
    d = np.abs(np.array([np.log(nl.A) - np.log(ll.A), nl.n - ll.n, nl.Q - ll.Q]))
    print("D6d: |difference| / std", d / ll.report.std_residual)
    assert np.all(d <= 0.1 * ll.report.std_residual)


def test_D6e_wrong_time_unit_is_caught_by_the_unit_check_and_shifts_A_by_3600():
    s, T = grid()
    r_h = norton_rate(s, T) * 3600.0                            # the same rates per hour
    labelled = creep_record(s, T, r_h, units={"stress": "Pa", "temperature": "K", "rate": "1/h"})
    with pytest.raises(ValueError, match="convert_to_si"):
        fit_norton_loglinear(labelled, S_REF, T_REF)
    good = fit_norton_loglinear(convert_to_si(labelled), S_REF, T_REF)
    assert rel(good.A, A_TRUE) <= 1e-8
    wrong = fit_norton_loglinear(creep_record(s, T, r_h), S_REF, T_REF)       # same numbers, label says 1/s
    assert rel(wrong.A, 3600.0 * A_TRUE) <= 1e-8 and rel(wrong.n, N_TRUE) <= 1e-8 and rel(wrong.Q, Q_TRUE) <= 1e-8


def test_cross_check_prediction_agrees_with_core_creep_arrhenius_factor():
    """Consistency (not independent): the rate law of the fit equals A f(T)(sigma/sigma_ref)^n with core ArrheniusFactor."""
    from toporlogy.core.creep import ArrheniusFactor
    fit = fit_norton_loglinear(creep_record(), S_REF, T_REF)
    s, T = np.array([60e6, 90e6]), np.array([570.0, 630.0])
    ref = fit.A * ArrheniusFactor(fit.Q, T_REF).factor(T) * (s / S_REF) ** fit.n
    assert rel(fit.predict_rate(s, T), ref) <= 1e-12


# ------------------------------------------------------------------------------------------------ D7 invalid data
SI_R = {"time": "s", "modulus": "Pa"}


def _mk(arrays=None, units=None, **kw):
    arrays = arrays or {"time": T_GRID, "modulus": relax(T_GRID)}
    args = dict(source="synthetic", origin="test generator")
    args.update(kw)
    return MaterialDataRecord("relaxation_modulus", arrays, units or SI_R, **args)


@pytest.mark.parametrize("units", [{"time": "furlong", "modulus": "Pa"}, {"time": "s", "modulus": "s"},
                                   {"time": "s"}, {"time": "s", "modulus": "psi"}])
def test_D7a_invalid_units(units):
    with pytest.raises(ValueError, match="unit"):
        _mk(units=units)


def test_D7a_celsius_is_refused():
    with pytest.raises(ValueError, match="unit"):
        creep_record(units={"stress": "Pa", "temperature": "degC", "rate": "1/s"})


def test_D7b_missing_or_invalid_provenance():
    arrays = {"time": T_GRID, "modulus": relax(T_GRID)}
    with pytest.raises(ValueError, match="origin"):
        MaterialDataRecord("relaxation_modulus", arrays, SI_R, "synthetic", "  ")
    with pytest.raises(ValueError, match="source"):
        MaterialDataRecord("relaxation_modulus", arrays, SI_R, "simulated", "x")
    with pytest.raises(ValueError, match="license"):
        MaterialDataRecord("relaxation_modulus", arrays, SI_R, "measured", "lab report 1")
    ok = MaterialDataRecord("relaxation_modulus", arrays, SI_R, "measured", "lab report 1", license="CC-BY-4.0 (test)")
    assert ok.source == "measured"                                      # the check is not a blanket refusal


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
def test_D7c_bad_values(bad):
    y = relax(T_GRID)
    y[3] = bad
    with pytest.raises(ValueError):
        _mk({"time": T_GRID, "modulus": y})
    t = T_GRID.copy()
    t[5] = bad
    with pytest.raises(ValueError):
        _mk({"time": t, "modulus": relax(T_GRID)})


def test_D7c_shape_checksum_and_box_errors():
    with pytest.raises(ValueError, match="length"):
        _mk({"time": T_GRID[:-1], "modulus": relax(T_GRID)})
    with pytest.raises(ValueError, match="non-empty 1-D"):
        _mk({"time": np.array([]), "modulus": np.array([])})
    with pytest.raises(ValueError, match="1-D"):
        _mk({"time": T_GRID.reshape(9, 9), "modulus": relax(T_GRID).reshape(9, 9)})
    with pytest.raises(ValueError, match="arrays must be exactly"):
        _mk({"time": T_GRID})
    good = _mk()
    with pytest.raises(ValueError, match="checksum"):
        _mk(checksum="0" * 64)
    _mk(checksum=good.checksum)                                         # the right checksum is accepted
    with pytest.raises(ValueError, match="does not contain"):
        _mk(box=ValidityBox({"time": (1e-2, 1e3)}))
    with pytest.raises(ValueError):
        ValidityBox({"time": (2.0, 1.0)})


def test_D7d_prediction_outside_the_box_raises():
    fit = fit_prony_fixed_times(_mk(), TAUS)
    assert fit.predict(1e5).shape == () or fit.predict(1e5).size == 1       # box edges are allowed
    fit.predict(1e-3)
    for t in (1e6, 1e-4, np.nan, np.inf):
        with pytest.raises(ValidityBoxError):
            fit.predict(t)
    with pytest.raises(ValidityBoxError):
        fit.predict(np.array([1.0, 1e6]))
    nf = fit_norton_loglinear(creep_record(), S_REF, T_REF)
    nf.predict_rate(100e6, 640.0)
    nf.predict_rate(50e6, 560.0)
    for s, T in ((200e6, 600.0), (75e6, 700.0), (75e6, 500.0), (10e6, 600.0), (np.nan, 600.0)):
        with pytest.raises(ValidityBoxError):
            nf.predict_rate(s, T)
    assert issubclass(ValidityBoxError, ValueError)
    with pytest.raises(ValueError, match="not a coordinate"):
        fit.box.check(stress=1.0)


def test_D7e_checksum_matches_independent_hash_and_detects_one_ulp():
    rec = _mk()
    h = hashlib.sha256()
    for name in ("modulus", "time"):                                        # sorted names
        a = np.ascontiguousarray(rec.arrays[name], dtype="<f8")
        h.update(name.encode() + b"\x00" + b"<f8" + repr(a.shape).encode() + b"\x00" + a.tobytes())
    assert rec.checksum == h.hexdigest()
    assert _mk().checksum == rec.checksum
    y = relax(T_GRID)
    y[10] = np.nextafter(y[10], np.inf)
    assert _mk({"time": T_GRID, "modulus": y}).checksum != rec.checksum
    rec.verify()
    assert not rec.arrays["time"].flags.writeable


def test_D7f_fits_check_quantity_start_sets_and_units():
    with pytest.raises(ValueError, match="creep_rate"):
        fit_norton_loglinear(_mk(), S_REF, T_REF)
    with pytest.raises(ValueError, match="relaxation_modulus"):
        fit_prony_fixed_times(creep_record(), TAUS)
    with pytest.raises(ValueError, match="inside tau_bounds"):
        fit_prony_free_times(_mk(), [(1e-9, 1.0, 1e3)])
    with pytest.raises(ValueError, match="SI"):
        fit_prony_fixed_times(_mk(arrays={"time": T_GRID / 3600.0, "modulus": relax(T_GRID) / 1e9},
                                  units={"time": "h", "modulus": "GPa"}), TAUS)
    with pytest.raises(ValueError, match="at least 3"):
        fit_norton_loglinear(creep_record(STRESSES[:2], TEMPS[:2]), S_REF, T_REF)
    with pytest.raises(ValueError):
        identifiability_report(np.ones((4, 2)), ("a",))
    assert Observations(np.ones(2), np.ones(2), "synthetic").kind == "synthetic"     # the reused type


@pytest.mark.skip(reason="D8 NOT VERIFIED: no licensed measured data is available; nothing here is real material data")
def test_D8_real_data_held_out_experiment():
    raise AssertionError("not run")
