"""Parameter identification of ply constants from laminate coupons: records I1-I7 of
``docs/olcumler/laminat_uq_onkayit.md``.

The data are SYNTHETIC: the CLT model of the library (optionally with an added shear nonlinearity) plus seeded
Gaussian noise. No measured data is used.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from toporlogy.core import laminate as L
from toporlogy.core.anisotropy import compliance_derivative
from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material import plane_stress_derivative
from toporlogy.optimization import identification as I

NAMES = ("E1", "E2", "G12", "nu12")
TRUE = np.array([140e3, 10e3, 5e3, 0.30])
LOW = np.array([50e3, 3e3, 1e3, 0.05])
HIGH = np.array([300e3, 30e3, 20e3, 0.45])
START = np.array([100e3, 8e3, 3e3, 0.25])
G23, NU23, T_PLY = 3.5e3, 0.4, 0.125

STACKS = {"0": [0.0] * 8, "90": [90.0] * 8, "pm45": [45, -45, 45, -45, -45, 45, -45, 45],
          "qi": [0, 45, 90, 90, 45, 0], "pm30": [30, -30, -30, 30]}
TRAIN = (("0", (50.0, 0, 0)), ("90", (20.0, 0, 0)), ("pm45", (50.0, 0, 0)))
HELD = (("qi", (150.0, 0, 0)), ("qi", (0, 150.0, 0)), ("pm30", (150.0, 0, 0)), ("pm30", (0, 60.0, 0)))


def material(theta: np.ndarray) -> ElasticTensor:
    E1, E2, G12, nu12 = theta
    return ElasticTensor.orthotropic(E1, E2, E2, G12, G12, G23, nu12, nu12, NU23)


def coupon(theta: np.ndarray, stack: str, load: tuple) -> np.ndarray:
    """Mid-plane strains (εx, εy) of one coupon."""
    lam = L.Laminate.from_angles(material(theta), STACKS[stack], T_PLY)
    return lam.response(N=load).eps0[:2]


def model_of(coupons: tuple):
    return lambda theta: np.concatenate([coupon(theta, s, n) for s, n in coupons])


def jacobian_of(coupons: tuple):
    """Analytic ∂(εx, εy)/∂θ: dS (engineering constants) → dC = −C dS C → dQ → dABD → −ABD⁻¹ dABD ABD⁻¹ r."""

    def jac(theta: np.ndarray) -> np.ndarray:
        E1, E2, _, nu12 = theta
        mat = material(theta)
        C = mat.stiffness()
        dS = [np.zeros((6, 6)) for _ in range(4)]
        dS[0][0, 0] = -1 / E1**2
        dS[0][0, 1] = dS[0][1, 0] = dS[0][0, 2] = dS[0][2, 0] = nu12 / E1**2
        dS[1][1, 1] = dS[1][2, 2] = -1 / E2**2
        dS[1][1, 2] = dS[1][2, 1] = NU23 / E2**2
        dS[2][4, 4] = dS[2][5, 5] = -1 / theta[2]**2
        dS[3][0, 1] = dS[3][1, 0] = dS[3][0, 2] = dS[3][2, 0] = -1 / E1
        dQ = [plane_stress_derivative(C, compliance_derivative(C, d)) for d in dS]    # compliance_derivative(C, dS) = dC
        rows = []
        for stack, load in coupons:
            lam = L.Laminate.from_angles(mat, STACKS[stack], T_PLY)
            K = lam.abd()
            e = np.linalg.solve(K, np.concatenate([load, np.zeros(3)]))
            cols = []
            for d in dQ:
                dK = L.abd_matrix(L.transformed_reduced_stiffness(d, lam.angles), lam.thicknesses)
                cols.append(-np.linalg.solve(K, dK @ e)[:2])
            rows.append(np.column_stack(cols))
        return np.vstack(rows)

    return jac


def noisy(values: np.ndarray, level: float, seed: int, description: str) -> I.Observations:
    sigma = level * np.abs(values)
    y = values + sigma * np.random.default_rng(seed).standard_normal(values.size)
    return I.Observations(y, sigma, "synthetic", description)


def test_i1_analytic_jacobian_matches_central_differences() -> None:
    for coupons in (TRAIN, HELD):
        err = I.check_jacobian(model_of(coupons), jacobian_of(coupons), START, START)
        assert err <= 1e-6, err


def test_i2_noise_free_recovery() -> None:
    y = model_of(TRAIN)(TRUE)
    obs = I.Observations(y, 0.01 * np.abs(y), "synthetic", "noise free")
    r = I.identify(model_of(TRAIN), obs, START, (LOW, HIGH), NAMES, jacobian=jacobian_of(TRAIN))
    assert r.success and r.identifiable and r.kind == "synthetic"
    assert np.abs(r.theta / TRUE - 1).max() <= 1e-8, r.theta


def noisy_identification() -> tuple[I.Identification, I.Observations]:
    obs = noisy(model_of(TRAIN)(TRUE), 0.01, 20, "1 % noise, training coupons")
    held = noisy(model_of(HELD)(TRUE), 0.01, 21, "1 % noise, held-out coupons")
    r = I.identify(model_of(TRAIN), obs, START, (LOW, HIGH), NAMES, jacobian=jacobian_of(TRAIN))
    return r, held


def test_i3_noisy_estimate_and_held_out_validation() -> None:
    r, held = noisy_identification()
    assert r.identifiable and r.std is not None
    assert np.abs((r.theta - TRUE) / r.std).max() <= 4.0, (r.theta - TRUE) / r.std
    rms = I.weighted_rms(model_of(HELD), held, r.theta)
    assert 0.5 <= rms <= 1.6, rms


def test_i4_a_zero_degree_coupon_alone_cannot_identify_e2_and_g12() -> None:
    coupons = (("0", (50.0, 0, 0)),)
    y = model_of(coupons)(TRUE)
    obs = I.Observations(y, 0.01 * np.abs(y), "synthetic")
    r = I.identify(model_of(coupons), obs, START, (LOW, HIGH), NAMES, jacobian=jacobian_of(coupons))
    assert r.rank == 2 and not r.identifiable
    assert r.covariance is None and r.std is None
    assert r.null_space.shape == (4, 2)
    for v in r.null_space.T:                                    # entirely in the (E2, G12) plane
        assert np.linalg.norm(v[[1, 2]]) >= 0.99, v


def test_i5_multi_start_finds_one_minimum() -> None:
    obs = noisy(model_of(TRAIN)(TRUE), 0.01, 20, "1 % noise")
    runs, agree = I.multi_start_identify(model_of(TRAIN), obs, (LOW, HIGH), NAMES, n_starts=16, seed=5,
                                         jacobian=jacobian_of(TRAIN))
    assert len(runs) == 16 and agree >= 14, (agree, [r.cost for r in runs])


def shear_softening(theta: np.ndarray, stack: str, load: tuple, tau0: float = 80.0) -> np.ndarray:
    """Truth with a secant shear modulus G12 / (1 + (τ/τ0)²), τ the largest ply shear stress (fixed point)."""
    g = theta[2]
    for _ in range(200):
        t = theta.copy()
        t[2] = g
        lam = L.Laminate.from_angles(material(t), STACKS[stack], T_PLY)
        state = lam.response(N=load)
        tau = float(np.abs(lam.ply_stresses(state).stress_material[:, :, 2]).max())
        new = theta[2] / (1.0 + (tau / tau0) ** 2)
        if abs(new - g) <= 1e-10 * g:
            return state.eps0[:2]
        g = 0.5 * (g + new)
    raise RuntimeError("fixed point did not converge")


def mismatch_case() -> tuple[I.Identification, float]:
    truth = lambda coupons: np.concatenate([shear_softening(TRUE, s, n) for s, n in coupons])   # noqa: E731
    obs = noisy(truth(TRAIN), 0.01, 30, "shear-softening truth, training coupons")
    held = noisy(truth(HELD), 0.01, 31, "shear-softening truth, held-out coupons at higher load")
    r = I.identify(model_of(TRAIN), obs, START, (LOW, HIGH), NAMES, jacobian=jacobian_of(TRAIN))
    return r, I.weighted_rms(model_of(HELD), held, r.theta)


def test_i6_model_mismatch_shows_in_held_out_residuals() -> None:
    _, rms = mismatch_case()
    assert rms >= 3.0, rms


def test_i7_prior_keeps_unidentified_parameters() -> None:
    coupons = (("0", (50.0, 0, 0)),)
    y = model_of(coupons)(TRUE)
    obs = I.Observations(y, 0.01 * np.abs(y), "synthetic")
    prior = I.GaussianPrior(np.array([120e3, 12e3, 4e3, 0.28]), np.array([50e3, 2e3, 1e3, 0.1]))
    r = I.identify(model_of(coupons), obs, START, (LOW, HIGH), NAMES, jacobian=jacobian_of(coupons), prior=prior)
    assert r.rank == 2 and r.covariance is not None
    for j in (1, 2):
        assert abs(r.theta[j] / prior.mean[j] - 1) <= 1e-6, r.theta


def test_bad_input_is_refused() -> None:
    y = model_of(TRAIN)(TRUE)
    with pytest.raises(ValueError):
        I.Observations(y, 0.0 * y, "synthetic")
    with pytest.raises(ValueError):
        I.Observations(y, 0.01 * np.abs(y), "made up")
    with pytest.raises(ValueError):
        I.Observations(np.array([math.nan]), np.array([1.0]), "synthetic")
    obs = I.Observations(y, 0.01 * np.abs(y), "synthetic")
    with pytest.raises(ValueError):
        I.identify(model_of(TRAIN), obs, START * 10, (LOW, HIGH), NAMES)          # start outside the bounds
    with pytest.raises(ValueError):
        I.identify(model_of(HELD), obs, START, (LOW, HIGH), NAMES)                # model / data size mismatch
    with pytest.raises(ValueError):
        I.identify(model_of(TRAIN), obs, START, (LOW, HIGH), NAMES[:2])
    # a wrong analytic Jacobian is caught
    wrong = lambda t: 1.01 * jacobian_of(TRAIN)(t)                                # noqa: E731
    assert I.check_jacobian(model_of(TRAIN), wrong, START, START) > 1e-3
