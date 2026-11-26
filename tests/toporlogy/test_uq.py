"""Uncertainty quantification and robust design: records U1-U10 of ``docs/olcumler/laminat_uq_onkayit.md``.

All distributions and parameter values are SYNTHETIC. Seeds are fixed, so the statistical checks are deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

import toporlogy as T
from toporlogy.optimization import uq

A, B = 2.0, 3.0
ALPHA = 0.95
Q95 = A + B * stats.norm.ppf(ALPHA)
CVAR95 = A + B * stats.norm.pdf(stats.norm.ppf(ALPHA)) / (1 - ALPHA)


def normal_space() -> uq.ParameterSpace:
    return uq.ParameterSpace((uq.Parameter.normal("x", 0.0, 1.0),))


def linear(p: dict[str, float]) -> float:
    return A + B * p["x"]


def values(n: int, method: str, seed: int) -> np.ndarray:
    return uq.propagate(linear, normal_space().sample(n, method, seed)).sample_values()


# --------------------------------------------------------------------------------------------------- U1-U2
def test_u1a_monte_carlo_against_the_normal_closed_forms() -> None:
    y = values(2**14, "mc", 1)
    m, v = uq.mean_estimate(y), uq.variance_estimate(y)
    assert abs(m.value - A) <= 4 * m.standard_error
    assert abs(v.value - B**2) <= 4 * v.standard_error
    assert abs(uq.quantile_estimate(y, ALPHA).value - Q95) <= 0.07 * B
    assert abs(uq.cvar_estimate(y, ALPHA).value - CVAR95) <= 0.08 * B


@pytest.mark.parametrize(("method", "var_tol"), [("sobol", 5e-3), ("lhs", 2e-2)])
def test_u1b_quasi_random_and_latin_hypercube(method: str, var_tol: float) -> None:
    y = values(2**14, method, 1)
    assert abs(y.mean() - A) <= 1e-3 * B
    assert abs(y.var(ddof=1) - B**2) <= var_tol * B**2
    assert abs(uq.quantile_estimate(y, ALPHA).value - Q95) <= 1e-2 * B
    assert abs(uq.cvar_estimate(y, ALPHA).value - CVAR95) <= 1e-2 * B


def test_u2_uniform_quantile_and_cvar() -> None:
    space = uq.ParameterSpace((uq.Parameter.uniform("x", 0.0, 1.0),))
    y = uq.propagate(lambda p: p["x"], space.sample(2**12, "sobol", 3)).sample_values()
    for a in (0.5, 0.9, 0.95):
        assert abs(uq.quantile_estimate(y, a).value - a) <= 1e-3
        assert abs(uq.cvar_estimate(y, a).value - (1 + a) / 2) <= 1e-3


# --------------------------------------------------------------------------------------------------- U3-U4
def test_u3_confidence_interval_coverage() -> None:
    space = uq.ParameterSpace((uq.Parameter.uniform("x", 0.0, 1.0),))
    hits = 0
    for seed in range(1000):
        e = uq.mean_estimate(space.sample(200, "mc", seed).values[:, 0] ** 2)
        hits += e.ci_low <= 1 / 3 <= e.ci_high
    assert 0.925 <= hits / 1000 <= 0.975


def ishigami_space() -> uq.ParameterSpace:
    return uq.ParameterSpace(tuple(uq.Parameter.uniform(f"x{i}", -math.pi, math.pi) for i in (1, 2, 3)))


def ishigami(v: np.ndarray) -> np.ndarray:
    return np.sin(v[:, 0]) + 7 * np.sin(v[:, 1]) ** 2 + 0.1 * v[:, 2] ** 4 * np.sin(v[:, 0])


def rmse_table(sizes: list[int], seeds: int = 40) -> dict[str, np.ndarray]:
    space = ishigami_space()
    out = {}
    for method in uq.SAMPLING_METHODS:
        out[method] = np.array([math.sqrt(np.mean([(ishigami(space.sample(n, method, s).values).mean() - 3.5) ** 2
                                                   for s in range(seeds)])) for n in sizes])
    return out


def test_u4_convergence_rates() -> None:
    sizes = [2**k for k in range(6, 14)]
    r = rmse_table(sizes)
    slope = {m: np.polyfit(np.log(sizes), np.log(v), 1)[0] for m, v in r.items()}
    assert -0.65 <= slope["mc"] <= -0.35, slope
    assert slope["sobol"] <= -0.8, slope
    assert np.all(r["lhs"] <= r["mc"]), (r["lhs"], r["mc"])


# --------------------------------------------------------------------------------------------------- U5
def test_u5_sobol_indices_of_ishigami() -> None:
    a, b = 7.0, 0.1
    V = a**2 / 8 + b * math.pi**4 / 5 + b**2 * math.pi**8 / 18 + 0.5
    V1 = 0.5 * (1 + b * math.pi**4 / 5) ** 2
    V2 = a**2 / 8
    V13 = 8 * b**2 * math.pi**8 / 225
    first, total = np.array([V1, V2, 0.0]) / V, np.array([V1 + V13, V2, V13]) / V
    r = uq.sobol_indices(lambda p: float(ishigami(np.array([[p["x1"], p["x2"], p["x3"]]]))[0]), ishigami_space(),
                         2**13, seed=4)
    assert r.n_evaluations == 2**13 * 5
    assert abs(r.variance - V) <= 0.02 * V
    assert np.abs(r.first - first).max() <= 0.02, r.first
    assert np.abs(r.total - total).max() <= 0.02, r.total


# --------------------------------------------------------------------------------------------------- U6-U7
def test_u6_truncated_distributions_stay_inside_and_inadmissible_support_is_refused() -> None:
    p = uq.Parameter.normal("E", 70.0, 10.0, truncate=(50.0, 85.0), admissible=(0.0, math.inf), unit="GPa")
    q = uq.Parameter.lognormal("k", 2.0, 0.3, truncate=(1.0, 4.0), admissible=(0.0, math.inf))
    s = uq.ParameterSpace((p, q)).sample(2**14, "sobol", 5)
    assert s.values[:, 0].min() >= 50.0 and s.values[:, 0].max() <= 85.0
    assert s.values[:, 1].min() >= 1.0 and s.values[:, 1].max() <= 4.0
    want = stats.truncnorm.mean(-2.0, 1.5, loc=70.0, scale=10.0)
    assert abs(s.values[:, 0].mean() - want) <= 1e-3 * want
    assert s.seed == 5 and s.method == "sobol" and set(s.versions) == {"numpy", "scipy"}
    assert p.metadata()["source"] == "synthetic" and p.metadata()["unit"] == "GPa"
    with pytest.raises(ValueError):                                            # an untruncated normal modulus
        uq.Parameter.normal("E", 70.0, 10.0, admissible=(0.0, math.inf))
    with pytest.raises(ValueError):                                            # nu = 0.5 is a limit, not an input
        uq.Parameter.uniform("nu", 0.2, 0.55, admissible=(-1.0, 0.5))
    with pytest.raises(ValueError):
        uq.Parameter.uniform("t", -0.1, 0.1, admissible=(0.0, math.inf))
    uq.Parameter.lognormal("k", 2.0, 0.3, admissible=(0.0, math.inf))          # open at 0: admissible


def test_u6_joint_admissibility_is_not_filtered_silently() -> None:
    space = uq.ParameterSpace((uq.Parameter.uniform("a", 0.0, 1.0), uq.Parameter.uniform("b", 0.0, 1.0)),
                              joint_check=lambda p: p["a"] + p["b"] < 1.5)
    with pytest.raises(uq.InadmissibleSampleError):
        space.sample(256, "mc", 0)


def test_u7_gaussian_copula_correlation() -> None:
    space = uq.ParameterSpace((uq.Parameter.normal("a", 1.0, 2.0), uq.Parameter.normal("b", -1.0, 0.5)),
                              correlation=np.array([[1.0, 0.7], [0.7, 1.0]]))
    v = space.sample(2**14, "sobol", 6).values
    assert abs(np.corrcoef(v.T)[0, 1] - 0.7) <= 0.02
    with pytest.raises(ValueError):
        uq.ParameterSpace(space.parameters, correlation=np.array([[1.0, 1.2], [1.2, 1.0]]))
    with pytest.raises(ValueError):
        uq.sobol_indices(linear, space, 64)


# --------------------------------------------------------------------------------------------------- U8-U9
def test_u8_failed_samples_are_never_dropped() -> None:
    space = uq.ParameterSpace((uq.Parameter.uniform("x", 0.0, 1.0),))

    def model(p: dict[str, float]) -> float:
        if p["x"] > 0.9:
            raise RuntimeError("solver diverged")
        return p["x"]

    r = uq.propagate(model, space.sample(1000, "lhs", 7))
    assert r.n_failed == 100 and np.isnan(r.values[r.failed]).all()
    assert all("solver diverged" in m for m in r.errors.values())
    with pytest.raises(uq.FailedSamplesError):
        r.sample_values()
    with pytest.raises(uq.FailedSamplesError):
        uq.mean_estimate(r.values)
    with pytest.raises(ValueError):
        r.sample_values("drop")
    y = r.sample_values("worst")
    assert uq.mean_estimate(y).value == math.inf and not uq.mean_estimate(y).reliable
    assert uq.cvar_estimate(y, 0.5).value == math.inf
    assert math.isfinite(uq.quantile_estimate(y, 0.5).value)
    assert uq.quantile_estimate(y, 0.95).value == math.inf
    nan = uq.propagate(lambda p: math.inf if p["x"] < 0.5 else 1.0, space.sample(16, "mc", 0))
    assert nan.n_failed > 0                                                    # non-finite output is a failure too


def test_u8_parallel_map_gives_the_same_values() -> None:
    from concurrent.futures import ThreadPoolExecutor

    s = normal_space().sample(64, "lhs", 2)
    with ThreadPoolExecutor(4) as ex:
        par = uq.propagate(linear, s, map_fn=ex.map).values
    assert np.array_equal(par, uq.propagate(linear, s).values)
    assert np.array_equal(s.values, normal_space().sample(64, "lhs", 2).values)  # seeded: reproducible


def test_u9_rare_events_are_flagged() -> None:
    threshold = stats.norm.ppf(0.999)
    y = normal_space().sample(1000, "mc", 10).values[:, 0]
    e = uq.exceedance_probability(y, threshold)
    assert not e.reliable
    assert e.ci_low <= 1e-3 <= e.ci_high
    assert not uq.quantile_estimate(y, 0.999).reliable
    assert not uq.cvar_estimate(y, 0.999).reliable
    common = uq.exceedance_probability(y, 0.0)
    assert common.reliable and common.ci_low <= 0.5 <= common.ci_high


def test_replicated_interval_for_quasi_monte_carlo() -> None:
    e = uq.replicated_mean(linear, normal_space(), 256, "sobol", range(10))
    assert e.method == "replicated-sobol" and e.ci_low <= A <= e.ci_high
    with pytest.raises(ValueError):
        uq.replicated_mean(linear, normal_space(), 256, "sobol", [1, 1])
    with pytest.raises(ValueError):
        normal_space().sample(100, "sobol", 0)                                # not a power of two
    with pytest.raises(ValueError):
        normal_space().sample(100, "halton", 0)


# --------------------------------------------------------------------------------------------------- U10
HALF_ANGLE = math.radians(30.0)


def angle_coefficients(n: int, seed: int) -> np.ndarray:
    """Unit loads (n, 2) = (sin θ, −cos θ) for θ ~ U(−30°, 30°) about the downward direction."""
    space = uq.ParameterSpace((uq.Parameter.uniform("theta", -HALF_ANGLE, HALF_ANGLE, unit="rad"),))
    th = space.sample(n, "sobol", seed).values[:, 0]
    return np.column_stack([np.sin(th), -np.cos(th)])


def cantilever(nx: int = 60, ny: int = 30):
    d = T.Domain((nx, ny))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    tip = d.nodes_where(lambda X: (X[:, 0] > nx - 1e-9) & (np.abs(X[:, 1] - ny / 2) < 1e-9))
    basis = np.zeros((d.ndof, 2))
    basis[d.dofs(tip, 0), 0] = 1.0
    basis[d.dofs(tip, 1), 1] = 1.0
    return d, fixed, basis


def study(d, fixed, F, objective) -> T.Study:
    chain = T.standard_chain(d, 2.0)
    return T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), objective, [T.Constraint(T.Volume(), 0.4)],
                   T.MMA(move=0.2))


def test_u10a_scenario_compliance_and_its_analytic_mean() -> None:
    d, fixed, basis = cantilever(20, 10)
    coef = angle_coefficients(64, 1)
    st = study(d, fixed, uq.load_scenarios(basis, coef), uq.ScenarioCompliance("mean"))
    x = np.random.default_rng(0).uniform(0.2, 0.8, st.n_var)
    ev = st.evaluate(x, gradients=False)
    unit = study(d, fixed, basis, T.Compliance()).evaluate(x, gradients=False).U
    G = basis.T @ unit
    per = np.einsum("ni,ij,nj->n", coef, G, coef)
    got = np.sum(uq.load_scenarios(basis, coef) * ev.U, axis=0)
    assert np.abs(got - per).max() <= 1e-9 * per.max()
    es2 = 0.5 - math.sin(2 * HALF_ANGLE) / (4 * HALF_ANGLE)
    ec2 = 1.0 - es2
    exact = G[0, 0] * es2 + G[1, 1] * ec2
    assert abs(ev.f0 - exact) <= 1e-3 * exact


@pytest.mark.parametrize(("measure", "kw"), [("mean_std", {"k": 3.0}), ("cvar", {"alpha": 0.8}), ("mean", {})])
def test_u10b_robust_gradients_match_finite_differences(measure: str, kw: dict) -> None:
    d, fixed, basis = cantilever(16, 8)
    st = study(d, fixed, uq.load_scenarios(basis, angle_coefficients(16, 1)), uq.ScenarioCompliance(measure, **kw))
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.9, st.n_var)
    direction = rng.uniform(-1, 1, st.n_var)
    ev = st.evaluate(x)
    h = 1e-6
    fd = (st.evaluate(x + h * direction, False).f0 - st.evaluate(x - h * direction, False).f0) / (2 * h)
    assert abs(float(ev.df0 @ direction) - fd) <= 1e-5 * abs(fd)


def robust_comparison(iterations: int = 80) -> dict[str, dict[str, float]]:
    """Nominal (vertical load) against robust (mean + 3 std over 16 common random scenarios) designs, both measured on
    512 independently seeded scenarios."""
    d, fixed, basis = cantilever()
    held = uq.load_scenarios(basis, angle_coefficients(512, 2))
    nominal = study(d, fixed, basis @ np.array([0.0, -1.0]), T.Compliance())
    robust = study(d, fixed, uq.load_scenarios(basis, angle_coefficients(16, 1)),
                   uq.ScenarioCompliance("mean_std", k=3.0))
    out = {}
    for name, st in (("nominal", nominal), ("robust", robust)):
        res = st.run(max_iter=iterations, tol=1e-3)
        E = st.interp(torch.as_tensor(res.xphys)).reshape(-1)[torch.as_tensor(d.active_elements)]
        U = st.physics.solve(E, held)
        c = torch.as_tensor(np.sum(held * U.numpy(), axis=0))
        nom = float(np.dot(basis @ np.array([0.0, -1.0]),
                           st.physics.solve(E, basis @ np.array([0.0, -1.0])).numpy()[:, 0]))
        out[name] = {"mean": float(c.mean()), "std": float(c.std()), "nominal_load": nom,
                     "mean_3std": float(uq.risk_measure(c, "mean_std", k=3.0)),
                     "cvar95": float(uq.risk_measure(c, "cvar", alpha=0.95)), "max": float(c.max()),
                     "volume": res.constraint_values["volume"], "iterations": res.iterations}
    return out


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="pre-registered U10c failed: on the held-out scenarios the robust design is "
                                       "slightly worse than the nominal one (see docstring)")
def test_u10c_robust_design_beats_the_nominal_one_on_held_out_scenarios() -> None:
    """Pre-registered U10c, first run (26 Nov 2026), not met: held-out mean + 3 std 102.23 (robust) against 101.80
    (nominal), CVaR_0.95 92.26 against 91.40. Post hoc: the robust run stops in a local minimum whose TRAINING
    objective (102.63) is above that of the nominal design (102.27), and warm-starting it from the nominal design
    gains only 0.02-0.4 %: with a horizontal tip-load component that the midspan cantilever carries axially, there is
    almost nothing to gain. See ``docs/olcumler/laminat_uq.md``."""
    r = robust_comparison()
    assert r["robust"]["mean_3std"] < r["nominal"]["mean_3std"], r
    assert r["robust"]["cvar95"] < r["nominal"]["cvar95"], r
    for k in ("nominal", "robust"):
        assert r[k]["volume"] <= 0.4 * (1 + 1e-3), r


def column_comparison(iterations: int = 100) -> dict[str, dict[str, float]]:
    """POST HOC (chosen after U10c failed): a column, base fixed, tip load straight down. The nominal design carries
    the nominal load only and is fragile to the lateral component; same scenarios, risk measure and optimiser."""
    d = T.Domain((30, 60))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 1] < 1e-9))
    top = d.nodes_where(lambda X: (np.abs(X[:, 0] - 15) < 1e-9) & (X[:, 1] > 60 - 1e-9))
    basis = np.zeros((d.ndof, 2))
    basis[d.dofs(top, 0), 0] = 1.0
    basis[d.dofs(top, 1), 1] = 1.0
    held = uq.load_scenarios(basis, angle_coefficients(512, 2))
    down = basis @ np.array([0.0, -1.0])
    out = {}
    for name, F, obj in (("nominal", down, T.Compliance()),
                         ("robust", uq.load_scenarios(basis, angle_coefficients(16, 1)),
                          uq.ScenarioCompliance("mean_std", k=3.0))):
        st = study(d, fixed, F, obj)
        res = st.run(max_iter=iterations, tol=1e-3)
        E = st.interp(torch.as_tensor(res.xphys)).reshape(-1)[torch.as_tensor(d.active_elements)]
        c = torch.as_tensor(np.sum(held * st.physics.solve(E, held).numpy(), axis=0))
        out[name] = {"mean": float(c.mean()), "std": float(c.std()),
                     "nominal_load": float(np.dot(down, st.physics.solve(E, down).numpy()[:, 0])),
                     "mean_3std": float(uq.risk_measure(c, "mean_std", k=3.0)),
                     "cvar95": float(uq.risk_measure(c, "cvar", alpha=0.95)), "max": float(c.max()),
                     "volume": res.constraint_values["volume"], "iterations": res.iterations}
    return out


@pytest.mark.slow
def test_post_hoc_robust_column_beats_the_nominal_one() -> None:
    """POST HOC check (not pre-registered). First measured: held-out mean + 3 std 37.4 against 172.7."""
    r = column_comparison()
    assert r["robust"]["mean_3std"] < 0.5 * r["nominal"]["mean_3std"], r
    assert r["robust"]["cvar95"] < 0.5 * r["nominal"]["cvar95"], r
    assert r["robust"]["nominal_load"] > r["nominal"]["nominal_load"]       # the price at the nominal load
    for k in ("nominal", "robust"):
        assert r[k]["volume"] <= 0.4 * (1 + 1e-3), r


def test_risk_measure_refuses_bad_input() -> None:
    v = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        uq.risk_measure(v, "worst")
    with pytest.raises(ValueError):
        uq.risk_measure(v[:1], "mean_std")
    with pytest.raises(ValueError):
        uq.risk_measure(v, "cvar", alpha=1.0)
    with pytest.raises(ValueError):
        uq.load_scenarios(np.zeros((4, 2)), np.zeros((3, 3)))
