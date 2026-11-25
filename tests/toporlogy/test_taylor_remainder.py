"""Taylor remainder test of the design gradients (docs/olcumler/dogrulama_turu.md, section T; pre-registered).

r(h) = |f(x + h d) - f(x) - h grad f . d|, h_k = 0.05 * 2^-k, k = 0..6, x in [0.3, 0.7], d random with max|d| = 1,
orders p_k = log2(r_k / r_(k+1)).

T1 (per response): two consecutive p_k both in [1.8, 2.2]  (a correct gradient gives p -> 2; a noise floor at the
    smallest h is expected and only the consecutive pair is required).
T2 (negative control): with the directional derivative scaled by 1.01 the last two p_k are <= 1.3 (p -> 1).
The thresholds are fixed; they are not to be loosened.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.configuration.build import study_from_dict

H0 = 0.05
KS = range(7)
T1_BAND = (1.8, 2.2)
T2_MAX = 1.3

ORTHO = {"E1": 40.0, "E2": 9.0, "E3": 9.0, "G12": 4.0, "G13": 4.0, "G23": 3.2, "nu12": 0.28, "nu13": 0.28,
         "nu23": 0.4}


def _cantilever(shape=(10, 5), constraints=None, material=None, **extra) -> dict:
    d3 = len(shape) == 3
    cfg = {"domain": {"type": "box", "shape": list(shape), "h": 1.0},
           "material": material or {"E0": 1.0, "nu": 0.3},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {
               "type": "point", "at": ["max", "min"] + (["min"] if d3 else [])},
               "force": [0, -1] + ([0] if d3 else [])}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}],
           "interpolation": {"type": "simp", "penal": 3.0},
           "objective": "compliance",
           "constraints": constraints if constraints is not None else [{"response": "volume", "upper": 0.5}],
           "optimizer": {"type": "mma", "move": 0.1}, "run": {"max_iter": 1}}
    cfg.update(extra)
    return cfg


def _angle_file(tmp_path, shape):
    f = tmp_path / "angle.npy"
    ix, iy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    np.save(f, 15.0 + 5.0 * ix - 3.0 * iy)
    return str(f)


# Each builder returns (cfg, kind, index): kind "f0" = objective, "g" = constraint number `index`,
# None = the objective and every constraint.
def _compliance_2d(tp):
    return _cantilever(), "f0", 0


def _compliance_3d(tp):
    return _cantilever((6, 3, 3)), "f0", 0


def _displacement_2d(tp):
    # measured away from the load and across it (at the load point along the unit load it would equal the
    # compliance, a self-adjoint response; the first version of this case did that)
    cfg = _cantilever(objective={"type": "displacement", "where": {"type": "point", "at": ["max", "max"]},
                                 "direction": [1, 0]})
    return cfg, "f0", 0


def _stress_pnorm_2d(tp):
    cfg = _cantilever(constraints=[{"response": "volume", "upper": 0.5},
                                   {"response": {"type": "stress", "sigma_lim": 2.0, "aggregate": "pnorm",
                                                 "parameter": 6.0, "adaptive": None}}])
    return cfg, "g", 1


def _tsai_wu_angle_field(tp):
    shape = (10, 4)
    mat = {"E0": 40.0, "nu": 0.3, "orthotropic": {**ORTHO, "angle_field": _angle_file(tp, shape)},
           "strength": {"Xt": 15.0, "Xc": 12.0, "Yt": 0.5, "Yc": 2.0, "S12": 0.7}}
    cfg = _cantilever(shape, material=mat,
                      constraints=[{"response": "volume", "upper": 0.5},
                                   {"response": {"type": "stress", "aggregate": "pnorm", "parameter": 6.0,
                                                 "adaptive": None}}])
    return cfg, "g", 1


def _buckling_ks_2d(tp):
    cfg = _cantilever((6, 14), material={"E0": 100.0, "nu": 0.3},
                      constraints=[{"response": "volume", "upper": 0.5},
                                   {"response": {"type": "buckling", "load_factor": 0.01, "n_modes": 4,
                                                 "rho": 20.0}}],
                      supports=[{"type": "fixed", "where": {"type": "plane", "axis": "y", "at": "min"}}],
                      load_cases=[{"name": "live", "loads": [{"type": "force", "where": {
                          "type": "plane", "axis": "y", "at": "max"}, "force": [0.02, -1.0]}]}])
    return cfg, "g", 1


def _eigenfrequency_2d(tp):
    cfg = _cantilever((12, 4), material={"E0": 1.0, "nu": 0.3, "rho": 1.0},
                      constraints=[{"response": "volume", "upper": 0.5}],
                      physics={"type": "modal", "n_modes": 3},
                      supports=[{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}},
                                {"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "max"}}],
                      interpolation={"type": "two_range_simp", "penal": 3.0},
                      objective={"type": "maximize", "response": {"type": "frequency", "k": 1}})
    cfg.pop("load_cases")
    return cfg, "f0", 0


def _conduction_connectivity(tp):
    keep = [{"type": "box", "lo": [2, 0], "hi": [5, 1]}, {"type": "box", "lo": [15, 0], "hi": [18, 1]},
            {"type": "box", "lo": [2, 5], "hi": [5, 8]}, {"type": "box", "lo": [15, 5], "hi": [18, 8]}]
    cfg = {"domain": {"type": "box", "shape": [20, 8], "keep": keep}, "material": {"E0": 1.0, "nu": 0.3},
           "supports": [{"type": "fixed", "where": {"type": "in_box", "lo": [2, 0], "hi": [5, 0]}, "name": "sa"},
                        {"type": "fixed", "where": {"type": "in_box", "lo": [15, 0], "hi": [18, 0]}, "name": "sb"}],
           "load_cases": [{"loads": [
               {"type": "force", "where": {"type": "in_box", "lo": [2, 5], "hi": [5, 8]}, "force": [0, -1]},
               {"type": "force", "where": {"type": "in_box", "lo": [15, 5], "hi": [18, 8]}, "force": [0, -1]}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}, {"type": "heaviside_projection", "beta": 1.0}],
           "interpolation": {"type": "simp", "penal": 3.0},
           "constraints": [{"response": "volume", "upper": 0.3},
                           {"response": {"type": "conduction_connectivity", "sink": "sa", "aggregate": "pnorm",
                                         "p": 6.0}, "upper": 1e6}],
           "optimizer": "mma", "run": {"max_iter": 1}}
    return cfg, "g", 1


def _multiphase(tp):
    cfg = _cantilever(
        (10, 5), material={"phases": [{"name": "soft", "E": 1.0, "density": 1.0, "cost": 1.0},
                                      {"name": "stiff", "E": 3.0, "density": 2.0, "cost": 5.0}]},
        interpolation="multiphase_simp",
        constraints=[{"response": {"type": "phase_volume", "phases": "soft"}, "upper": 0.3},
                     {"response": "cost", "upper": 20.0}])
    return cfg, None, None          # f0, g0 (soft volume), g1 (cost): all three are checked


def _nonlinear_end_compliance(tp):
    cfg = _cantilever((10, 4), physics={"type": "nonlinear_elasticity", "rtol": 1e-12},
                      objective="end_compliance",
                      load_cases=[{"name": "tip", "loads": [{"type": "force", "where": {
                          "type": "point", "at": ["max", "min"]}, "force": [0, -0.0005]}]}],
                      material={"E0": 1.0, "nu": 0.3})
    return cfg, "f0", 0


def _compliance_3d_gmg(tp):
    cfg = _cantilever((6, 3, 3), physics={"type": "elasticity_gmg", "tol": 1e-12, "direct_below": 0})
    return cfg, "f0", 0


def _t2_xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


T1_CASES = [
    pytest.param(_compliance_2d, id="compliance_2d"),
    pytest.param(_compliance_3d, id="compliance_3d"),
    pytest.param(_displacement_2d, id="displacement_2d"),
    pytest.param(_stress_pnorm_2d, id="stress_pnorm_2d"),
    pytest.param(_tsai_wu_angle_field, id="tsai_wu_angle_field"),
    pytest.param(_buckling_ks_2d, id="buckling_ks_2d"),
    pytest.param(_eigenfrequency_2d, id="eigenfrequency_2d"),
    pytest.param(_conduction_connectivity, id="conduction_connectivity"),
    pytest.param(_multiphase, id="multiphase_compliance_volume_cost"),
    pytest.param(_nonlinear_end_compliance, id="nonlinear_end_compliance"),
    pytest.param(_compliance_3d_gmg, id="compliance_3d_gmg"),
]
# T2 is the same list; the cases marked xfail below fail the pre-registered negative control (see _T2_NOTE).
_T2_NOTE = ("pre-registered T2 FAILED (the control, not the gradient: T1 passes): the 1 % error h*0.01*g only "
            "overtakes the curvature term c*h^2 below h = 0.01*|g|/c, which is under the smallest h = 7.8e-4 here")
T2_CASES = [
    pytest.param(_compliance_2d, id="compliance_2d"),
    pytest.param(_compliance_3d, id="compliance_3d"),
    pytest.param(_displacement_2d, id="displacement_2d", marks=_t2_xfail(
        _T2_NOTE + "; measured p* = [1.91, 1.85, 1.74, 1.59, 1.42, 1.27]")),
    pytest.param(_stress_pnorm_2d, id="stress_pnorm_2d", marks=_t2_xfail(
        _T2_NOTE + "; measured p* = [1.83, 1.78, 1.68, 1.53, 1.36, 1.22]")),
    pytest.param(_tsai_wu_angle_field, id="tsai_wu_angle_field"),
    pytest.param(_buckling_ks_2d, id="buckling_ks_2d", marks=_t2_xfail(
        "pre-registered T2 FAILED: the 1 % error cancels the signed remainder near h = 1.6e-3 "
        "(measured p* = [2.06, 2.1, 2.19, 2.47, 4.14, -0.53]; T1 passes)")),
    pytest.param(_eigenfrequency_2d, id="eigenfrequency_2d"),
    pytest.param(_conduction_connectivity, id="conduction_connectivity", marks=_t2_xfail(
        _T2_NOTE + "; measured p* = [1.97, 1.89, 1.79, 1.64, 1.47, 1.31]")),
    pytest.param(_multiphase, id="multiphase_compliance_volume_cost", marks=_t2_xfail(
        _T2_NOTE + "; objective f0: measured p* = [1.96, 1.94, 1.89, 1.8, 1.66, 1.5]")),
    pytest.param(_nonlinear_end_compliance, id="nonlinear_end_compliance"),
    pytest.param(_compliance_3d_gmg, id="compliance_3d_gmg"),
]

_CACHE: dict = {}


def _pick(ev, kind, idx):
    return (ev.f0, ev.df0) if kind == "f0" else (ev.g[idx], ev.dg[idx])


def _orders(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(r[:-1] / r[1:])


def _consecutive_in_band(p):
    return any(T1_BAND[0] <= p[k] <= T1_BAND[1] and T1_BAND[0] <= p[k + 1] <= T1_BAND[1] for k in range(len(p) - 1))


def _measure(builder, tmp_path_factory) -> list[dict]:
    """Per tested response of the study: r_k, p_k (true gradient) and p_k* (gradient x 1.01). Cached per builder."""
    if builder in _CACHE:
        return _CACHE[builder]
    cfg, kind, idx = builder(tmp_path_factory.mktemp("taylor"))
    st = study_from_dict(cfg)
    n = len(st.default_start())
    rng = np.random.default_rng(2026)
    x = 0.3 + 0.4 * rng.random(n)
    d = rng.uniform(-1.0, 1.0, n)
    d /= np.abs(d).max()
    assert (x + H0 * d).min() >= 0.0 and (x + H0 * d).max() <= 1.0
    ev = st.evaluate(x)
    hs = np.array([H0 * 2.0 ** -k for k in KS])
    evs = [st.evaluate(x + h * d, gradients=False) for h in hs]
    targets = [(kind, idx)] if kind is not None else [("f0", 0), ("g", 0), ("g", 1)]
    out = []
    for kd, ix in targets:
        f0, grad = _pick(ev, kd, ix)
        g = float(np.dot(grad, d))
        fs = np.array([float(_pick(e, kd, ix)[0]) for e in evs])
        r1, r2 = np.abs(fs - f0 - hs * g), np.abs(fs - f0 - hs * 1.01 * g)
        res = {"label": f"{kd}{ix if kd == 'g' else ''}", "r": r1, "r2": r2, "p": _orders(r1), "p2": _orders(r2)}
        print(f"\n  [{res['label']}] n={n} f={float(f0):.6e} grad.d={g:.6e}")
        print(f"    r_k  = {np.array2string(r1, precision=3)}")
        print(f"    p_k  = {np.array2string(res['p'], precision=3)}   (T1: two consecutive in {T1_BAND})")
        print(f"    p_k* = {np.array2string(res['p2'], precision=3)}   (T2, 1.01 grad: last two <= {T2_MAX})")
        out.append(res)
    _CACHE[builder] = out
    return out


@pytest.mark.parametrize("builder", T1_CASES)
def test_t1_taylor_orders(builder, tmp_path_factory):
    bad = [f"{r['label']}: p = {np.round(r['p'], 2).tolist()}" for r in _measure(builder, tmp_path_factory)
           if not _consecutive_in_band(r["p"])]
    assert not bad, "T1 failed: " + "; ".join(bad)


@pytest.mark.parametrize("builder", T2_CASES)
def test_t2_negative_control(builder, tmp_path_factory):
    bad = [f"{r['label']}: p* = {np.round(r['p2'], 2).tolist()}" for r in _measure(builder, tmp_path_factory)
           if not np.all(r["p2"][-2:] <= T2_MAX)]
    assert not bad, "T2 failed: " + "; ".join(bad)

