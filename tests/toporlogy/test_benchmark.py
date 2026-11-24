"""0/1 evaluation and comparison of results (optimization/benchmark.py)."""
from __future__ import annotations

import numpy as np
import pytest

import toporlogy as T
from toporlogy.optimization.benchmark import binarize, compare_runs, discrete_report, format_table
from toporlogy.optimization.robust import RobustCompliance, RobustSet


def _domain() -> T.Domain:
    keep = np.zeros((8, 4), bool)
    keep[0, :] = True
    void = np.zeros((8, 4), bool)
    void[7, 3] = True
    frac = np.ones((8, 4))
    frac[4, 0] = 0.0
    return T.Domain((8, 4), frac=frac, keep=keep, void=void)


def test_binarize_rules() -> None:
    d = _domain()
    x = np.random.default_rng(0).random(d.shape)
    b = binarize(x, d, volume=0.5)
    assert set(np.unique(b)) <= {0.0, 1.0}
    assert b[d.keep].all() and not b[d.void].any() and b[4, 0] == 0
    assert b[d.active].sum() == round(0.5 * d.n_active)                   # exact count, keep included
    free = d.design
    assert x[free][b[free] == 1].min() >= x[free][b[free] == 0].max()     # the largest densities are kept
    t = binarize(x, d, threshold=0.3)
    assert np.array_equal(t[free] == 1, x[free] > 0.3) and t[d.keep].all()
    assert np.array_equal(binarize(x, d, volume=0.5), b)                  # deterministic
    with pytest.raises(ValueError):
        binarize(x[:3], d)
    with pytest.raises(ValueError):
        binarize(x, d, volume=0.0)


def _mbb(optimizer, constraints=None, objective=None):
    d = T.Domain((30, 10))
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    roller = d.nodes_where(lambda X: (X[:, 0] == 30) & (X[:, 1] == 0))
    top_left = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == 10))
    F = d.load_vector()[:, 0]
    F[d.dofs(top_left, 1)] = -1.0
    chain = T.standard_chain(d, 1.5, beta=1.0)
    return T.Study(d, T.Elasticity(d, np.concatenate([d.dofs(left, 0), d.dofs(roller, 1)])), F, chain,
                   T.SIMP(d, emin_ratio=1e-9), objective(chain) if objective else T.Compliance(),
                   constraints or [T.Constraint(T.Volume(), 0.5)], optimizer)


def test_discrete_report_and_compare() -> None:
    runs = {}
    for name, opt in (("oc", T.OC()), ("mma", T.MMA())):
        st = _mbb(opt)
        runs[name] = (st, st.run(max_iter=25))
    rows = compare_runs(runs)
    for r, (st, res) in zip(rows, runs.values(), strict=True):
        assert r["objective"] == res.objective and r["iterations"] == 25
        assert r["evaluations"] == st.n_evaluations == 26                 # 25 iterations + the final re-evaluation
        assert r["volume_binary"] == pytest.approx(0.5) and r["constraints"]["volume"]["satisfied"] is True
        # the grey design uses intermediate stiffness the 0/1 design does not have: values differ, both positive
        assert r["objective_binary"] > 0 and r["ratio"] == pytest.approx(r["objective_binary"] / r["objective"])
        assert 0 < r["m_nd"] < 1 and r["threshold_rule"] == "volume 0.5" and r["not_evaluated"] == {}
    direct = runs["oc"][0].evaluate_density(binarize(runs["oc"][1].xphys, runs["oc"][0].domain, 0.5))
    assert rows[0]["objective_binary"] == direct.values["compliance"]
    table = format_table(rows)
    assert table.count("\n") == 3 and "1.000" in table and "yes" in table
    fixed = discrete_report(*runs["oc"], volume=None, threshold=0.5)
    assert fixed["threshold_rule"] == "threshold 0.5"
    with pytest.raises(ValueError):
        discrete_report(*runs["oc"], volume="auto")


def test_chain_dependent_responses_are_not_evaluated_not_passed() -> None:
    rs = RobustSet(0.6)
    st = _mbb(T.MMA(), objective=lambda chain: RobustCompliance(rs))
    rep = discrete_report(st, st.run(max_iter=3))
    assert rep["objective_binary"] is None and rep["ratio"] is None and "robust_compliance" in rep["not_evaluated"]
    assert "-" in format_table([{"name": "robust", **rep}])
