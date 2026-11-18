"""G7: what a run reports about itself (docs/gelistirme_plani.md B-02, B-06).

``Result.converged`` used to mean only "design change below tol and continuation finished": a run with violated
constraints, or one whose last beta step was never optimised, could still say converged. The result now states
separately why the loop stopped (``termination``), whether the returned design satisfies its constraints
(``feasible``, ``max_violation``), how far it is from a KKT point (``kkt_residual``), and the run summary lists
every finding, unverified parts included, as machine-readable ``flags``.

Thresholds below are measured on the 30 x 10 MBB of ``_mbb`` (17-18 Nov 2026): KKT residual 0.77 (MMA) / 0.82 (OC)
after one iteration, 1.8e-4 (MMA, 57 iterations) / 2.1e-2 (OC, 179 iterations) at convergence (tol 1e-3).
"""
from __future__ import annotations

import numpy as np
import pytest

import toporlogy as T
from toporlogy.execution.runs import flags_of, status_of
from toporlogy.optim.base import Optimizer


def _mbb(opt, nx=30, ny=10, vf=0.5, projection=False, schedule=None):
    d = T.Domain((nx, ny))
    X = d.node_coords()
    fixed = np.r_[d.dofs(np.flatnonzero(X[:, 0] < 1e-9), 0),
                  d.dofs(np.flatnonzero((X[:, 0] > nx - 1e-9) & (X[:, 1] < 1e-9)), 1)]
    F = d.load_vector()
    F[d.dofs(np.flatnonzero((X[:, 0] < 1e-9) & (X[:, 1] > ny - 1e-9)), 1), 0] = -1.0
    chain = T.standard_chain(d, 1.5, projection=projection, beta=1.0)
    sch = schedule(chain) if schedule else None
    return T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(), [T.Constraint(T.Volume(), vf)],
                   opt, sch)


# ---------------------------------------------------------------------------------------------- termination
def test_max_iter_is_not_converged():
    res = _mbb(T.MMA(move=0.2)).run(max_iter=3, tol=1e-3)
    assert res.iterations == 3 and res.termination == "max_iter" and res.converged is False


def test_design_change_stop_is_converged_and_early():
    res = _mbb(T.MMA(move=0.2)).run(max_iter=400, tol=1e-3)
    assert res.termination == "design_change" and res.converged is True
    assert res.iterations < 400 and res.history["change"][-1] <= 1e-3


def test_beta_step_in_the_last_iteration_is_not_converged():
    """B-06: the schedule steps beta when the change falls to its tol. With that step in the last allowed iteration
    the loop ended by `continue`; change <= tol and the schedule finished, so the old rule said converged although
    no iteration ran at the final beta. Here beta 8 -> 16 = beta_max happens in iteration 1 of 1."""
    def sch(chain):
        return T.BetaContinuation(chain, 8.0, 16.0, every=1000, tol=10.0)   # change <= 10 always: step at once
    st = _mbb(T.MMA(move=0.2), projection=True, schedule=sch)
    res = st.run(max_iter=1, tol=10.0)
    assert st.schedule.beta == 16.0 and st.schedule.finished              # the old rule's two conditions hold
    assert res.history["change"][-1] <= 10.0
    assert res.termination == "schedule_step_at_max_iter" and res.converged is False
    res2 = st.run(max_iter=2, tol=10.0)              # one iteration at beta 16 after the step: a genuine stop
    assert res2.termination == "design_change" and res2.converged is True and res2.iterations == 2


# ---------------------------------------------------------------------------------------------- feasibility
def _volume_at(x0_value, upper):
    st = _mbb(T.MMA(move=0.2), vf=upper)
    return st, st.run(x0=np.full(st.n_var, x0_value), max_iter=0)     # no iteration: the start design is returned


def test_feasibility_uses_the_summary_rule_at_its_boundary():
    """value <= upper + tol * |upper| (tol 1e-3): just inside passes, just outside fails, and max_violation is the
    relative excess, computed here independently."""
    _, ref = _volume_at(0.4, 0.5)
    v = ref.final.values["volume"]
    for factor, expect in ((1 + 0.5e-3, True), (1 + 2e-3, False)):
        upper = v / factor
        _, res = _volume_at(0.4, upper)
        assert res.final.values["volume"] == v                                # same design: deterministic
        assert res.feasible is expect, (factor, res.feasible)
        assert res.max_violation == pytest.approx((v - upper) / upper, rel=1e-12)
    _, ok = _volume_at(0.4, v * 1.5)
    assert ok.feasible is True and ok.max_violation == 0.0


def test_a_converged_but_infeasible_design_says_both():
    """Converged and feasible are separate: a design-change stop does not imply the constraints hold."""
    st = _mbb(T.OC(move=0.2, dv_scale=300.0), vf=0.3)
    res = st.run(x0=np.full(st.n_var, 0.9), max_iter=1, tol=10.0)     # tol 10: the first step counts as converged
    assert res.converged is True and res.termination == "design_change"
    assert res.feasible is False and res.max_violation > 0.5
    v = res.final.values["volume"]
    assert res.max_violation == pytest.approx((v - 0.3) / 0.3, rel=1e-12)


# ---------------------------------------------------------------------------------------------- KKT residual
@pytest.mark.parametrize("make,converged_max,ratio_max", [
    (lambda: T.MMA(move=0.2), 1e-3, 1e-2),
    (lambda: T.OC(move=0.2, dv_scale=300.0), 5e-2, 1e-1),
])
def test_kkt_residual_falls_by_orders_at_convergence(make, converged_max, ratio_max):
    first = _mbb(make()).run(max_iter=1, tol=1e-3).kkt_residual
    res = _mbb(make()).run(max_iter=400, tol=1e-3)
    assert res.termination == "design_change"
    assert first > 0.5                                   # one iteration from the uniform start is far from KKT
    assert res.kkt_residual < converged_max and res.kkt_residual < ratio_max * first


def test_kkt_residual_detects_wrong_multipliers():
    """The residual is a real test of the multipliers, not a number that is always small: at the converged MMA
    design, doubling or zeroing the multipliers makes it large."""
    opt = T.MMA(move=0.2)
    st = _mbb(opt)
    res = st.run(max_iter=400, tol=1e-3)
    lam = opt.kkt_multipliers()
    ev = st.evaluate(res.x)
    assert st._kkt_residual(res.x, ev, lam) == pytest.approx(res.kkt_residual, rel=1e-9)
    # measured: 2x -> > 100x, 0x -> 0.0997 = ~550x the true residual 1.8e-4
    assert st._kkt_residual(res.x, ev, 2 * lam) > 100 * res.kkt_residual
    assert st._kkt_residual(res.x, ev, 0 * lam) > 100 * res.kkt_residual


def test_optimizer_without_multipliers_reports_none():
    class Plain(Optimizer):
        name = "plain"

        def step(self, x, f0, df0, g, dg, constraint_fn=None):
            return np.clip(x - 0.01 * np.sign(df0), 0.0, 1.0)
    res = _mbb(Plain()).run(max_iter=2)
    assert res.kkt_residual is None and res.feasible in (True, False) and res.termination == "max_iter"


# ---------------------------------------------------------------------------------------------- summary flags
def _summary(**over):
    s = {"converged": True, "termination": "design_change",
         "constraints": [{"name": "volume", "value": 0.3, "upper": 0.3, "satisfied": True}], "post": {}}
    s.update(over)
    return s


def test_flags_name_every_finding_once_and_nothing_for_a_clean_run():
    assert flags_of(_summary()) == [] and status_of(_summary()) == "ok"
    s = _summary(converged=False, termination="schedule_step_at_max_iter",
                 constraints=[{"name": "v", "satisfied": False}, {"name": "w", "satisfied": False}],
                 post={"connectivity_gate": {"passed": False, "repaired_design": {"feasible": None}},
                       "connectivity_gate_2": {"passed": False, "repaired_design": {"feasible": None}},
                       "reverify": {"bc_fidelity": "unsupported"}})
    assert flags_of(s) == ["not_converged", "schedule_step_at_max_iter", "constraint_violated",
                           "connectivity_gate_failed", "repair_unverified", "connectivity_gate_2_failed",
                           "bc_mapping_unsupported"]


def test_unverified_parts_are_flagged_even_when_the_status_is_ok():
    """An approximate re-analysis fails nothing, so the status line stays "ok"; the flags still say that the
    re-analysis did not solve the optimisation's exact supports and loads."""
    s = _summary(post={"reverify": {"bc_fidelity": "approximate", "summary": "..."}})
    assert status_of(s) == "ok" and flags_of(s) == ["bc_mapping_approximate"]
    assert flags_of(_summary(post={"reverify": {"bc_fidelity": "equivalent"}})) == []


def test_run_summary_carries_the_result_contract(tmp_path):
    from test_config import mbb

    from toporlogy import config as C
    s = C.build_setup(mbb(max_iter=3))
    sm = C.run_setup(s, tmp_path / "run")
    assert sm["termination"] == "max_iter" and sm["converged"] is False and "not_converged" in sm["flags"]
    assert sm["feasible"] == all(c["satisfied"] for c in sm["constraints"])
    assert sm["max_violation"] >= 0.0 and sm["kkt_residual"] is not None and sm["kkt_residual"] > 0.0
    import json
    rj = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    assert rj["termination"] == sm["termination"] and rj["feasible"] == sm["feasible"]
