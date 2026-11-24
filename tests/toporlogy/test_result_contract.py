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
from toporlogy.optimizers.base import Optimizer


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
    v = res.final.values["volume"]
    assert res.kkt_residual is None and res.termination == "max_iter"
    assert res.feasible is bool(v <= 0.5 + 1e-3 * 0.5)                  # computed here, not taken from the run
    assert res.max_violation == pytest.approx(max(0.0, (v - 0.5) / 0.5), abs=1e-15)


# ---------------------------------------------------------------------------------------------- summary flags
def _summary(**over):
    s = {"converged": True, "termination": "design_change", "objective_source": "returned design",
         "constraints": [{"name": "volume", "value": 0.3, "upper": 0.3, "verdict": "PASS",
                          "source": "returned design"}], "post": {}}
    s.update(over)
    return s


def test_flags_name_every_finding_once_and_nothing_for_a_clean_run():
    assert flags_of(_summary()) == [] and status_of(_summary()) == "ok"
    s = _summary(converged=False, termination="schedule_step_at_max_iter",
                 constraints=[{"name": "v", "verdict": "FAIL"}, {"name": "w", "verdict": "FAIL"}],
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

    from toporlogy.configuration.build import build_setup
    from toporlogy.execution.runner import run_setup
    s = build_setup(mbb(max_iter=3))
    sm = run_setup(s, tmp_path / "run")
    assert sm["termination"] == "max_iter" and sm["converged"] is False and "not_converged" in sm["flags"]
    assert sm["verdict"] == ("PASS" if all(c["verdict"] == "PASS" for c in sm["constraints"]) else "FAIL")
    want = max([0.0] + [(c["value"] - c["upper"]) / max(abs(c["upper"]), 1e-12) for c in sm["constraints"]])
    assert sm["max_violation"] == pytest.approx(want, abs=1e-15)
    # the KKT residual recomputed from the returned design and the optimizer's last multipliers
    x = np.load(tmp_path / "run" / "result.npz")["x"]
    ev = s.study.evaluate(x)
    assert sm["kkt_residual"] == pytest.approx(s.study._kkt_residual(x, ev, s.study.optimizer.kkt_multipliers()),
                                               rel=1e-9)
    import json
    rj = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    assert rj["termination"] == sm["termination"] and rj["feasible"] == (sm["verdict"] == "PASS")


# ---------------------------------------------------------------------------------------------- read_run
def test_read_run_reports_the_returned_designs_constraints(tmp_path):
    """A run recorded from Python (no summary.json): read_run must report the constraint values of the design the
    run returned (result.json), not those of the last history row, which belong to the design before the final
    update. Before the fix it read history[-1]."""
    import json

    from toporlogy.execution.runs import read_run
    st = _mbb(T.MMA(move=0.1), vf=0.3)                  # from 0.6 the volume falls ~0.1 per step: designs differ
    st.recorder = T.Recorder(tmp_path / "run")
    res = st.run(x0=np.full(st.n_var, 0.6), max_iter=2, tol=0.0)
    s = read_run(tmp_path / "run")
    v = {c["name"]: c["value"] for c in s["constraints"]}
    again = st.evaluate(res.x).values                                   # the returned design, re-evaluated
    assert v["volume"] == res.constraint_values["volume"] == pytest.approx(again["volume"], rel=1e-12)
    last = s["history"][-1]["volume"]
    assert abs(last - v["volume"]) > 1e-3                               # the old source really is a different design
    assert [c["source"] for c in s["constraints"]] == ["returned design"]
    assert s["termination"] == res.termination and s["verdict"] == ("PASS" if res.feasible else "FAIL")
    assert s["max_violation"] == pytest.approx(res.max_violation, abs=1e-15)
    assert [c["verdict"] for c in s["constraints"]] == ["PASS" if res.feasible else "FAIL"]
    # a result.json written before "constraints" existed: the fallback is used and says what it is
    rj = tmp_path / "run" / "result.json"
    old = json.loads(rj.read_text(encoding="utf-8"))
    del old["constraints"]
    rj.write_text(json.dumps(old), encoding="utf-8")
    s_old = read_run(tmp_path / "run")
    assert s_old["constraints"][0]["value"] == last and s_old["constraints"][0]["source"] == "last iterate"


# ---------------------------------------------------------------------------------------------- PR-08 verdicts
# Every constraint of a run summary carries a verdict PASS / FAIL / UNVERIFIED and the source of its value; a value
# that does not belong to the returned design (result.npz x) is UNVERIFIED, never "satisfied", and the run status is
# then never "ok" (docs/yol_haritasi_2026-11-18.md §3, PR-08).
def _old_python_run(folder, volumes, upper=0.5, converged=True):
    """A run folder as Recorder wrote it before 14 Nov 2026 (99d8d67): result.json without "constraints", its
    objective = history f0[-1]; the constraint values exist only in history.jsonl, for the iterates."""
    import json
    folder.mkdir()
    rows = [{"it": i + 1, "t": 0.1 * i, "f0": 10.0 - i, "change": 0.5 / (i + 1), "volume": v}
            for i, v in enumerate(volumes)]
    (folder / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res = {"iterations": len(rows), "converged": converged, "seconds": 1.0, "objective": rows[-1]["f0"],
           "settings": {"objective": "compliance", "constraints": [["volume", upper]]}}
    (folder / "result.json").write_text(json.dumps(res), encoding="utf-8")


def test_old_result_json_is_unverified_not_ok(tmp_path):
    """Red before PR-08: the last history row satisfies the constraint and the run converged, so read_run said
    "ok" and "satisfied" although that row belongs to the design BEFORE the last update; the returned design's
    constraint value was never recorded. Now the constraint is UNVERIFIED and the status says so."""
    from toporlogy.execution.runs import read_run
    _old_python_run(tmp_path / "run", [0.7, 0.55, 0.499])
    s = read_run(tmp_path / "run")
    assert s["status"] != "ok"
    c = s["constraints"][0]
    assert c["value"] == 0.499 and c["verdict"] == "UNVERIFIED" and c["source"] == "last iterate"
    assert "satisfied" not in c
    assert s["verdict"] == "UNVERIFIED" and s["objective_source"] == "last iterate"
    assert s["status"] == "constraint unverified, objective unverified"
    assert "constraint_unverified" in s["flags"] and "objective_unverified" in s["flags"]
    # a last iterate that violates is not a FAIL of the returned design either: still UNVERIFIED
    _old_python_run(tmp_path / "run2", [0.7, 0.6])
    assert read_run(tmp_path / "run2")["constraints"][0]["verdict"] == "UNVERIFIED"


def test_old_summary_json_is_unverified_not_ok(tmp_path):
    """Red before PR-08: a summary.json written before the G7 contract (18 Nov 2026, no "flags") was returned as
    stored, status "ok". Before 42f593c (17 Nov 2026) run_setup took the constraint values from history[-1], and
    such a file cannot be told apart from one written in between, so its constraints are UNVERIFIED."""
    import json

    from toporlogy.execution.runs import read_run
    f = tmp_path / "run"
    f.mkdir()
    old = {"name": "mbb", "objective_name": "compliance", "objective": 5.0, "iterations": 40, "max_iter": 100,
           "converged": True, "seconds": 2.0, "grey_fraction": 0.1, "post": {}, "status": "ok",
           "constraints": [{"name": "volume", "value": 0.5, "upper": 0.5, "satisfied": True}]}
    (f / "summary.json").write_text(json.dumps(old), encoding="utf-8")
    s = read_run(f)
    assert s["status"] != "ok"
    assert [(c["verdict"], c["source"]) for c in s["constraints"]] == [("UNVERIFIED", "unknown")]
    assert s["verdict"] == "UNVERIFIED" and s["objective_source"] == "unknown"
    # with "flags" (written by G7 or later, after 42f593c) the stored values are the returned design's
    old.update(flags=[], termination="design_change", feasible=True)
    (f / "summary.json").write_text(json.dumps(old), encoding="utf-8")
    s = read_run(f)
    assert s["status"] == "ok" and s["verdict"] == "PASS"
    assert [(c["verdict"], c["source"]) for c in s["constraints"]] == [("PASS", "returned design")]
    assert "feasible" not in s and "satisfied" not in s["constraints"][0]          # one name per concept


def test_read_run_judges_with_the_runs_own_feasibility_tol(tmp_path):
    """Red before PR-08: read_run judged result.json constraints with a fixed 1e-3 while Study.run used its
    feasibility_tol, so the same design was "feasible" in result.json and "constraint violated" in read_run.
    Volume 0.6 against 0.5 is 20 % over: inside feasibility_tol 0.5, outside 1e-3."""
    import json

    from toporlogy.execution.runs import read_run
    st = _mbb(T.MMA(move=0.2), vf=0.5)
    st.recorder = T.Recorder(tmp_path / "run")
    res = st.run(x0=np.full(st.n_var, 0.6), max_iter=0, feasibility_tol=0.5)
    assert res.feasible is True and res.constraint_values["volume"] == pytest.approx(0.6, rel=1e-12)
    s = read_run(tmp_path / "run")
    assert "constraint violated" not in s["status"]
    assert json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))["feasibility_tol"] == 0.5
    assert [(c["verdict"], c["source"]) for c in s["constraints"]] == [("PASS", "returned design")]
    assert s["verdict"] == "PASS" and s["status"] == "not converged"


def test_summary_result_and_read_run_report_the_same_design(tmp_path):
    """summary.json, result.json, read_run (from summary.json and, without it, from result.json) give the same
    objective, constraint values and verdicts, those of the returned design (result.npz x re-evaluated), not those
    of the last iterate."""
    import json
    import shutil

    from test_config import mbb

    from toporlogy.configuration.build import build_setup
    from toporlogy.execution.runner import run_setup
    from toporlogy.execution.runs import read_run
    s = build_setup(mbb(max_iter=3))
    sm = run_setup(s, tmp_path / "run")
    ev = s.study.evaluate(np.load(tmp_path / "run" / "result.npz")["x"], gradients=False)
    rj = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    from_summary = read_run(tmp_path / "run")
    shutil.copytree(tmp_path / "run", tmp_path / "py")
    (tmp_path / "py" / "summary.json").unlink()
    from_result = read_run(tmp_path / "py")
    last = from_summary["history"][-1]
    assert abs(last["f0"] - ev.f0) > 1e-6 * abs(ev.f0)          # the last iterate is a different design
    assert abs(last["volume"] - ev.values["volume"]) > 1e-6
    for rep in (sm, from_summary, from_result):
        assert rep["objective"] == pytest.approx(ev.f0, rel=1e-12) and rep["objective_source"] == "returned design"
        assert [(c["name"], c["source"]) for c in rep["constraints"]] == [("volume", "returned design")]
        assert rep["constraints"][0]["value"] == pytest.approx(ev.values["volume"], rel=1e-12)
        assert rep["verdict"] == ("PASS" if rj["feasible"] else "FAIL")
        assert rep["status"] == sm["status"]
    assert rj["objective"] == pytest.approx(ev.f0, rel=1e-12)
    assert rj["constraints"]["volume"] == pytest.approx(ev.values["volume"], rel=1e-12)
    assert [c["verdict"] for c in from_result["constraints"]] == [c["verdict"] for c in sm["constraints"]]


def test_cli_never_shows_an_unverified_run_as_ok(tmp_path, capsys):
    """show prints the verdict and the source of an unverified value and exits 1; compare does not pick an
    unverified run as the best feasible one."""
    from toporlogy.execution.cli import main
    _old_python_run(tmp_path / "old", [0.7, 0.55, 0.499])
    code = main(["show", str(tmp_path / "old")])
    out = capsys.readouterr().out
    assert code == 1
    status = next(ln for ln in out.splitlines() if ln.startswith("status"))
    assert "ok" not in status.split() and "unverified" in status
    line = next(ln for ln in out.splitlines() if ln.startswith("constraint"))
    assert "UNVERIFIED" in line and "last iterate" in line and "satisfied" not in line
    obj = next(ln for ln in out.splitlines() if ln.startswith("objective"))
    assert "UNVERIFIED" in obj and "last iterate" in obj
    _old_python_run(tmp_path / "old2", [0.7, 0.55, 0.45])
    main(["compare", str(tmp_path / "old"), str(tmp_path / "old2")])
    rows = capsys.readouterr().out.splitlines()
    assert not any(r.startswith("*") for r in rows)                     # no run is verified feasible
