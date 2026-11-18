"""connectivity.one_body: layered protection (frozen ties + eroded-field conduction constraint + gate + repair).

The benchmark splits into two bodies under plain compliance minimisation (tests/toporlogy/test_connectivity.py measures
that); with the one-call plan the result must be one body holding every interface and PASS the gate, including the
0.3/0.7 thresholds and the one-element erosion.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import (MMA, BetaContinuation, Compliance, Constraint, Elasticity, SIMP, Study, Volume,
                          standard_chain)
from toporlogy.optimization.connectivity import analyse_connectivity, one_body, split_benchmark


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _study(domain, b, extra):
    chain = standard_chain(domain, 1.5, beta=1.0)
    cons = [Constraint(Volume(), 0.15)] + list(extra)
    return Study(domain, Elasticity(domain, b.fixed), b.loads, chain, SIMP(domain), Compliance(), cons, MMA(),
                 BetaContinuation(chain, 1.0, 16.0, every=20))


def test_one_body_plan_gives_one_robust_body():
    b = split_benchmark(2, 16)
    plan = one_body(b.domain, b.interfaces, ties=1, constraint="conduction", sink="support_a")
    assert plan.ties is not None and plan.ties.any()
    assert len(plan.constraints) == 1 and plan.constraints[0].key == "connectivity"
    res = _study(plan.domain, b, plan.constraints).run(max_iter=80)
    design, rep, br = plan.finish(res.xphys)
    print("\n" + rep.summary())
    assert rep.connected and rep.n_bodies == 1
    assert rep.verdict == "PASS", rep.reasons
    assert br is None                       # no repair needed


def test_finish_repairs_a_split_design():
    b = split_benchmark(2, 16)
    res = _study(b.domain, b, []).run(max_iter=80)
    before = analyse_connectivity(b.domain, res.xphys, b.interfaces)
    assert before.n_bodies >= 2 and not before.connected          # the benchmark really splits
    plan = one_body(b.domain, b.interfaces, ties=None, constraint=None)
    design, rep, br = plan.finish(res.xphys)
    assert br is not None and br.n_bridges >= 1
    assert rep.connected and rep.n_bodies == 1
    assert np.all(design[br.added] == 1.0)


def test_interfaces_as_node_sets_and_bad_constraint():
    b = split_benchmark(2, 12)
    nodes = {k: np.unique(b.domain.edof[np.flatnonzero(m.ravel())] // 2) for k, m in b.interfaces.items()}
    plan = one_body(b.domain, nodes, ties=1, constraint=None)
    assert set(plan.interfaces) == set(b.interfaces)
    with pytest.raises(ValueError, match='constraint:\\ conduction'):
        one_body(b.domain, b.interfaces, constraint="magic")
