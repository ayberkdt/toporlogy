"""Multi-start for the time-aware topology optimisation (docs/foundations/prereg/MS_multistart.md, MS1-MS4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.optimization.viscoelastic_topology import (RobustViscoelastic, multistart, optimize_viscoelastic,
                                                          perturbed_starts)

from test_viscoelastic_robust import _mbb
from test_viscoelastic_uncertain import SAMPLES, _set


def test_ms1_ms2_best_of_runs_and_deterministic() -> None:
    st, _ = _mbb()
    starts = [None] + perturbed_starts(2, 0.5, st.ne, 0.1, seed=0)
    a = multistart(st, starts, 0.5, 1.5, max_iter=8)
    b = multistart(st, starts, 0.5, 1.5, max_iter=8)
    objs = [r[0] for r in a.runs if r[3]]
    assert a.best.objective == min(objs) and a.runs[a.best_index][0] == a.best.objective
    assert np.array_equal(a.best.x, b.best.x)


def test_ms4_perturbed_starts() -> None:
    xs = perturbed_starts(5, 0.4, 300, 0.2, seed=3)
    for x in xs:
        assert abs(x.mean() - 0.4) <= 1e-12 and x.min() >= 0.0 and x.max() <= 1.0
    assert not np.array_equal(xs[0], xs[1])


@pytest.mark.slow
def test_ms3_robust_multistart_not_worse_than_nominal() -> None:
    st, times = _mbb()
    sts = _set(st.mesh, times, st.loads, np.setdiff1d(np.arange(st.ndof), st.free), SAMPLES)
    kw = dict(max_iter=1500, kkt_tol=1e-4)
    nom = optimize_viscoelastic(sts[4], 0.5, 1.5, **kw)
    rob = RobustViscoelastic(sts, kappa=2.0)
    ms = multistart(rob, [None, nom.x] + perturbed_starts(2, 0.5, st.ne, 0.1, seed=0), 0.5, 1.5, **kw)

    def robust_value(xp):
        with torch.no_grad():
            J = np.array([float(s.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))).compliance[-1]) for s in sts])
        return float(J.mean() + 2 * J.std())
    r_ms, r_nom = robust_value(ms.best.xphys), robust_value(nom.xphys)
    print(f"MS3 runs {ms.runs}; best index {ms.best_index}; robust value multistart {r_ms:.6e} vs nominal {r_nom:.6e}")
    assert r_ms <= r_nom
