"""KKT-residual stopping for the viscoelastic optimisations (docs/foundations/prereg/KS_kkt_stop.md, KS1-KS4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from toporlogy.optimization.viscoelastic_topology import RobustViscoelastic, optimize_viscoelastic

from test_viscoelastic_robust import _mbb, _step_of
from test_viscoelastic_uncertain import SAMPLES, _set

REPO = Path(__file__).resolve().parents[2]


FINGERPRINT = REPO / "tests" / "toporlogy" / "baseline" / "ks1_fingerprint_78bedbb.json"


def test_ks1_default_unchanged() -> None:
    """x after 5 iterations equals the output of the function before the KKT stop (78bedbb). The fingerprint was
    written by loading that commit's source and running it (``docs/foundations/prereg/KS_kkt_stop.md``)."""
    st, _ = _mbb()
    new = optimize_viscoelastic(st, 0.5, 1.5, max_iter=5)
    old_x = np.array(json.loads(FINGERPRINT.read_text(encoding="utf-8"))["x"])
    assert np.array_equal(new.x, old_x)
    assert new.termination == "max_iter" and new.kkt_residual is None


def _w(xp):
    return torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="pre-registered KS2 FAILED: with both runs KKT-stopped (residuals 6.9e-5, "
                   "8.1e-5) the robust design's mean+2std is 1.408537e-4 vs 1.404853e-4 for the nominal design (+0.26 %): "
                   "different stationary points, not a stopping artefact")
def test_ks2_ks4_robust_with_kkt_stop() -> None:
    st, times = _mbb()
    sts = _set(st.mesh, times, st.loads, np.setdiff1d(np.arange(st.ndof), st.free), SAMPLES)
    rob = optimize_viscoelastic(RobustViscoelastic(sts, kappa=2.0), 0.5, 1.5, max_iter=1500, kkt_tol=1e-4)
    nom = optimize_viscoelastic(sts[4], 0.5, 1.5, max_iter=1500, kkt_tol=1e-4)
    with torch.no_grad():
        def stat(xp):
            J = np.array([float(s.response(_w(xp)).compliance[-1]) for s in sts])
            return J.mean() + 2 * J.std()
        r, n = stat(rob.xphys), stat(nom.xphys)
    print(f"KS2 robust mean+2std {r:.6e} ({rob.iterations} it, {rob.termination}, kkt {rob.kkt_residual}); "
          f"nominal {n:.6e} ({nom.iterations} it, {nom.termination}, kkt {nom.kkt_residual})")
    for res in (rob, nom):                                                               # KS4
        assert (res.converged and res.kkt_residual <= 1e-4) or (not res.converged and res.iterations == 1500)
    assert r <= n * (1 + 1e-4)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="pre-registered KS3 FAILED: all four runs KKT-stopped (residual <= 9.2e-5); "
                   "min-max worst 1.215313e-4 vs 1.211352e-4 for the t = 100 s design (+0.33 %): a different "
                   "stationary point")
def test_ks3_ks4_minmax_with_kkt_stop() -> None:
    st, times = _mbb()
    ks = [0, _step_of(times, 10.0), len(times) - 1]
    mm = optimize_viscoelastic(st, 0.5, 1.5, steps=ks, max_iter=1500, kkt_tol=1e-4)
    singles = [optimize_viscoelastic(st, 0.5, 1.5, step_index=k, max_iter=1500, kkt_tol=1e-4) for k in ks]
    with torch.no_grad():
        def worst(xp):
            c = st.response(_w(xp)).compliance
            return max(float(c[k]) for k in ks)
        wm, ws = worst(mm.xphys), [worst(r.xphys) for r in singles]
    print(f"KS3 min-max worst {wm:.6e} ({mm.iterations} it, {mm.termination}, kkt {mm.kkt_residual}); singles "
          f"{[f'{w:.6e}' for w in ws]} {[(r.iterations, r.termination) for r in singles]}")
    for res in [mm, *singles]:
        assert (res.converged and res.kkt_residual <= 1e-4) or (not res.converged and res.iterations == 1500)
    assert wm <= (1 + 1e-3) * min(ws)
