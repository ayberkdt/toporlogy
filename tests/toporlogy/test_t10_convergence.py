"""PR-07: T10 mesh convergence of one finished geometry on at least three really different meshes.

Before: the post-processing sweep re-checked the finished part at two tet sizes and called a ratio change <= 5 %
"converged" (docs/olcumler/robustness.md §4.2). The stored records (robustness.json, "post") show why that is not a
convergence test: r2 passed with 0.93 % although the finer size added only 1.8 % elements (140 550 -> 143 062), and
r1 changed by 12.4 % while the tet-mesh volume itself changed from 0.988 to 0.998 of the voxel volume (the surface
was remeshed per size, so the geometry was not the same). The records are not changed by this PR.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from toporlogy.post.verify import CONVERGED, NOT_CONVERGED, assess_convergence

RECORDS = Path(__file__).resolve().parents[2] / "docs" / "olcumler" / "robustness.json"


def _stored_t10(name: str) -> list[dict]:
    post = json.loads(RECORDS.read_text(encoding="utf-8"))["post"]
    rows = post if isinstance(post, list) else list(post.values())
    return next(r for r in rows if isinstance(r, dict) and r.get("input") == name)["t10"]["runs"]


@pytest.mark.parametrize("name", ["r1_cantilever", "r2_mbb3d"])
def test_the_stored_two_size_rechecks_are_not_convergence(name):
    runs = _stored_t10(name)
    old = abs(runs[1]["ratio"] - runs[0]["ratio"]) / runs[1]["ratio"] <= 0.05          # the old criterion
    h = [r["elements"] ** (-1 / 3) for r in runs]                     # h ~ elements^(-1/3)
    new = assess_convergence(h, [r["compliance_t10"] for r in runs])
    assert new["verdict"] == NOT_CONVERGED and "at least 3" in new["reasons"][0]
    if name == "r2_mbb3d":
        assert old                                                     # the old check called this converged
        three = assess_convergence(h + [h[1] / 1.006], [r["compliance_t10"] for r in runs] + [12.9])
        assert any("not really different" in s for s in three["reasons"])


def _series(h, f_inf=10.0, c=3.0, p=2.0):
    return [f_inf + c * x ** p for x in h]


def test_a_clean_power_law_converges_with_its_order_and_limit():
    h = [0.4, 0.25, 0.16, 0.1]
    r = assess_convergence(h, _series(h))
    assert r["verdict"] == CONVERGED and not r["reasons"]
    assert r["order"] == pytest.approx(2.0, rel=1e-9) and r["order_coarse"] == pytest.approx(2.0, rel=1e-9)
    assert r["extrapolated"] == pytest.approx(10.0, rel=1e-12)
    h2 = [0.4, 0.27, 0.15]                                             # unequal refinement factors (iterated q(p))
    r2 = assess_convergence(h2, _series(h2, p=1.5))
    assert r2["order"] == pytest.approx(1.5, rel=1e-9) and r2["extrapolated"] == pytest.approx(10.0, rel=1e-12)


def test_not_converged_cases_name_their_reason():
    h = [0.4, 0.25, 0.16]
    osc = assess_convergence(h, [10.0, 10.3, 10.1])
    assert osc["verdict"] == NOT_CONVERGED and "oscillatory" in osc["reasons"][0]
    close = assess_convergence([0.4, 0.38, 0.36], _series([0.4, 0.38, 0.36]))
    assert any("not really different" in s for s in close["reasons"])
    big = assess_convergence(h, _series(h, c=300.0))                   # still far from the limit
    assert any("GCI" in s for s in big["reasons"])
    low = assess_convergence(h, _series(h, c=0.01, p=0.2))
    assert any("apparent order" in s for s in low["reasons"])
    h4 = [0.4, 0.25, 0.16, 0.1]
    f = _series(h4)
    f[0] += 0.3                                                        # coarsest mesh off the power law
    unstable = assess_convergence(h4, f)
    assert any("not stable" in s or "coarser triplet" in s for s in unstable["reasons"])


def test_mesh_convergence_needs_three_sizes():
    from toporlogy.post.verify import mesh_convergence
    with pytest.raises(ValueError, match="at least 3"):
        mesh_convergence((np.zeros((4, 3)), np.zeros((4, 3), int)), [], {}, E=1.0, sizes=(1.0, 0.5))


def _box_surface(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


@pytest.mark.gmsh
@pytest.mark.slow
def test_the_same_cantilever_converges_on_fine_meshes_and_not_on_coarse_ones():
    """Box cantilever 20 x 4 x 4, clamped end, uniform tip traction; one surface for every size (volumes equal to
    1e-14). Measured 24 Nov 2026: sizes 2.0 / 1.25 / 0.8 / 0.5 give compliance 126.541 / 126.830 / 126.941 / 127.069,
    NOT CONVERGED (apparent order 0.11 on the finest triplet, 2.44 on the coarser one); sizes 0.8 / 0.5 / 0.32 give
    order 1.47 and fine-grid GCI 0.07 %, CONVERGED (with 0.2 added: order 1.94 vs 1.47, GCI 0.02 %, CONVERGED)."""
    pytest.importorskip("gmsh")
    from toporlogy.post.verify import mesh_convergence
    V, F = _box_surface((0, 0, 0), (20, 4, 4))
    sup = [lambda X: X[:, 0] < 1e-6]
    loads = {"tip": [(lambda X: X[:, 0] > 20 - 1e-6, np.array([0.0, 0.0, -1.0]))]}
    coarse = mesh_convergence((V, F), sup, loads, E=1.0, sizes=(2.0, 1.25, 0.8, 0.5), surface_edge=0)
    fine = mesh_convergence((V, F), sup, loads, E=1.0, sizes=(0.8, 0.5, 0.32), surface_edge=0)
    assert coarse["verdict"] == NOT_CONVERGED and fine["verdict"] == CONVERGED
    assert coarse["volume_spread"] < 1e-10 and fine["volume_spread"] < 1e-10
    assert all(m["quality"]["q_min"] > 0.2 for m in coarse["meshes"] + fine["meshes"])
    print(f"\n  coarse: {coarse['cases']['tip']['reasons']}; fine: order {fine['cases']['tip']['order']:.2f}, "
          f"GCI {100 * fine['cases']['tip']['gci_fine']:.3f} %")
