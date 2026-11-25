"""The L-bracket stress examples (docs/olcumler/gerilme_onkayit.md §5 and §7).

§5 (K1-K4, pre-registered 2026-11-26 00:24 +0300) measured the examples as they were; their returned designs left
the load in void (gerilme.md §3.2). §7 (Y1-Y5, pre-registered 01:56 +0300 before the first run) gave both examples a
solid load pad and the library's one-body protection (``connectivity``). K1 compared with the historical ratios of
the old examples and was removed with them (its FAIL stays in the record). Tests named ``posthoc`` were added after
a first result and are not pre-registered evidence. The optimisation tests are slow."""
from __future__ import annotations

import pytest

import _lbracket_stress as B


# ----------------------------------------------------------------------------------------------- pre-registered
@pytest.mark.slow
def test_k3_solid_bracket_peak():
    got = B.refinement("solid")["levels"][0]["peak_centroid"]["ratio"]
    print(f"\n  K3: {got:.4f}")
    assert abs(got - B.HISTORICAL_SOLID) <= 0.005


@pytest.mark.slow
def test_k4_solid_bracket_refinement_is_singular():
    r = B.refinement("solid")
    print("\n  K4 solid: " + " ".join(f"{lv['peak_centroid']['ratio']:.3f}" for lv in r["levels"]))
    assert r["status"] == "singular — max stress not converged"


# ----------------------------------------------------------------------------------------------- post-hoc
@pytest.mark.slow
@pytest.mark.parametrize("method", list(B.METHODS))
def test_posthoc_k2_returned_design_against_the_independent_implementation(method):
    out = B.run(method)
    print(f"\n  K2 (post-hoc) {method}: {out['oracle_relative_difference']:.2e}")
    assert out["oracle_relative_difference"] <= 1e-12
    for variant in ("grey", "binary"):                       # recovery and constraint module: one quantity
        row = out[variant]
        assert row["raw_solid_max"]["ratio"] == pytest.approx(row["raw_solid_max_constraint_module"], rel=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("method", list(B.METHODS))
def test_posthoc_k4_refined_binary_design_is_singular(method):
    r = B.refinement(method)
    print(f"\n  K4 (post-hoc) {method}: " + " ".join(f"{lv['peak_centroid']['ratio']:.3f}" for lv in r["levels"])
          + f" | {r['status']}")
    assert r["status"] == "singular — max stress not converged"


# ----------------------------------------------------------------------------------------------- load path (§7)
@pytest.mark.slow
@pytest.mark.parametrize("method", list(B.METHODS))
def test_y1_y3_y4_the_returned_design_carries_its_load(method):
    out = B.run(method)                                      # Y1: completes
    c0 = B.solid_compliance(method)
    for variant in ("grey", "binary"):
        lp = out[variant]["load_path"]
        print(f"\n  Y {method} {variant}: attached {lp['load_attached_to_supported_solid']}, "
              f"compliance / solid {lp['compliance'] / c0:.2f}, raw stress {out[variant]['raw_solid_max']['ratio']:.3f}")
        assert lp["load_attached_to_supported_solid"]                                 # Y3
    assert out["binary"]["load_path"]["compliance"] / c0 <= 10.0                       # Y4


@pytest.mark.slow
@pytest.mark.parametrize("method", [
    "aggregated",
    pytest.param("augmented_lagrangian", marks=pytest.mark.xfail(strict=True, reason=(
        "pre-registered Y2 FAILED on the first run: the gate says FAIL for 2 floating 8-cell islands; the support "
        "and the load are in one body (Y3 passes)")))])
def test_y2_one_body_gate_on_the_returned_design(method):
    rep = B.gate(method, B.run(method)["arrays"]["x_grey"])
    print(f"\n  Y2 {method}: {rep['summary']}")
    assert rep["verdict"] in ("PASS", "FRAGILE")


@pytest.mark.slow
@pytest.mark.parametrize("method", list(B.METHODS))
def test_y5_the_gate_rejects_the_old_designs(method):
    rep = B.gate(method, B.old_design(method))
    print(f"\n  Y5 {method}: {rep['summary']}")
    assert rep["verdict"] == "FAIL"
    assert "L1 force" in rep["unattached"]
