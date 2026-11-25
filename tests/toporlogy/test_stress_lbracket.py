"""Reproduction of the L-bracket stress experiment (docs/olcumler/gerilme_onkayit.md §5, K1-K4, pre-registered
2026-11-26 00:24 +0300 before the first run).

The first run stopped with non-finite moduli (NaN gradient of the qp relaxation at x̄ = 0); since the library fix
(gerilme_onkayit.md §6, ``test_stress_zero_density.py``) the unchanged examples run and reproduce the earlier
post-hoc numbers bit for bit (F5). K1 then fails on its own threshold and stays a strict expected failure. The tests
named ``posthoc`` were added after the first result; they are not pre-registered evidence. The optimisation tests are
slow."""
from __future__ import annotations

import pytest

import _lbracket_stress as B


# ----------------------------------------------------------------------------------------------- pre-registered
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="pre-registered K1 FAILED once the examples ran (after the §6 fix): raw solid "
                   "ratios 1.229 / 1.744 (aggregated, grey / 0-1) and 1.169 / 1.025 (augmented Lagrangian) against "
                   "1.131 / 1.715 and 1.103 / 1.025; the grey differences 0.098 and 0.066 exceed 0.05")
@pytest.mark.parametrize("method", list(B.METHODS))
def test_k1_historical_ratios(method):
    out = B.run(method)
    for variant in ("grey", "binary"):
        got = out[variant]["raw_solid_max"]["ratio"]
        want = B.HISTORICAL[method][variant]
        print(f"\n  K1 {method} {variant}: {got:.4f} (historical {want})")
        assert abs(got - want) <= 0.05


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
@pytest.mark.xfail(strict=True, reason="the returned designs of both examples leave the loaded nodes in void "
                                       "elements (post-hoc finding, docs/olcumler/gerilme.md)")
@pytest.mark.parametrize("method", list(B.METHODS))
def test_posthoc_returned_design_carries_its_load_through_solid_material(method):
    row = B.run(method)["grey"]["load_path"]
    print(f"\n  load path {method}: max x̄ at load {row['max_density_at_load']:.3f}, compliance {row['compliance']:.4g}")
    assert row["load_attached_to_supported_solid"]


@pytest.mark.slow
@pytest.mark.parametrize("method", list(B.METHODS))
def test_posthoc_k4_refined_binary_design_is_singular(method):
    r = B.refinement(method)
    print(f"\n  K4 (post-hoc) {method}: " + " ".join(f"{lv['peak_centroid']['ratio']:.3f}" for lv in r["levels"])
          + f" | {r['status']}")
    assert r["status"] == "singular — max stress not converged"
