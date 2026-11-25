"""Reproduction of the L-bracket stress experiment (docs/olcumler/gerilme_onkayit.md §5, K1-K4, pre-registered
2026-11-26 00:24 +0300 before the first run).

What the first run found: the two unchanged examples stop with non-finite moduli (NaN gradient of the qp-relaxed
stress at x̄ = 0), so K1, and K2 / K4 on their designs, could not be evaluated. K1 stays as a strict expected failure
(``docs/olcumler/gerilme.md``). The tests named ``posthoc`` were added after that result and use the experiment-only
zero-safe subclass of ``_lbracket_stress``; they are not pre-registered evidence. The optimisation tests are slow."""
from __future__ import annotations

import pytest
import torch

import _lbracket_stress as B
from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.mechanisms import InvalidModuliError
from toporlogy.optimization.local import Aggregated
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.stress import VonMises

NAN_TORCH = B.torch_turns_zero_times_inf_into_nan()
WHY = "NaN gradient of the qp stress relaxation at x̄ = 0 (optimization/stress.py, VonMises); docs/olcumler/gerilme.md"


# ----------------------------------------------------------------------------------------------- the defect
def _relaxed_gradient(cls: type[VonMises]) -> torch.Tensor:
    d = Domain((6, 4))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    F = d.load_vector()
    F[2 * d.nodes_where(lambda X: (X[:, 0] == 6) & (X[:, 1] == 0))[0] + 1, 0] = -1.0
    x = torch.full(d.shape, 0.6, dtype=torch.float64)
    x[2, 3] = 0.0
    x.requires_grad_(True)
    E = 1e-6 + x ** 3
    U = Elasticity(d, fixed, 0.3).solve(E.reshape(-1), F)
    value = Aggregated(cls(d, 1.0, 0.3, E0=1.0, q=0.5), "pnorm", 8.0)(EvalContext(d, x, E, U, F))
    (g,) = torch.autograd.grad(value, x)
    return g


@pytest.mark.xfail(NAN_TORCH, reason=WHY, strict=True)
def test_relaxed_stress_gradient_is_finite_at_zero_density():
    assert bool(torch.isfinite(_relaxed_gradient(VonMises)).all())


def test_posthoc_zero_safe_subclass_changes_only_the_zero_density_entry():
    g = _relaxed_gradient(B.ZeroSafeVonMises)
    assert bool(torch.isfinite(g).all())
    ref = _relaxed_gradient(VonMises)
    finite = torch.isfinite(ref)
    assert int(finite.sum()) >= ref.numel() - 1
    assert torch.allclose(g[finite], ref[finite], rtol=1e-12, atol=0.0)


# ----------------------------------------------------------------------------------------------- pre-registered
@pytest.mark.slow
@pytest.mark.xfail(NAN_TORCH, reason=WHY, strict=True, raises=InvalidModuliError)
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
    out = B.run(method, True)
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
    row = B.run(method, True)["grey"]["load_path"]
    print(f"\n  load path {method}: max x̄ at load {row['max_density_at_load']:.3f}, compliance {row['compliance']:.4g}")
    assert row["load_attached_to_supported_solid"]


@pytest.mark.slow
@pytest.mark.parametrize("method", list(B.METHODS))
def test_posthoc_k4_refined_binary_design_is_singular(method):
    r = B.refinement(method)
    print(f"\n  K4 (post-hoc) {method}: " + " ".join(f"{lv['peak_centroid']['ratio']:.3f}" for lv in r["levels"])
          + f" | {r['status']}")
    assert r["status"] == "singular — max stress not converged"
