"""PR-06: SDF / CSG contract — degenerate shapes and offsets that remove a shape are refused, not silently accepted.

Before this PR (measured 24 Nov 2026): ``SpherePrimitive(c, 1).offset(-2)`` had radius −1 and sdf > 0 everywhere (the
shape vanished without a message), ``CylinderPrimitive`` behaved the same, ``BoxPrimitive((0,0,0), (1,1,1)).offset(-0.8)``
gave lo = 0.8 > hi = 0.2 (empty), and a zero cylinder axis gave NaN distances with only a RuntimeWarning. A bore Cut
with machining stock ≥ its radius would have vanished from the print variant.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy import geometry as G
from toporlogy.post.primitives import BoxPrimitive, CylinderPrimitive, PlanePrimitive, SpherePrimitive

RNG = np.random.default_rng(6)
X = RNG.uniform(-3.0, 3.0, (400, 3))


# ------------------------------------------------------------------------------------------- construction
@pytest.mark.parametrize("make", [
    lambda: SpherePrimitive((0, 0, 0), 0.0),
    lambda: SpherePrimitive((0, 0, 0), -1.0),
    lambda: SpherePrimitive((0, 0, 0), float("nan")),
    lambda: SpherePrimitive((0, 0), 1.0),
    lambda: SpherePrimitive((0, float("inf"), 0), 1.0),
    lambda: CylinderPrimitive((0, 0, 0), (0, 0, 1), 0.0),
    lambda: CylinderPrimitive((0, 0, 0), (0, 0, 0), 1.0),
    lambda: CylinderPrimitive((0, 0, 0), (0, float("nan"), 1), 1.0),
    lambda: CylinderPrimitive((0, 0, 0), (0, 0, 1), 1.0, length=0.0),
    lambda: CylinderPrimitive((0, 0, 0), (0, 0, 1), 1.0, length=-2.0),
    lambda: PlanePrimitive((0, 0, 0), (0, 0, 0)),
    lambda: BoxPrimitive((0, 0, 0), (1, 0, 1)),
    lambda: BoxPrimitive((0, 0, 0), (1, -1, 1)),
    lambda: G.HalfSpace((0, 0, 0), (0, 0, 0)),
], ids=["sphere-r0", "sphere-r<0", "sphere-r-nan", "sphere-2d", "sphere-inf-centre", "cyl-r0", "cyl-zero-axis",
        "cyl-nan-axis", "cyl-length0", "cyl-length<0", "plane-zero-normal", "box-flat", "box-inverted",
        "halfspace-zero-normal"])
def test_degenerate_shapes_are_refused(make):
    with pytest.raises(ValueError):
        make()


# ------------------------------------------------------------------------------------------------ offsets
def test_offsets_that_remove_the_shape_are_refused():
    with pytest.raises(ValueError, match="removes"):
        SpherePrimitive((0, 0, 0), 1.0).offset(-1.0)
    with pytest.raises(ValueError, match="removes"):
        CylinderPrimitive((0, 0, 0), (0, 0, 1), 1.0).offset(-2.0)
    with pytest.raises(ValueError, match="removes"):
        CylinderPrimitive((0, 0, 0), (0, 0, 1), 5.0, length=2.0).offset(-1.0)       # caps meet
    with pytest.raises(ValueError, match="removes"):
        BoxPrimitive((0, 0, 0), (1, 1, 1)).offset(-0.5)


def test_offsets_match_the_analytic_shapes():
    c = np.array([0.3, -0.2, 0.1])
    assert np.allclose(SpherePrimitive(c, 1.5).offset(-0.4).sdf(X), np.linalg.norm(X - c, axis=1) - 1.1, atol=1e-14)
    cyl = CylinderPrimitive(c, (1, 1, 0), 1.2, length=3.0).offset(-0.3)
    assert np.allclose(cyl.sdf(X), CylinderPrimitive(c, (1, 1, 0), 0.9, length=2.4).sdf(X), atol=1e-14)
    lo, hi = np.array([-1.0, -0.5, -2.0]), np.array([1.5, 0.5, 1.0])
    box = BoxPrimitive(lo, hi)
    assert np.allclose(box.offset(-0.2).sdf(X), BoxPrimitive(lo + 0.2, hi - 0.2).sdf(X), atol=1e-14)
    assert np.allclose(box.offset(0.3).sdf(X), box.sdf(X) - 0.3, atol=1e-14)        # rounded box, exact offset


def test_bore_stock_larger_than_the_radius_fails_before_the_pipeline_runs():
    from toporlogy import Domain
    from toporlogy.post import Cut, PostParams, finalize
    d = Domain((8, 8, 4), h=1.0)
    bore = Cut(CylinderPrimitive((4, 4, 2), (0, 0, 1), 1.5), stock=2.0, name="bore")
    with pytest.raises(ValueError, match="bore"):
        finalize(d, np.ones(d.shape), preserve=[bore], params=PostParams(upsample=2))


# --------------------------------------------------------------------------------------- element_fraction
def _exact_plane_fraction(domain, n, p0):
    """Volume fraction of every element below the plane n·x ≤ n·p0 (all n_k > 0): inclusion–exclusion over the
    cube corners, V = Σ_v (−1)^|v| max(0, n·p0 − n·x_v)³ / (6 n₁n₂n₃)."""
    import itertools
    h = domain.h
    x0 = np.indices(domain.shape).reshape(3, -1).T * h + np.asarray(domain.origin, float)
    out = np.zeros(len(x0))
    for v in itertools.product((0, 1), repeat=3):
        out += (-1) ** sum(v) * np.maximum(n @ p0 - (x0 + h * np.array(v)) @ n, 0.0) ** 3
    return (out / (6.0 * np.prod(n)) / h ** 3).reshape(domain.shape)


def test_element_fraction_converges_to_the_exact_planar_cut():
    """Midpoint rule on samples³ points: per element error ≤ 3 / samples for a plane (each of the samples² columns
    along the dominant normal axis crosses at most 3 sub-cells). Measured on this oblique plane: 0.115 (2),
    0.0096 (6, the default), 0.0014 (16), 0.00027 (32)."""
    from toporlogy import Domain
    from toporlogy.post.primitives import element_fraction
    d = Domain((10, 10, 10), h=1.0)
    n = np.array([1.0, 0.37, 0.21])
    n /= np.linalg.norm(n)
    p0 = np.array([5.13, 5.0, 5.0])
    exact = _exact_plane_fraction(d, n, p0)
    err = {s: float(np.abs(element_fraction(d, PlanePrimitive(p0, n), samples=s) - exact).max())
           for s in (2, 6, 16, 32)}
    assert all(e <= 3.0 / s for s, e in err.items())
    assert err[32] < err[16] < err[6] < err[2]
    assert err[6] < 0.012                                         # the default


@pytest.mark.parametrize("samples", [0, -1, 2.5])
def test_element_fraction_refuses_a_bad_sample_count(samples):
    from toporlogy import Domain
    from toporlogy.post.primitives import element_fraction
    with pytest.raises(ValueError, match="samples"):
        element_fraction(Domain((2, 2, 2), h=1.0), SpherePrimitive((1, 1, 1), 0.7), samples=samples)
