"""POD-Galerkin reduced model: records R1-R7 of ``docs/olcumler/laminat_uq_onkayit.md``.

Problem: cantilever 40 × 20, one modulus per quadrant (θ_q in [0.5, 2], normalised values), fixed tip load. The
independent side is the library's own full-order solver (``Elasticity.solve_numpy``) and ``AffineSystem.solve``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import qmc

import toporlogy as T
from toporlogy.optimization import rom

NX, NY = 40, 20
LOW, HIGH = 0.5, 2.0
RANK = 15


def thetas(n: int, seed: int) -> np.ndarray:
    return LOW + (HIGH - LOW) * qmc.LatinHypercube(4, seed=seed).random(n)


def problem():
    d = T.Domain((NX, NY))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    physics = T.Elasticity(d, fixed)
    ijk = d.element_ijk(physics.elements)
    regions = (ijk[:, 0] >= NX // 2).astype(np.int64) + 2 * (ijk[:, 1] >= NY // 2).astype(np.int64)
    F = d.load_vector(2)
    tip = d.nodes_where(lambda X: (X[:, 0] > NX - 1e-9) & (np.abs(X[:, 1] - NY / 2) < 1e-9))
    F[d.dofs(tip, 1), 0] = -1.0
    F[d.dofs(tip, 0), 1] = 1.0
    return physics, regions, F, rom.AffineSystem.from_elasticity(physics, regions, F[:, 0])


@pytest.fixture(scope="module")
def case():
    physics, regions, F, system = problem()
    return physics, regions, F, system, rom.PODModel(system, thetas(40, 1), RANK)


def energy_error(system: rom.AffineSystem, theta: np.ndarray, u_r: np.ndarray, load=None) -> tuple[float, float]:
    """(‖u − u_r‖_K, ‖u‖_K)."""
    u = system.solve(theta, load)
    K = system.stiffness(theta)
    e = u - u_r
    return math.sqrt(max(float(e @ (K @ e)), 0.0)), math.sqrt(float(u @ (K @ u)))


def test_affine_system_reproduces_the_library_solver(case) -> None:
    physics, regions, F, system, _ = case
    t = np.array([0.7, 1.3, 1.9, 0.6])
    ref = physics.solve_numpy(t[regions], F[:, 0])[physics.free, 0]
    assert np.abs(system.solve(t) - ref).max() <= 1e-10 * np.abs(ref).max()


def test_r1_projection_error_equals_the_discarded_singular_values(case) -> None:
    model = case[4]
    S = model.snapshot_matrix
    _, sv = rom.pod_basis(S)
    total = float((sv**2).sum())
    for r in (1, 3, 6, 10, 15):
        V, _ = rom.pod_basis(S, r)
        assert abs(rom.projection_error(S, V) - float((sv[r:] ** 2).sum())) <= 1e-12 * total, r
        assert np.abs(V.T @ V - np.eye(r)).max() <= 1e-12


def test_r2_full_rank_basis_reproduces_the_snapshots(case) -> None:
    system = case[3]
    train = thetas(40, 1)
    full = rom.PODModel(system, train)
    worst = 0.0
    for t in train:
        e, nu = energy_error(system, t, full.solve(t).u)
        worst = max(worst, e / nu)
    assert worst <= 1e-6, (worst, full.rank)


def test_r3_r4_bound_is_rigorous_and_the_test_error_is_small(case) -> None:
    system, model = case[3], case[4]
    errors = []
    for t in thetas(50, 2):
        sol = model.solve(t)
        e, nu = energy_error(system, t, sol.u)
        gamma = math.sqrt(np.max(t / model.reference) / np.min(t / model.reference))
        eff = sol.error_bound / e
        assert 1 - 1e-6 <= eff <= gamma * (1 + 1e-6), (eff, gamma)
        assert e / nu <= sol.relative_error_bound * (1 + 1e-6)
        assert sol.approximate and not sol.extrapolation or bool(np.any((t < model.train_low) | (t > model.train_high)))
        errors.append(e / nu)
    assert max(errors) <= 1e-2, max(errors)                                   # R4


def compliance_rows(system: rom.AffineSystem, model: rom.PODModel) -> list[tuple[float, float, float]]:
    """(s, s − s_r, e²) for every test point."""
    out = []
    for t in thetas(50, 2):
        sol = model.solve(t)
        s = float(system.load @ system.solve(t))
        e, _ = energy_error(system, t, sol.u)
        out.append((s, s - sol.compliance, e**2))
    return out


def test_r5a_compliance_is_a_lower_bound(case) -> None:
    for s, gap, _ in compliance_rows(case[3], case[4]):
        assert gap >= -1e-10 * s


@pytest.mark.xfail(strict=True, reason="pre-registered R5 identity failed: s - s_r is a difference of two nearly equal "
                                       "numbers and carries rounding above 1e-6 of e^2 (see docstring)")
def test_r5b_compliance_gap_equals_the_energy_error(case) -> None:
    """Pre-registered R5, first run (26 Nov 2026), not met: |(s − s_r) − e²| / e² up to 2.28e-5, above 1e-6 at 30 of 50
    test points. The tolerance was set relative to e², which is about 1e-7 of s here; the absolute difference is at
    most 1.3e-12 of s (rounding of the two solves). The post hoc test below states that."""
    for _, gap, e2 in compliance_rows(case[3], case[4]):
        assert abs(gap - e2) <= 1e-6 * e2


def test_post_hoc_r5_identity_to_rounding_of_the_compliance(case) -> None:
    """POST HOC: the identity holds to 1e-11 of the compliance itself."""
    for s, gap, e2 in compliance_rows(case[3], case[4]):
        assert abs(gap - e2) <= 1e-11 * s


def other_load_error(case) -> tuple[float, rom.ReducedSolution, float]:
    """(relative energy error, reduced solution, absolute error) of the vertical-load basis under the axial load."""
    physics, _, F, system, model = case
    other = F[physics.free, 1]
    t = np.array([1.0, 1.0, 1.0, 1.0])
    sol = model.solve(t, load=other)
    e, nu = energy_error(system, t, sol.u, load=other)
    return e / nu, sol, e


@pytest.mark.xfail(strict=True, reason="pre-registered R6 error level not reached: relative error 0.293 < 0.5")
def test_r6_a_basis_for_one_load_fails_on_another() -> None:
    """Pre-registered R6, first run (26 Nov 2026), not met: the relative energy error of the other load is 0.293, not
    >= 0.5. The failure is still large (29 % against 0.3 % worst case on the trained load) and the bound reports it
    (next test)."""
    physics, regions, F, system = problem()
    model = rom.PODModel(system, thetas(40, 1), RANK)
    rel, _, _ = other_load_error((physics, regions, F, system, model))
    assert rel >= 0.5, rel


def test_r6_the_bound_flags_the_untrained_load(case) -> None:
    rel, sol, e = other_load_error(case)
    assert sol.error_bound >= e * (1 - 1e-6)
    assert sol.relative_error_bound >= rel
    assert rel >= 0.2 and sol.relative_error_bound >= 0.2      # POST HOC level (first measured 0.293 and 0.308)


def test_r6_limits_are_enforced(case) -> None:
    system, model = case[3], case[4]
    assert model.solve([3.0, 1.0, 1.0, 1.0]).extrapolation
    assert not model.solve([1.0, 1.0, 1.0, 1.0]).extrapolation
    with pytest.raises(ValueError):
        model.solve([0.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        model.solve([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        model.solve([1.0, 1.0])
    with pytest.raises(ValueError):                                            # more modes than snapshots carry
        rom.PODModel(system, thetas(5, 3), rank=6)
    with pytest.raises(ValueError):
        rom.pod_basis(np.zeros((10, 3)))
    physics, regions, F, _ = problem()
    with pytest.raises(ValueError):
        rom.AffineSystem.from_elasticity(physics, regions + 1, F[:, 0])
