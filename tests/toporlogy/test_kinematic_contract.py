"""PR-02: the assumptions of the support (kinematic) check are enforced, not only documented.

The check in ``fem/mechanisms.py`` is exact for an elastic part whose every active element has a positive, finite
modulus (then an element's stiffness has exactly the rigid motions as null space). Before this PR nothing enforced
that: a negative modulus gave an indefinite K that the direct solver still solved, so the result passed silently.
Now moduli are validated before every factorisation (``InvalidModuliError``), the support set of a physics object
is frozen (the cached check cannot go stale), and the three failure kinds carry distinct codes.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy import Domain, Elasticity
from toporlogy.fem.mechanisms import InvalidModuliError, MechanismError, SolveResidualError, zero_energy_modes
from toporlogy.solvers import DirectSolver


def _cantilever(shape=(8, 4)):
    d = Domain(shape)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    F = d.load_vector()
    F[d.dofs(d.nodes_where(lambda X: X[:, 0] > shape[0] - 1e-9), 1), 0] = -1.0
    return d, fixed, F


@pytest.mark.parametrize("case", ["nan", "inf", "zero", "negative", "shape"])
def test_invalid_moduli_are_refused_before_the_factorisation(case):
    """A negative modulus makes K indefinite; the direct solver still returns an answer with a small residual, so
    before this PR it passed silently. Every invalid case is now refused with its reason."""
    d, fixed, F = _cantilever()
    phys = Elasticity(d, fixed, solver=DirectSolver("splu"))
    E = np.ones(phys.n_elements)
    if case == "shape":
        E = np.ones(phys.n_elements + 1)
    else:
        E[5] = {"nan": np.nan, "inf": np.inf, "zero": 0.0, "negative": -1.0}[case]
    want = {"nan": "not finite", "inf": "not finite", "zero": "positive", "negative": "positive", "shape": "shape"}
    with pytest.raises(InvalidModuliError, match=want[case]):
        phys.solve_numpy(E, F)


def test_a_negative_modulus_used_to_pass_the_residual_check():
    """Evidence for the policy: with the moduli validation bypassed, a negative modulus is factorised and solved with
    a residual far below RESIDUAL_TOL — the residual check alone cannot catch a wrong material model."""
    d, fixed, F = _cantilever()
    phys = Elasticity(d, fixed, solver=DirectSolver("splu"))
    E = np.ones(phys.n_elements)
    E[5] = -1.0
    A = phys.matrix(E)
    K = (A + A.T - sp.diags(A.diagonal())).toarray()
    assert np.linalg.eigvalsh(K).min() < 0                      # indefinite
    x = spla.splu(sp.csc_matrix(K)).solve(F[phys.free, 0])
    assert np.linalg.norm(K @ x - F[phys.free, 0]) / np.linalg.norm(F[phys.free, 0]) < 1e-10


def test_tiny_positive_moduli_are_allowed_and_solved_correctly():
    """SIMP's void elements (E ~ 1e-9 E0) are legal: high contrast is ill-conditioning, not a mechanism. The solve is
    accepted (backward error at the rounding level), it agrees with an independent sparse LU to the accuracy the
    conditioning allows (measured 3e-8 relative), and its relative residual alone (> 1e-6 here) would have refused it
    under the PR-01 rule."""
    d, fixed, F = _cantilever()
    phys = Elasticity(d, fixed, solver=DirectSolver("splu"))
    E = np.where(np.random.default_rng(0).random(phys.n_elements) < 0.5, 1e-9, 1.0)
    U = phys.solve_numpy(E, F)[:, 0]
    A = phys.matrix(E)
    K = (A + A.T - sp.diags(A.diagonal())).tocsc()
    ref = spla.splu(K).solve(F[phys.free, 0])
    assert np.linalg.norm(U[phys.free] - ref) / np.linalg.norm(ref) < 1e-6
    rep = phys.last_residual_report
    assert rep.backward_error_per_rhs[0] < 1e-15 and rep.relative_per_rhs[0] > 1e-6


def test_the_support_check_runs_once_and_the_support_set_is_frozen(monkeypatch):
    """E changes do not need a new check (positive moduli keep the null space of every element); the supports of a
    physics object cannot change in place, so the cached result cannot go stale."""
    import toporlogy.fem.elastic as EL
    calls = []
    real = EL.check_supports
    monkeypatch.setattr(EL, "check_supports", lambda *a: (calls.append(1), real(*a))[1])
    d, fixed, F = _cantilever()
    phys = Elasticity(d, fixed, solver=DirectSolver("splu"))
    for s in (1.0, 0.3, 2.0):
        phys.solve_numpy(np.full(phys.n_elements, s), F)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="read-only"):
        phys.fixed[0] = 1
    with pytest.raises(ValueError, match="read-only"):
        phys.free[0] = 0


def test_failure_kinds_have_distinct_codes():
    codes = {MechanismError.code, SolveResidualError.code, InvalidModuliError.code}
    assert codes == {"mechanism", "solve_residual", "invalid_moduli"}


def test_modal_and_multigrid_validate_their_inputs_too():
    """Modal keeps its exemption from the support check (rigid modes are legitimate in free vibration) but not from
    the moduli and density validation; the multigrid path validates the moduli as the direct path does."""
    from toporlogy.fem.elastic_gmg import ElasticityGMG
    from toporlogy.fem.modal import Modal
    d = Domain((6, 3, 3))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    m = Modal(d, fixed)
    good = np.ones(d.n_active)
    bad = good.copy()
    bad[2] = -1.0
    with pytest.raises(InvalidModuliError, match="positive"):
        m.solve_numpy(bad, good, 2)
    with pytest.raises(InvalidModuliError, match="density: 1 value.s. negative"):
        m.solve_numpy(good, bad, 2)
    massless = good.copy()
    massless[2] = 0.0                                                   # two-range mass: void elements are massless
    lam, _ = m.solve_numpy(good, massless, 2)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0)
    free = Modal(Domain((4, 2, 2)), np.zeros(0, np.int64))             # free-free: six rigid modes are allowed
    lam, _ = free.solve_numpy(np.ones(16), np.ones(16), 7)
    assert np.sum(np.abs(lam) < 1e-6 * lam.max()) == 6
    g = ElasticityGMG(d, fixed, direct_below=0, device="cpu")
    with pytest.raises(InvalidModuliError, match="positive"):
        g.solve_numpy(bad, d.load_vector())


# ---------------------------------------------------------------------------------------------- supports
def _reference(phys) -> int:
    """Zero eigenvalues of the dense reduced matrix. Threshold 1e-12 λmax: a true zero mode is at the rounding level
    (~eps λmax ≈ 1e-11 here), while stiff springs (total up to 1e6) leave genuine soft modes near 1e-4 that a looser
    1e-9 λmax threshold miscounted as zero (seen in this sweep, seed 7, trial 17: λmin 7.8e-5, next 5.4e-2)."""
    A = phys.matrix(np.ones(phys.n_elements)).toarray()
    K = A + A.T - np.diag(np.diag(A))
    w = np.linalg.eigvalsh(K)
    return int((w <= 1e-12 * w.max()).sum())


def test_overlapping_and_random_oblique_supports_match_the_assembled_system():
    """Several supports on the same DOFs, and random oblique rollers and springs on random edges: the kinematic count
    equals the zero eigenvalues of the reduced matrix the solver really factorises (Tᵀ (K + S) T on kept DOFs)."""
    from toporlogy import boundary as B
    d = Domain((6, 3))
    edges = [B.OnPlane([0, 0], 0), B.OnPlane([6, 0], 0), B.OnPlane([0, 0], 1), B.OnPlane([0, 3], 1)]
    same = B.BoundaryConditions(d, [B.Fixed(edges[0]), B.Fixed(edges[0]), B.Roller(edges[0], [1.0, 0.0])],
                                [B.LoadCase("c", [])]).physics()
    assert zero_energy_modes(d, same.elements, same._kinematic_rows())[0] == _reference(same) == 0
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(40):
        sup = []
        for _ in range(int(rng.integers(1, 4))):
            where = edges[int(rng.integers(4))]
            n = rng.standard_normal(2)
            sup.append(B.Roller(where, n) if rng.random() < 0.7 else
                       B.ElasticSupport(where, direction=n, total=float(rng.uniform(0.1, 1e6))))
        phys = B.BoundaryConditions(d, sup, [B.LoadCase("c", [])]).physics()
        want = _reference(phys)
        assert zero_energy_modes(d, phys.elements, phys._kinematic_rows())[0] == want
        seen.add(min(want, 1))
    assert seen == {0, 1}


def test_a_failed_solve_is_never_a_result():
    """The support check passes but the solve is wrong: the optimisation stops with the error, it does not return a
    Result that could be mistaken for a solution."""
    import toporlogy as T

    class Wrong(DirectSolver):
        def solve(self, b):
            return 1.01 * np.asarray(super().solve(b))
    d, fixed, F = _cantilever()
    st = T.Study(d, T.Elasticity(d, fixed, solver=Wrong("splu")), F, T.standard_chain(d, 1.5), T.SIMP(d),
                 T.Compliance(), [T.Constraint(T.Volume(), 0.5)], T.OC())
    with pytest.raises(SolveResidualError, match="load case 0"):
        st.run(max_iter=2)
