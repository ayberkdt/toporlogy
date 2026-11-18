"""PR-01: the linear-solve residual is checked per load case (right-hand-side column), zero loads included.

Before: ``checked_residual`` used one Frobenius norm over all columns, ‖KX − B‖_F / ‖B‖_F, so a wrong solution of a
small load case hid behind a large one, and it returned 0 for B = 0 without looking at X. The matrices here are real
reduced stiffness matrices (unit moduli) and the exact solutions come from an independent sparse LU (scipy splu).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy import Domain, Elasticity
from toporlogy.fem.elastic import RESIDUAL_TOL, checked_residual
from toporlogy.fem.mechanisms import SolveResidualError
from toporlogy.solvers import DirectSolver


def _system(shape=(6, 3)):
    d = Domain(shape)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    phys = Elasticity(d, fixed, solver=DirectSolver("splu"))
    A = phys.matrix(np.ones(phys.n_elements))
    K = (A + A.T - sp.diags(A.diagonal())).tocsc()
    return d, phys, A, K


def _loads(n, rng, scales):
    B = rng.standard_normal((n, len(scales)))
    return B / np.linalg.norm(B, axis=0) * np.asarray(scales, float)


def test_small_load_case_cannot_hide_behind_a_large_one():
    """‖b₁‖ = 1e8 solved exactly, ‖b₂‖ = 1 solved 1 % wrong: the combined norm is ~1e-10 (passes), the second column's
    own relative residual is ~1e-2 (must fail, and the error names that column)."""
    _, _, A, K = _system()
    B = _loads(K.shape[0], np.random.default_rng(1), [1e8, 1.0])
    X = spla.splu(K).solve(B)
    X[:, 1] *= 1.01
    combined = np.linalg.norm(K @ X - B) / np.linalg.norm(B)
    assert combined < RESIDUAL_TOL                       # the old metric would accept this
    with pytest.raises(SolveResidualError, match=r"load case 1: .*1\.\d+e-02"):
        checked_residual(A, X, B)


def test_zero_load_with_a_nonzero_answer_is_refused():
    """b = 0 has the unique answer x = 0 (K is nonsingular); the old check returned 0 without looking at x."""
    _, _, A, K = _system()
    x = np.random.default_rng(2).standard_normal((K.shape[0], 1))
    with pytest.raises(SolveResidualError, match="load case 0: zero load"):
        checked_residual(A, x, np.zeros_like(x))


def test_zero_load_with_zero_answer_and_mixed_columns_pass():
    _, _, A, K = _system()
    B = _loads(K.shape[0], np.random.default_rng(3), [1.0, 1.0])
    B[:, 1] = 0.0
    X = spla.splu(K).solve(B)
    rep = checked_residual(A, X, B)
    assert rep is not None and rep.worst_rhs == 0
    assert np.isnan(rep.relative_per_rhs[1]) and rep.absolute_per_rhs[1] == 0.0
    assert rep.max_relative == rep.relative_per_rhs[0] < 1e-12
    single = checked_residual(A, np.zeros((K.shape[0], 1)), np.zeros((K.shape[0], 1)))
    assert single is not None and single.max_relative == 0.0


def test_relative_and_backward_errors_are_the_column_values():
    """Each reported number is recomputed here from its definition: r_j = ‖K x_j − b_j‖ / ‖b_j‖ and
    η_j = ‖K x_j − b_j‖ / (‖K‖₁ ‖x_j‖ + ‖b_j‖) (‖K‖₁ = max column sum, an upper bound of ‖K‖₂ for symmetric K)."""
    _, _, A, K = _system()
    B = _loads(K.shape[0], np.random.default_rng(4), [3.0, 0.5, 7.0])
    X = spla.splu(K).solve(B)
    X += 1e-9 * np.random.default_rng(5).standard_normal(X.shape)   # a measurable, acceptable error
    rep = checked_residual(A, X, B)
    R = K @ X - B
    want_rel = np.linalg.norm(R, axis=0) / np.linalg.norm(B, axis=0)
    k1 = float(abs(K).sum(axis=0).max())
    want_eta = np.linalg.norm(R, axis=0) / (k1 * np.linalg.norm(X, axis=0) + np.linalg.norm(B, axis=0))
    assert np.allclose(rep.relative_per_rhs, want_rel, rtol=1e-10)
    assert np.allclose(rep.absolute_per_rhs, np.linalg.norm(R, axis=0), rtol=1e-10)
    assert np.allclose(rep.backward_error_per_rhs, want_eta, rtol=1e-10)
    assert rep.worst_rhs == int(np.argmax(want_rel)) and rep.max_relative == pytest.approx(want_rel.max(), rel=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_solutions_fail_closed(bad):
    _, _, A, K = _system()
    B = _loads(K.shape[0], np.random.default_rng(6), [1.0, 1.0])
    X = spla.splu(K).solve(B)
    X[3, 1] = bad
    with pytest.raises(SolveResidualError, match="load case 1: the solution is not finite"):
        checked_residual(A, X, B)


class _CorruptColumn(DirectSolver):
    """A direct solver that returns the second column 1 % too large (the first one exact)."""

    def solve(self, b):
        x = np.asarray(super().solve(b)).reshape(b.shape).copy()
        if x.ndim == 2 and x.shape[1] > 1:
            x[:, 1] *= 1.01
        return x


def _two_cases(d):
    F = d.load_vector(2)
    tip = d.nodes_where(lambda X: X[:, 0] > d.shape[0] - 1e-9)
    F[d.dofs(tip, 1), 0] = -1e8 / len(tip)
    F[d.dofs(tip, 0), 1] = 1.0 / len(tip)
    return F


def test_elasticity_reports_per_load_case_and_refuses_the_wrong_one():
    d, _, _, _ = _system()
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    ok = Elasticity(d, fixed, solver=DirectSolver("splu"))
    ok.solve_numpy(np.ones(ok.n_elements), _two_cases(d))
    rep = ok.last_residual_report
    assert rep is not None and rep.relative_per_rhs.shape == (2,) and ok.last_residual == rep.max_relative < 1e-12
    bad = Elasticity(d, fixed, solver=_CorruptColumn("splu"))
    with pytest.raises(SolveResidualError, match="load case 1"):
        bad.solve_numpy(np.ones(bad.n_elements), _two_cases(d))


def test_constrained_elasticity_checks_its_reduced_system_per_load_case():
    """The oblique-constraint physics solves Tᵀ K T y = Tᵀ b on the kept local DOFs; the check runs on that system."""
    from toporlogy import boundary as B
    d = Domain((6, 3))
    bc = B.BoundaryConditions(d, [B.Fixed(B.OnPlane([0, 0], 0)), B.Roller(B.OnPlane([6, 0], 0), [1.0, 1.0])],
                              [B.LoadCase("c", [])])
    good = bc.physics(solver_factory=lambda: DirectSolver("splu"))
    bad = bc.physics(solver_factory=lambda: _CorruptColumn("splu"))
    assert type(good).__name__ == "ConstrainedElasticity"
    good.solve_numpy(np.ones(good.n_elements), _two_cases(d))
    assert good.last_residual_report.relative_per_rhs.shape == (2,) and good.last_residual < 1e-12
    with pytest.raises(SolveResidualError, match="load case 1"):
        bad.solve_numpy(np.ones(bad.n_elements), _two_cases(d))


def test_multigrid_reports_the_same_per_load_case_residual():
    """ElasticityGMG measures each column's true relative residual inside PCG. It is exposed under the same names, and
    it equals ‖K x_j − b_j‖ / ‖b_j‖ recomputed here with the direct path's assembled K; a zero load column gives
    x_j = 0 and NaN (ratio undefined)."""
    from toporlogy.fem.elastic_gmg import ElasticityGMG
    d = Domain((8, 4, 4))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    F = d.load_vector(3)
    tip = d.nodes_where(lambda X: X[:, 0] > 8 - 1e-9)
    F[d.dofs(tip, 2), 0] = -1e6 / len(tip)
    F[d.dofs(tip, 1), 1] = 1.0 / len(tip)
    E = np.ones(d.n_active)
    g = ElasticityGMG(d, fixed, direct_below=0, device="cpu", tol=1e-9)
    U = g.solve_numpy(E, F)
    rep = g.last_residual_report
    ref = Elasticity(d, fixed, solver=DirectSolver("splu"))
    A = ref.matrix(E)
    K = A + A.T - sp.diags(A.diagonal())
    free = ref.free
    want = np.linalg.norm(K @ U[free] - F[free], axis=0)[:2] / np.linalg.norm(F[free], axis=0)[:2]
    assert rep is not None and np.allclose(rep.relative_per_rhs[:2], want, rtol=1e-3)
    assert np.all(rep.relative_per_rhs[:2] <= 1e-9) and np.isnan(rep.relative_per_rhs[2])
    assert np.abs(U[:, 2]).max() == 0.0 and g.last_residual == rep.max_relative
