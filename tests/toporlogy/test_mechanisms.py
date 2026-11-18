"""B-22: a singular stiffness (floating piece, hinge, free rotation) must not be solved silently.

Before: ``Elasticity.solve_numpy`` on a part with an unsupported piece returned finite displacements of order 1e15.
Now the supports are checked kinematically before the first factorisation (``toporlogy.fem.mechanisms``) and every
solve checks its scaled residual. The reference for the number of zero-energy motions is independent of that model:
the number of (numerically) zero eigenvalues of the dense reduced stiffness with unit moduli.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy import Domain, Elasticity
from toporlogy.fem.mechanisms import MechanismError, SolveResidualError, zero_energy_modes
from toporlogy.solvers import DirectSolver


def _zero_eigs(A_upper) -> int:
    """Zero eigenvalues of the full symmetric matrix behind an upper triangle (dense reference)."""
    A = A_upper.toarray()
    K = A + A.T - np.diag(np.diag(A))
    w = np.linalg.eigvalsh(K)
    return int((w <= 1e-9 * w.max()).sum())


def _reference(phys) -> int:
    return _zero_eigs(phys.matrix(np.ones(phys.n_elements)))


def _model(phys) -> int:
    return zero_energy_modes(phys.domain, phys.elements, phys._kinematic_rows())[0]


def _clamped_left(d):
    return d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))


# ---------------------------------------------------------------------------------------------- worked cases
def test_floating_piece_is_refused_instead_of_solved():
    """Two blocks separated by a void column, only the left one clamped: the right one floats (3 motions in 2-D).
    The old behaviour, measured here with the check bypassed, is a finite 'solution' of enormous size."""
    frac = np.ones((9, 4))
    frac[4] = 0.0
    d = Domain((9, 4), frac=frac)
    F = d.load_vector()
    F[d.dofs(d.nodes_where(lambda X: X[:, 0] > 9 - 1e-9), 1), 0] = -1.0
    phys = Elasticity(d, _clamped_left(d), solver=DirectSolver("splu"))
    assert _model(phys) == _reference(phys) == 3
    with pytest.raises(MechanismError, match=r"3 zero-energy motion.*16 element\(s\) around \(7, 2\)"):
        phys.solve_numpy(np.ones(phys.n_elements), F)
    phys._kinematics_checked = True                     # what used to happen: no error, but no usable answer
    try:
        U = phys.solve_numpy(np.ones(phys.n_elements), F)
        assert np.all(np.isfinite(U)) and np.abs(U).max() > 1e6      # "finite" is not "correct"
    except (SolveResidualError, RuntimeError):
        pass                                            # the residual guard or the factorisation caught it


@pytest.mark.parametrize("case,want", [("pin", 1), ("hinge2d", 1), ("supported", 0)])
def test_two_dimensional_joints(case, want):
    if case == "hinge2d":                               # two blocks touching at one node: rotation about it
        frac = np.zeros((6, 6))
        frac[:3, :3] = 1.0
        frac[3:, 3:] = 1.0
        d = Domain((6, 6), frac=frac)
        fixed = _clamped_left(d)
    else:
        d = Domain((6, 3))
        if case == "pin":                               # one node held in x and y: the part can rotate about it
            fixed = d.dofs(d.nodes_where(lambda X: (X[:, 0] < 1e-9) & (X[:, 1] < 1e-9)))
        else:                                           # pin + roller: statically determinate
            fixed = np.r_[d.dofs(d.nodes_where(lambda X: (X[:, 0] < 1e-9) & (X[:, 1] < 1e-9))),
                          d.dofs(d.nodes_where(lambda X: (X[:, 0] > 6 - 1e-9) & (X[:, 1] < 1e-9)), 1)]
    phys = Elasticity(d, fixed)
    assert _model(phys) == _reference(phys) == want


@pytest.mark.parametrize("case,want", [("floating", 6), ("ball", 3), ("hinge3d", 1), ("clamped", 0)])
def test_three_dimensional_joints(case, want):
    """Cubes touching at a node (ball joint: 3 rotations) or along an edge (hinge: 1), a floating cube (6)."""
    frac = np.zeros((4, 4, 4))
    frac[:2, :2, :2] = 1.0
    if case == "floating":
        frac[3:, 3:, 3:] = 1.0
    elif case == "ball":
        frac[2:, 2:, 2:] = 1.0
    elif case == "hinge3d":
        frac[2:, 2:, :2] = 1.0
    d = Domain((4, 4, 4), frac=frac)
    phys = Elasticity(d, _clamped_left(d))
    assert _model(phys) == _reference(phys) == want


def test_random_parts_and_supports_match_the_dense_reference():
    """Random void patterns (face, edge and node contacts, islands) and random fixed DOFs, 2-D and 3-D: the
    kinematic count equals the zero-eigenvalue count of the assembled reduced stiffness in every case."""
    rng = np.random.default_rng(22)
    seen = set()
    for trial in range(160):
        dim = 2 if trial % 2 else 3
        shape = (5, 4) if dim == 2 else (3, 3, 3)
        frac = (rng.random(shape) < rng.uniform(0.45, 0.9)).astype(float)
        if frac.sum() < 2:
            continue
        d = Domain(shape, frac=frac)
        act = d.active_dofs
        fixed = rng.choice(act, size=int(rng.integers(0, min(len(act) - 1, 3 * dim + 4))), replace=False)
        phys = Elasticity(d, fixed)
        want = _reference(phys)
        assert _model(phys) == want, (trial, dim)
        seen.add(min(want, 1))
    assert seen == {0, 1}                               # both outcomes occurred


# ---------------------------------------------------------------------------------------------- other physics
def test_oblique_constraints_and_springs_count_as_supports():
    """ConstrainedElasticity: a roller with an oblique normal and a spring remove motions exactly as the dense
    reduced matrix (Tᵀ (K + S) T without the constrained local DOFs) says."""
    from toporlogy import boundary as B
    d = Domain((6, 3))
    left = B.OnPlane([0, 0], 0)
    # a whole edge on an oblique roller: slides along (1, -1); it cannot turn (at x = 0 a rotation moves the
    # nodes by w(-y, 0), whose (1, 1) component is not zero)
    for supports, want in (([B.Roller(left, [1.0, 1.0])], 1),
                           ([B.Roller(left, [1.0, 1.0]), B.Roller(B.OnPlane([6, 0], 0), [1.0, 0.0])], 0),
                           ([B.ElasticSupport(B.OnPlane([6, 0], 0), 0.1)], 1),    # normal springs: slides
                           ([B.ElasticSupport(B.OnPlane([6, 0], 0), 0.1, direction="all")], 0)):
        bc = B.BoundaryConditions(d, supports, [B.LoadCase("c", [])])
        phys = bc.physics()
        assert _model(phys) == _reference(phys) == want, supports


def test_residual_guard_rejects_a_wrong_solution():
    class Wrong(DirectSolver):
        def solve(self, b):
            return 1.001 * super().solve(b)            # 0.1 % off
    d = Domain((6, 3))
    F = d.load_vector()
    F[d.dofs(d.nodes_where(lambda X: X[:, 0] > 6 - 1e-9), 1), 0] = -1.0
    ok = Elasticity(d, _clamped_left(d), solver=DirectSolver("splu"))
    ok.solve_numpy(np.ones(ok.n_elements), F)
    assert ok.last_residual is not None and ok.last_residual < 1e-12
    bad = Elasticity(d, _clamped_left(d), solver=Wrong("splu"))
    with pytest.raises(SolveResidualError, match="residual"):
        bad.solve_numpy(np.ones(bad.n_elements), F)


def test_multigrid_path_checks_the_supports_too():
    from toporlogy.fem.elastic_gmg import ElasticityGMG
    frac = np.ones((8, 4, 4))
    frac[4] = 0.0
    d = Domain((8, 4, 4), frac=frac)
    phys = ElasticityGMG(d, _clamped_left(d))
    with pytest.raises(MechanismError, match="6 zero-energy"):
        phys.solve_numpy(np.ones(phys.n_elements), d.load_vector())


def test_count_does_not_depend_on_the_scale_of_a_support():
    """A stiff spring (k = 1e10, a penalty support) next to unit fixed-DOF rows: the count is that of k = 1, which
    the dense reference gives (found by test_bc.test_oblique_constraint_matches_penalty, which the first version of
    the check refused)."""
    from toporlogy import boundary as B
    d = Domain((6, 3))
    counts = []
    for k in (1.0, 1e10):
        bc = B.BoundaryConditions(d, [B.Roller(B.OnPlane([0, 0], 0), 0),
                                      B.ElasticSupport(B.OnPlane([0, 0], 1), direction=[0.0, 1.0], total=k)],
                                  [B.LoadCase("c", [])])
        phys = bc.physics()
        counts.append(_model(phys))
        if k == 1.0:
            counts.append(_reference(phys))
    assert counts == [0, 0, 0]


# ---------------------------------------------------------------------------------------------- PR-03: scaling
def _islands(k: int, supported: bool):
    """k separate 2x2x2 cubes on a grid (gap 1), optionally clamped at their base; expected motions 0 or 6 k."""
    import math
    n = math.ceil(k ** (1 / 3))
    frac = np.zeros((3 * n, 3 * n, 3 * n))
    bases = [(3 * i, 3 * j, 3 * m) for i in range(n) for j in range(n) for m in range(n)][:k]
    for x, y, z in bases:
        frac[x:x + 2, y:y + 2, z:z + 2] = 1.0
    d = Domain(frac.shape, frac=frac)
    X = d.node_coords()
    sel = np.zeros(len(X), bool)
    if supported:
        for x, y, z in bases:
            sel |= ((np.abs(X[:, 2] - z) < 1e-9) & (X[:, 0] > x - 1e-9) & (X[:, 0] < x + 2 + 1e-9)
                    & (X[:, 1] > y - 1e-9) & (X[:, 1] < y + 2 + 1e-9))
    return d, d.dofs(np.flatnonzero(sel))


def _chain(k: int):
    """k unit cubes, each touching the next at one node (ball joint), the first clamped: 3 (k - 1) motions, all in
    one contact component."""
    frac = np.zeros((k, 2, 2))
    for i in range(k):
        frac[i, i % 2, i % 2] = 1.0
    d = Domain(frac.shape, frac=frac)
    X = d.node_coords()
    return d, d.dofs(np.flatnonzero((X[:, 0] < 1e-9) & (X[:, 1] < 1 + 1e-9) & (X[:, 2] < 1 + 1e-9)))


def test_disjoint_islands_cost_grows_linearly_not_cubically():
    """1000 separately supported islands: one tiny rank problem per contact component. Before PR-03 the whole
    6000 x 6000 normal matrix was decomposed: measured 28.9 s and 574 MB peak; the bound here is 3 s."""
    import time

    from toporlogy.fem.mechanisms import fixed_dof_rows
    d, fixed = _islands(1000, True)
    t = time.perf_counter()
    n, _ = zero_energy_modes(d, d.active_elements, fixed_dof_rows(fixed, 3))
    assert n == 0 and time.perf_counter() - t < 3.0
    d, fixed = _islands(300, False)
    assert zero_energy_modes(d, d.active_elements, {})[0] == 1800


def test_one_large_contact_component_is_indeterminate_not_silent():
    """A single contact component above the size limit is not decided by a long dense computation or by guessing:
    MechanismCheckIndeterminate (code mechanism_indeterminate) says so; with a larger limit it is decided exactly."""
    from toporlogy.fem.mechanisms import MechanismCheckIndeterminate, fixed_dof_rows
    d, fixed = _chain(60)
    rows = fixed_dof_rows(fixed, 3)
    with pytest.raises(MechanismCheckIndeterminate, match="60 clusters.*360 unknowns.*limit 300"):
        zero_energy_modes(d, d.active_elements, rows, max_unknowns=300)
    assert MechanismCheckIndeterminate.code == "mechanism_indeterminate"
    assert zero_energy_modes(d, d.active_elements, rows, max_unknowns=360)[0] == 3 * 59


def test_counts_are_exact_for_islands_and_joint_chains():
    from toporlogy.fem.mechanisms import fixed_dof_rows
    for k in (1, 7, 40):
        d, fixed = _chain(k)
        assert zero_energy_modes(d, d.active_elements, fixed_dof_rows(fixed, 3))[0] == 3 * (k - 1)
        d, fixed = _islands(k, True)
        assert zero_energy_modes(d, d.active_elements, fixed_dof_rows(fixed, 3))[0] == 0
