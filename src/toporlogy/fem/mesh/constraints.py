"""General linear constraints ``C u = d`` on the general mesh: the transform ``u = T z + u0``.

A set of linear equations ``C u = d`` (rows: prescribed values, multi-point constraints, periodicity) is brought
to the transform ``u = T z + u0`` by sparse Gauss-Jordan elimination (:meth:`LinearConstraints.eliminate`): every
row, after substituting the slaves found so far, is solved for its largest remaining coefficient (that DOF becomes a
slave, the others are free "masters"), and the new slave is substituted into the earlier ones, so every slave is a
combination of masters only. The reduced problem is

``K_r = Tᵀ K T``,  ``f_r = Tᵀ (f − K u0)``,  ``K_r z = f_r``,  ``u = T z + u0``.

Dirichlet rows only (``T`` a column selection) take the exact path of :meth:`ElasticProblem.solve`: ``K_r`` is the
sub-matrix ``K[free][:, free]`` and ``f_r = f_free − K_fc u_c``, so ``u`` and the reactions are bitwise those of the
elimination path. DOFs that no cell touches are not unknowns (a constraint on one is refused).

Multiplier and reaction. ``r = K u − f`` is the residual of the ORIGINAL system, the force the constraints exert on the
body. At the solution ``Tᵀ r = 0``, so ``r = Cᵀ λ`` with a unique ``λ`` (the independent rows are kept):
stationarity of ``½ uᵀ K u − fᵀ u − λᵀ (C u − d)``. For a Dirichlet row ``λ = r_i``. The saddle-point (Lagrange)
system ``[K Cᵀ; C 0][u; μ] = [f; d]`` has ``μ = −λ``.

Detection. A row that is empty after the substitution is either inconsistent
(:class:`ContradictoryConstraints`, the right-hand side ``|d_i − …|`` exceeds ``rtol`` times the magnitudes added)
or dependent on earlier rows (:class:`RedundantConstraints`: duplicates, cycles ``a = b, b = c, c = a``; rank
deficiency of ``C``). ``redundant="drop"`` accepts consistent dependent rows (periodic cells list their corner nodes
several times) and records them; contradictions are always errors. The messages name the rows and the rows they
depend on. A constrained system whose rigid motions are not all held by the rows raises
:class:`~toporlogy.fem.mechanisms.MechanismError` (periodicity without a pin leaves the translations).

Constraint types built on it: :meth:`LinearConstraints.fix`, :meth:`tie` (``u_a − u_b = d``), :meth:`mpc` (one row of
any length), :func:`periodic` / :func:`periodic_lattice` (matched node sets, any offset vector including skew ones,
prescribed jump ``H a`` for a macro displacement gradient ``H``). Alternatives to elimination, with the condition
number (1-norm estimate) of the system they solve: :func:`solve_lagrange` (saddle point) and :func:`solve_penalty`
(``K + α s CᵀC``, error O(1/α), conditioning grows with α).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from toporlogy.fem.mechanisms import MechanismError
from toporlogy.fem.mesh.linear import Factorization, condition_1norm
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import Dirichlet, rigid_modes

__all__ = ["ConstraintError", "ContradictoryConstraints", "RedundantConstraints", "LinearConstraints", "Reduction",
           "ConstrainedSolution", "AlternativeSolution", "match_nodes", "periodic", "periodic_lattice", "macro_jump",
           "solve_constrained", "solve_lagrange", "solve_penalty"]


class ConstraintError(ValueError):
    """A set of linear constraints that cannot be used (base of the two detection errors)."""

    code = "constraint"


class ContradictoryConstraints(ConstraintError):
    """The right-hand sides of dependent rows disagree: no ``u`` satisfies ``C u = d``."""

    code = "constraint_contradictory"


class RedundantConstraints(ConstraintError):
    """Rows that depend on earlier rows (duplicates, cycles): ``C`` is rank deficient."""

    code = "constraint_redundant"


def _as_rhs(d: Any, m: int) -> np.ndarray:
    """Right-hand side (m, k): scalar, (m,) or (m, k)."""
    a = np.asarray(d, float)
    if a.ndim == 0:
        return np.full((m, 1), float(a))
    if a.ndim == 1:
        if len(a) != m:
            raise ValueError(f"{m} constraint row(s) but {len(a)} right-hand-side value(s)")
        return a[:, None].copy()
    if a.ndim == 2 and a.shape[0] == m:
        return a.copy()
    raise ValueError(f"right-hand side must be a scalar, ({m},) or ({m}, k), got {a.shape}")


class LinearConstraints:
    """Rows ``C u = d`` (C sparse (m, n), d (m, k): k right-hand-side columns, one per load case)."""

    def __init__(self, C: sp.spmatrix, d: Any = 0.0, labels: list[str] | None = None) -> None:
        self.C = sp.csr_matrix(C, dtype=float)
        self.C.sum_duplicates()
        self.d = _as_rhs(d, self.C.shape[0])
        self.labels = list(labels) if labels is not None else [f"row {i}" for i in range(self.C.shape[0])]
        if len(self.labels) != self.C.shape[0]:
            raise ValueError("one label per row")

    @property
    def n(self) -> int:
        return self.C.shape[1]

    @property
    def m(self) -> int:
        return self.C.shape[0]

    @property
    def n_cases(self) -> int:
        return self.d.shape[1]

    # ------------------------------------------------------------------ builders
    @classmethod
    def empty(cls, n: int) -> LinearConstraints:
        return cls(sp.csr_matrix((0, int(n))), np.zeros((0, 1)))

    @classmethod
    def mpc(cls, n: int, dofs: Any, coefs: Any, rhs: Any = 0.0, label: str | None = None) -> LinearConstraints:
        """One row ``Σ coefs[k] u[dofs[k]] = rhs`` (a multi-point constraint)."""
        dofs = np.asarray(dofs, np.int64).ravel()
        coefs = np.asarray(coefs, float).ravel()
        if len(dofs) != len(coefs) or not len(dofs):
            raise ValueError("an MPC row needs as many coefficients as DOFs (at least one)")
        _check_range(dofs, n)
        C = sp.csr_matrix((coefs, (np.zeros(len(dofs), np.int64), dofs)), shape=(1, int(n)))
        return cls(C, _as_rhs(rhs, 1), [label or "mpc"])

    @classmethod
    def tie(cls, n: int, a: Any, b: Any, d: Any = 0.0) -> LinearConstraints:
        """Rows ``u_a − u_b = d`` for the DOF arrays ``a``, ``b`` (equal length); d scalar, (m,) or (m, k)."""
        a = np.asarray(a, np.int64).ravel()
        b = np.asarray(b, np.int64).ravel()
        if len(a) != len(b):
            raise ValueError("tie: a and b must have the same length")
        _check_range(a, n)
        _check_range(b, n)
        m = len(a)
        rows = np.repeat(np.arange(m), 2)
        C = sp.csr_matrix((np.tile([1.0, -1.0], m), (rows, np.column_stack([a, b]).ravel())), shape=(m, int(n)))
        return cls(C, _as_rhs(d, m), [f"tie u[{x}] - u[{y}]" for x, y in zip(a.tolist(), b.tolist(), strict=True)])

    @classmethod
    def fix(cls, n: int, dofs: Any, values: Any = 0.0) -> LinearConstraints:
        """Rows ``u_i = value`` (prescribed DOFs; a DOF listed twice gives two rows, which the elimination checks)."""
        dofs = np.asarray(dofs, np.int64).ravel()
        _check_range(dofs, n)
        m = len(dofs)
        C = sp.csr_matrix((np.ones(m), (np.arange(m), dofs)), shape=(m, int(n)))
        return cls(C, _as_rhs(values, m), [f"fix u[{i}]" for i in dofs.tolist()])

    @classmethod
    def from_dirichlet(cls, bc: Dirichlet, n: int) -> LinearConstraints:
        """The rows of a :class:`Dirichlet` (values (n_c, n_cases))."""
        out = cls.fix(n, bc.dofs, bc.values)
        return out

    def __add__(self, other: LinearConstraints) -> LinearConstraints:
        if self.n != other.n:
            raise ValueError(f"constraint sets on {self.n} and {other.n} DOFs cannot be combined")
        k1, k2 = self.n_cases, other.n_cases
        if k1 != k2 and 1 not in (k1, k2):
            raise ValueError(f"right-hand sides with {k1} and {k2} columns cannot be combined")
        k = max(k1, k2)
        d = np.vstack([np.broadcast_to(self.d, (self.m, k)), np.broadcast_to(other.d, (other.m, k))])
        return LinearConstraints(sp.vstack([self.C, other.C]).tocsr(), d, self.labels + other.labels)

    # ------------------------------------------------------------------ elimination
    def eliminate(self, universe: Any = None, rtol: float = 1e-10, redundant: str = "error") -> Reduction:
        """Sparse Gauss-Jordan elimination to ``u = T z + u0`` (see the module docstring).

        ``universe``: the DOFs that are unknowns (default all); a constraint on any other DOF is a ``ConstraintError``.
        ``redundant``: ``"error"`` (default) or ``"drop"`` for consistent dependent rows."""
        if redundant not in ("error", "drop"):
            raise ValueError("redundant: 'error' | 'drop'")
        n, k = self.n, self.n_cases
        uni = np.arange(n) if universe is None else np.unique(np.asarray(universe, np.int64))
        in_uni = np.zeros(n, bool)
        in_uni[uni] = True
        C = self.C
        terms: dict[int, dict[int, float]] = {}
        const: dict[int, np.ndarray] = {}
        deps: dict[int, set[int]] = {}
        users: dict[int, set[int]] = {}
        pivot_rows: list[int] = []
        dropped: list[int] = []
        for i in range(self.m):
            lo, hi = C.indptr[i], C.indptr[i + 1]
            row = {int(j): float(v) for j, v in zip(C.indices[lo:hi], C.data[lo:hi], strict=True) if v != 0.0}
            outside = [j for j in row if not in_uni[j]]
            if outside:
                raise ConstraintError(f"{self.labels[i]} (row {i}) constrains DOF(s) {outside[:5]} that belong to no "
                                      "cell (not unknowns of the problem)")
            rhs = self.d[i].copy()
            mag = np.abs(rhs).copy()
            cmag = max((abs(v) for v in row.values()), default=0.0)
            used = {i}
            for dof in [j for j in row if j in terms]:
                a = row.pop(dof)
                for kk, v in terms[dof].items():
                    row[kk] = row.get(kk, 0.0) + a * v
                    cmag = max(cmag, abs(a * v))
                rhs = rhs - a * const[dof]
                mag = mag + np.abs(a) * np.abs(const[dof])
                used |= deps[dof]
            row = {j: v for j, v in row.items() if abs(v) > rtol * cmag}
            if not row:
                others = sorted(used - {i})
                tol = rtol * np.maximum(mag, np.finfo(float).tiny)
                if (np.abs(rhs) > tol).any():
                    raise ContradictoryConstraints(
                        f"{self.labels[i]} (row {i}) contradicts "
                        f"{[self.labels[j] + f' (row {j})' for j in others] or 'itself'}: after substitution it "
                        f"reads 0 = {rhs.tolist()}")
                if redundant == "error":
                    raise RedundantConstraints(
                        f"{self.labels[i]} (row {i}) is a linear combination of "
                        f"{[self.labels[j] + f' (row {j})' for j in others] or 'no other row (an empty row)'}: the "
                        "constraint matrix is rank deficient (duplicate or cyclic constraints); redundant='drop' "
                        "accepts consistent ones")
                dropped.append(i)
                continue
            p = max(row, key=lambda j: (abs(row[j]), j))
            cp = row.pop(p)
            new_terms = {j: -v / cp for j, v in row.items()}
            new_const = rhs / cp
            for s in list(users.get(p, ())):
                a = terms[s].pop(p)
                for kk, v in new_terms.items():
                    terms[s][kk] = terms[s].get(kk, 0.0) + a * v
                    users.setdefault(kk, set()).add(s)
                const[s] = const[s] + a * new_const
                deps[s] |= used
            users.pop(p, None)
            terms[p] = new_terms
            const[p] = new_const
            deps[p] = used
            for j in new_terms:
                users.setdefault(j, set()).add(p)
            pivot_rows.append(i)
        slaves = np.array(sorted(terms), np.int64)
        masters = np.setdiff1d(uni, slaves)
        col = np.full(n, -1, np.int64)
        col[masters] = np.arange(len(masters))
        ri, ci, vv = [masters], [np.arange(len(masters))], [np.ones(len(masters))]
        u0 = np.zeros((n, k))
        pure = True
        for s in slaves.tolist():
            t = terms[s]
            if t:
                pure = False
                ri.append(np.full(len(t), s))
                ci.append(col[np.array(list(t), np.int64)])
                vv.append(np.array(list(t.values())))
            u0[s] = const[s]
        T = sp.csr_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))), shape=(n, len(masters)))
        T.sort_indices()
        return Reduction(n, uni, masters, slaves, T, u0, pure, self.C[pivot_rows], self.d[pivot_rows],
                         [self.labels[i] for i in pivot_rows], pivot_rows, dropped)


def _check_range(dofs: np.ndarray, n: int) -> None:
    if len(dofs) and (dofs.min() < 0 or dofs.max() >= n):
        raise ValueError(f"constrained DOF out of range 0 … {n - 1}")


@dataclass
class Reduction:
    """``u = T z + u0`` of an elimination: ``masters`` (the z DOFs, ascending), ``slaves``, ``T`` (n, n_masters) CSR,
    ``u0`` (n, k), the independent rows kept (``C_used``, ``d_used``, ``rows``) and the consistent rows dropped."""

    n: int
    universe: np.ndarray
    masters: np.ndarray
    slaves: np.ndarray
    T: sp.csr_matrix
    u0: np.ndarray
    pure_selection: bool
    C_used: sp.csr_matrix
    d_used: np.ndarray
    labels: list[str]
    rows: list[int]
    dropped: list[int] = field(default_factory=list)

    def expand(self, z: np.ndarray) -> np.ndarray:
        """``u = T z + u0`` for z (n_masters,) or (n_masters, k)."""
        z = np.asarray(z, float)
        Z = z[:, None] if z.ndim == 1 else z
        U = np.asarray(self.T @ Z) + self.u0
        return U[:, 0] if z.ndim == 1 and U.shape[1] == 1 else U

    def reduce_matrix(self, K: sp.spmatrix, general: bool = False) -> sp.csr_matrix:
        """``K_r = Tᵀ K T``; for Dirichlet rows only the exact sub-matrix ``K[m][:, m]`` (``general=True`` forces the
        triple product)."""
        K = sp.csr_matrix(K)
        if self.pure_selection and not general:
            return sp.csr_matrix(K[self.masters][:, self.masters])
        Kr = sp.csr_matrix(self.T.T @ K @ self.T)
        Kr.sort_indices()
        return Kr

    def reduce_load(self, K: sp.spmatrix, F: np.ndarray, general: bool = False) -> np.ndarray:
        """``f_r = Tᵀ (f − K u0)`` for f (n,) or (n, k); (n_masters, k)."""
        K = sp.csr_matrix(K)
        F = np.asarray(F, float)
        F = F[:, None] if F.ndim == 1 else F
        k = F.shape[1]
        if self.u0.shape[1] not in (1, k):
            raise ValueError(f"the constraints carry {self.u0.shape[1]} right-hand-side column(s), the load {k}")
        U0 = np.broadcast_to(self.u0, (self.n, k))
        if self.pure_selection and not general:
            return F[self.masters] - K[self.masters][:, self.slaves] @ U0[self.slaves]
        return np.asarray(self.T.T @ (F - K @ U0))

    def multipliers(self, r: np.ndarray, rtol: float = 1e-8) -> tuple[np.ndarray, float]:
        """``λ`` (m_used, k) with ``r = C_usedᵀ λ`` for the residual r = K u − f (n, k), and the relative residual
        ``‖C_usedᵀ λ − r‖ / ‖r‖`` (small when ``Tᵀ r = 0``, i.e. at a solution)."""
        R = np.asarray(r, float)
        R = R[:, None] if R.ndim == 1 else R
        Cu = self.C_used
        if Cu.shape[0] == 0:
            return np.zeros((0, R.shape[1])), float(np.linalg.norm(R) > 0)
        G = sp.csc_matrix(Cu @ Cu.T)
        lam = spla.splu(G).solve(np.asarray(Cu @ R))
        nr = np.linalg.norm(R)
        res = float(np.linalg.norm(Cu.T @ lam - R) / (nr if nr > 0 else 1.0))
        return lam, res


@dataclass
class ConstrainedSolution:
    """``u`` (ndof, k), ``reactions`` = ``r = K u − f`` over all DOFs (nonzero at the constrained ones), the
    multipliers ``λ`` of the independent rows (``r = Cᵀ λ``), the multiplier residual, the reduction used."""

    u: np.ndarray
    reactions: np.ndarray
    multipliers: np.ndarray
    multiplier_residual: float
    reduction: Reduction
    z: np.ndarray

    @property
    def constraint_forces(self) -> np.ndarray:
        """``C_usedᵀ λ`` (ndof, k): the force of the constraints rebuilt from the multipliers."""
        return np.asarray(self.reduction.C_used.T @ self.multipliers)


def _free_rigid_motions(problem: Any, C: sp.csr_matrix) -> None:
    """Raise :class:`MechanismError` when the rows ``C`` leave a rigid motion of a connected piece free (the
    generalisation of the support-rank check of ``_Linear._check_supports`` to arbitrary rows)."""
    mesh = problem.mesh
    nc = problem.ncomp
    n_piece, lab = mesh.components()
    robin = getattr(problem, "_robin_cells", np.zeros(0, np.int64))
    held_pieces = set(lab[np.asarray(robin, np.int64)].tolist()) if len(robin) else set()
    cols, rows_i, vals = [], [], []
    owner: list[int] = []
    ncol = 0
    for c in range(n_piece):
        if c in held_pieces:
            continue
        nodes = np.unique(mesh.cells[lab == c])
        R = rigid_modes(mesh.points[nodes], nc)
        dofs = (nc * nodes[:, None] + np.arange(nc)).ravel()
        for j in range(R.shape[1]):
            rows_i.append(dofs)
            cols.append(np.full(len(dofs), ncol + j))
            vals.append(R[:, j])
            owner.append(c)
        ncol += R.shape[1]
    if ncol == 0:
        return
    Rg = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols))),
                       shape=(problem.ndof, ncol))
    scale = np.maximum(np.asarray(abs(C).max(axis=1).todense()).ravel(), 1e-300)
    M = np.asarray((sp.diags(1.0 / scale) @ C @ Rg).todense()) if C.shape[0] else np.zeros((0, ncol))
    sv = np.linalg.svd(M, compute_uv=False) if M.size else np.zeros(0)
    rank = int((sv > 1e-8).sum())
    free = ncol - rank
    if free > 0:
        raise MechanismError(f"the constraints leave {free} free rigid motion(s) of the {n_piece} piece(s) (rank "
                             f"{rank} of {ncol}): the system is singular (periodicity needs a pin)")


def _load_matrix(problem: Any, f: np.ndarray) -> np.ndarray:
    F = np.asarray(f, float)
    F = F[:, None] if F.ndim == 1 else F
    if F.shape[0] != problem.ndof:
        raise ValueError(f"load must have {problem.ndof} rows, got {F.shape[0]}")
    return F


def solve_constrained(problem: Any, f: np.ndarray, constraints: LinearConstraints, rtol: float = 1e-10,
                      redundant: str = "error") -> ConstrainedSolution:
    """Solve ``K u = f`` of an :class:`ElasticProblem` / :class:`HeatProblem` under ``constraints`` by elimination.

    Returns ``u``, ``r = K u − f`` (all DOFs), the multipliers and the reduction. ``f`` (ndof,) or (ndof, k)."""
    F = _load_matrix(problem, f)
    if constraints.n != problem.ndof:
        raise ValueError(f"constraints on {constraints.n} DOFs, the problem has {problem.ndof}")
    red = constraints.eliminate(problem.active, rtol, redundant)
    _free_rigid_motions(problem, red.C_used)
    K = problem.K
    Kr = red.reduce_matrix(K)
    fr = red.reduce_load(K, F)
    if len(red.masters):
        z = Factorization(Kr, "spd").solve(fr).reshape(fr.shape)
    else:
        z = np.zeros((0, F.shape[1]))
    U = red.expand(z)
    U = U if U.ndim == 2 else U[:, None]
    if U.shape[1] != F.shape[1]:
        U = np.broadcast_to(U, F.shape).copy()
    r = np.asarray(K @ U) - F
    lam, res = red.multipliers(r)
    return ConstrainedSolution(U, r, lam, res, red, z)


# ---------------------------------------------------------------------------------------------- alternatives
@dataclass
class AlternativeSolution:
    """Result of :func:`solve_lagrange` / :func:`solve_penalty`: ``u`` (ndof, k), the multipliers in the convention of
    the method (Lagrange: ``μ = −λ``; penalty: ``α s (Ĉ u − d̂)``, see :func:`solve_penalty`), the 1-norm condition
    estimate of the matrix that was factored and its size."""

    u: np.ndarray
    multipliers: np.ndarray
    cond1: float
    size: int
    method: str


def _independent(problem: Any, constraints: LinearConstraints, rtol: float, redundant: str) -> Reduction:
    red = constraints.eliminate(problem.active, rtol, redundant)
    _free_rigid_motions(problem, red.C_used)
    return red


def solve_lagrange(problem: Any, f: np.ndarray, constraints: LinearConstraints, rtol: float = 1e-10,
                   redundant: str = "error") -> AlternativeSolution:
    """Saddle-point solution of ``[K Cᵀ; C 0][u; μ] = [f; d]`` on the unknown DOFs (SuperLU with pivoting);
    ``multipliers`` = μ = −λ of :func:`solve_constrained`. ``cond1`` is the estimate of the saddle matrix."""
    F = _load_matrix(problem, f)
    red = _independent(problem, constraints, rtol, redundant)
    act = problem.active
    Ka = sp.csr_matrix(problem.K[act][:, act])
    Ca = sp.csr_matrix(red.C_used[:, act])
    m = Ca.shape[0]
    A = sp.bmat([[Ka, Ca.T], [Ca, None]], format="csr") if m else Ka
    k = F.shape[1]
    d = np.broadcast_to(red.d_used, (m, k))
    rhs = np.vstack([F[act], d])
    fact = Factorization(A, "symmetric_indefinite")
    X = fact.solve(rhs).reshape(rhs.shape)
    U = np.zeros_like(F)
    U[act] = X[:len(act)]
    return AlternativeSolution(U, X[len(act):], condition_1norm(A, fact), A.shape[0], "lagrange")


def solve_penalty(problem: Any, f: np.ndarray, constraints: LinearConstraints, alpha: float = 1e8,
                  rtol: float = 1e-10, redundant: str = "error") -> AlternativeSolution:
    """Penalty solution of ``(K + α s Ĉᵀ Ĉ) u = f + α s Ĉᵀ d̂`` (Ĉ, d̂: the independent rows scaled to unit largest
    coefficient; s = max diag K). The constraint error is O(1/α); ``cond1`` grows with α. ``multipliers`` are
    ``α s (Ĉ u − d̂)`` (force per unit scaled row, ≈ λ·scale)."""
    if not alpha > 0:
        raise ValueError("alpha > 0")
    F = _load_matrix(problem, f)
    red = _independent(problem, constraints, rtol, redundant)
    act = problem.active
    Ka = sp.csr_matrix(problem.K[act][:, act])
    Ca = sp.csr_matrix(red.C_used[:, act])
    k = F.shape[1]
    if Ca.shape[0]:
        sc = np.asarray(abs(Ca).max(axis=1).todense()).ravel()
        Ca = sp.csr_matrix(sp.diags(1.0 / sc) @ Ca)
        d = np.broadcast_to(red.d_used / sc[:, None], (Ca.shape[0], k))
    else:
        d = np.zeros((0, k))
    s = float(Ka.diagonal().max())
    A = sp.csr_matrix(Ka + alpha * s * (Ca.T @ Ca))
    rhs = F[act] + alpha * s * np.asarray(Ca.T @ d)
    fact = Factorization(A, "spd")
    X = fact.solve(rhs).reshape(rhs.shape)
    U = np.zeros_like(F)
    U[act] = X
    lam = alpha * s * (np.asarray(Ca @ X) - d)
    return AlternativeSolution(U, lam, condition_1norm(A, fact), A.shape[0], "penalty")


# ---------------------------------------------------------------------------------------------- periodicity
def match_nodes(points: np.ndarray, minus: Any, plus: Any, offset: Any, tol: float | None = None) -> np.ndarray:
    """Pairs ``(a, b)`` (n, 2) of node numbers with ``x_b = x_a + offset`` (``a`` from ``minus``, ``b`` from ``plus``).

    Every node of both sets must be matched exactly once (same size, one partner each); otherwise
    :class:`ConstraintError` names the unmatched or ambiguous nodes. ``tol`` default ``1e-8 ×`` the extent of the
    points."""
    P = np.asarray(points, float)
    a = np.asarray(minus, np.int64).ravel()
    b = np.asarray(plus, np.int64).ravel()
    off = np.asarray(offset, float).ravel()
    if off.shape != (P.shape[1],):
        raise ValueError(f"offset must have {P.shape[1]} components")
    if len(a) != len(b):
        raise ConstraintError(f"periodic node sets of different size: {len(a)} and {len(b)}")
    tol = 1e-8 * float(np.ptp(P, axis=0).max()) if tol is None else float(tol)
    tree = cKDTree(P[b])
    dist, j = tree.query(P[a] + off)
    bad = dist > tol
    if bad.any():
        raise ConstraintError(f"{int(bad.sum())} node(s) have no partner at x + offset within {tol:.2e}, first: "
                              f"{a[bad][:5].tolist()}")
    if len(np.unique(j)) != len(j):
        raise ConstraintError("two nodes of the first set map to the same partner (ambiguous matching)")
    return np.column_stack([a, b[j]])


def macro_jump(H: Any, offset: Any) -> np.ndarray:
    """Jump ``H a`` of a displacement ``u = H x + fluctuation`` across the lattice vector ``a`` (H: (dim, dim))."""
    return np.asarray(H, float) @ np.asarray(offset, float)


def _periodic_rows(n_nodes: int, pairs: np.ndarray, jump: np.ndarray, ncomp: int, label: str) -> LinearConstraints:
    dofs_a = (ncomp * pairs[:, 0][:, None] + np.arange(ncomp)).ravel()
    dofs_b = (ncomp * pairs[:, 1][:, None] + np.arange(ncomp)).ravel()
    J = np.asarray(jump, float)
    J = np.full((ncomp, 1), float(J)) if J.ndim == 0 else (J[:, None] if J.ndim == 1 else J)
    if J.shape[0] != ncomp:
        raise ValueError(f"jump must have {ncomp} components, got {J.shape}")
    out = LinearConstraints.tie(n_nodes * ncomp, dofs_b, dofs_a, np.tile(J, (len(pairs), 1)))
    out.labels = [f"{label} comp {i % ncomp} node {pairs[i // ncomp, 1]} - node {pairs[i // ncomp, 0]}"
                  for i in range(out.m)]
    return out


def periodic(mesh: Mesh, minus: Any, plus: Any, offset: Any, jump: Any = 0.0, ncomp: int | None = None,
             tol: float | None = None) -> LinearConstraints:
    """Rows ``u_b − u_a = jump`` for every matched pair ``x_b = x_a + offset`` (``a`` in ``minus``, ``b`` in
    ``plus``), every component; ``offset`` any vector (skew cells), ``jump`` a scalar, a (ncomp,) vector (e.g.
    :func:`macro_jump`) or (ncomp, k). DOFs are node-major with ``ncomp`` components (default the mesh dimension)."""
    nc = mesh.dim if ncomp is None else int(ncomp)
    pairs = match_nodes(mesh.points, minus, plus, offset, tol)
    return _periodic_rows(mesh.n_nodes, pairs, jump, nc, "periodic")


def periodic_lattice(mesh: Mesh, lattice: Any, H: Any = None, ncomp: int | None = None,
                     tol: float | None = None) -> LinearConstraints:
    """Periodicity of a cell with lattice vectors ``lattice`` (n_vec, dim): for each vector ``a`` every used node
    whose translate ``x + a`` is also a node gives ``u_b − u_a = H a`` (H default zero: periodic displacement).
    Corner nodes produce dependent rows, so solve with ``redundant="drop"``. Each vector must match at least one
    pair, else :class:`ConstraintError`."""
    nc = mesh.dim if ncomp is None else int(ncomp)
    L = np.atleast_2d(np.asarray(lattice, float))
    used = mesh.used_nodes()
    P = mesh.points[used]
    tol = 1e-8 * float(np.ptp(P, axis=0).max()) if tol is None else float(tol)
    tree = cKDTree(P)
    out = LinearConstraints.empty(mesh.n_nodes * nc)
    for a in L:
        dist, j = tree.query(P + a)
        ok = dist <= tol
        if not ok.any():
            raise ConstraintError(f"no node pair is a lattice vector {a.tolist()} apart")
        pairs = np.column_stack([used[ok], used[j[ok]]])
        jump = np.zeros(nc) if H is None else macro_jump(H, a)
        out = out + _periodic_rows(mesh.n_nodes, pairs, jump, nc, f"periodic a={a.tolist()}")
    return out
