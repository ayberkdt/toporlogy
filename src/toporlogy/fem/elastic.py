"""Linear elasticity on the regular grid with an autograd-aware linear solve (docs/mimari_plan.md §2, D1).

``K(E) U = F`` with ``K = Σ_e E_e · s · KE`` over the active elements, where ``s = h`` in 3-D (unit cube scaled to
edge h: B ∝ 1/h, volume h³) and ``s = thickness`` in 2-D (plane stress, square element of any size). ``E_e`` already
contains the element's solid fraction (the interpolation multiplies by ``frac``).

:meth:`Elasticity.solve` is a ``torch.autograd.Function``: the forward pass factorises and solves; the backward pass
solves the adjoint system with the same factorisation, ``K λ = ∂J/∂U``, and returns
``∂J/∂E_e = −s · λ_eᵀ KE u_e``. When ``∂J/∂U`` is a multiple of ``F`` (compliance), ``λ = a·U`` is used without a
solve (self-adjoint shortcut, exact).

Element matrices: :mod:`toporlogy.fem.elements` (H8, H8IM statically condensed, Q4, Q4IM).

Per-element material (fibre orientation): ``elasticity`` may be (n_active, k, k), one unit-modulus D per active
element (order of ``domain.active_elements``). The element matrices are then ``KE[ke_index[e]]``, built once per
distinct D (:func:`toporlogy.fem.elements.element_matrix_field`), and ``K = Σ_e E_e · s · KE_e``. With a single D
(k, k) or none the original path is used unchanged (one ``KE``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.mechanisms import SolveResidualError, check_moduli, check_supports, fixed_dof_rows
from toporlogy.solvers.direct import DirectSolver
from toporlogy.solvers.ordering import nested_dissection_blocks

__all__ = ["Elasticity", "element_matrix", "ORDERINGS", "GROUPED_MAX", "RESIDUAL_TOL", "BACKWARD_TOL", "ResidualReport",
           "checked_residual", "matrix_norm1"]


from toporlogy.fem.elements import element_matrix  # noqa: E402  (re-exported)
from toporlogy.fem.elements import (D_iso, D_plane_stress, element_matrix_field,  # noqa: E402
                                    element_matrix_from_D, indexed_matmul)


class Elasticity:
    """Static linear elasticity on a :class:`Domain` with fixed DOFs.

    Parameters
    ----------
    domain : Domain
    fixed : sequence of int
        Box DOF numbers with zero displacement. DOFs not touched by any active element are removed as well.
    nu : float
        Poisson's ratio.
    element : str, optional
        "H8" (default in 3-D), "H8IM", or "Q4" (2-D).
    solver : object with ``factor(A_upper)`` / ``solve(B)``, optional
        Default :class:`DirectSolver` ("auto").
    elasticity : array, optional
        Unit-modulus elasticity matrix D (6 × 6 in 3-D, 3 × 3 in 2-D; :func:`toporlogy.core.material.unit_elasticity`)
        for orthotropic, rotated or plane-strain material, or one per active element (n_active, k, k;
        :func:`toporlogy.core.material.unit_elasticity_field`). None = isotropic plane stress / 3-D from ``nu``; when
        it is given, ``nu`` is not used and :attr:`nu` is None.
    ordering : {"nested_dissection", "solver"}
        Elimination order handed to a :class:`DirectSolver` (SuperLU): a geometric nested dissection of the node grid
        (default; :mod:`toporlogy.solvers.ordering`) or the solver's own (``MMD_AT_PLUS_A``, the order before
        25 Nov 2026). PARDISO and other solvers ignore it. Settable attribute.

    Attributes
    ----------
    D : numpy.ndarray
        The unit-modulus elasticity the element matrices were built from, (k, k) or (n_active, k, k) (stress and
        buckling use it).
    KE : numpy.ndarray
        Unit element stiffness (nloc, nloc); per-element material: the distinct matrices (m, nloc, nloc).
    ke_index : numpy.ndarray or None
        Per-element material: (n_active,) index of each element's matrix in ``KE`` (and of its D in ``D_distinct``);
        None for one material.
    """

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, nu: float = 0.3, element: str | None = None,
                 solver: Any = None, elasticity: np.ndarray | None = None,
                 ordering: str = "nested_dissection") -> None:
        self.ordering = ordering
        self.domain = domain
        self.element = element or ("H8" if domain.dim == 3 else "Q4")
        self.ke_index: np.ndarray | None = None
        self.D_distinct: np.ndarray | None = None
        self._group_cache: list[tuple[int, np.ndarray]] | None = None
        if elasticity is None:
            self.nu: float | None = float(nu)
            self.D = D_iso(nu) if domain.dim == 3 else D_plane_stress(nu)
            self.KE = element_matrix(domain.dim, self.element, nu)
        else:
            self.nu = None
            self.D = np.asarray(elasticity, float).copy()
            if self.D.ndim == 3:
                if len(self.D) != domain.n_active:
                    raise ValueError(f"per-element elasticity has {len(self.D)} matrices, the domain "
                                     f"{domain.n_active} active elements")
                self.KE, self.ke_index = element_matrix_field(domain.dim, self.element, self.D)
                self.D_distinct = self.D[np.unique(self.ke_index, return_index=True)[1]]
            else:
                self.KE = element_matrix_from_D(domain.dim, self.element, self.D)
        self.geom = domain.h if domain.dim == 3 else domain.thickness
        self.elements = domain.active_elements
        self.edof = domain.edof[self.elements]                      # (na, nloc)
        fixed_arr = np.unique(np.asarray(fixed, np.int64))
        if fixed_arr.size and (fixed_arr.min() < 0 or fixed_arr.max() >= domain.ndof):
            raise ValueError("fixed DOF out of range")
        self.fixed = fixed_arr
        self.free = np.setdiff1d(domain.active_dofs, fixed_arr)
        # the support set of a physics object is frozen: the support check runs once and must not go stale
        self.fixed.setflags(write=False)
        self.free.setflags(write=False)
        if self.free.size == 0:
            raise ValueError("no free DOF")
        nd, nf, nl = domain.ndof, len(self.free), self.KE.shape[-1]
        gmap = np.full(nd, -1, np.int64)
        gmap[self.free] = np.arange(nf)
        r = gmap[np.repeat(self.edof, nl, axis=1).ravel()]
        c = gmap[np.tile(self.edof, (1, nl)).ravel()]
        keep = (r >= 0) & (c >= 0) & (r <= c)
        self._sel = np.flatnonzero(keep)
        key = r[keep] * nf + c[keep]
        ukey, inv = np.unique(key, return_inverse=True)
        self._inv = inv.astype(np.int64)
        self._P = sp.csr_matrix((np.zeros(len(ukey)), (ukey // nf, ukey % nf)), shape=(nf, nf))
        self.solver = solver if solver is not None else DirectSolver()
        self._order: np.ndarray | None = None                      # nested dissection of the reduced system, lazy
        self._blocks: np.ndarray | None = None                     # its parts (unknowns per part), with _order
        self._token = 0                                            # increments with every factorisation
        self._factored_E: np.ndarray | None = None
        self.n_solves = 0
        self.n_adjoint_solves = 0
        self._kinematics_checked = False
        self._A: sp.csr_matrix | None = None                     # reduced matrix of the last factorisation
        self._A_norm1: float | None = None                         # its ‖K‖₁ (for the backward error)
        self.last_residual: float | None = None                    # worst load case ‖K x − b‖ / ‖b‖, last solve
        self.last_residual_report: ResidualReport | None = None   # every load case of the last solve

    # ------------------------------------------------------------------ assembly and solve (NumPy)
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def per_element(self) -> bool:
        """True when every element has its own unit elasticity (``ke_index`` set)."""
        return self.ke_index is not None

    def element_stiffness(self) -> np.ndarray:
        """(na, nloc, nloc) unit element matrices (one material: ``KE`` repeated)."""
        if self.ke_index is None:
            return np.broadcast_to(self.KE, (self.n_elements,) + self.KE.shape).copy()
        return self.KE[self.ke_index]

    def element_forces(self, Ue: np.ndarray) -> np.ndarray:
        """(na, nloc, nc) ``KE_e u_e`` for element displacements ``Ue`` (na, nloc, nc) (unit modulus, no geom)."""
        if self.ke_index is None:
            return np.einsum("ab,ebc->eac", self.KE, Ue)
        if len(self.KE) <= GROUPED_MAX:
            out = np.empty(Ue.shape[:1] + (self.KE.shape[1],) + Ue.shape[2:])
            for j, sel in self._groups():
                out[sel] = np.einsum("ab,ebc->eac", self.KE[j], Ue[sel])
            return out
        return indexed_matmul(self.KE, self.ke_index, Ue)

    def _groups(self) -> list[tuple[int, np.ndarray]]:
        """(matrix number, its elements) of a per-element model."""
        if self._group_cache is None:
            idx = self.ke_index
            assert idx is not None
            self._group_cache = [(j, np.flatnonzero(idx == j)) for j in range(len(self.KE))]
        return self._group_cache

    def matrix(self, E_active: np.ndarray) -> sp.csr_matrix:
        """Upper triangle of the reduced stiffness for element moduli ``E_active`` (na,)."""
        s = np.broadcast_to(np.asarray(E_active, float) * self.geom, (self.n_elements,))
        if self.ke_index is None:
            w = (s[:, None] * self.KE.ravel()[None, :]).ravel()[self._sel]
        else:
            w = (s[:, None, None] * self.KE[self.ke_index]).ravel()[self._sel]
        A = self._P.copy()
        A.data = np.bincount(self._inv, weights=w, minlength=self._P.nnz)
        return A

    @property
    def ordering(self) -> str:
        return self._ordering

    @ordering.setter
    def ordering(self, value: str) -> None:
        if value not in ORDERINGS:
            raise ValueError(f"ordering: {' | '.join(ORDERINGS)}, got {value!r}")
        self._ordering = value

    def elimination_order(self) -> np.ndarray:
        """Nested dissection order of the reduced system (index into :attr:`free`), computed once."""
        if self._order is None:
            d = self.domain
            self._order, self._blocks = nested_dissection_blocks(tuple(n + 1 for n in d.shape), d.dim, self.free)
        return self._order

    def elimination_blocks(self) -> np.ndarray:
        """Unknowns per part (leaf box, separator plane) of :meth:`elimination_order`, consecutive in that order: the
        dense blocks of the supernodal Cholesky (``DirectSolver("cholesky")``). A subclass that changes the order
        must give its own blocks."""
        if type(self).elimination_order is not Elasticity.elimination_order:
            raise NotImplementedError(f"{type(self).__name__} changes the elimination order but gives no "
                                      "elimination_blocks(): the 'cholesky' direct solver needs both")
        self.elimination_order()
        assert self._blocks is not None
        return self._blocks

    def _kinematic_rows(self) -> dict[int, np.ndarray]:
        """Support rows per node acting on its displacement (fixed DOFs: unit rows); see :mod:`.mechanisms`."""
        return fixed_dof_rows(self.fixed, self.domain.dim)

    def check_kinematics(self, max_unknowns: int | None = None) -> None:
        """Raise :class:`~toporlogy.fem.mechanisms.MechanismError` if the supports leave a zero-energy motion (a
        floating piece, a hinge, a free rotation): the stiffness would be singular and a direct solve would return
        finite nonsense. Runs once, before the first factorisation."""
        lim = {} if max_unknowns is None else {"max_unknowns": int(max_unknowns)}   # call before the first solve
        check_supports(self.domain, self.elements, self._kinematic_rows(), **lim)
        self._kinematics_checked = True

    def factor(self, E_active: np.ndarray) -> None:
        E_active = check_moduli(E_active, self.n_elements)        # before the support check: it assumes E > 0
        if not self._kinematics_checked:
            self.check_kinematics()
        self._A = self.matrix(E_active)
        self._A_norm1 = matrix_norm1(self._A)
        if isinstance(self.solver, DirectSolver) and self.ordering == "nested_dissection":
            blocks = self.elimination_blocks() if self.solver.method == "cholesky" else None
            self.solver.factor(self._A, order=self.elimination_order(), blocks=blocks)
        else:
            self.solver.factor(self._A)
        self._factored_E = E_active.copy()
        self._token += 1

    def _solve_free(self, B_full: np.ndarray) -> np.ndarray:
        X = np.zeros_like(B_full)
        b = B_full[self.free]
        x = np.asarray(self.solver.solve(b)).reshape(b.shape)
        self._record_residual(checked_residual(self._A, x, b, self._A_norm1))
        X[self.free] = x
        return X

    def _loads(self, F: np.ndarray) -> np.ndarray:
        """Loads as (ndof, n_cases), refused with a clear message when the rows do not match or a value is not
        finite (docs/vv_fault_injection.md: inf_load_solve, load_wrong_rows_solve)."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        n = self.domain.ndof
        if F2.ndim != 2 or F2.shape[0] != n:
            raise ValueError(f"load F has shape {np.shape(F)}; expected {n} rows (DOFs), one column per load case")
        if not np.isfinite(F2).all():
            raise ValueError("load F must be finite (NaN or Inf found)")
        return F2

    def solve_numpy(self, E_active: np.ndarray, F: np.ndarray) -> np.ndarray:
        """U (ndof, n_cases) for moduli ``E_active`` and loads ``F`` (ndof,) or (ndof, n_cases); no autograd."""
        F2 = self._loads(F)
        self.factor(E_active)
        self.n_solves += 1
        return self._solve_free(F2)

    def _record_residual(self, rep: ResidualReport | None) -> None:
        self.last_residual_report = rep
        self.last_residual = None if rep is None else rep.max_relative

    def element_energy(self, U: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        """(na, n_cases) ``v_eᵀ KE u_e`` per active element and case (unit modulus, without ``geom``)."""
        U = U[:, None] if U.ndim == 1 else U
        V = U if V is None else (V[:, None] if V.ndim == 1 else V)
        Ue, Ve = U[self.edof], V[self.edof]                         # (na, nloc, nc)
        if self.ke_index is None:
            return np.einsum("eac,ab,ebc->ec", Ve, self.KE, Ue)
        if len(self.KE) <= GROUPED_MAX:                             # same arithmetic as one KE, element by element
            out = np.empty((len(Ue), Ue.shape[2]))
            for j, sel in self._groups():
                out[sel] = np.einsum("eac,ab,ebc->ec", Ve[sel], self.KE[j], Ue[sel])
            return out
        return np.einsum("eac,eac->ec", Ve, indexed_matmul(self.KE, self.ke_index, Ue))

    # ------------------------------------------------------------------ autograd
    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        """Differentiable ``U(E)``: (ndof, n_cases) float64 tensor. ``E_active`` (na,) tensor."""
        return _LinearSolve.apply(E_active, self, self._loads(F))


ORDERINGS = ("nested_dissection", "solver")

GROUPED_MAX = 32
"""Per-element material with at most this many distinct element matrices: element-level products run group by group
with the single-matrix arithmetic (a uniform or layered field then gives the single-D results bit for bit); above it
the gathered matrices are multiplied in chunks (:func:`toporlogy.fem.elements.indexed_matmul`)."""

RESIDUAL_TOL = 1e-6        # relative residual ‖Kx − b‖ / ‖b‖ accepted outright
BACKWARD_TOL = 1e-14       # backward error accepted outright (~45 eps); see checked_residual


@dataclass(frozen=True)
class ResidualReport:
    """Residuals of one linear solve K X = B, one entry per load case (right-hand-side column j).

    ``relative_per_rhs``: ‖K x_j − b_j‖ / ‖b_j‖, NaN for a zero load column (the ratio is undefined there).
    ``absolute_per_rhs``: ‖K x_j − b_j‖.
    ``backward_error_per_rhs``: η_j = ‖K x_j − b_j‖ / (‖K‖₁ ‖x_j‖ + ‖b_j‖), 0 when x_j = b_j = 0. ‖K‖₁ (largest
    absolute column sum) is an upper bound of ‖K‖₂ for symmetric K, so η_j is not overestimated. A small η_j says the
    solver solved a nearby system; it does not bound the forward error of an ill-conditioned K.
    ``worst_rhs``: column with the largest relative residual (0 when every load column is zero).
    ``max_relative``: that largest relative residual (0.0 when every load column is zero).
    """

    relative_per_rhs: np.ndarray
    absolute_per_rhs: np.ndarray
    backward_error_per_rhs: np.ndarray
    worst_rhs: int
    max_relative: float


def matrix_norm1(A: sp.csr_matrix) -> float:
    """‖K‖₁ (largest absolute column sum) of K = A + Aᵀ − diag(A), A its upper triangle; an upper bound of ‖K‖₂."""
    absA = abs(A)
    return float((np.asarray(absA.sum(axis=0)).ravel() + np.asarray(absA.sum(axis=1)).ravel()
                  - abs(A.diagonal())).max(initial=0.0))


def checked_residual(A: sp.csr_matrix | None, x: np.ndarray, b: np.ndarray,
                     norm1: float | None = None) -> ResidualReport | None:
    """Check every load case of K X = B (K = A + Aᵀ − diag(A), A: upper triangle) and return the residuals.

    Raises :class:`~toporlogy.fem.mechanisms.SolveResidualError`, naming the load case, when a solution column is not
    finite, when a nonzero load column has BOTH ‖K x_j − b_j‖ / ‖b_j‖ > :data:`RESIDUAL_TOL` and backward error
    η_j > :data:`BACKWARD_TOL`, or when a zero load column has
    a nonzero solution (K is nonsingular after the support check, so x_j = 0 is the only answer; noise up to
    RESIDUAL_TOL × the largest solution norm of the loaded columns is accepted). None when there is no matrix (a
    custom solver path that bypasses factor()). ``norm1``: :func:`matrix_norm1` of A when the caller has it (it
    changes only with the factorisation; computing it costs about three products with A).
    """
    if A is None:
        return None
    X = x.reshape(len(x), -1)
    Bm = b.reshape(X.shape)
    finite = np.isfinite(X).all(axis=0)
    if not finite.all():
        j = int(np.flatnonzero(~finite)[0])
        raise SolveResidualError(f"load case {j}: the solution is not finite (NaN or Inf)")
    R = A @ X + A.T @ X - A.diagonal()[:, None] * X - Bm
    absr = np.linalg.norm(R, axis=0)
    nb = np.linalg.norm(Bm, axis=0)
    nx = np.linalg.norm(X, axis=0)
    k1 = matrix_norm1(A) if norm1 is None else float(norm1)
    den = k1 * nx + nb
    eta = np.divide(absr, den, out=np.zeros_like(absr), where=den > 0)
    loaded = nb > 0
    rel = np.full(len(nb), np.nan)
    rel[loaded] = absr[loaded] / nb[loaded]
    # Why both: with high stiffness contrast (SIMP voids) ‖K‖‖x‖ ≫ ‖b‖, so a correct, backward-stable solve has a
    # large relative residual (measured 18 Nov 2026: up to 4e-4 at contrast 1e-9 on a 120 x 60 grid) while its
    # backward error stays at the rounding level (≤ 1.1e-16 over PARDISO and SuperLU, 2-D and 3-D, contrast 1 to
    # 1e-9). A solution 1 % off has η = 4e-5 at unit moduli and 1.5e-13 at contrast 1e-9, so it is still refused.
    for j in np.flatnonzero(loaded).tolist():
        if not (rel[j] <= RESIDUAL_TOL or eta[j] <= BACKWARD_TOL):
            raise SolveResidualError(f"load case {j}: linear solve residual ‖Kx − b‖/‖b‖ = {rel[j]:.2e} > "
                                     f"{RESIDUAL_TOL:g} and backward error {eta[j]:.2e} > {BACKWARD_TOL:g}: the "
                                     "system is singular or numerically not solvable at these moduli")
    x_scale = float(nx[loaded].max(initial=0.0))
    for j in np.flatnonzero(~loaded).tolist():
        if nx[j] > RESIDUAL_TOL * x_scale:
            raise SolveResidualError(f"load case {j}: zero load but the solution is not zero (‖x‖ = {nx[j]:.2e}, "
                                     f"‖Kx‖ = {absr[j]:.2e}, backward error {eta[j]:.2e})")
    worst = int(np.nanargmax(rel)) if loaded.any() else 0
    return ResidualReport(rel, absr, eta, worst, float(rel[worst]) if loaded.any() else 0.0)


class _LinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E_active: torch.Tensor, model: Elasticity, F: np.ndarray) -> torch.Tensor:
        E_np = E_active.detach().cpu().double().numpy()
        U = model.solve_numpy(E_np, F)
        ctx.model, ctx.F, ctx.U, ctx.E, ctx.token = model, F, U, E_np, model._token
        ctx.device, ctx.dtype = E_active.device, E_active.dtype
        return torch.from_numpy(U).to(device=E_active.device, dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, None, None]:
        model: Elasticity = ctx.model
        G = gU.detach().cpu().double().numpy()
        U, F = ctx.U, ctx.F
        lam = np.zeros_like(U)
        need = []
        fr = model.free
        for c in range(U.shape[1]):
            g, f = G[fr, c], F[fr, c]
            ff = float(f @ f)
            if ff > 0:
                a = float(g @ f) / ff
                if np.linalg.norm(g - a * f) <= 1e-12 * max(float(np.linalg.norm(g)), 1e-300):
                    lam[:, c] = a * U[:, c]                     # self-adjoint: K λ = a F  →  λ = a U
                    continue
            if np.any(g):
                need.append(c)
        if need:
            if model._token != ctx.token:                      # another forward refactorised: restore ours
                model.factor(ctx.E)
            sol = model._solve_free(G[:, need])
            lam[:, need] = sol
            model.n_adjoint_solves += 1
        dE = -model.geom * model.element_energy(U, lam).sum(axis=1)
        return torch.from_numpy(dE).to(device=ctx.device, dtype=ctx.dtype), None, None
