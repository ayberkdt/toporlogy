"""Reliability checks and diagnostics for generalized symmetric eigenproblems (WP05 L06, eigen part;
``docs/foundations/EIGEN_RELIABILITY.md``, preregistration ``docs/foundations/prereg/L6e_eigen.md``).

Problems covered: modal ``K φ = λ M φ`` (K symmetric, M SPD) and the buckling pencil ``(K + λ K_G) φ = 0`` with K SPD
and K_G symmetric indefinite, solved as ``(−K_G) φ = μ K φ``, ``μ = 1/λ``. This module does not import
``toporlogy.fem``; the production solves in :mod:`toporlogy.fem.modal` and :mod:`toporlogy.fem.buckling` stay as they
are and can be checked with :func:`eigen_report`.

* :func:`eigen_report`: per pair the backward error ``‖Kφ − λMφ‖ / ((‖K‖ + |λ|‖M‖)‖φ‖)``, the Rayleigh-quotient
  consistency, ``‖ΦᵀMΦ − I‖_F``, and a cluster-aware part: eigenvalues closer than ``cluster_rtol`` form a cluster whose
  vectors are not unique; the cluster is judged by a basis-independent block residual and, with a reference, by
  principal angles (:func:`subspace_angles`), never by comparing single vectors.
* :func:`mac`, :func:`subspace_angles`, :func:`track_sweep`: modal assurance criterion, principal angles in the M inner
  product, and identity of modes across a parameter change (the sorted index is not an identity at a crossing).
* :func:`shift_invert_modes`: Lanczos shift-invert with an explicit shift; a shift at or numerically at an eigenvalue is
  refused (:class:`ShiftError`); a semidefinite K (free-free) is handled by a non-zero shift and its zero modes are
  reported.
* :func:`buckling_load_factors`: the buckling pencil with its assumptions checked and the load convention stated.

Units follow the matrices: λ has the unit of ``K``/``M``; the buckling load factor is dimensionless (a multiple of the
reference load).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["ShiftError", "EigenReport", "eigen_report", "mac", "subspace_angles", "Crossing", "TrackResult",
           "track_sweep", "ShiftInvertResult", "shift_invert_modes", "BucklingResult", "buckling_load_factors"]

SYM_TOL = 1e-12          #: ‖A − Aᵀ‖_F ≤ SYM_TOL ‖A‖_F for a matrix declared symmetric
_DENSE_NORM_N = 400      #: matrix 2-norm computed densely up to this size


class ShiftError(ValueError):
    """The shift lies at (or numerically at) an eigenvalue, or K − σM cannot be factorised."""


# ---------------------------------------------------------------------------------------------------- input checks
def _as_matrix(A: Any, name: str) -> Any:
    A = A.tocsr() if sp.issparse(A) else np.asarray(A, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")
    data = A.data if sp.issparse(A) else A
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return A


def _fro(A: Any) -> float:
    return float(spla.norm(A)) if sp.issparse(A) else float(np.linalg.norm(A))


def _check_sym(A: Any, name: str) -> None:
    num, den = _fro(A - A.T), _fro(A)
    if num > SYM_TOL * max(den, 1e-300):
        raise ValueError(f"{name} must be symmetric (‖A − Aᵀ‖_F / ‖A‖_F = {num / max(den, 1e-300):.2e})")


def _check_pair(K: Any, M: Any) -> tuple[Any, Any]:
    K, M = _as_matrix(K, "K"), _as_matrix(M, "M")
    if K.shape != M.shape:
        raise ValueError(f"K {K.shape} and M {M.shape} differ in shape")
    _check_sym(K, "K")
    _check_sym(M, "M")
    return K, M


def _norm2(A: Any) -> float:
    """Spectral norm of a symmetric matrix: dense for small n, a Lanczos estimate (never above the true value)
    otherwise."""
    n = A.shape[0]
    if n <= _DENSE_NORM_N:
        return float(np.linalg.norm(A.toarray() if sp.issparse(A) else A, 2))
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    return float(abs(spla.eigsh(A, k=1, which="LM", tol=1e-4, return_eigenvectors=False)[0]))


def _norm1(A: Any) -> float:
    return float(abs(A).sum(axis=0).max()) if sp.issparse(A) else float(np.abs(A).sum(axis=0).max())


def _chol_m(M: Any) -> np.ndarray:
    """Upper factor R with RᵀR = M (dense), ValueError if M is not SPD."""
    try:
        return np.linalg.cholesky(M.toarray() if sp.issparse(M) else M).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("M is not symmetric positive definite (Cholesky failed)") from exc


def _vecs(Phi: np.ndarray, n: int, name: str = "Phi") -> np.ndarray:
    Phi = np.asarray(Phi, float)
    if Phi.ndim == 1:
        Phi = Phi[:, None]
    if Phi.ndim != 2 or Phi.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows, got shape {Phi.shape}")
    if not np.isfinite(Phi).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return Phi


# ---------------------------------------------------------------------------------------------------- MAC, angles
def mac(Phi_a: np.ndarray, Phi_b: np.ndarray, M: Any = None) -> np.ndarray:
    """Modal assurance criterion matrix ``MAC_ij = (a_iᵀ M b_j)² / ((a_iᵀ M a_i)(b_jᵀ M b_j))`` (ka × kb), in [0, 1];
    ``M=None`` is the identity. Invariant to the sign and the scale of every vector."""
    Phi_a = np.asarray(Phi_a, float)
    Phi_a = Phi_a[:, None] if Phi_a.ndim == 1 else Phi_a
    n = Phi_a.shape[0]
    if np.asarray(Phi_b).shape[0] != n:
        raise ValueError(f"Phi_a and Phi_b must have the same n (rows), got {Phi_a.shape} and {np.shape(Phi_b)}")
    Phi_b = _vecs(Phi_b, n, "Phi_b")
    Phi_a = _vecs(Phi_a, n, "Phi_a")
    if M is None:
        Ma, Mb = Phi_a, Phi_b
    else:
        M = _as_matrix(M, "M")
        if M.shape[0] != n:
            raise ValueError(f"M has {M.shape[0]} rows, vectors have {n}")
        Ma, Mb = M @ Phi_a, M @ Phi_b
    cross = Phi_a.T @ Mb
    na, nb = np.einsum("ij,ij->j", Phi_a, Ma), np.einsum("ij,ij->j", Phi_b, Mb)
    if np.any(na <= 0.0) or np.any(nb <= 0.0):
        raise ValueError("MAC needs vectors of positive M-norm (M positive definite on the given modes, "
                         "no zero vector)")
    return cross ** 2 / np.outer(na, nb)


def _m_orthonormal(Phi: np.ndarray, M: Any) -> np.ndarray:
    """Basis of span(Phi) with QᵀMQ = I (Cholesky of the Gram matrix)."""
    G = Phi.T @ (M @ Phi if M is not None else Phi)
    G = 0.5 * (G + G.T)
    try:
        R = np.linalg.cholesky(G).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("the vectors are linearly dependent (or M is not SPD): no M-orthonormal basis") from exc
    return np.linalg.solve(R.T, Phi.T).T


def subspace_angles(Phi_a: np.ndarray, Phi_b: np.ndarray, M: Any = None) -> np.ndarray:
    """Principal angles (ascending, radians) between span(Phi_a) and span(Phi_b) in the M inner product (identity for
    ``M=None``); both must have the same number of columns. The sines come from the residual block without squaring
    (module notes of the preregistration), so angles down to about 1e-15 are resolved. Individual vectors of an
    eigenvalue cluster are not unique; their span is."""
    Phi_a = np.asarray(Phi_a, float)
    Phi_a = Phi_a[:, None] if Phi_a.ndim == 1 else Phi_a
    n = Phi_a.shape[0]
    Phi_a, Phi_b = _vecs(Phi_a, n, "Phi_a"), _vecs(Phi_b, n, "Phi_b")
    if Phi_a.shape[1] != Phi_b.shape[1]:
        raise ValueError(f"subspaces of different dimension: {Phi_a.shape[1]} and {Phi_b.shape[1]}")
    if M is not None:
        M = _as_matrix(M, "M")
        if M.shape[0] != n:
            raise ValueError(f"M has {M.shape[0]} rows, vectors have {n}")
    Qa, Qb = _m_orthonormal(Phi_a, M), _m_orthonormal(Phi_b, M)
    MQb = Qb if M is None else M @ Qb
    C = Qa.T @ MQb
    cos = np.sort(np.linalg.svd(C, compute_uv=False))[::-1]
    S = Qb - Qa @ C
    Qs, Rs = np.linalg.qr(S)
    H = Qs.T @ (Qs if M is None else M @ Qs)
    try:
        Lc = np.linalg.cholesky(0.5 * (H + H.T))
        sin = np.linalg.svd(Lc.T @ Rs, compute_uv=False)
    except np.linalg.LinAlgError:                       # S numerically zero: the subspaces coincide
        sin = np.linalg.svd(Rs, compute_uv=False)
    sin = np.sort(sin)
    return np.arctan2(sin, np.clip(cos, 0.0, None))


# ---------------------------------------------------------------------------------------------------- report
def _clusters(lam: np.ndarray, rtol: float, scale: float) -> list[tuple[int, ...]]:
    """Index groups of sorted eigenvalues closer than rtol·max(|λ|) + 1e-12·scale (scale = ‖K‖/‖M‖)."""
    order = np.argsort(lam, kind="stable")
    groups: list[list[int]] = [[int(order[0])]] if len(lam) else []
    for a, b in zip(order[:-1], order[1:], strict=True):
        if abs(lam[b] - lam[a]) <= rtol * max(abs(lam[a]), abs(lam[b])) + 1e-12 * scale:
            groups[-1].append(int(b))
        else:
            groups.append([int(b)])
    return [tuple(g) for g in groups]


@dataclass
class EigenReport:
    """Result of :func:`eigen_report` (definitions in the module notes of the preregistration)."""

    residual: np.ndarray                    #: η_i, backward error of every pair (k,)
    rayleigh: np.ndarray                    #: r_i, Rayleigh-quotient consistency (k,)
    rayleigh_quotient: np.ndarray           #: φᵀKφ / φᵀMφ (k,)
    orth_error: float                       #: ‖ΦᵀMΦ − I‖_F
    clusters: list[tuple[int, ...]]         #: index groups of (nearly) equal eigenvalues, sorted by eigenvalue
    cluster_residual: np.ndarray            #: basis-independent block residual per cluster
    #: largest principal angle to the reference per cluster (None without reference)
    cluster_angle: np.ndarray | None
    norm_K: float
    norm_M: float
    notes: list[str] = field(default_factory=list)

    @property
    def max_residual(self) -> float:
        return float(self.residual.max()) if self.residual.size else 0.0

    @property
    def max_rayleigh(self) -> float:
        return float(self.rayleigh.max()) if self.rayleigh.size else 0.0

    @property
    def max_cluster_angle(self) -> float | None:
        return None if self.cluster_angle is None else float(self.cluster_angle.max())

    def passed(self, tol_residual: float = 1e-8, tol_orth: float = 1e-8, tol_rayleigh: float = 1e-8,
               tol_angle: float = 1e-8) -> bool:
        """True when every measure is within its tolerance (angle only if a reference was given)."""
        ok = (self.max_residual <= tol_residual and self.orth_error <= tol_orth and self.max_rayleigh <= tol_rayleigh
              and (self.cluster_residual.size == 0 or float(self.cluster_residual.max()) <= tol_residual))
        if self.cluster_angle is not None:
            ok = ok and self.max_cluster_angle <= tol_angle      # type: ignore[operator]
        return bool(ok)


def eigen_report(K: Any, M: Any, lam: np.ndarray, Phi: np.ndarray, *, cluster_rtol: float = 1e-6,
                 reference: tuple[np.ndarray, np.ndarray] | None = None, norm_K: float | None = None,
                 norm_M: float | None = None) -> EigenReport:
    """Reliability report of computed eigenpairs ``(lam (k,), Phi (n, k))`` of ``K φ = λ M φ``.

    Per pair: η_i = ‖Kφ − λMφ‖ / ((‖K‖ + |λ|‖M‖)‖φ‖) (2-norms; a backward error), r_i = |φᵀKφ − λφᵀMφ| / ((‖K‖ +
    |λ|‖M‖)‖φ‖²) (the eigenvalue agrees with the Rayleigh quotient of its vector). Overall ‖ΦᵀMΦ − I‖_F (the vectors are
    expected M-orthonormal). Clusters (sorted eigenvalues closer than ``cluster_rtol``) get a basis-independent block
    residual ‖KΦ_c − MΦ_c Θ‖_F / ((‖K‖ + max|λ|‖M‖)‖Φ_c‖_F), Θ = (Φ_cᵀMΦ_c)⁻¹Φ_cᵀKΦ_c. With ``reference = (lam_ref,
    Phi_ref)`` (the same count of pairs, e.g. a dense solve or the analytic subspace) each cluster is compared with the
    reference columns of the same sorted positions by principal angles; single vectors of a cluster are never compared.
    ``norm_K`` and ``norm_M`` skip the norm computation."""
    K, M = _check_pair(K, M)
    n = K.shape[0]
    lam = np.asarray(lam, float).reshape(-1)
    Phi = _vecs(Phi, n)
    if Phi.shape[1] != lam.size:
        raise ValueError(f"{lam.size} eigenvalues but {Phi.shape[1]} vectors")
    if not np.isfinite(lam).all():
        raise ValueError("eigenvalues contain NaN or Inf")
    nK = _norm2(K) if norm_K is None else float(norm_K)
    nM = _norm2(M) if norm_M is None else float(norm_M)
    KP, MP = K @ Phi, M @ Phi
    pn = np.linalg.norm(Phi, axis=0)
    if np.any(pn == 0.0):
        raise ValueError("a vector is zero")
    denom = nK + np.abs(lam) * nM
    res = np.linalg.norm(KP - MP * lam, axis=0) / (denom * pn)
    phiK, phiM = np.einsum("ij,ij->j", Phi, KP), np.einsum("ij,ij->j", Phi, MP)
    ray = np.abs(phiK - lam * phiM) / (denom * pn ** 2)
    rq = phiK / np.where(phiM != 0.0, phiM, np.nan)
    G = Phi.T @ MP
    orth = float(np.linalg.norm(0.5 * (G + G.T) - np.eye(lam.size), "fro"))
    groups = _clusters(lam, cluster_rtol, nK / max(nM, 1e-300))
    notes: list[str] = []
    cres = np.zeros(len(groups))
    for c, g in enumerate(groups):
        P = Phi[:, g]
        Gm = P.T @ MP[:, g]
        Gk = P.T @ KP[:, g]
        try:
            Theta = np.linalg.solve(Gm, Gk)
        except np.linalg.LinAlgError:
            cres[c] = np.inf
            notes.append(f"cluster {g}: Gram matrix singular (dependent vectors)")
            continue
        R = KP[:, g] - MP[:, g] @ Theta
        cres[c] = float(np.linalg.norm(R, "fro") / ((nK + np.abs(lam[list(g)]).max() * nM) * np.linalg.norm(P, "fro")))
    ang = None
    if reference is not None:
        lam_r, Phi_r = reference
        lam_r = np.asarray(lam_r, float).reshape(-1)
        Phi_r = _vecs(Phi_r, n, "reference Phi")
        if Phi_r.shape[1] != lam.size or lam_r.size != lam.size:
            raise ValueError("the reference must have the same number of pairs")
        ia, ir = np.argsort(lam, kind="stable"), np.argsort(lam_r, kind="stable")
        ang = np.zeros(len(groups))
        pos = {int(j): p for p, j in enumerate(ia)}
        for c, g in enumerate(groups):
            cols_ref = [int(ir[pos[j]]) for j in g]
            ang[c] = float(subspace_angles(Phi[:, g], Phi_r[:, cols_ref], M).max())
    return EigenReport(res, ray, rq, orth, groups, cres, ang, nK, nM, notes)


# ---------------------------------------------------------------------------------------------------- tracking
@dataclass
class Crossing:
    """Two tracks whose eigenvalue order reversed between ``step - 1`` and ``step``."""

    step: int
    track_a: int
    track_b: int


@dataclass
class TrackResult:
    """Result of :func:`track_sweep`. Track t is the mode that is column t at step 0."""

    order: list[np.ndarray]                 #: order[s][t] = column of step s that belongs to track t
    lam: np.ndarray                         #: (steps, k) eigenvalues arranged by track
    mac_to_previous: np.ndarray             #: (steps − 1, k) MAC of every track with itself at the previous step
    crossings: list[Crossing]
    ambiguous: list[tuple[int, int]]        #: (step, track) with low or non-unique MAC
    mac_matrices: list[np.ndarray]          #: (steps − 1) MAC matrices (previous columns × current columns)


def track_sweep(lams: Sequence[np.ndarray], Phis: Sequence[np.ndarray], M: Any, *, mac_min: float = 0.9,
                margin: float = 0.2) -> TrackResult:
    """Follow modes through a parameter sweep by MAC; the sorted index is not an identity.

    ``lams[s]`` (k,) ascending and ``Phis[s]`` (n, k) are the solution at step s; ``M`` is one matrix or one per step
    (the mean of the two steps' matrices is used for a pair). Consecutive steps are matched one-to-one by the assignment
    that maximises the sum of the MAC. ``crossings`` lists track pairs whose column order reversed; ``ambiguous`` marks
    a track with MAC below ``mac_min`` or within ``margin`` of the next best MAC in its row. Limit: a mode that enters
    the window from outside is invisible; track k + extra modes and trust the first k."""
    S = len(lams)
    if S < 2 or len(Phis) != S:
        raise ValueError("need at least two steps and one eigenvector block per step")
    Ms = list(M) if isinstance(M, (list, tuple)) else [M] * S
    if len(Ms) != S:
        raise ValueError("M must be one matrix or one per step")
    lam0 = np.asarray(lams[0], float).reshape(-1)
    k = lam0.size
    n = np.asarray(Phis[0]).shape[0]
    for s in range(S):
        if np.asarray(lams[s]).size != k or np.asarray(Phis[s]).shape != (n, k):
            raise ValueError(f"step {s}: expected {k} modes of length {n}")
    order = [np.arange(k)]
    lam = np.zeros((S, k))
    lam[0] = lam0
    macd = np.zeros((S - 1, k))
    mats: list[np.ndarray] = []
    crossings: list[Crossing] = []
    amb: list[tuple[int, int]] = []
    for s in range(1, S):
        Mp = Ms[s - 1] if Ms[s - 1] is Ms[s] else 0.5 * (Ms[s - 1] + Ms[s])
        Q = mac(np.asarray(Phis[s - 1], float), np.asarray(Phis[s], float), Mp)       # rows: previous columns
        mats.append(Q)
        prev = order[s - 1]                                  # prev[t] = column at step s-1 of track t
        rows, cols = sopt.linear_sum_assignment(-Q)
        col_of_row = np.empty(k, int)
        col_of_row[rows] = cols
        cur = col_of_row[prev]                               # column at step s of track t
        order.append(cur)
        lam[s] = np.asarray(lams[s], float).reshape(-1)[cur]
        for t in range(k):
            row = Q[prev[t]]
            best = row[cur[t]]
            macd[s - 1, t] = best
            second = np.partition(np.delete(row, cur[t]), -1)[-1] if k > 1 else 0.0
            if best < mac_min or best - second < margin:
                amb.append((s, t))
        for a in range(k):
            for b in range(a + 1, k):
                if (prev[a] < prev[b]) != (cur[a] < cur[b]):
                    crossings.append(Crossing(s, a, b))
    return TrackResult(order, lam, macd, crossings, amb, mats)


# ---------------------------------------------------------------------------------------------------- shift-invert
@dataclass
class ShiftInvertResult:
    """Result of :func:`shift_invert_modes` (k pairs sorted ascending by λ; the eigenvalues nearest ``sigma``)."""

    lam: np.ndarray
    Phi: np.ndarray                         #: M-orthonormal (n, k)
    sigma: float
    scale: float                            #: s = ‖K‖₁/‖M‖₁, the spectral scale used for the tolerances
    shift_distance: float                   #: 1/g ≤ min|λ − σ| (lower bound from the 1-norm estimate g)
    zero_idx: np.ndarray                    #: positions in ``lam`` with |λ| ≤ zero_tol·s
    n_zero: int
    zero_residual: float                    #: max ‖Kφ‖/(‖K‖‖φ‖) over the zero modes (0 if none)
    zero_count_ok: bool | None              #: n_zero == expected_zero (None without expectation)
    cluster_cut: bool                       #: the (k+1)-th pair belongs to the cluster of a returned one
    lam_next: float
    report: EigenReport
    notes: list[str] = field(default_factory=list)


def _shift_factor(K: Any, M: Any, sigma: float, scale: float, refuse_rtol: float) -> tuple[Any, float]:
    A = sp.csc_matrix(K - sigma * M)
    try:
        lu = spla.splu(A)
    except RuntimeError as exc:
        raise ShiftError(f"K − σM is exactly singular at σ = {sigma:.6g}: the shift is an eigenvalue") from exc
    n = A.shape[0]
    Mc = sp.csr_matrix(M) if sp.issparse(M) else np.asarray(M)
    op = spla.LinearOperator((n, n), matvec=lambda v: lu.solve(np.asarray(Mc @ np.asarray(v, float).reshape(-1))),
                             rmatvec=lambda v: Mc @ lu.solve(np.asarray(v, float).reshape(-1)), dtype=float)
    g = float(spla.onenormest(op)) if n > 1 else float(abs(op.matvec(np.ones(1))[0]))
    dist = 1.0 / g if g > 0.0 and np.isfinite(g) else 0.0
    if dist <= refuse_rtol * scale:
        raise ShiftError(f"σ = {sigma:.6g} is at (or numerically at) an eigenvalue: min|λ − σ| ≤ {dist:.3e}, below "
                         f"refuse_rtol · s = {refuse_rtol * scale:.3e}")
    return lu, dist


def shift_invert_modes(K: Any, M: Any, k: int, sigma: float | None = None, *, assume_psd: bool = False,
                       expected_zero: int | None = None, zero_tol: float = 1e-9, refuse_rtol: float = 1e-8,
                       cluster_rtol: float = 1e-6, tol: float = 0.0, seed: int = 0) -> ShiftInvertResult:
    """The ``k`` eigenpairs of ``K φ = λ M φ`` nearest ``sigma`` by Lanczos shift-invert, with the safeguards of the
    module notes.

    * ``sigma`` is an explicit choice. ``sigma=None`` is accepted only with ``assume_psd=True`` (K positive
      semidefinite, e.g. free-free): the shift is then ``−1e-6 s`` (s = ‖K‖₁/‖M‖₁), below the spectrum, so K − σM is
      SPD, the k nearest are the k lowest, and the zero modes are found instead of a singular factorisation.
    * A shift at or numerically at an eigenvalue raises :class:`ShiftError` (exactly singular LU, or the 1-norm estimate
      of (K − σM)⁻¹M gives min|λ − σ| ≤ ``refuse_rtol``·s). ``sigma=0`` on a free-free structure is therefore refused.
    * One extra pair is computed; ``cluster_cut`` reports that the k-th returned eigenvalue shares its cluster with it
      (the returned subspace of that cluster is incomplete).
    * Zero modes (|λ| ≤ ``zero_tol``·s) are reported with their null-space residual; ``expected_zero`` is compared, not
      enforced.
    * The pairs are cleaned by Rayleigh-Ritz in their span (exact M-orthonormality), sorted ascending by λ."""
    K, M = _check_pair(K, M)
    n = K.shape[0]
    if not 1 <= k < n - 1:
        raise ValueError(f"k must lie in [1, {n - 2}], got {k}")
    if n <= 3000:
        _chol_m(M)
    s = _norm1(K) / max(_norm1(M), 1e-300)
    if sigma is None:
        if not assume_psd:
            raise ValueError("choose sigma explicitly, or pass assume_psd=True (K positive semidefinite) "
                             "for the default shift")
        sigma = -1e-6 * s
    sigma = float(sigma)
    if not np.isfinite(sigma):
        raise ShiftError("sigma must be finite")
    notes: list[str] = []
    lu, dist = _shift_factor(K, M, sigma, s, refuse_rtol)
    Kc = sp.csr_matrix(K)
    Mc = sp.csr_matrix(M)
    Opinv = spla.LinearOperator((n, n), matvec=lambda b: lu.solve(np.asarray(b, float).reshape(-1)), dtype=float)
    kk = k + 1
    v0 = np.random.default_rng(seed).standard_normal(n)
    ncv = min(n, max(2 * kk + 1, kk + 16))
    _, V = spla.eigsh(Kc, k=kk, M=Mc, sigma=sigma, which="LM", OPinv=Opinv, tol=tol, ncv=ncv, v0=v0)
    Kr, Mr = V.T @ (Kc @ V), V.T @ (Mc @ V)
    lam_all, Q = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    Phi_all = V @ Q
    near = np.argsort(np.abs(lam_all - sigma), kind="stable")
    keep, extra = near[:k], int(near[k])
    lam_next = float(lam_all[extra])
    srt = np.sort(keep)
    lam, Phi = lam_all[srt], Phi_all[:, srt]
    close = lambda a, b: abs(a - b) <= cluster_rtol * max(abs(a), abs(b)) + 1e-12 * s       # noqa: E731
    dk = np.abs(lam - sigma).max()
    cut = bool(any(close(lam_next, x) for x in lam) or abs(abs(lam_next - sigma) - dk) <= cluster_rtol * dk + 1e-12 * s)
    if cut:
        notes.append("the extra eigenpair lies in the cluster of a returned one: the subspace of that cluster is "
                     "incomplete (ask for more modes, or compare subspaces only for complete clusters)")
    zero_idx = np.flatnonzero(np.abs(lam) <= zero_tol * s)
    nK = _norm2(K)
    if zero_idx.size:
        Z = Phi[:, zero_idx]
        zres = float((np.linalg.norm(Kc @ Z, axis=0) / (nK * np.linalg.norm(Z, axis=0))).max())
    else:
        zres = 0.0
    if assume_psd and np.any(lam < -zero_tol * s):
        notes.append(f"assume_psd=True but λ_min = {lam.min():.3e} < −zero_tol·s: K has a negative eigenvalue")
    zok = None if expected_zero is None else bool(zero_idx.size == int(expected_zero))
    rep = eigen_report(K, M, lam, Phi, cluster_rtol=cluster_rtol, norm_K=nK)
    return ShiftInvertResult(lam, Phi, sigma, s, dist, zero_idx, int(zero_idx.size), zres, zok, cut, lam_next, rep,
                             notes)


# ---------------------------------------------------------------------------------------------------- buckling
@dataclass
class BucklingResult:
    """Result of :func:`buckling_load_factors`. Load factor λ: buckling at λ × the reference load."""

    load_factors: np.ndarray                #: lowest positive λ = 1/μ, ascending (≤ k)
    mu: np.ndarray                          #: the corresponding μ (descending)
    Phi: np.ndarray                         #: K-orthonormal modes (n, len(load_factors))
    report: EigenReport | None              #: report of (−K_G, K, μ, Φ)
    reverse_load_factors: np.ndarray | None  #: |λ| for μ < 0 (buckling under the reversed load), dense path only
    kg_inertia: tuple[int, int, int] | None  #: (n₊, n₋, n₀) of K_G, dense path only
    spd_evidence: str                       #: "cholesky" (proof up to rounding) or "lanczos_min_eigenvalue" (evidence)
    path: str                               #: "dense" or "lanczos"
    cluster_cut: bool
    notes: list[str] = field(default_factory=list)


def _inertia_sym(A: np.ndarray, tol: float) -> tuple[int, int, int]:
    w = np.linalg.eigvalsh(A)
    return int((w > tol).sum()), int((w < -tol).sum()), int((np.abs(w) <= tol).sum())


def buckling_load_factors(K: Any, KG: Any, k: int = 1, *, dense_max: int = 300, spd_dense_max: int = 2500,
                          mu_rtol: float = 1e-10, cluster_rtol: float = 1e-6, tol: float = 0.0,
                          seed: int = 0) -> BucklingResult:
    """Lowest positive load factors of ``(K + λ K_G) φ = 0``, solved as ``(−K_G) φ = μ K φ``, ``μ = 1/λ``.

    Assumptions (checked): K symmetric positive definite (Cholesky up to ``spd_dense_max`` unknowns, otherwise the
    smallest eigenvalue by Lanczos shift-invert, which is evidence, not proof); K_G symmetric, in general indefinite
    (its inertia is reported on the dense path). Convention: K_G belongs to the reference load, tension positive, so
    compression makes K_G negative along the mode and μ > 0; the load factor is 1/μ for μ > ``mu_rtol``·max|μ|; μ < 0 is
    a buckling under the reversed load, reported separately (dense path); μ ≈ 0 does not buckle. No positive μ gives an
    empty result and a note. Modes are K-orthonormal, ``μ = −φᵀK_Gφ``. Up to ``dense_max`` unknowns the whole spectrum
    is computed densely, above it the ``k`` largest μ (plus one extra for the cluster-cut flag) by Lanczos with K⁻¹ as
    the inverse."""
    K, KG = _check_pair(K, KG)
    n = K.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must lie in [1, {n - 1}], got {k}")
    notes: list[str] = []
    if n <= spd_dense_max:
        try:
            np.linalg.cholesky(K.toarray() if sp.issparse(K) else K)
        except np.linalg.LinAlgError as exc:
            raise ValueError("K is not positive definite: the buckling solver assumes a stable unloaded "
                             "structure") from exc
        evidence = "cholesky"
    else:
        lmin = float(spla.eigsh(sp.csc_matrix(K), k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0])
        if lmin <= 1e-12 * _norm1(K):
            raise ValueError(f"K is not positive definite (smallest eigenvalue {lmin:.3e})")
        evidence = "lanczos_min_eigenvalue"
        notes.append("K positive definiteness is Lanczos evidence (smallest eigenvalue > 0), not a factorisation proof")
    A = -KG
    if n <= dense_max:
        Kd = K.toarray() if sp.issparse(K) else K
        Ad = A.toarray() if sp.issparse(A) else A
        mu_all, V = sla.eigh(0.5 * (Ad + Ad.T), 0.5 * (Kd + Kd.T))
        o = np.argsort(-mu_all)
        mu_all, V = mu_all[o], V[:, o]
        mmax = float(np.abs(mu_all).max())
        pos = mu_all > mu_rtol * max(mmax, 1e-300)
        neg = mu_all < -mu_rtol * max(mmax, 1e-300)
        mu, Phi = mu_all[pos][:k], V[:, pos][:, :k]
        rev = np.sort(1.0 / np.abs(mu_all[neg]))
        kg_in = _inertia_sym(0.5 * (Ad + Ad.T), mu_rtol * max(float(np.abs(Ad).max()), 1e-300))
        cut = bool(mu.size == k and np.count_nonzero(pos) > k
                   and abs(mu_all[pos][k] - mu[-1]) <= cluster_rtol * abs(mu[-1]))
        path = "dense"
    else:
        Kc = sp.csc_matrix(K)
        lu = spla.splu(Kc)
        Minv = spla.LinearOperator((n, n), matvec=lambda b: lu.solve(np.asarray(b, float).reshape(-1)), dtype=float)
        kk = min(k + 1, n - 2)
        v0 = np.random.default_rng(seed).standard_normal(n)
        mu_l, V = spla.eigsh(sp.csr_matrix(A), k=kk, M=Kc, Minv=Minv, which="LA", tol=tol, v0=v0,
                             ncv=min(n - 1, max(2 * kk + 2, 24)), maxiter=20 * n)
        Kr = V.T @ (Kc @ V)
        Gr = V.T @ (sp.csr_matrix(A) @ V)
        mu_r, Q = sla.eigh(0.5 * (Gr + Gr.T), 0.5 * (Kr + Kr.T))
        o = np.argsort(-mu_r)
        mu_r, Phi_r = mu_r[o], (V @ Q)[:, o]
        mmax = float(np.abs(mu_r).max())
        pos = mu_r > mu_rtol * max(mmax, 1e-300)
        mu_p, Phi_p = mu_r[pos], Phi_r[:, pos]
        mu, Phi = mu_p[:k], Phi_p[:, :k]
        cut = bool(mu_p.size > k and abs(mu_p[k] - mu[-1]) <= cluster_rtol * abs(mu[-1]))
        rev, kg_in, path = None, None, "lanczos"
    if cut:
        notes.append("the next mode shares the cluster of the last returned one: the subspace of that cluster is "
                     "incomplete")
    if mu.size == 0:
        notes.append("no positive mu: the reference load does not buckle the structure (tension, or no compressive "
                     "stress)")
        return BucklingResult(np.zeros(0), mu, np.zeros((n, 0)), None, rev, kg_in, evidence, path, False, notes)
    rep = eigen_report(A, K, mu, Phi, cluster_rtol=cluster_rtol)
    lam = 1.0 / mu
    o = np.argsort(lam)
    return BucklingResult(lam[o], mu[o], Phi[:, o], rep, rev, kg_in, evidence, path, cut, notes)
