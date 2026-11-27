"""Spectral functions of symmetric 2 × 2 and 3 × 3 tensors and their first and second (Fréchet) derivatives.

For A = Q diag(λ) Qᵀ and a scalar function f, ``f(A) = Q diag(f(λ)) Qᵀ``. The directional derivative in a
**symmetric** direction H is

    L_A[H] = Q [ D ⊙ (Qᵀ H Q) ] Qᵀ,    D_ij = (f(λ_i) − f(λ_j)) / (λ_i − λ_j),   D_ii = f'(λ_i),

with ⊙ the element-wise product. D is evaluated by forms that stay accurate for equal and nearly equal eigenvalues
(the eigenvalues are never perturbed or separated):

    sqrt        1 / (√a + √b)
    inv_sqrt    −1 / (√a √b (√a + √b))
    log         log1p((hi − lo)/lo) / (hi − lo)             hi = max(a, b), lo = min(a, b); a = b → 1/a
    exp         e^hi · (−expm1(−(hi − lo))) / (hi − lo)     a = b → e^a
    power(p)    hi^(p−1) · expm1(p ℓ) / expm1(ℓ),  ℓ = −log1p((hi − lo)/lo);   a = b → p a^(p−1)

Second derivative (``prereg/T2_second_derivative.md``). With H̃ = QᵀHQ, K̃ = QᵀKQ and the second divided difference
F_ijk = f[λ_i, λ_j, λ_k] (symmetric in i, j, k),

    d²f(A)[H, K] = Q [ Σ_k F_ijk (H̃_ik K̃_kj + K̃_ik H̃_kj) ]_ij Qᵀ        (full mixed derivative, no factor ½,
                                                                           symmetric in H, K).

F = (D(hi, mid) − D(mid, lo)) / (hi − lo) for the sorted nodes where the spread is not small, and a Taylor series
about the mean of the nodes (coefficients f^{(j)}(m)/j!, complete homogeneous polynomials of the offsets) where it is
small; all nodes equal gives exactly f''/2. sqrt and inv_sqrt have cancellation-free closed forms
(−1/Π(√x+√y) and (Σ√x)/(Π√x · Π(√x+√y))). The Hessian-vector product of φ(A) = ⟨G, f(A)⟩ in direction H is
d²f(A)[H, G] (:func:`spectral_hvp`).

Contract (``docs/foundations/TENSOR_CONTRACT.md``)
--------------------------------------------------
* Inputs: float64 arrays (..., n, n), n ∈ {2, 3}, finite and symmetric to :data:`toporlogy.core.tensor.SYMMETRY_RTOL`
  (otherwise ValueError; no silent symmetrisation of the input; ``eigh`` then reads the symmetrised copy).
* Domain: ``log``, ``sqrt``, ``inv_sqrt`` and ``power`` need a symmetric **positive definite** A: a computed eigenvalue
  ≤ 0 raises ValueError (nothing is clipped). ``exp`` has no restriction. ``eigh`` returns small eigenvalues with an
  absolute error of the order ε λ_max, so for a condition number near 1/ε a positive but tiny eigenvalue can come out
  ≤ 0 (then the call refuses) or with few correct digits: no accuracy is claimed beyond κ ≈ 1e8.
* Only **symmetric** perturbations H (and cotangents G) are supported: L is the derivative of f on the space of
  symmetric matrices. A non-symmetric H raises ValueError; the derivative on general matrices is not defined here.
* JVP = :func:`spectral_frechet`. For symmetric A the map L_A is self-adjoint under the Frobenius product,
  ⟨L_A[H], G⟩ = ⟨H, L_A[G]⟩, so the VJP :func:`spectral_vjp` is the same map applied to the (symmetric) cotangent; it
  is the gradient of ⟨G, f(A)⟩ with respect to a symmetric A.
* Batching: leading axes of A and H broadcast against each other (NumPy rules).
* Second derivative: :func:`spectral_second` (symmetric H, K), :func:`spectral_hvp`; same input rules. There is no
  torch path (a double backward through ``eigh`` would differentiate eigenvectors, which are not unique at repeated
  eigenvalues).

Verification: ``tests/toporlogy/test_spectral.py`` (``docs/foundations/prereg/T_spectral.md``) and
``tests/toporlogy/test_spectral_second.py`` (``docs/foundations/prereg/T2_second_derivative.md``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from toporlogy.core import tensor as T

__all__ = ["SPECTRAL_FUNCTIONS", "spectral_function", "spectral_frechet", "spectral_vjp", "spectral_second",
           "spectral_hvp"]

#: supported scalar functions (``power`` needs the exponent ``p``)
SPECTRAL_FUNCTIONS = ("exp", "log", "sqrt", "inv_sqrt", "power")
_NEEDS_SPD = ("log", "sqrt", "inv_sqrt", "power")


def _check_spec(f: str, p: float | None) -> float | None:
    if f not in SPECTRAL_FUNCTIONS:
        raise ValueError(f"f must be one of {SPECTRAL_FUNCTIONS}, got {f!r}")
    if f == "power":
        if p is None or isinstance(p, bool) or not math.isfinite(float(p)):
            raise ValueError("f='power' needs a finite exponent p")
        return float(p)
    if p is not None:
        raise ValueError(f"p is only used with f='power', not with f={f!r}")
    return None


def _as_sym(A: Any, what: str) -> np.ndarray:
    a = np.asarray(A, dtype=float)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or a.shape[-1] not in (2, 3):
        raise ValueError(f"{what} must be (..., n, n) with n in (2, 3), got {a.shape}")
    if a.size == 0:
        raise ValueError(f"{what} must not be empty")
    return np.asarray(T.check_symmetric(a, T.SYMMETRY_RTOL, what))


def _eig(A: Any, f: str) -> tuple[np.ndarray, np.ndarray]:
    a = _as_sym(A, "A")
    w, Q = np.linalg.eigh(0.5 * (a + np.swapaxes(a, -1, -2)))
    if f in _NEEDS_SPD and not bool((w > 0.0).all()):
        bad = int((w <= 0.0).any(axis=-1).sum())
        raise ValueError(f"f={f!r} needs a symmetric positive definite A; {bad} matrix(es) with an eigenvalue ≤ 0 "
                         f"(smallest {float(w.min()):.3e}); no clipping is applied")
    return w, Q


def _values(w: np.ndarray, f: str, p: float | None) -> np.ndarray:
    if f == "exp":
        return np.exp(w)
    if f == "log":
        return np.log(w)
    if f == "sqrt":
        return np.sqrt(w)
    if f == "inv_sqrt":
        return 1.0 / np.sqrt(w)
    return w ** p


def _dd_pair(a: np.ndarray, b: np.ndarray, f: str, p: float | None) -> np.ndarray:
    """First divided difference f[a, b] (elementwise, broadcast) by the stable forms of the module notes."""
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    d = hi - lo
    eq = d == 0.0
    ds = np.where(eq, 1.0, d)                       # safe divisor; the equal entries are replaced below
    if f == "sqrt":
        return 1.0 / (np.sqrt(a) + np.sqrt(b))
    if f == "inv_sqrt":
        sa, sb = np.sqrt(a), np.sqrt(b)
        return -1.0 / (sa * sb * (sa + sb))
    if f == "log":
        return np.where(eq, 1.0 / hi, np.log1p(ds / lo) / ds)
    if f == "exp":
        return np.exp(hi) * np.where(eq, 1.0, -np.expm1(-ds) / ds)
    if p is None:
        raise ValueError(f"spectral function {f!r}: unknown, or power without an exponent")
    ell = -np.log1p(np.where(eq, 1.0, d / lo))      # ℓ = log(lo/hi) < 0 (placeholder where equal, replaced below)
    return np.where(eq, p * hi ** (p - 1.0), hi ** (p - 1.0) * np.expm1(p * ell) / np.expm1(ell))


def _divided_differences(w: np.ndarray, f: str, p: float | None) -> np.ndarray:
    """D_ij of the module notes for eigenvalues ``w`` (..., n) → (..., n, n)."""
    return _dd_pair(w[..., :, None], w[..., None, :], f, p)


def spectral_function(A: Any, f: str, p: float | None = None) -> np.ndarray:
    """f(A) of symmetric A (..., n, n), n ∈ {2, 3}; ``f`` ∈ {"exp", "log", "sqrt", "inv_sqrt", "power"} (``p`` the
    exponent of "power"). SPD is required for all but "exp" (ValueError otherwise, no clipping). Module notes for
    the contract."""
    p = _check_spec(f, p)
    w, Q = _eig(A, f)
    return (Q * _values(w, f, p)[..., None, :]) @ np.swapaxes(Q, -1, -2)


def _apply(A: Any, H: Any, f: str, p: float | None, what: str) -> np.ndarray:
    p = _check_spec(f, p)
    h = _as_sym(H, what)
    w, Q = _eig(A, f)
    if h.shape[-1] != w.shape[-1]:
        raise ValueError(f"{what} (n = {h.shape[-1]}) and A (n = {w.shape[-1]}) must have the same size")
    D = _divided_differences(w, f, p)
    Qt = np.swapaxes(Q, -1, -2)
    return Q @ (D * (Qt @ h @ Q)) @ Qt


def spectral_frechet(A: Any, H: Any, f: str, p: float | None = None) -> np.ndarray:
    """JVP: the Fréchet derivative L_A[H] = d/dt f(A + tH) at t = 0 for a **symmetric** direction H (..., n, n),
    broadcast against A. Exact at repeated and nearly repeated eigenvalues (module notes)."""
    return _apply(A, H, f, p, "H")


def spectral_vjp(A: Any, G: Any, f: str, p: float | None = None) -> np.ndarray:
    """VJP: the gradient of ⟨G, f(A)⟩ (Frobenius product) with respect to the symmetric A, for a **symmetric**
    cotangent G. L_A is self-adjoint on symmetric matrices, so this is L_A[G]; a general cotangent must be
    symmetrised by the caller (that is the gradient on the symmetric subspace)."""
    return _apply(A, G, f, p, "G")


# ---------------------------------------------------------------------------------------- second derivative
_TAYLOR_TERMS = 12          # series terms about the node mean where the spread is small
_EXP_SMALL = 1e-2           # exp: absolute spread below which the series is used
_REL_SMALL = 1e-3           # log, power: spread / mean below which the series is used


def _taylor_second(x: np.ndarray, f: str, p: float | None) -> np.ndarray:
    """f[x0, x1, x2] = Σ_k f^{(k+2)}(m)/(k+2)! h_k(x − m), m the mean of the nodes; ``x`` (..., 3)."""
    m = x.mean(axis=-1)
    d = x - m[..., None]
    if f == "exp":
        c = 0.5 * np.exp(m)
    elif f == "log":
        c = -0.5 / (m * m)
    else:
        if p is None:
            raise ValueError(f"spectral function {f!r}: unknown, or power without an exponent")
        c = 0.5 * p * (p - 1.0) * m ** (p - 2.0)
    pv = 0.0 if p is None else float(p)
    h1 = np.ones_like(m)
    h2 = np.ones_like(m)
    h3 = np.ones_like(m)
    out = c * h3
    for k in range(1, _TAYLOR_TERMS):
        j = k + 1                                   # c currently holds f^{(j)}(m)/j!
        if f == "exp":
            c = c / (j + 1)
        elif f == "log":
            c = -c * j / ((j + 1) * m)
        else:
            c = c * (pv - j) / ((j + 1) * m)
        h1 = h1 * d[..., 0]
        h2 = d[..., 1] * h2 + h1
        h3 = d[..., 2] * h3 + h2
        out = out + c * h3
    return out


def _second_divided_differences(w: np.ndarray, f: str, p: float | None) -> np.ndarray:
    """F_ijk = f[w_i, w_j, w_k] for eigenvalues ``w`` (..., n) → (..., n, n, n), stable at equal nodes."""
    xi, xj, xk = np.broadcast_arrays(w[..., :, None, None], w[..., None, :, None], w[..., None, None, :])
    if f == "sqrt":
        si, sj, sk = np.sqrt(xi), np.sqrt(xj), np.sqrt(xk)
        return -1.0 / ((si + sj) * (si + sk) * (sj + sk))
    if f == "inv_sqrt":
        si, sj, sk = np.sqrt(xi), np.sqrt(xj), np.sqrt(xk)
        return (si + sj + sk) / (si * sj * sk * (si + sj) * (si + sk) * (sj + sk))
    if f == "power" and p is None:
        raise ValueError("spectral function 'power' needs an exponent")
    x = np.sort(np.stack([xi, xj, xk], axis=-1), axis=-1)
    lo, mid, hi = x[..., 0], x[..., 1], x[..., 2]
    spread = hi - lo
    with np.errstate(all="ignore"):
        closed = (_dd_pair(hi, mid, f, p) - _dd_pair(mid, lo, f, p)) / np.where(spread == 0.0, 1.0, spread)
        series = _taylor_second(x, f, p)
    small = spread < _EXP_SMALL if f == "exp" else spread < _REL_SMALL * x.mean(axis=-1)
    return np.where(small, series, closed)


def _second(A: Any, H: Any, K: Any, f: str, p: float | None, names: tuple[str, str]) -> np.ndarray:
    p = _check_spec(f, p)
    h = _as_sym(H, names[0])
    k = _as_sym(K, names[1])
    w, Q = _eig(A, f)
    for arr, nm in ((h, names[0]), (k, names[1])):
        if arr.shape[-1] != w.shape[-1]:
            raise ValueError(f"{nm} (n = {arr.shape[-1]}) and A (n = {w.shape[-1]}) must have the same size")
    F = _second_divided_differences(w, f, p)
    Qt = np.swapaxes(Q, -1, -2)
    ht, kt = Qt @ h @ Q, Qt @ k @ Q
    M = np.einsum("...ijk,...ik,...kj->...ij", F, ht, kt) + np.einsum("...ijk,...ik,...kj->...ij", F, kt, ht)
    return Q @ M @ Qt


def spectral_second(A: Any, H: Any, K: Any, f: str, p: float | None = None) -> np.ndarray:
    """Second Fréchet derivative d²f(A)[H, K] = ∂²/∂s∂t f(A + sH + tK) at 0 for **symmetric** directions H, K
    (..., n, n), broadcast against A; symmetric in H and K, no factor ½. Exact at repeated and nearly repeated
    eigenvalues (module notes)."""
    return _second(A, H, K, f, p, ("H", "K"))


def spectral_hvp(A: Any, G: Any, H: Any, f: str, p: float | None = None) -> np.ndarray:
    """Hessian-vector product of φ(A) = ⟨G, f(A)⟩ (Frobenius product, symmetric G) in the symmetric direction H:
    d/dt ∇φ(A + tH) = d²f(A)[H, G], so ⟨K, hvp⟩ = d²φ(A)[H, K] and the Hessian is symmetric in (H, K)."""
    return _second(A, H, G, f, p, ("H", "G"))
