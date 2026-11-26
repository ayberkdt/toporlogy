"""Spectral functions of symmetric 2 × 2 and 3 × 3 tensors and their first (Fréchet) derivative.

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
* Second derivatives are **not implemented**; neither is a torch path (a double backward through ``eigh`` would
  differentiate eigenvectors, which are not unique at repeated eigenvalues).

Verification: ``tests/toporlogy/test_spectral.py`` (preregistration ``docs/foundations/prereg/T_spectral.md``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from toporlogy.core import tensor as T

__all__ = ["SPECTRAL_FUNCTIONS", "spectral_function", "spectral_frechet", "spectral_vjp"]

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


def _divided_differences(w: np.ndarray, f: str, p: float | None) -> np.ndarray:
    """D_ij of the module notes for eigenvalues ``w`` (..., n) → (..., n, n)."""
    a, b = w[..., :, None], w[..., None, :]
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
    ell = -np.log1p(np.where(eq, 1.0, d / lo))      # ℓ = log(lo/hi) < 0 (placeholder where equal, replaced below)
    return np.where(eq, p * hi ** (p - 1.0), hi ** (p - 1.0) * np.expm1(p * ell) / np.expm1(ell))


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
