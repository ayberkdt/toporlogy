"""Directional elastic properties and Poisson-ratio analysis of linear elastic materials.

All functions take a Voigt **compliance** S (6, 6; engineering shear, :mod:`toporlogy.core.tensor`) in the frame in
which the directions are given, and unit vectors n (load direction) and m (transverse direction, m ⊥ n):

    1/E(n)   = S_ijkl n_i n_j n_k n_l                       Young's modulus for uniaxial stress along n
    ν(n, m)  = −E(n) S_ijkl m_i m_j n_k n_l                  contraction along m under uniaxial stress along n
    1/G(n,m) = 4 S_ijkl n_i m_j n_k m_l                      shear modulus for the shear stress σ_nm
    β(n)     = S_ijkk n_i n_j                                linear compressibility (strain along n per unit pressure;
                                                             negative values are admissible: NLC)
    β_V      = S_iijj                                        volumetric compressibility (= 1/K)

with S_ijkl the tensor compliance. For an isotropic material E(n) = E, ν(n, m) = ν, G(n, m) = G, β(n) = 1/(3K).

Extrema
-------
For a fixed n, m ranges over the unit circle of the plane ⊥ n; with an orthonormal basis P (3 × 2) of that plane and
A_ij(n) = S_ijkl n_k n_l, mᵀ A m is a quadratic form on the circle, so ν(n, ·) has its extrema at the eigenvectors
of the 2 × 2 matrix Pᵀ A P (:func:`poisson_extrema`, analytic). The same holds for G(n, ·) with
B_jl(n) = S_ijkl n_i n_k. Over n there is no closed form: :func:`directional_extrema` samples a Fibonacci sphere,
then refines the best samples by local minimisation (Nelder–Mead on spherical angles) from several starts. The result
is labelled ``"refined"`` (or ``"sampled"``) and is the best value found, **not** a proven global extremum.

Admissible anisotropic Poisson ratios are unbounded: for a positive definite S, the Cauchy–Schwarz inequality on the
energy inner product gives only |ν(n, m)| ≤ √(E(n)/E(m)), so |ν| can be arbitrarily large when E(n)/E(m) is
(Ting & Chen 2005, QJMAM 58(1) 73-82, doi:10.1093/qjmamj/hbh021: abstract read, text not available here; the
example family of ``tests/toporlogy/test_anisotropy.py`` is an orthotropic SPD family built for this purpose, not
catalogue data).

Sensitivities
-------------
:func:`isotropic_stiffness_derivatives` (∂C/∂E, ∂C/∂ν), :func:`compliance_derivative` (dS = −S dC S),
:func:`rotated_stiffness_derivative` (dC' for a change dR of the frame), :func:`young_modulus_derivative`
(dE(n) = −E² n⊗n⊗n⊗n : dS). These are derivatives of smooth maps inside the admissible set (S positive definite);
they blow up like the condition number near its boundary (ν → 1/2, ν → −1). :func:`cholesky_parameters` /
:func:`stiffness_from_cholesky` parametrise SPD stiffnesses as M_C = L Lᵀ (L lower triangular, log-diagonal), which
stays SPD for every parameter value, unlike the engineering constants.

Verification: ``tests/toporlogy/test_anisotropy.py`` (records P1-P11 of ``docs/olcumler/tensor_elastisite.md``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.linear_elastic import (check_positive_definite, fibonacci_sphere, isotropic_compliance,
                                           isotropic_moduli, isotropic_stiffness)
from toporlogy.references import cite

__all__ = ["young_modulus", "poisson_ratio", "shear_modulus", "linear_compressibility",
           "volumetric_compressibility", "plane_basis", "poisson_extrema", "shear_extrema", "directional_extrema",
           "QUANTITIES", "isotropic_sweep", "isotropic_stiffness_derivatives", "compliance_derivative",
           "rotated_stiffness_derivative", "young_modulus_derivative", "cholesky_parameters",
           "stiffness_from_cholesky"]

QUANTITIES = ("young", "poisson", "shear", "linear_compressibility")


def _S4(S6: Any) -> np.ndarray:
    S = np.asarray(S6, float)
    if S.shape != (6, 6):
        raise ValueError(f"S must be a Voigt compliance (6, 6), got {S.shape}")
    return T.compliance_to_tensor(T.check_symmetric(S, T.SYMMETRY_RTOL, "S"))


def _unit(n: Any, what: str) -> np.ndarray:
    v = np.asarray(n, float)
    if v.shape[-1] != 3:
        raise ValueError(f"{what} must be (..., 3), got {v.shape}")
    r = np.linalg.norm(v, axis=-1)
    if not np.all(np.abs(r - 1.0) <= 1e-10):
        raise ValueError(f"{what} must be unit vectors (max | |{what}| - 1 | = {np.abs(r - 1).max():.3e})")
    return v


def young_modulus(S6: Any, n: Any) -> np.ndarray:
    """E(n) (...,) (module notes)."""
    n = _unit(n, "n")
    return 1.0 / np.einsum("ijkl,...i,...j,...k,...l->...", _S4(S6), n, n, n, n)


def poisson_ratio(S6: Any, n: Any, m: Any) -> np.ndarray:
    """ν(n, m) (...,), m ⊥ n (module notes)."""
    n, m = _unit(n, "n"), _unit(m, "m")
    if not np.all(np.abs(np.einsum("...i,...i->...", n, m)) <= 1e-10):
        raise ValueError("m must be perpendicular to n")
    S4 = _S4(S6)
    return (-np.einsum("ijkl,...i,...j,...k,...l->...", S4, m, m, n, n)
            / np.einsum("ijkl,...i,...j,...k,...l->...", S4, n, n, n, n))


def shear_modulus(S6: Any, n: Any, m: Any) -> np.ndarray:
    """G(n, m) (...,), m ⊥ n (module notes)."""
    n, m = _unit(n, "n"), _unit(m, "m")
    if not np.all(np.abs(np.einsum("...i,...i->...", n, m)) <= 1e-10):
        raise ValueError("m must be perpendicular to n")
    return 1.0 / (4.0 * np.einsum("ijkl,...i,...j,...k,...l->...", _S4(S6), n, m, n, m))


def linear_compressibility(S6: Any, n: Any) -> np.ndarray:
    """β(n) = S_ijkk n_i n_j (...,)."""
    n = _unit(n, "n")
    return np.einsum("ijkk,...i,...j->...", _S4(S6), n, n)


def volumetric_compressibility(S6: Any) -> float:
    """β_V = S_iijj (sum of the upper-left 3 × 3 block of the Voigt compliance)."""
    return float(np.einsum("iijj->", _S4(S6)))


def plane_basis(n: Any) -> np.ndarray:
    """(..., 3, 2) orthonormal basis of the plane ⊥ n (columns p, q with p × q = n)."""
    n = _unit(n, "n")
    a = np.where((np.abs(n[..., 0]) < 0.9)[..., None], np.array([1.0, 0, 0]), np.array([0.0, 1, 0]))
    p = a - np.einsum("...i,...i->...", a, n)[..., None] * n
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    q = np.cross(n, p)
    return np.stack([p, q], axis=-1)


def _circle_extrema(A: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(w_lo, m_lo, w_hi, m_hi): extrema of mᵀ A m over unit m ⊥ n (2 × 2 eigenproblem)."""
    P = plane_basis(n)
    B = np.einsum("...ia,...ij,...jb->...ab", P, A, P)
    w, V = np.linalg.eigh(0.5 * (B + np.swapaxes(B, -1, -2)))
    m = np.einsum("...ia,...ab->...ib", P, V)
    return w[..., 0], m[..., :, 0], w[..., 1], m[..., :, 1]


@cite("ting_chen_2005", note="anisotropic Poisson ratio unbounded under positive definiteness; abstract read only")
def poisson_extrema(S6: Any, n: Any) -> dict[str, np.ndarray]:
    """Analytic extrema of ν(n, m) over m ⊥ n: {"nu_min", "m_min", "nu_max", "m_max"} (module notes)."""
    n = _unit(n, "n")
    S4 = _S4(S6)
    A = np.einsum("ijkl,...k,...l->...ij", S4, n, n)
    E = 1.0 / np.einsum("...ij,...i,...j->...", A, n, n)
    lo, m_lo, hi, m_hi = _circle_extrema(A, n)
    return {"nu_min": -E * hi, "m_min": m_hi, "nu_max": -E * lo, "m_max": m_lo}


def shear_extrema(S6: Any, n: Any) -> dict[str, np.ndarray]:
    """Analytic extrema of G(n, m) over m ⊥ n: {"G_min", "m_min", "G_max", "m_max"}."""
    n = _unit(n, "n")
    B = np.einsum("ijkl,...i,...k->...jl", _S4(S6), n, n)
    lo, m_lo, hi, m_hi = _circle_extrema(B, n)
    return {"G_min": 1.0 / (4.0 * hi), "m_min": m_hi, "G_max": 1.0 / (4.0 * lo), "m_max": m_lo}


def _sph(a: np.ndarray) -> np.ndarray:
    th, ph = a[..., 0], a[..., 1]
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def _ang(n: np.ndarray) -> np.ndarray:
    return np.array([math.acos(float(np.clip(n[2], -1.0, 1.0))), math.atan2(float(n[1]), float(n[0]))])


def _profile(S4: np.ndarray, quantity: str, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None,
                                                                    np.ndarray | None]:
    """(low, high, m_low, m_high) of ``quantity`` at unit directions n for a tensor compliance S4: for "poisson" /
    "shear" the analytic extrema over m ⊥ n (the formulas of the public functions, without their input checks)."""
    if quantity == "young":
        v = 1.0 / np.einsum("ijkl,...i,...j,...k,...l->...", S4, n, n, n, n)
        return v, v, None, None
    if quantity == "linear_compressibility":
        v = np.einsum("ijkk,...i,...j->...", S4, n, n)
        return v, v, None, None
    if quantity == "poisson":
        Am = np.einsum("ijkl,...k,...l->...ij", S4, n, n)
        E = 1.0 / np.einsum("...ij,...i,...j->...", Am, n, n)
        lo, m_lo, hi, m_hi = _circle_extrema(Am, n)
        return -E * hi, -E * lo, m_hi, m_lo
    if quantity == "shear":
        B = np.einsum("ijkl,...i,...k->...jl", S4, n, n)
        lo, m_lo, hi, m_hi = _circle_extrema(B, n)
        return 1.0 / (4.0 * hi), 1.0 / (4.0 * lo), m_hi, m_lo
    raise ValueError(f"quantity: one of {QUANTITIES}")


def directional_extrema(S6: Any, quantity: str, n_samples: int = 2000, n_starts: int = 8,
                        refine: bool = True) -> dict[str, Any]:
    """Smallest and largest ``quantity`` over unit n (and m ⊥ n for "poisson" / "shear"): Fibonacci sample of
    ``n_samples`` directions, then Nelder–Mead from the ``n_starts`` best samples (module notes).

    Returns {"min", "n_min", "m_min", "max", "n_max", "m_max", "sampled_min", "sampled_max", "label",
    "n_samples", "n_starts"}; ``label`` is "refined" or "sampled" (never a global guarantee)."""
    S4 = _S4(S6)
    N = fibonacci_sphere(n_samples)
    lo, hi, mlo, mhi = _profile(S4, quantity, N)
    out: dict[str, Any] = {"sampled_min": float(lo.min()), "sampled_max": float(hi.max()),
                           "n_samples": int(n_samples), "n_starts": int(n_starts),
                           "label": "refined" if refine else "sampled"}
    for kind, vals, sign in (("min", lo, 1.0), ("max", hi, -1.0)):
        order = np.argsort(sign * vals)
        best_n = N[order[0]]
        best = float(vals[order[0]])
        if refine:
            from scipy.optimize import minimize

            def f(a: np.ndarray, kind: str = kind, sign: float = sign) -> float:
                lo_, hi_, _, _ = _profile(S4, quantity, _sph(a)[None, :])
                return sign * float((lo_ if kind == "min" else hi_)[0])
            for k in order[:n_starts]:
                r = minimize(f, _ang(N[k]), method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-14 * max(1.0, abs(best)), "maxiter": 2000})
                if float(r.fun) < sign * best:              # r.fun = sign * value
                    best, best_n = sign * float(r.fun), _sph(r.x)
        _, _, ml, mh = _profile(S4, quantity, best_n[None, :])
        out[kind] = best
        out[f"n_{kind}"] = best_n
        mm = ml if kind == "min" else mh
        out[f"m_{kind}"] = None if mm is None else mm[0]
    return out


# ============================================================================================== isotropic sweep
def isotropic_sweep(nus: Any, fixed: str = "E", value: float = 1.0, rho: float | None = 1.0) -> list[dict[str, Any]]:
    """One row per ν of an isotropic material with ``fixed`` = "E" or "G" held at ``value`` (normalised, synthetic):
    moduli, K/G, Mandel eigenvalues and condition numbers (computed and closed form 3K/2G or its inverse), wave
    speeds c_L = √(M/ρ), c_T = √(G/ρ) (``rho`` None: omitted), strain energy density of a unit uniaxial strain
    (M/2) and of a unit engineering shear (G/2), and the float32 / float64 round-off of S = C⁻¹ (‖S C − I‖_max)."""
    if fixed not in ("E", "G"):
        raise ValueError("fixed: 'E' or 'G'")
    rows = []
    for nu in np.asarray(nus, float).ravel():
        m = isotropic_moduli(E=value, nu=nu) if fixed == "E" else isotropic_moduli(G=value, nu=nu)
        C = isotropic_stiffness(m.E, float(nu))
        S = isotropic_compliance(m.E, float(nu))
        wC = np.linalg.eigvalsh(T.stiffness_to_mandel(C))
        wS = np.linalg.eigvalsh(T.compliance_to_mandel(S))
        cond_cf = max(3 * m.K, 2 * m.G) / min(3 * m.K, 2 * m.G)
        eps_u = np.array([1.0, 0, 0, 0, 0, 0])
        eps_s = np.array([0.0, 0, 0, 0, 0, 1.0])
        rnd = {}
        for name, dt in (("float64", "float64"), ("float32", "float32")):
            Cd: Any = C.astype(dt)
            Sd = np.linalg.inv(Cd)
            rnd[name] = float(np.abs(Sd.astype(np.float64) @ C - np.eye(6)).max())
        row = {"nu": float(nu), "fixed": fixed, "E": m.E, "G": m.G, "K": m.K, "lam": m.lam, "M": m.M,
               "K_over_G": m.K / m.G, "mandel_C": wC.tolist(), "mandel_S": wS.tolist(),
               "cond_C": float(wC[-1] / wC[0]), "cond_S": float(wS[-1] / wS[0]), "cond_closed_form": cond_cf,
               "energy_uniaxial_strain": 0.5 * float(eps_u @ C @ eps_u),
               "energy_unit_shear": 0.5 * float(eps_s @ C @ eps_s),
               "inverse_error_float64": rnd["float64"], "inverse_error_float32": rnd["float32"]}
        if rho is not None:
            if not rho > 0:
                raise ValueError("rho > 0 is required")
            row["c_L"], row["c_T"] = math.sqrt(m.M / rho), math.sqrt(m.G / rho)
        rows.append(row)
    return rows


# ============================================================================================== sensitivities
def isotropic_stiffness_derivatives(E: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """(∂C/∂E, ∂C/∂ν) of :func:`toporlogy.core.linear_elastic.isotropic_stiffness` (Voigt 6 × 6):
    C = c [ν on, (1 − ν) diagonal of the normal block] + G [shear], c = E/((1 + ν)(1 − 2ν)), G = E/(2(1 + ν))."""
    C = isotropic_stiffness(E, nu)
    dE = C / E
    d = (1 + nu) * (1 - 2 * nu)
    c = E / d
    dc = E * (1 + 4 * nu) / d ** 2                 # d' = (1 - 2 nu) - 2 (1 + nu) = -(1 + 4 nu)
    dG = -E / (2 * (1 + nu) ** 2)
    dn = np.zeros((6, 6))
    dn[:3, :3] = dc * nu + c
    np.fill_diagonal(dn[:3, :3], dc * (1 - nu) - c)
    dn[3:, 3:] = np.eye(3) * dG
    return dE, dn


def compliance_derivative(S6: Any, dC6: Any) -> np.ndarray:
    """dS = −S dC S (Voigt; valid because S_voigt = C_voigt⁻¹ in the engineering convention)."""
    S = np.asarray(S6, float)
    return -S @ np.asarray(dC6, float) @ S


def rotated_stiffness_derivative(C6: Any, R: Any, dR: Any) -> np.ndarray:
    """d/dt of the Voigt stiffness R ⋆ C (:func:`toporlogy.core.tensor.rotate4`) for R(t) with dR/dt = ``dR``: the sum
    of the four terms with one R replaced by dR."""
    C4 = T.stiffness_to_tensor(np.asarray(C6, float))
    R = np.asarray(R, float)
    dR = np.asarray(dR, float)
    d = (np.einsum("ip,jq,kr,ls,pqrs->ijkl", dR, R, R, R, C4) + np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, dR, R, R, C4)
         + np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, dR, R, C4) + np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, dR, C4))
    return T.tensor_to_stiffness(d)


def young_modulus_derivative(S6: Any, n: Any, dS6: Any) -> np.ndarray:
    """dE(n) = −E(n)² n⊗n⊗n⊗n : dS for a change dS of the Voigt compliance."""
    n = _unit(n, "n")
    E = young_modulus(S6, n)
    return -E ** 2 * np.einsum("ijkl,...i,...j,...k,...l->...", T.compliance_to_tensor(np.asarray(dS6, float)),
                               n, n, n, n)


_TRIL = np.tril_indices(6, -1)


def cholesky_parameters(C6: Any) -> np.ndarray:
    """θ (21,) with :func:`stiffness_from_cholesky` (θ) = C6: log of the Cholesky diagonal of the Mandel matrix (6),
    then its strictly lower entries row by row (15). C6 must be SPD."""
    check_positive_definite(C6, "stiffness", "C")
    L = np.linalg.cholesky(T.stiffness_to_mandel(np.asarray(C6, float)))
    return np.concatenate([np.log(np.diag(L)), L[_TRIL]])


def stiffness_from_cholesky(theta: Any) -> np.ndarray:
    """Voigt stiffness of M_C = L Lᵀ, L lower triangular with diag exp(θ[:6]) and strict lower part θ[6:] (row by
    row): SPD for every finite θ (module notes)."""
    th = np.asarray(theta, float).reshape(21)
    L = np.zeros((6, 6))
    L[np.arange(6), np.arange(6)] = np.exp(th[:6])
    L[_TRIL] = th[6:]
    M = L @ L.T
    return T.mandel_to_stiffness(0.5 * (M + M.T))
