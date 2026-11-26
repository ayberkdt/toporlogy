"""Linear elastic materials: isotropic, cubic, transversely isotropic, orthotropic and general anisotropic stiffness,
admissibility (positive definiteness, sampled strong ellipticity) and the conversions between engineering constants,
C and S.

Conventions are those of :mod:`toporlogy.core.tensor`: Voigt ``[xx, yy, zz, yz, xz, xy]``, engineering shear
strains, C_voigt = S_voigt⁻¹; eigenvalues and condition numbers from the Mandel form. Units: any consistent set
(nothing is assumed; E = 1 inputs are normalised values, not a unit).

Canonical form
--------------
:class:`ElasticTensor` holds one stiffness in **material axes** (``C_material``, Voigt 6 × 6) and the material
frame ``R`` (columns = material axes in global coordinates; None = aligned). Every constructor (engineering
constants, C or S) ends there; :meth:`ElasticTensor.stiffness` is the global (physical) stiffness. Construction
refuses a matrix that is not symmetric to :data:`toporlogy.core.tensor.SYMMETRY_RTOL` (no silent symmetrisation)
or not positive definite (:func:`check_positive_definite`).

Admissibility
-------------
* Positive definite strain energy (3-D, isotropic): G > 0 and K > 0, i.e. E > 0 and −1 < ν < 1/2. λ < 0 alone is
  admissible (every ν < 0 has λ < 0); auxetic ν is accepted. ν = 1/2 (incompressible) and ν = −1 are limits, not
  inputs: an incompressible material needs a mixed / constrained formulation and is refused here.
* Orthotropic: E_i > 0, G_ij > 0 and S positive definite. Reciprocity ν_ij / E_i = ν_ji / E_j (ν_ij: contraction
  along j under stress along i, ε_j = −ν_ij ε_i) is built in: S_ji = −ν_ij / E_i. A single ν_ij may exceed 1/2 or
  lie below −1 (only ν_ij² < E_i / E_j, and the 3 × 3 determinant condition, are needed): the isotropic bounds do not
  apply to anisotropic Poisson ratios.
* :func:`check_positive_definite`: symmetry, Cholesky of the Mandel matrix, and the scaled smallest eigenvalue
  λ_min / λ_max of the Mandel matrix (refused below 64 eps: numerically singular).
* Strong ellipticity (needed for real wave speeds and for well-posedness of the displacement problem) is weaker than
  positive definiteness: the acoustic tensor Q_ik(n) = C_ijkl n_j n_l must be positive definite for every unit n.
  :func:`strong_ellipticity` **samples** n (Fibonacci sphere) and refines the worst sample; its result is labelled
  ``"sampled"`` / ``"refined"`` and is not a global guarantee. Example (tested): isotropic λ = −1.5, μ = 1 is
  strongly elliptic (min(μ, λ + 2μ) = 0.5 > 0) but not positive definite (K = λ + 2μ/3 < 0). For a nonlinear
  material the same test applies to the consistent tangent of a stated stress / strain pair and configuration.

Verification: ``tests/toporlogy/test_linear_elastic.py`` (records L1-L7 of ``docs/olcumler/tensor_elastisite.md``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toporlogy.core import tensor as T

__all__ = ["NotPositiveDefiniteError", "check_positive_definite", "IsotropicModuli", "isotropic_moduli",
           "isotropic_stiffness", "isotropic_compliance", "isotropic_plane_stress", "isotropic_plane_strain",
           "plane_bulk_modulus", "orthotropic_compliance", "transversely_isotropic_compliance", "cubic_stiffness",
           "engineering_constants", "symmetry_class", "acoustic_tensor", "fibonacci_sphere", "strong_ellipticity",
           "wave_speeds", "ElasticTensor", "SINGULAR_RCOND", "poisson_uniaxial"]

#: scaled smallest Mandel eigenvalue below which a matrix counts as numerically singular
SINGULAR_RCOND = 64 * np.finfo(float).eps


class NotPositiveDefiniteError(ValueError):
    """An elasticity matrix whose strain energy is not positive for every strain."""


def check_positive_definite(A6: Any, kind: str = "stiffness", what: str = "C") -> dict[str, float]:
    """Diagnostics {"min_eigenvalue", "max_eigenvalue", "rcond", "condition_number"} of a Voigt stiffness or
    compliance (Mandel eigenvalues) if it is symmetric and positive definite (Cholesky succeeds and
    λ_min / λ_max > :data:`SINGULAR_RCOND`); raises ValueError (asymmetric) or :class:`NotPositiveDefiniteError`."""
    A = T.check_symmetric(np.asarray(A6, float), T.SYMMETRY_RTOL, what)
    if A.shape not in ((6, 6), (3, 3)):
        raise ValueError(f"{what} must be a Voigt matrix (6, 6) or (3, 3), got {A.shape}")
    M = T.stiffness_to_mandel(A) if kind == "stiffness" else T.compliance_to_mandel(A)
    M = 0.5 * (M + M.T)                            # rounding-level asymmetry only (checked above)
    w = np.linalg.eigvalsh(M)
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise NotPositiveDefiniteError(f"{what} is not positive definite (Cholesky fails; smallest Mandel eigenvalue "
                                       f"{w[0]:.6g})") from None
    rc = float(w[0] / w[-1])
    if not rc > SINGULAR_RCOND:
        raise NotPositiveDefiniteError(f"{what} is numerically singular (λ_min / λ_max = {rc:.3e})")
    return {"min_eigenvalue": float(w[0]), "max_eigenvalue": float(w[-1]), "rcond": rc,
            "condition_number": float(w[-1] / w[0])}


# ============================================================================================== isotropic
@dataclass(frozen=True)
class IsotropicModuli:
    """Every isotropic modulus of one material: E, ν, G = μ, K, λ, M = λ + 2μ (P-wave modulus)."""

    E: float
    nu: float
    G: float
    K: float
    lam: float
    M: float

    @property
    def mu(self) -> float:
        return self.G


def isotropic_moduli(**two: float) -> IsotropicModuli:
    """All moduli from exactly two of E, nu, G (or mu), K, lam, M; refuses G ≤ 0 or K ≤ 0 (module notes).

    Supported pairs: (E, nu), (E, G), (E, K), (G, K), (lam, G), (G, M), (K, nu), (G, nu), (lam, nu)."""
    if "mu" in two:
        if "G" in two:
            raise ValueError("give G or mu, not both")
        two["G"] = two.pop("mu")
    keys = frozenset(two)
    if len(keys) != 2:
        raise ValueError(f"give exactly two moduli, got {sorted(keys)}")
    v = {k: float(x) for k, x in two.items()}
    if not all(math.isfinite(x) for x in v.values()):
        raise ValueError("moduli must be finite")
    if keys == {"E", "nu"}:
        E, nu = v["E"], v["nu"]
        if not (E > 0 and -1 < nu < 0.5):
            raise ValueError("E > 0 and -1 < nu < 0.5 are required (nu = 0.5 / -1 are limits, not inputs)")
        G, K = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    elif keys == {"E", "G"}:
        E, G = v["E"], v["G"]
        if not (E > 0 and G > 0):
            raise ValueError("E > 0 and G > 0 are required")
        K = E * G / (3 * (3 * G - E))
    elif keys == {"E", "K"}:
        E, K = v["E"], v["K"]
        if not (E > 0 and K > 0):
            raise ValueError("E > 0 and K > 0 are required")
        G = 3 * K * E / (9 * K - E)
    elif keys == {"G", "K"}:
        G, K = v["G"], v["K"]
    elif keys == {"lam", "G"}:
        G, K = v["G"], v["lam"] + 2 * v["G"] / 3
    elif keys == {"G", "M"}:
        G, K = v["G"], v["M"] - 4 * v["G"] / 3
    elif keys == {"K", "nu"}:
        K, nu = v["K"], v["nu"]
        if not -1 < nu < 0.5:
            raise ValueError("-1 < nu < 0.5 is required")
        G = 3 * K * (1 - 2 * nu) / (2 * (1 + nu))
    elif keys == {"G", "nu"}:
        G, nu = v["G"], v["nu"]
        if not -1 < nu < 0.5:
            raise ValueError("-1 < nu < 0.5 is required")
        K = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))
    elif keys == {"lam", "nu"}:
        lam, nu = v["lam"], v["nu"]
        if not (-1 < nu < 0.5 and nu != 0):
            raise ValueError("-1 < nu < 0.5, nu != 0 is required (lam = 0 at nu = 0 does not fix the scale)")
        G = lam * (1 - 2 * nu) / (2 * nu)
        K = lam * (1 + nu) / (3 * nu)
    else:
        raise ValueError(f"unsupported pair {sorted(keys)}")
    if not (G > 0 and K > 0) or not (math.isfinite(G) and math.isfinite(K)):
        raise ValueError(f"not positive definite: G = {G:.6g}, K = {K:.6g} (G > 0 and K > 0 are required)")
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    if "E" in v:
        E = v["E"]
    if "nu" in v:
        nu = v["nu"]
    lam = K - 2 * G / 3
    return IsotropicModuli(E=E, nu=nu, G=G, K=K, lam=lam, M=lam + 2 * G)


def _check_e_nu(E: float, nu: float) -> None:
    if not -1 < nu < 0.5:
        raise ValueError("-1 < nu < 0.5 is required")
    if not E > 0:
        raise ValueError("E > 0 is required")


def isotropic_stiffness(E: float, nu: float) -> np.ndarray:
    """3-D isotropic Voigt stiffness (6, 6). Written as c = E / ((1 + ν)(1 − 2ν)), c ν off the diagonal,
    c (1 − ν) on it, E / (2(1 + ν)) for shear: the arithmetic of the library's unit-modulus ``D_iso``."""
    _check_e_nu(E, nu)
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = c * nu
    np.fill_diagonal(D[:3, :3], c * (1 - nu))
    D[3:, 3:] = np.eye(3) * (E / (2 * (1 + nu)))
    return D


def isotropic_compliance(E: float, nu: float) -> np.ndarray:
    """3-D isotropic Voigt compliance (6, 6), engineering shear: 1/E, −ν/E, 1/G."""
    _check_e_nu(E, nu)
    S = np.full((3, 3), -nu / E)
    np.fill_diagonal(S, 1.0 / E)
    out = np.zeros((6, 6))
    out[:3, :3] = S
    out[3:, 3:] = np.eye(3) * (2 * (1 + nu) / E)
    return out


def isotropic_plane_stress(E: float, nu: float) -> np.ndarray:
    """In-plane (3, 3) [xx, yy, xy] stiffness with σzz = σyz = σxz = 0."""
    _check_e_nu(E, nu)
    return np.array([[E, E * nu, 0.0], [E * nu, E, 0.0], [0.0, 0.0, E * (1 - nu) / 2]]) / (1 - nu ** 2)


def isotropic_plane_strain(E: float, nu: float) -> np.ndarray:
    """In-plane (3, 3) [xx, yy, xy] stiffness with εzz = εyz = εxz = 0."""
    _check_e_nu(E, nu)
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, (1 - 2 * nu) / 2]])


def plane_bulk_modulus(E: float, nu: float, plane: str = "stress") -> float:
    """2-D (in-plane) bulk modulus κ, σ_αα / 2 = κ ε_αα: plane stress E / (2(1 − ν)); plane strain λ + μ =
    E / (2(1 + ν)(1 − 2ν))."""
    _check_e_nu(E, nu)
    if plane == "stress":
        return E / (2 * (1 - nu))
    if plane == "strain":
        return E / (2 * (1 + nu) * (1 - 2 * nu))
    raise ValueError("plane: 'stress' or 'strain'")


# ============================================================================================== anisotropic
def orthotropic_compliance(E1: float, E2: float, E3: float, G12: float, G13: float, G23: float,
                           nu12: float, nu13: float, nu23: float) -> np.ndarray:
    """S (6, 6) in the material axes 1, 2, 3 (Voigt [11, 22, 33, 23, 13, 12], engineering shear); S_ji = −ν_ij/E_i
    (reciprocity built in). Only E_i > 0 and G_ij > 0 are checked here; positive definiteness by the caller."""
    vals = {"E1": E1, "E2": E2, "E3": E3, "G12": G12, "G13": G13, "G23": G23}
    for k, x in vals.items():
        if not float(x) > 0:
            raise ValueError(f"{k} > 0 is required")
    E1, E2, E3 = float(E1), float(E2), float(E3)
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -float(nu12) / E1
    S[0, 2] = S[2, 0] = -float(nu13) / E1
    S[1, 2] = S[2, 1] = -float(nu23) / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / float(G23), 1 / float(G13), 1 / float(G12)
    return S


# proper cyclic permutations whose third column (the local symmetry axis z) is the global axis
_AXIS_PERM = {2: np.eye(3), 0: np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]),
              1: np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])}


def transversely_isotropic_compliance(E_p: float, E_t: float, nu_p: float, nu_pt: float, G_t: float,
                                      axis: int = 2) -> np.ndarray:
    """S (6, 6) of a transversely isotropic material with symmetry axis ``axis`` (0, 1, 2 = x, y, z): in-plane
    modulus E_p and Poisson ratio ν_p (G_p = E_p / (2(1 + ν_p)) follows), axial modulus E_t, ν_pt (stress in the
    plane, contraction along the axis), transverse shear modulus G_t. Five independent constants."""
    if not -1 < nu_p < 1:
        raise ValueError("-1 < nu_p < 1 is required")
    S = orthotropic_compliance(E_p, E_p, E_t, E_p / (2 * (1 + nu_p)), G_t, G_t, nu_p, nu_pt, nu_pt)
    if axis == 2:
        return S
    if axis not in _AXIS_PERM:
        raise ValueError("axis: 0, 1 or 2")
    Rp = _AXIS_PERM[axis]                          # proper cyclic permutation: local z -> global axis
    return T.tensor_to_compliance(T.rotate4(T.compliance_to_tensor(S), Rp))


def cubic_stiffness(C11: float, C12: float, C44: float) -> np.ndarray:
    """C (6, 6) of cubic symmetry in its crystal axes (three constants; C44 is the Voigt shear stiffness)."""
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3:, 3:] = np.eye(3) * C44
    return C


def engineering_constants(S6: Any) -> dict[str, Any]:
    """Engineering constants of a Voigt compliance in its own frame: E1, E2, E3 = 1/S_ii, G23, G13, G12 = 1/S_44,55,66,
    nu_ij = −S_ji / S_ii for i ≠ j (all six, so reciprocity can be checked), and ``coupling`` = the components of S
    that vanish for orthotropy in this frame (max |.| over max |S|)."""
    S = T.check_symmetric(np.asarray(S6, float), T.SYMMETRY_RTOL, "S")
    out: dict[str, Any] = {"E1": 1 / S[0, 0], "E2": 1 / S[1, 1], "E3": 1 / S[2, 2],
                           "G23": 1 / S[3, 3], "G13": 1 / S[4, 4], "G12": 1 / S[5, 5]}
    for i in range(3):
        for j in range(3):
            if i != j:
                out[f"nu{i + 1}{j + 1}"] = -S[j, i] / S[i, i]
    mask = np.ones((6, 6), bool)
    mask[:3, :3] = False
    mask[np.arange(6), np.arange(6)] = False
    out["coupling"] = float(np.abs(S[mask]).max() / np.abs(S).max())
    return out


def poisson_uniaxial(C: Any, load: int, transverse: int) -> Any:
    """Poisson ratio ν_{load,transverse} = −ε_transverse / ε_load under uniaxial stress along ``load`` of the stiffness
    ``C`` (Voigt, engineering shear; 3×3 in 2-D, 6×6 in 3-D; NumPy or torch, differentiable in torch). With
    S = C⁻¹: ν = −S[transverse, load] / S[load, load], the column S[:, load] from a linear solve of C x = e_load (no
    explicit inverse). ``load`` and ``transverse`` are distinct normal components (2-D: 0, 1; 3-D: 0, 1, 2). The
    stiffness ratio C12/C11 is not this quantity: for an uncoupled 2-D C it is ν21, and with normal–shear coupling it
    is no Poisson ratio at all (``docs/foundations/prereg/E04_poisson.md``)."""
    nv = int(C.shape[-1])
    if tuple(C.shape[-2:]) != (nv, nv) or nv not in (3, 6):
        raise ValueError(f"C must be (3, 3) or (6, 6) Voigt, got {tuple(C.shape)}")
    nn = 2 if nv == 3 else 3
    i, j = int(load), int(transverse)
    if i == j or not (0 <= i < nn and 0 <= j < nn):
        raise ValueError(f"load and transverse must be distinct normal components in 0..{nn - 1}, got {load}, "
                         f"{transverse}")
    if T._is_torch(C):
        import torch
        e = torch.zeros(nv, dtype=C.dtype, device=C.device)
        e[i] = 1.0
        x = torch.linalg.solve(C, e)
    else:
        e = np.zeros(nv)
        e[i] = 1.0
        x = np.linalg.solve(np.asarray(C, float), e)
    return -x[j] / x[i]


def symmetry_class(C6: Any, rtol: float = 1e-10) -> str:
    """Material symmetry of a Voigt stiffness **in the given frame**: "isotropic", "cubic", "transversely_isotropic_x"
    | "_y" | "_z", "orthotropic" or "anisotropic" (lower symmetry, or a higher one seen in a rotated frame; except
    isotropy, which no rotation hides)."""
    C = T.check_symmetric(np.asarray(C6, float), T.SYMMETRY_RTOL, "C")
    s = float(np.abs(C).max())
    tol = rtol * s

    def eq(a: float, b: float) -> bool:
        return abs(a - b) <= tol
    mask = np.ones((6, 6), bool)
    mask[:3, :3] = False
    mask[np.arange(6), np.arange(6)] = False
    if np.abs(C[mask]).max() > tol:
        return "anisotropic"
    d, o, g = np.diag(C)[:3], (C[1, 2], C[0, 2], C[0, 1]), np.diag(C)[3:]
    if (eq(d[0], d[1]) and eq(d[1], d[2]) and eq(o[0], o[1]) and eq(o[1], o[2])
            and eq(g[0], g[1]) and eq(g[1], g[2])):
        return "isotropic" if eq(g[0], 0.5 * (d[0] - o[0])) else "cubic"
    for ax in range(3):                           # isotropy plane ⊥ axis ax: in-plane indices p, q
        p, q = [k for k in range(3) if k != ax]
        # Voigt shear of the plane (p, q) is index 3 + ax; the two transverse shears are 3 + p, 3 + q
        if (eq(d[p], d[q]) and eq(C[ax, p], C[ax, q]) and eq(g[p], g[q])
                and eq(g[ax], 0.5 * (d[p] - C[p, q]))):
            return f"transversely_isotropic_{'xyz'[ax]}"
    return "orthotropic"


def acoustic_tensor(C6: Any, n: Any) -> np.ndarray:
    """Q_ik(n) = C_ijkl n_j n_l (..., 3, 3) for unit directions n (..., 3) (C need not be positive definite)."""
    C4 = T.stiffness_to_tensor(np.asarray(C6, float))
    n = np.asarray(n, float)
    return np.einsum("ijkl,...j,...l->...ik", C4, n, n)


def fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) nearly uniform unit vectors (golden-angle spiral); deterministic."""
    if n < 1:
        raise ValueError("n >= 1 is required")
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _sph(a: np.ndarray) -> np.ndarray:
    th, ph = a[..., 0], a[..., 1]
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def strong_ellipticity(C6: Any, n_directions: int = 2000, refine: bool = True) -> dict[str, Any]:
    """Smallest eigenvalue of the acoustic tensor over unit n: Fibonacci sample, then (``refine``) a local
    minimisation from the worst sample. {"min_eigenvalue", "direction", "label": "sampled" | "refined",
    "n_directions"}; a positive value is evidence, not a proof, of strong ellipticity (module notes)."""
    C6 = np.asarray(C6, float)
    N = fibonacci_sphere(n_directions)
    lam = np.linalg.eigvalsh(acoustic_tensor(C6, N))[:, 0]
    k = int(np.argmin(lam))
    best, nbest, label = float(lam[k]), N[k], "sampled"
    if refine:
        from scipy.optimize import minimize

        def f(a: np.ndarray) -> float:
            return float(np.linalg.eigvalsh(acoustic_tensor(C6, _sph(a)))[0])
        a0 = np.array([math.acos(np.clip(nbest[2], -1, 1)), math.atan2(nbest[1], nbest[0])])
        r = minimize(f, a0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14 * max(1.0, abs(best))})
        if r.fun < best:
            best, nbest = float(r.fun), _sph(r.x)
        label = "refined"
    return {"min_eigenvalue": best, "direction": nbest, "label": label, "n_directions": int(n_directions)}


def wave_speeds(C6: Any, n: Any, rho: float) -> np.ndarray:
    """Plane-wave phase speeds √(eig Q(n) / ρ) (..., 3), ascending; NaN where an eigenvalue is negative."""
    if not rho > 0:
        raise ValueError("rho > 0 is required")
    w = np.linalg.eigvalsh(acoustic_tensor(C6, n))
    with np.errstate(invalid="ignore"):
        return np.sqrt(w / rho)


# ============================================================================================== canonical form
@dataclass(frozen=True, eq=False)
class ElasticTensor:
    """A linear elastic material: stiffness ``C_material`` (Voigt 6 × 6) in its own axes and the material frame
    ``R`` (columns = material axes in global coordinates; None = aligned). Symmetric and positive definite (module
    notes); ``diagnostics`` holds the Mandel eigenvalue summary of :func:`check_positive_definite`."""

    C_material: np.ndarray
    R: np.ndarray | None = None
    diagnostics: dict[str, float] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        C = np.array(self.C_material, float)
        if C.shape != (6, 6):
            raise ValueError(f"C_material must be (6, 6), got {C.shape}")
        diag = check_positive_definite(C, "stiffness", "C")
        C.setflags(write=False)
        object.__setattr__(self, "C_material", C)
        if self.R is not None:
            R = np.array(T.check_rotation(self.R), float)
            R.setflags(write=False)
            object.__setattr__(self, "R", R)
        object.__setattr__(self, "diagnostics", diag)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_stiffness(cls, C6: Any, R: Any = None) -> ElasticTensor:
        return cls(np.asarray(C6, float), R)

    @classmethod
    def from_compliance(cls, S6: Any, R: Any = None) -> ElasticTensor:
        """From a Voigt compliance in material axes (symmetric and positive definite, checked before inversion)."""
        S = np.asarray(S6, float)
        check_positive_definite(S, "compliance", "S")
        C = np.linalg.inv(S)
        return cls(0.5 * (C + C.T), R)

    @classmethod
    def isotropic(cls, E: float, nu: float) -> ElasticTensor:
        return cls(isotropic_stiffness(E, nu))

    @classmethod
    def cubic(cls, C11: float, C12: float, C44: float, R: Any = None) -> ElasticTensor:
        return cls(cubic_stiffness(C11, C12, C44), R)

    @classmethod
    def transversely_isotropic(cls, E_p: float, E_t: float, nu_p: float, nu_pt: float, G_t: float, axis: int = 2,
                               R: Any = None) -> ElasticTensor:
        return cls.from_compliance(transversely_isotropic_compliance(E_p, E_t, nu_p, nu_pt, G_t, axis), R)

    @classmethod
    def orthotropic(cls, E1: float, E2: float, E3: float, G12: float, G13: float, G23: float, nu12: float,
                    nu13: float, nu23: float, R: Any = None) -> ElasticTensor:
        return cls.from_compliance(orthotropic_compliance(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23), R)

    # ------------------------------------------------------------------ views
    def stiffness(self) -> np.ndarray:
        """Global Voigt stiffness (material frame rotated by R)."""
        if self.R is None:
            return np.array(self.C_material)
        C = T.tensor_to_stiffness(T.rotate4(T.stiffness_to_tensor(self.C_material), self.R))
        return 0.5 * (C + C.T)

    def compliance(self) -> np.ndarray:
        """Global Voigt compliance (engineering shear)."""
        S = np.linalg.inv(self.stiffness())
        return 0.5 * (S + S.T)

    def compliance_material(self) -> np.ndarray:
        S = np.linalg.inv(self.C_material)
        return 0.5 * (S + S.T)

    def tensor(self) -> np.ndarray:
        """Global C_ijkl (3, 3, 3, 3)."""
        return T.stiffness_to_tensor(self.stiffness())

    def mandel(self) -> np.ndarray:
        return T.stiffness_to_mandel(self.stiffness())

    def eigenvalues(self) -> np.ndarray:
        """Mandel eigenvalues (frame independent)."""
        return np.linalg.eigvalsh(T.stiffness_to_mandel(self.C_material))

    def condition_number(self) -> float:
        w = self.eigenvalues()
        return float(w[-1] / w[0])

    def engineering_constants(self) -> dict[str, Any]:
        """Engineering constants in the material axes (:func:`engineering_constants`)."""
        return engineering_constants(self.compliance_material())

    def symmetry_class(self, rtol: float = 1e-10) -> str:
        """Symmetry in the material axes (:func:`symmetry_class`)."""
        return symmetry_class(self.C_material, rtol)

    def rotated(self, R: Any) -> ElasticTensor:
        """The same material with its frame turned by R (new frame R · self.R)."""
        Rn = T.check_rotation(R)
        return ElasticTensor(self.C_material, Rn if self.R is None else Rn @ self.R)

    def strong_ellipticity(self, n_directions: int = 2000) -> dict[str, Any]:
        return strong_ellipticity(self.stiffness(), n_directions)
