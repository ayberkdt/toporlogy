"""Element matrices of the regular grid (unit cube / unit square, E = 1).

* ``H8``: trilinear brick, 2×2×2 Gauss integration (the element of top3d, Liu & Tovar 2014). Locks in bending when a
  member is only one or two elements thick.
* ``H8IM``: the same brick with the nine Wilson incompatible modes ``1 − ξ², 1 − η², 1 − ζ²`` per displacement
  component, statically condensed (Taylor, Beresford & Wilson 1976; Cook et al. 2002 §6.7). All elements are the
  same cube, so the Jacobian is constant and the element passes the patch test without Taylor's correction.
* ``Q4``: bilinear plane-stress square, closed form of top88 (Andreassen et al. 2011, lines 8-12). Locks in
  bending like H8 (Ferrari & Sigmund 2019, Fig. 1, report +49 % on the Euler load of a column 4 elements thick).
* ``Q4IM``: the bilinear square with Wilson's two incompatible modes ``1 − ξ², 1 − η²`` per component ("Q6",
  Wilson et al. 1973), statically condensed exactly as H8IM; constant Jacobian, so the patch test holds. It
  represents pure bending of the square exactly (``tests/toporlogy/test_material.py``).

The isotropic matrices ``D_iso``, ``D_plane_stress``, ``D_plane_strain`` are the unit-modulus (E = 1) cases of
:mod:`toporlogy.core.linear_elastic` (one implementation). Any other elasticity (orthotropic, rotated, plane strain;
:mod:`toporlogy.core.material`) goes through
:func:`element_matrix_from_D`: the same 2×2(×2) Gauss rule with the given unit-modulus D (H8IM condensed as above).
For the isotropic plane-stress D it reproduces ``ke_q4`` to rounding, for D_iso ``ke_h8`` exactly (same code path).
A different D per element (fibre orientation field) goes through :func:`element_matrix_field`: the distinct D
matrices are found exactly and :func:`element_matrix_from_D` runs once per distinct D (incompatible modes condensed
per element), so elements sharing a D share the matrix bit for bit; :func:`indexed_matmul` applies such an indexed
set of matrices to element vectors in chunks.

A physical cube of edge h and modulus E has stiffness ``E·h·KE``; a square of thickness t has ``E·t·KE``.
Voigt order in 3-D: [εx, εy, εz, γyz, γxz, γxy]; in 2-D: [εx, εy, γxy].
Corner order 3-D: (−1,−1,−1), (1,−1,−1), (1,1,−1), (−1,1,−1), then the same at ζ = +1 (:data:`H8_CORNER`);
2-D: counter-clockwise from the lower-left corner.

Ported from the repository's validated element code (patch test 3e-14, cantilever 0.986–0.990 of Timoshenko for
H8IM); the library keeps its own copy so it does not depend on application packages.
"""
from __future__ import annotations

import math

import numpy as np

from toporlogy.core.linear_elastic import (NotPositiveDefiniteError, check_positive_definite, isotropic_plane_strain,
                                           isotropic_plane_stress, isotropic_stiffness)
from toporlogy.references import cite

__all__ = ["H8_CORNER", "Q4_CORNER", "D_iso", "D_plane_stress", "D_plane_strain", "ke_h8", "ke_h8_from_D", "ke_q4",
           "ke_q4_from_D", "ke_q4im_from_D", "element_matrix", "element_matrix_from_D", "b_centroid_h8",
           "b_centroid_q4", "element_matrix_field", "mode_recovery_from_D", "indexed_matmul", "ELEMENTS_2D",
           "ELEMENTS_3D"]

ELEMENTS_2D = ("Q4", "Q4IM")
ELEMENTS_3D = ("H8", "H8IM")

H8_CORNER = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
Q4_CORNER = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_G = 1.0 / math.sqrt(3.0)
_GAUSS3 = np.array([[a, b, c] for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)])


def D_iso(nu: float) -> np.ndarray:
    """Isotropic 3-D elasticity matrix for E = 1 (Voigt [εx, εy, εz, γyz, γxz, γxy])."""
    return isotropic_stiffness(1.0, nu)


def D_plane_stress(nu: float) -> np.ndarray:
    """Plane-stress elasticity matrix for E = 1 (Voigt [εx, εy, γxy])."""
    return isotropic_plane_stress(1.0, nu)


def D_plane_strain(nu: float) -> np.ndarray:
    """Plane-strain elasticity matrix for E = 1 (Voigt [εx, εy, γxy]; εz = 0)."""
    return isotropic_plane_strain(1.0, nu)


def _check_D(D: np.ndarray, n: int) -> np.ndarray:
    """D (n, n) if symmetric and positive definite (:func:`toporlogy.core.linear_elastic.check_positive_definite`)."""
    D = np.asarray(D, float)
    if D.shape != (n, n):
        raise ValueError(f"D must be {n} x {n}, got {D.shape}")
    try:
        check_positive_definite(D, "stiffness", "D")
    except NotPositiveDefiniteError as e:
        raise ValueError(f"D must be positive definite: {e}") from None
    except ValueError as e:
        raise ValueError(f"D must be symmetric: {e}") from None
    return D


def _B_from_grad(dN: np.ndarray) -> np.ndarray:
    n = dN.shape[1]
    B = np.zeros((6, 3 * n))
    B[0, 0::3] = dN[0]
    B[1, 1::3] = dN[1]
    B[2, 2::3] = dN[2]
    B[3, 1::3] = dN[2]
    B[3, 2::3] = dN[1]
    B[4, 0::3] = dN[2]
    B[4, 2::3] = dN[0]
    B[5, 0::3] = dN[1]
    B[5, 1::3] = dN[0]
    return B


def _B_unit_h8(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B_u (6, 24), B_α (6, 9)) of the unit cube at natural point p (∂/∂x = 2 ∂/∂ξ)."""
    s = H8_CORNER
    a = 1 + s * p
    dN = 0.125 * np.vstack([s[:, 0] * a[:, 1] * a[:, 2], s[:, 1] * a[:, 0] * a[:, 2],
                            s[:, 2] * a[:, 0] * a[:, 1]]) * 2.0
    dP = np.diag(-2.0 * p * 2.0)
    return _B_from_grad(dN), _B_from_grad(dP)


@cite("taylor_beresford_wilson_1976", "cook_2002", "liu_tovar_2014",
      note="H8 and H8 with incompatible modes, statically condensed")
def ke_h8(nu: float = 0.3, element: str = "H8") -> tuple[np.ndarray, np.ndarray | None]:
    """Unit-cube stiffness (24, 24) for E = 1 and, for ``"H8IM"``, the mode recovery matrix R (9, 24), α = R u."""
    return ke_h8_from_D(D_iso(nu), element)


def ke_h8_from_D(D: np.ndarray, element: str = "H8") -> tuple[np.ndarray, np.ndarray | None]:
    """As :func:`ke_h8` for any unit-modulus 6 × 6 elasticity matrix D (orthotropic, rotated)."""
    if element not in ("H8", "H8IM"):
        raise ValueError("element must be 'H8' or 'H8IM'")
    D = _check_D(D, 6)
    Kuu, Kua, Kaa = np.zeros((24, 24)), np.zeros((24, 9)), np.zeros((9, 9))
    for p in _GAUSS3:
        Bu, Ba = _B_unit_h8(p)
        w = 0.125                                  # det J = 1/8, weight 1
        Kuu += Bu.T @ D @ Bu * w
        Kua += Bu.T @ D @ Ba * w
        Kaa += Ba.T @ D @ Ba * w
    if element == "H8":
        return Kuu, None
    R = -np.linalg.solve(Kaa, Kua.T)
    K = Kuu + Kua @ R
    return 0.5 * (K + K.T), R


@cite("andreassen_2011", note="top88, lines 8-12")
def ke_q4(nu: float = 0.3) -> np.ndarray:
    """Unit-square bilinear plane-stress stiffness (8, 8) for E = 1, t = 1 (DOF order of top88)."""
    if not -1 < nu < 0.5:
        raise ValueError("−1 < ν < 0.5 is required")
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float)
    return 1 / (1 - nu ** 2) / 24 * (np.block([[A11, A12], [A12.T, A11]])
                                     + nu * np.block([[B11, B12], [B12.T, B11]]))


def ke_q4_from_D(D: np.ndarray) -> np.ndarray:
    """Unit-square bilinear stiffness (8, 8), t = 1, for any unit-modulus 3 × 3 in-plane D (plane stress or plane
    strain, orthotropic); 2 × 2 Gauss rule, DOF order of :func:`ke_q4`."""
    D = _check_D(D, 3)
    K = np.zeros((8, 8))
    s = Q4_CORNER
    for p in ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)):
        a = 1 + s * np.asarray(p)
        dN = 0.25 * np.vstack([s[:, 0] * a[:, 1], s[:, 1] * a[:, 0]]) * 2.0      # ∂/∂x = 2 ∂/∂ξ
        B = np.zeros((3, 8))
        B[0, 0::2] = dN[0]
        B[1, 1::2] = dN[1]
        B[2, 0::2] = dN[1]
        B[2, 1::2] = dN[0]
        K += B.T @ D @ B * 0.25                                                 # det J = 1/4, weight 1
    return 0.5 * (K + K.T)


def _B_unit_q4(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B_u (3, 8), B_α (3, 4)) of the unit square at natural point p; α = [a1_u, a1_v, a2_u, a2_v] are the
    amplitudes of the modes 1 − ξ², 1 − η² (∂/∂x = 2 ∂/∂ξ)."""
    s = Q4_CORNER
    a = 1 + s * p
    dN = 0.25 * np.vstack([s[:, 0] * a[:, 1], s[:, 1] * a[:, 0]]) * 2.0
    dP = np.diag(-2.0 * p * 2.0)                          # ∂(1 − ξ²)/∂x, ∂(1 − η²)/∂y; cross terms are 0

    def bmat(g: np.ndarray) -> np.ndarray:
        n = g.shape[1]
        B = np.zeros((3, 2 * n))
        B[0, 0::2] = g[0]
        B[1, 1::2] = g[1]
        B[2, 0::2] = g[1]
        B[2, 1::2] = g[0]
        return B
    return bmat(dN), bmat(dP)


@cite("wilson_1973", note="Q6: bilinear square with two incompatible modes per component, condensed")
def ke_q4im_from_D(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-square Q4 with incompatible modes (``Q4IM``), t = 1, for a unit-modulus 3 × 3 D: (K (8, 8), R (4, 8))
    with the mode amplitudes α = R u."""
    D = _check_D(D, 3)
    Kuu, Kua, Kaa = np.zeros((8, 8)), np.zeros((8, 4)), np.zeros((4, 4))
    for p in ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)):
        Bu, Ba = _B_unit_q4(np.asarray(p))
        Kuu += Bu.T @ D @ Bu * 0.25
        Kua += Bu.T @ D @ Ba * 0.25
        Kaa += Ba.T @ D @ Ba * 0.25
    R = -np.linalg.solve(Kaa, Kua.T)
    K = Kuu + Kua @ R
    return 0.5 * (K + K.T), R


def b_centroid_h8() -> np.ndarray:
    """Strain matrix (6, 24) at the unit-cube centroid (exact for H8 and H8IM: the mode derivatives vanish there)."""
    return _B_unit_h8(np.zeros(3))[0]


def b_centroid_q4() -> np.ndarray:
    """Strain matrix (3, 8) at the unit-square centroid."""
    s = Q4_CORNER
    dN = 0.25 * np.vstack([s[:, 0], s[:, 1]]) * 2.0          # at ξ = η = 0; ∂/∂x = 2 ∂/∂ξ
    B = np.zeros((3, 8))
    B[0, 0::2] = dN[0]
    B[1, 1::2] = dN[1]
    B[2, 0::2] = dN[1]
    B[2, 1::2] = dN[0]
    return B


def element_matrix_from_D(dim: int, element: str, D: np.ndarray) -> np.ndarray:
    """Unit-modulus element stiffness for a given elasticity matrix D (6 × 6 in 3-D, 3 × 3 in 2-D)."""
    if dim == 3:
        if element not in ELEMENTS_3D:
            raise ValueError("3-D element: H8 | H8IM")
        return np.asarray(ke_h8_from_D(D, element)[0], float)
    if element not in ELEMENTS_2D:
        raise ValueError("2-D element: Q4 | Q4IM")
    return ke_q4_from_D(D) if element == "Q4" else ke_q4im_from_D(D)[0]


def element_matrix(dim: int, element: str, nu: float) -> np.ndarray:
    """Unit-modulus element stiffness: 3-D "H8" | "H8IM" (24×24); 2-D "Q4" (8×8)."""
    if dim == 3:
        if element not in ELEMENTS_3D:
            raise ValueError("3-D element: H8 | H8IM")
        return np.asarray(ke_h8(nu, element)[0], float)
    if element not in ELEMENTS_2D:
        raise ValueError("2-D element: Q4 | Q4IM")
    return np.asarray(ke_q4(nu), float) if element == "Q4" else ke_q4im_from_D(D_plane_stress(nu))[0]


def element_matrix_field(dim: int, element: str, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-element unit-modulus stiffness for per-element D (n, k, k): (distinct matrices (m, nloc, nloc), index (n,))
    with element e's matrix ``KE[index[e]]``; each distinct D goes through :func:`element_matrix_from_D` once."""
    from toporlogy.core.material import unique_matrices
    Dn = np.asarray(D, float)
    k = 6 if dim == 3 else 3
    if Dn.ndim != 3 or Dn.shape[1:] != (k, k):
        raise ValueError(f"per-element D must be (n, {k}, {k}) in {dim}-D, got {Dn.shape}")
    Du, idx = unique_matrices(Dn)
    return np.array([element_matrix_from_D(dim, element, Dk) for Dk in Du]), idx


def mode_recovery_from_D(dim: int, D: np.ndarray) -> np.ndarray:
    """Mode recovery matrix R (α = R u_e) of the incompatible-mode element (H8IM in 3-D, Q4IM in 2-D) for one D;
    (n_modes, nloc)."""
    if dim == 3:
        R = ke_h8_from_D(D, "H8IM")[1]
        assert R is not None
        return R
    return ke_q4im_from_D(D)[1]


def indexed_matmul(M: np.ndarray, index: np.ndarray, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """(n, a, c) with entry e = ``M[index[e]] @ X[e]`` for matrices M (m, a, b) and X (n, b, c), in chunks of
    ``chunk`` elements (the gathered matrices are never all in memory at once)."""
    n = len(index)
    out = np.empty((n, M.shape[1], X.shape[2]))
    for s in range(0, n, chunk):
        out[s:s + chunk] = np.matmul(M[index[s:s + chunk]], X[s:s + chunk])
    return out
