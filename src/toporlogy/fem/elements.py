"""Element matrices of the regular grid (unit cube / unit square, E = 1).

* ``H8``: trilinear brick, 2×2×2 Gauss integration (the element of top3d, Liu & Tovar 2014). Locks in bending when a
  member is only one or two elements thick.
* ``H8IM``: the same brick with the nine Wilson incompatible modes ``1 − ξ², 1 − η², 1 − ζ²`` per displacement
  component, statically condensed (Taylor, Beresford & Wilson 1976; Cook et al. 2002 §6.7). All elements are the
  same cube, so the Jacobian is constant and the element passes the patch test without Taylor's correction.
* ``Q4``: bilinear plane-stress square, closed form of top88 (Andreassen et al. 2011, lines 8-12).

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

from toporlogy.references import cite

__all__ = ["H8_CORNER", "Q4_CORNER", "D_iso", "D_plane_stress", "ke_h8", "ke_q4", "element_matrix",
           "b_centroid_h8", "b_centroid_q4"]

H8_CORNER = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
Q4_CORNER = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_G = 1.0 / math.sqrt(3.0)
_GAUSS3 = np.array([[a, b, c] for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)])


def D_iso(nu: float) -> np.ndarray:
    """Isotropic 3-D elasticity matrix for E = 1 (Voigt [εx, εy, εz, γyz, γxz, γxy])."""
    if not -1 < nu < 0.5:
        raise ValueError("-1 < nu < 0.5 is required")
    c = 1.0 / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = c * nu
    np.fill_diagonal(D[:3, :3], c * (1 - nu))
    D[3:, 3:] = np.eye(3) / (2 * (1 + nu))
    return D


def D_plane_stress(nu: float) -> np.ndarray:
    """Plane-stress elasticity matrix for E = 1 (Voigt [εx, εy, γxy])."""
    if not -1 < nu < 0.5:
        raise ValueError("-1 < nu < 0.5 is required")
    return np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2]]) / (1 - nu ** 2)


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
    if element not in ("H8", "H8IM"):
        raise ValueError("element must be 'H8' or 'H8IM'")
    D = D_iso(nu)
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


def element_matrix(dim: int, element: str, nu: float) -> np.ndarray:
    """Unit-modulus element stiffness: 3-D "H8" | "H8IM" (24×24); 2-D "Q4" (8×8)."""
    if dim == 3:
        if element not in ("H8", "H8IM"):
            raise ValueError("3-D element: H8 | H8IM")
        return np.asarray(ke_h8(nu, element)[0], float)
    if element != "Q4":
        raise ValueError("2-D element: Q4")
    return np.asarray(ke_q4(nu), float)
