"""Material data: elasticity (isotropic, orthotropic, rotated, plane stress / plane strain) and strengths.

Voigt order is the library's (:mod:`toporlogy.fem.elements`): 3-D ``[xx, yy, zz, yz, xz, xy]`` with engineering shear
strains, 2-D ``[xx, yy, xy]``. Everything here is plain NumPy (base layer); the finite elements, the stress fields and
the failure criteria use these matrices.

Elasticity
----------
:class:`Orthotropic` holds the nine engineering constants in the material axes 1, 2, 3 and builds the compliance

    S = [[ 1/E1,  −ν21/E2, −ν31/E3, 0, 0, 0 ],          ν21/E2 = ν12/E1, ν31/E3 = ν13/E1, ν32/E3 = ν23/E2
         [ −ν12/E1, 1/E2,  −ν32/E3, 0, 0, 0 ],
         [ −ν13/E1, −ν23/E2, 1/E3,  0, 0, 0 ],
         [ 0, 0, 0, 1/G23, 0, 0 ], [ 0, 0, 0, 0, 1/G13, 0 ], [ 0, 0, 0, 0, 0, 1/G12 ]]

and C = S⁻¹; it refuses constants for which C is not positive definite (the material would release energy). An
isotropic material is the special case E1 = E2 = E3 = E, G = E / (2(1 + ν)) (:meth:`Orthotropic.isotropic`).

The material axes are given by a rotation ``R`` whose columns are the material axes 1, 2, 3 in global coordinates
(:func:`rotation_from_axes`, :func:`rotation_about_z`). The stiffness is rotated as the fourth-order tensor,
C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs, not by Voigt transformation matrices, so there is no factor-of-two bookkeeping.

2-D: :func:`plane_stress` eliminates σzz = σyz = σxz = 0 through the compliance (exact for any orientation);
:func:`plane_strain` keeps the in-plane rows and columns of C (εzz = εyz = εxz = 0); :func:`out_of_plane_row` is the
row of C that gives σzz from the in-plane strains under plane strain (failure criteria need it).

Normalisation: the finite elements are built for a unit modulus and scaled by the interpolation's E0. A material's
``reference_modulus`` is that E0 (E for isotropic, E1 for orthotropic); :func:`unit_elasticity` returns D / E0.

Strength
--------
:class:`Strength` (isotropic: tension, compression, optional shear, safety factor) and :class:`OrthotropicStrength`
(Xt, Xc, Yt, Yc, Zt, Zc, S23, S13, S12 in the material axes, Tsai-Wu interaction f12*). Allowables are strengths
divided by the safety factor; the failure criteria are in :mod:`toporlogy.optimization.strength`.

Verification: ``tests/toporlogy/test_material.py`` (isotropic case equals ``fem.elements.D_iso`` / ``D_plane_stress``;
rotations leave an isotropic C unchanged and are consistent with 1/E(n) = n_i n_j n_k n_l S_ijkl; plane-stress
reduction of an isotropic C gives D_plane_stress; plane strain gives the Lamé form; positive-definiteness checks).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["VOIGT_3D", "Orthotropic", "Strength", "OrthotropicStrength", "rotation_from_axes", "rotation_about_z",
           "rotate_stiffness", "plane_stress", "plane_strain", "out_of_plane_row", "unit_elasticity",
           "elasticity_from_config", "strength_from_config", "PLANE_MODES"]

#: tensor index pairs of the 3-D Voigt components (toporlogy order)
VOIGT_3D = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
PLANE_MODES = ("stress", "strain")
_IN_PLANE = [0, 1, 5]                       # xx, yy, xy inside the 3-D Voigt vector


@dataclass(frozen=True)
class Orthotropic:
    """Orthotropic linear elastic material in its own axes 1, 2, 3 (module notes). Units: any consistent set.

    ``nu_ij`` is the Poisson ratio for stress along i (contraction along j = −nu_ij · ε_i).
    """

    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float

    def __post_init__(self) -> None:
        for k in ("E1", "E2", "E3", "G12", "G13", "G23"):
            if not float(getattr(self, k)) > 0:
                raise ValueError(f"{k} > 0 is required")
        w = np.linalg.eigvalsh(0.5 * (self.compliance() + self.compliance().T))
        if not w.min() > 0:
            raise ValueError("the elastic constants do not give a positive definite stiffness (check the Poisson "
                             f"ratios: smallest compliance eigenvalue {w.min():.3g})")

    @classmethod
    def isotropic(cls, E: float, nu: float) -> Orthotropic:
        """The isotropic material E, ν (G = E / (2(1 + ν)))."""
        if not -1 < nu < 0.5:
            raise ValueError("-1 < nu < 0.5 is required")
        G = E / (2 * (1 + nu))
        return cls(E, E, E, G, G, G, nu, nu, nu)

    @property
    def reference_modulus(self) -> float:
        return float(self.E1)

    def compliance(self) -> np.ndarray:
        """S (6, 6) in the material axes, Voigt [11, 22, 33, 23, 13, 12], engineering shear strains."""
        E1, E2, E3 = float(self.E1), float(self.E2), float(self.E3)
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
        S[0, 1] = S[1, 0] = -float(self.nu12) / E1
        S[0, 2] = S[2, 0] = -float(self.nu13) / E1
        S[1, 2] = S[2, 1] = -float(self.nu23) / E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / float(self.G23), 1 / float(self.G13), 1 / float(self.G12)
        return S

    def stiffness(self, R: np.ndarray | None = None) -> np.ndarray:
        """C (6, 6) in global axes for material axes ``R`` (columns = axes 1, 2, 3; None = aligned)."""
        C = np.linalg.inv(self.compliance())
        C = 0.5 * (C + C.T)
        return C if R is None else rotate_stiffness(C, R)


def rotation_from_axes(axis1: Any, axis2: Any) -> np.ndarray:
    """R with columns e1, e2, e3: e1 = axis1 / |axis1|, e2 = the part of axis2 normal to e1 (normalised),
    e3 = e1 × e2. Raises when axis2 is (nearly) parallel to axis1."""
    a = np.asarray(axis1, float).reshape(3)
    b = np.asarray(axis2, float).reshape(3)
    if not np.linalg.norm(a) > 0:
        raise ValueError("axis1 must not be zero")
    e1 = a / np.linalg.norm(a)
    b = b - (b @ e1) * e1
    if not np.linalg.norm(b) > 1e-9 * max(1.0, float(np.linalg.norm(axis2))):
        raise ValueError("axis2 must not be parallel to axis1")
    e2 = b / np.linalg.norm(b)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def rotation_about_z(angle_deg: float) -> np.ndarray:
    """R for material axis 1 at ``angle_deg`` from global x in the x-y plane (axis 3 = z)."""
    t = np.deg2rad(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _tensor(C6: np.ndarray) -> np.ndarray:
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT_3D):
        for b, (k, m) in enumerate(VOIGT_3D):
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, m), (m, k)}:
                    C[p, q, r, s] = C6[a, b]
    return C


def rotate_stiffness(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    """C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs in Voigt form (engineering shear strains on both sides)."""
    Rm = np.asarray(R, float)
    if Rm.shape != (3, 3) or not np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-12):
        raise ValueError("R must be a 3 x 3 orthogonal matrix")
    Ct = np.einsum("ip,jq,kr,ls,pqrs->ijkl", Rm, Rm, Rm, Rm, _tensor(np.asarray(C6, float)))
    out = np.array([[Ct[i, j, k, m] for (k, m) in VOIGT_3D] for (i, j) in VOIGT_3D])
    return 0.5 * (out + out.T)


def plane_stress(C6: np.ndarray) -> np.ndarray:
    """In-plane stiffness (3, 3) [xx, yy, xy] with σzz = σyz = σxz = 0."""
    S = np.linalg.inv(np.asarray(C6, float))
    Q = np.linalg.inv(S[np.ix_(_IN_PLANE, _IN_PLANE)])
    return 0.5 * (Q + Q.T)


def plane_strain(C6: np.ndarray) -> np.ndarray:
    """In-plane stiffness (3, 3) [xx, yy, xy] with εzz = εyz = εxz = 0."""
    return np.asarray(C6, float)[np.ix_(_IN_PLANE, _IN_PLANE)].copy()


def out_of_plane_row(C6: np.ndarray) -> np.ndarray:
    """(3,) row with σzz = row · [εxx, εyy, γxy] under plane strain."""
    return np.asarray(C6, float)[2, _IN_PLANE].copy()


def unit_elasticity(material: Orthotropic, dim: int, plane: str = "stress",
                    R: np.ndarray | None = None) -> np.ndarray:
    """D / E0 for the element matrices: (6, 6) in 3-D, (3, 3) in 2-D (``plane`` stress | strain)."""
    C = material.stiffness(R) / material.reference_modulus
    if dim == 3:
        return C
    if dim != 2:
        raise ValueError("dim must be 2 or 3")
    if plane not in PLANE_MODES:
        raise ValueError(f"plane: one of {PLANE_MODES}")
    return plane_stress(C) if plane == "stress" else plane_strain(C)


@dataclass(frozen=True)
class Strength:
    """Isotropic strengths (same unit as the stresses). ``compression`` None = ``tension``; ``shear`` None = not
    given (Tresca and von Mises derive it). ``safety_factor`` divides every strength (allowables)."""

    tension: float
    compression: float | None = None
    shear: float | None = None
    safety_factor: float = 1.0

    def __post_init__(self) -> None:
        if not float(self.tension) > 0:
            raise ValueError("strength.tension > 0 is required")
        if self.compression is not None and not float(self.compression) > 0:
            raise ValueError("strength.compression > 0 is required (a magnitude)")
        if self.shear is not None and not float(self.shear) > 0:
            raise ValueError("strength.shear > 0 is required")
        if not float(self.safety_factor) >= 1.0:
            raise ValueError("safety_factor >= 1 is required")

    @property
    def allowable_tension(self) -> float:
        return float(self.tension) / float(self.safety_factor)

    @property
    def allowable_compression(self) -> float:
        c = self.tension if self.compression is None else self.compression
        return float(c) / float(self.safety_factor)


@dataclass(frozen=True)
class OrthotropicStrength:
    """Strengths in the material axes (magnitudes): Xt, Xc along 1, Yt, Yc along 2, Zt, Zc along 3 (None = Y),
    shear S23, S13, S12 (S13 None = S12, S23 None = S12), Tsai-Wu interaction ``f12`` (F_ij = f12 sqrt(F_ii F_jj);
    −1/2 is a common choice, not a measured value), ``safety_factor`` on all strengths."""

    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    Zt: float | None = None
    Zc: float | None = None
    S13: float | None = None
    S23: float | None = None
    f12: float = -0.5
    safety_factor: float = 1.0

    def __post_init__(self) -> None:
        for k in ("Xt", "Xc", "Yt", "Yc", "S12", "Zt", "Zc", "S13", "S23"):
            v = getattr(self, k)
            if v is not None and not float(v) > 0:
                raise ValueError(f"strength.{k} > 0 is required (a magnitude)")
        if not -1.0 < float(self.f12) < 1.0:
            raise ValueError("-1 < f12 < 1 is required")
        if not float(self.safety_factor) >= 1.0:
            raise ValueError("safety_factor >= 1 is required")

    def allowables(self) -> dict[str, float]:
        """Xt … S12 divided by the safety factor, with the defaults for Z, S13, S23 filled in."""
        sf = float(self.safety_factor)
        v = {"Xt": self.Xt, "Xc": self.Xc, "Yt": self.Yt, "Yc": self.Yc, "S12": self.S12,
             "Zt": self.Yt if self.Zt is None else self.Zt, "Zc": self.Yc if self.Zc is None else self.Zc,
             "S13": self.S12 if self.S13 is None else self.S13, "S23": self.S12 if self.S23 is None else self.S23}
        return {k: float(x) / sf for k, x in v.items()}


# ============================================================================================== configuration
_ORTHO_KEYS = ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")


def elasticity_from_config(material: dict[str, Any], dim: int) -> tuple[Orthotropic, np.ndarray | None]:
    """(material, R) of a resolved ``material`` section: isotropic from E0 / nu unless ``orthotropic`` is given
    ({E1, E2, E3, G12, G13, G23, nu12, nu13, nu23} plus ``angle`` (degrees about z) or ``axes`` [axis1, axis2]).
    2-D orthotropy is limited to rotations about z (out-of-plane shear would couple to the in-plane state)."""
    spec = material.get("orthotropic")
    if spec is None:
        return Orthotropic.isotropic(float(material["E0"]), float(material["nu"])), None
    if not isinstance(spec, dict):
        raise ValueError("material.orthotropic must be a mapping")
    unknown = set(spec) - set(_ORTHO_KEYS) - {"angle", "axes"}
    if unknown:
        raise ValueError(f"material.orthotropic: unknown keys {sorted(unknown)} (keys: {', '.join(_ORTHO_KEYS)}, "
                         "angle, axes)")
    missing = [k for k in _ORTHO_KEYS if k not in spec]
    if missing:
        raise ValueError(f"material.orthotropic: missing {missing}")
    mat = Orthotropic(*(float(spec[k]) for k in _ORTHO_KEYS))
    if "angle" in spec and "axes" in spec:
        raise ValueError("material.orthotropic: give angle or axes, not both")
    R = None
    if "angle" in spec:
        R = rotation_about_z(float(spec["angle"]))
    elif "axes" in spec:
        if dim == 2:
            raise ValueError("material.orthotropic.axes: 2-D allows a rotation about z only (use angle)")
        ax = spec["axes"]
        if not (isinstance(ax, list | tuple) and len(ax) == 2):
            raise ValueError("material.orthotropic.axes: [axis1, axis2]")
        R = rotation_from_axes(ax[0], ax[1])
    return mat, R


def strength_from_config(material: dict[str, Any]) -> Strength | OrthotropicStrength | None:
    """The ``strength`` mapping of a resolved material section: {tension, compression, shear, safety_factor}
    (isotropic) or {Xt, Xc, Yt, Yc, S12, Zt, Zc, S13, S23, f12, safety_factor} (orthotropic); None if absent."""
    spec = material.get("strength")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("material.strength must be a mapping")
    iso = {"tension", "compression", "shear", "safety_factor"}
    ortho = {"Xt", "Xc", "Yt", "Yc", "S12", "Zt", "Zc", "S13", "S23", "f12", "safety_factor"}
    keys = set(spec)
    kw: dict[str, Any] = {k: (None if spec[k] is None else float(spec[k])) for k in keys}
    if keys <= iso and "tension" in keys:
        return Strength(**kw)
    if keys <= ortho and {"Xt", "Xc", "Yt", "Yc", "S12"} <= keys:
        return OrthotropicStrength(**kw)
    raise ValueError("material.strength: either {tension, compression, shear, safety_factor} or "
                     "{Xt, Xc, Yt, Yc, S12, Zt, Zc, S13, S23, f12, safety_factor}")
