"""Material data: elasticity (isotropic, orthotropic, rotated, plane stress / plane strain) and strengths.

Voigt order is the library's (:data:`toporlogy.core.tensor.VOIGT_3D`): 3-D ``[xx, yy, zz, yz, xz, xy]`` with
engineering shear strains, 2-D ``[xx, yy, xy]``. Everything here is plain NumPy (base layer); the finite elements, the
stress fields and the failure criteria use these matrices. The tensor algebra (Voigt <-> tensor, rotation, symmetry
and positive-definiteness checks) is :mod:`toporlogy.core.tensor` and :mod:`toporlogy.core.linear_elastic`; this
module holds the configuration-facing material records and the 3-D -> 2-D reductions.

Elasticity
----------
:class:`Orthotropic` holds the nine engineering constants in the material axes 1, 2, 3 and builds the compliance

    S = [[ 1/E1,  −ν21/E2, −ν31/E3, 0, 0, 0 ],          ν21/E2 = ν12/E1, ν31/E3 = ν13/E1, ν32/E3 = ν23/E2
         [ −ν12/E1, 1/E2,  −ν32/E3, 0, 0, 0 ],
         [ −ν13/E1, −ν23/E2, 1/E3,  0, 0, 0 ],
         [ 0, 0, 0, 1/G23, 0, 0 ], [ 0, 0, 0, 0, 1/G13, 0 ], [ 0, 0, 0, 0, 0, 1/G12 ]]

(:func:`toporlogy.core.linear_elastic.orthotropic_compliance`) and C = S⁻¹; it refuses constants for which S is not
positive definite (:func:`toporlogy.core.linear_elastic.check_positive_definite`: Cholesky and the scaled Mandel
eigenvalue; the material would release energy). An isotropic material is the special case E1 = E2 = E3 = E,
G = E / (2(1 + ν)) (:meth:`Orthotropic.isotropic`). A single ν_ij may lie outside the isotropic range (−1, 1/2).

The material axes are given by a rotation ``R`` whose columns are the material axes 1, 2, 3 in global coordinates
(:func:`rotation_from_axes`, :func:`rotation_about_z`). The stiffness is rotated as the fourth-order tensor,
C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs (:func:`toporlogy.core.tensor.rotate4`), not by Voigt transformation matrices,
so there is no factor-of-two bookkeeping. An input C that is not symmetric is refused, not symmetrised; the output is
averaged with its transpose only to remove rounding-level asymmetry of the einsum.

2-D: :func:`plane_stress` eliminates σzz = σyz = σxz = 0 through the compliance, Q = (S_pp)⁻¹, which is the Schur
complement C_pp − C_po C_oo⁻¹ C_op of all three out-of-plane stresses (exact for any orientation; with coupling of
the out-of-plane shears, as after a rotation that is not about z, eliminating σzz alone is wrong, see
``tests/toporlogy/test_linear_elastic.py``). :func:`plane_stress_recovery` gives the out-of-plane strains
[εzz, γyz, γxz] and :func:`plane_stress_derivative` dQ for a change dC. :func:`plane_strain` keeps the in-plane rows
and columns of C (εzz = εyz = εxz = 0); :func:`plane_strain_recovery` gives [σzz, σyz, σxz] from the in-plane
strains and :func:`out_of_plane_row` its σzz row (failure criteria need it). The 2-D kinematics are assumptions: plane
stress for thin plates loaded in their plane, plane strain for long prismatic bodies; neither represents bending or
out-of-plane warping.

Normalisation: the finite elements are built for a unit modulus and scaled by the interpolation's E0. A material's
``reference_modulus`` is that E0 (E for isotropic, E1 for orthotropic); :func:`unit_elasticity` returns D / E0.

Orientation per element (fibre angle): :func:`rotations_about_z` (2-D angle field) and :func:`rotations_from_axes`
(3-D axis field) give (n, 3, 3) rotations; :func:`unit_elasticity_field` the (n, k, k) unit-modulus D of every
element, computed once per distinct rotation with :func:`unit_elasticity` (an element with the same R as another
gets the same matrix bit for bit, so a uniform field reproduces the single-D path exactly). The physics accept the
(n_active, k, k) array as ``elasticity`` (:class:`toporlogy.fem.Elasticity`). Not modelled: the fibre angle as a
design variable (CFAO), laminate (ply stacking) theory, damage progression.

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

from toporlogy.core import tensor as T
from toporlogy.core.linear_elastic import check_positive_definite, isotropic_moduli, orthotropic_compliance

__all__ = ["Orthotropic", "Strength", "OrthotropicStrength", "rotation_from_axes", "rotation_about_z",
           "rotations_from_axes", "rotations_about_z", "rotate_stiffness", "plane_stress", "plane_strain",
           "out_of_plane_row", "out_of_plane_rows", "plane_stress_recovery", "plane_strain_recovery",
           "plane_stress_derivative", "unit_elasticity", "unit_elasticity_field", "unique_matrices",
           "elasticity_from_config", "strength_from_config", "PLANE_MODES"]

PLANE_MODES = ("stress", "strain")
_IN_PLANE = [0, 1, 5]                       # xx, yy, xy inside the 3-D Voigt vector
_OUT_OF_PLANE = [2, 3, 4]                   # zz, yz, xz


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
        S = self.compliance()                       # checks E_i > 0, G_ij > 0
        try:
            check_positive_definite(S, "compliance", "S")
        except ValueError as e:
            raise ValueError(f"the elastic constants do not give a positive definite stiffness (check the Poisson "
                             f"ratios; reciprocity nu_ji = nu_ij E_j / E_i is built in): {e}") from None

    @classmethod
    def isotropic(cls, E: float, nu: float) -> Orthotropic:
        """The isotropic material E, ν (G = E / (2(1 + ν)))."""
        G = isotropic_moduli(E=E, nu=nu).G                  # refuses E <= 0 and nu outside (-1, 1/2)
        return cls(E, E, E, G, G, G, nu, nu, nu)

    @property
    def reference_modulus(self) -> float:
        return float(self.E1)

    def compliance(self) -> np.ndarray:
        """S (6, 6) in the material axes, Voigt [11, 22, 33, 23, 13, 12], engineering shear strains."""
        return orthotropic_compliance(self.E1, self.E2, self.E3, self.G12, self.G13, self.G23, self.nu12, self.nu13,
                                      self.nu23)

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


def rotations_about_z(angles_deg: Any) -> np.ndarray:
    """(n, 3, 3) rotations for material axis 1 at ``angles_deg[i]`` from global x (as :func:`rotation_about_z`)."""
    a = np.asarray(angles_deg, float).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the angle field must be finite")
    return np.array([rotation_about_z(float(t)) for t in a]).reshape(len(a), 3, 3)


def rotations_from_axes(axis1: Any, axis2: Any) -> np.ndarray:
    """(n, 3, 3) rotations from per-element axes ``axis1`` (n, 3) and ``axis2`` (n, 3) or one vector (3,), each as
    :func:`rotation_from_axes`. Raises, naming the first element, when axis2 is parallel to axis1 there."""
    a1 = np.asarray(axis1, float).reshape(-1, 3)
    b = np.asarray(axis2, float)
    a2 = np.broadcast_to(b.reshape(3), a1.shape) if b.ndim == 1 else b.reshape(-1, 3)
    if a2.shape != a1.shape:
        raise ValueError(f"axis2 must be (3,) or {a1.shape}, got {b.shape}")
    if not (np.isfinite(a1).all() and np.isfinite(a2).all()):
        raise ValueError("the axis field must be finite")
    out = np.empty((len(a1), 3, 3))
    for i in range(len(a1)):
        try:
            out[i] = rotation_from_axes(a1[i], a2[i])
        except ValueError as e:
            raise ValueError(f"element {i}: {e}") from None
    return out


def unique_matrices(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distinct matrices (m, ...), index (n,)) with ``A == distinct[index]`` exactly."""
    A = np.ascontiguousarray(np.asarray(A, float))
    if A.ndim < 2:
        raise ValueError("A must be (n, ...)")
    u, idx = np.unique(A.reshape(len(A), -1), axis=0, return_inverse=True)
    return u.reshape((len(u),) + A.shape[1:]), np.asarray(idx, np.int64).reshape(-1)


def rotate_stiffness(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    """C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs in Voigt form (engineering shear strains on both sides). ``R`` (3, 3)
    orthogonal to 1e-12 (a reflection gives the same even-order tensor as −R and is accepted); C6 symmetric."""
    Rm = np.asarray(R, float)
    if Rm.shape != (3, 3):
        raise ValueError("R must be a 3 x 3 orthogonal matrix")
    T.check_rotation(Rm, proper=False)
    C = _check_C6(C6)
    out = T.tensor_to_stiffness(T.rotate4(T.stiffness_to_tensor(C), Rm))
    return 0.5 * (out + out.T)


def _check_C6(C6: np.ndarray) -> np.ndarray:
    C = np.asarray(C6, float)
    if C.shape != (6, 6):
        raise ValueError(f"C must be (6, 6), got {C.shape}")
    return T.check_symmetric(C, T.SYMMETRY_RTOL, "C")


def plane_stress(C6: np.ndarray) -> np.ndarray:
    """In-plane stiffness (3, 3) [xx, yy, xy] with σzz = σyz = σxz = 0: Q = (S_pp)⁻¹ (module notes)."""
    S = np.linalg.inv(_check_C6(C6))
    Q = np.linalg.inv(S[np.ix_(_IN_PLANE, _IN_PLANE)])
    return 0.5 * (Q + Q.T)


def plane_stress_recovery(C6: np.ndarray) -> np.ndarray:
    """(3, 3) matrix giving the out-of-plane strains [εzz, γyz, γxz] = S_op Q [εxx, εyy, γxy] under plane stress."""
    S = np.linalg.inv(_check_C6(C6))
    return S[np.ix_(_OUT_OF_PLANE, _IN_PLANE)] @ plane_stress(C6)


def plane_stress_derivative(C6: np.ndarray, dC6: np.ndarray) -> np.ndarray:
    """dQ of :func:`plane_stress` for a symmetric change dC: dQ = Q S_p: dC S_:p Q with S = C⁻¹ (from dS = −S dC S
    and dQ = −Q dS_pp Q)."""
    S = np.linalg.inv(_check_C6(C6))
    Q = plane_stress(C6)
    Sp = S[_IN_PLANE, :]
    return Q @ Sp @ np.asarray(dC6, float) @ Sp.T @ Q


def plane_strain(C6: np.ndarray) -> np.ndarray:
    """In-plane stiffness (3, 3) [xx, yy, xy] with εzz = εyz = εxz = 0."""
    return np.asarray(C6, float)[np.ix_(_IN_PLANE, _IN_PLANE)].copy()


def plane_strain_recovery(C6: np.ndarray) -> np.ndarray:
    """(3, 3) rows [σzz, σyz, σxz] = rows · [εxx, εyy, γxy] under plane strain (σyz, σxz vanish unless the
    out-of-plane shears couple to the in-plane strains)."""
    return np.asarray(C6, float)[np.ix_(_OUT_OF_PLANE, _IN_PLANE)].copy()


def out_of_plane_row(C6: np.ndarray) -> np.ndarray:
    """(3,) row with σzz = row · [εxx, εyy, γxy] under plane strain (first row of :func:`plane_strain_recovery`)."""
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


def unit_elasticity_field(material: Orthotropic, dim: int, plane: str = "stress",
                          R: np.ndarray | None = None) -> np.ndarray:
    """(n, k, k) unit-modulus D of every element for per-element material axes ``R`` (n, 3, 3), each computed with
    :func:`unit_elasticity` once per distinct rotation (module notes). ``R`` None or (3, 3) returns (k, k)."""
    if R is None or np.ndim(R) == 2:
        return unit_elasticity(material, dim, plane, R)
    Rs, idx = unique_matrices(np.asarray(R, float).reshape(-1, 3, 3))
    Du = np.array([unit_elasticity(material, dim, plane, r) for r in Rs])
    return Du[idx]


def out_of_plane_rows(material: Orthotropic, R: np.ndarray | None = None) -> np.ndarray:
    """Plane-strain σzz rows of the unit-modulus stiffness: (3,) for one R (or None), (n, 3) for R (n, 3, 3)."""
    E0 = material.reference_modulus
    if R is None or np.ndim(R) == 2:
        return out_of_plane_row(material.stiffness(R) / E0)
    Rs, idx = unique_matrices(np.asarray(R, float).reshape(-1, 3, 3))
    return np.array([out_of_plane_row(material.stiffness(r) / E0) for r in Rs])[idx]


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


def elasticity_from_config(material: dict[str, Any], dim: int,
                           domain: Any = None) -> tuple[Orthotropic, np.ndarray | None]:
    """(material, R) of a resolved ``material`` section: isotropic from E0 / nu unless ``orthotropic`` is given
    ({E1, E2, E3, G12, G13, G23, nu12, nu13, nu23} plus at most one orientation: ``angle`` (degrees about z),
    ``axes`` [axis1, axis2] (3-D), ``angle_field`` (2-D) or ``axis_field`` (3-D)). 2-D orthotropy is limited to
    rotations about z (out-of-plane shear would couple to the in-plane state).

    Fields (per element; need ``domain``): ``angle_field`` a ``.npy`` path or array of the element grid shape
    (degrees about z); ``axis_field`` a ``.npy`` path or array (*shape, 3) of axis 1, with ``axis2`` (one vector)
    for the second axis, or (*shape, 2, 3) with both. Values outside the part are ignored. R is then
    (n_active, 3, 3) in the order of ``domain.active_elements``. A relative path is taken from the working directory
    (as for the STL domain)."""
    spec = material.get("orthotropic")
    if spec is None:
        E0 = float(material["E0"])
        if not (np.isfinite(E0) and E0 > 0):
            raise ValueError(f"E0 must be a finite number > 0, got {material['E0']!r}")
        return Orthotropic.isotropic(E0, float(material["nu"])), None
    if not isinstance(spec, dict):
        raise ValueError("material.orthotropic must be a mapping")
    orient = ("angle", "axes", "angle_field", "axis_field")
    unknown = set(spec) - set(_ORTHO_KEYS) - set(orient) - {"axis2"}
    if unknown:
        raise ValueError(f"material.orthotropic: unknown keys {sorted(unknown)} (keys: {', '.join(_ORTHO_KEYS)}, "
                         f"{', '.join(orient)}, axis2)")
    missing = [k for k in _ORTHO_KEYS if k not in spec]
    if missing:
        raise ValueError(f"material.orthotropic: missing {missing}")
    mat = Orthotropic(*(float(spec[k]) for k in _ORTHO_KEYS))
    given = [k for k in orient if k in spec]
    if len(given) > 1:
        raise ValueError(f"material.orthotropic: give one of {', '.join(orient)}, not {' and '.join(given)}")
    if "axis2" in spec and given != ["axis_field"]:
        raise ValueError("material.orthotropic.axis2 belongs to axis_field")
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
    elif "angle_field" in spec:
        if dim != 2:
            raise ValueError("material.orthotropic.angle_field is 2-D (use axis_field in 3-D)")
        R = rotations_about_z(_element_field(spec["angle_field"], domain, (), "angle_field"))
    elif "axis_field" in spec:
        if dim != 3:
            raise ValueError("material.orthotropic.axis_field is 3-D (use angle_field in 2-D)")
        raw = _load_field(spec["axis_field"], "axis_field")
        if domain is not None and raw.shape == tuple(domain.shape) + (2, 3):
            ax2 = _element_field(raw, domain, (2, 3), "axis_field")
            R = rotations_from_axes(ax2[:, 0], ax2[:, 1])
        else:
            if "axis2" not in spec:
                raise ValueError("material.orthotropic.axis_field of shape (*shape, 3) needs axis2 (the second "
                                 "material axis, one vector); or give (*shape, 2, 3)")
            ax1 = _element_field(raw, domain, (3,), "axis_field")
            R = rotations_from_axes(ax1, np.asarray(spec["axis2"], float).reshape(3))
    return mat, R


def _load_field(v: Any, what: str) -> np.ndarray:
    if isinstance(v, str):
        from pathlib import Path
        p = Path(v)
        if not p.is_file():
            raise ValueError(f"material.orthotropic.{what}: file not found: {p.resolve()}")
        return np.asarray(np.load(p, allow_pickle=False), float)
    return np.asarray(v, float)


def _element_field(v: Any, domain: Any, tail: tuple[int, ...], what: str) -> np.ndarray:
    """(n_active, *tail) values of a field given on the element grid (*domain.shape, *tail)."""
    if domain is None:
        raise ValueError(f"material.orthotropic.{what} needs the domain")
    a = _load_field(v, what)
    want = tuple(domain.shape) + tail
    if a.shape != want:
        raise ValueError(f"material.orthotropic.{what}: shape {a.shape}, the element grid needs {want}")
    out = a.reshape((-1,) + tail)[domain.active_elements]
    if not np.isfinite(out).all():
        raise ValueError(f"material.orthotropic.{what}: values inside the part must be finite")
    return out


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
