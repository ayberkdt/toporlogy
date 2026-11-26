"""Classical lamination theory (CLT): reduced and transformed ply stiffness, ABD matrices, thermal resultants,
mid-plane response and ply stress recovery, with design derivatives in ply angles and thicknesses.

Conventions
-----------
* Plies are listed from the bottom (z = −h/2) to the top (z = +h/2); z is measured from the geometric mid-plane
  (the reference surface; no offset reference is offered).
* Ply axes: 1 = fibre, 2 = transverse in the laminate plane, 3 = thickness. ``angle`` (degrees) turns axis 1 from the
  global x towards y, about +z.
* In-plane Voigt order ``[xx, yy, xy]`` (:data:`toporlogy.core.tensor.VOIGT_2D`), engineering shear strain γ_xy;
  curvatures κ = [κ_x, κ_y, κ_xy] with κ_xy the engineering twist, so that ε(z) = ε⁰ + z κ.
* Resultants per unit width: N = ∫σ dz, M = ∫σ z dz. With free thermal strain ᾱ ΔT(z), ΔT(z) = ΔT0 + z ΔT1:

      [N]   [A  B] [ε⁰]   [N_T]           N_T = ∫ Q̄ ᾱ ΔT dz,  M_T = ∫ Q̄ ᾱ ΔT z dz
      [M] = [B  D] [κ ] − [M_T]

* Units: any consistent set; nothing is assumed.

Reuse
-----
The ply material is a 3-D :class:`toporlogy.core.linear_elastic.ElasticTensor` in ply axes. The reduced stiffness
Q is its plane-stress reduction (:func:`toporlogy.core.material.plane_stress`); the transformed stiffness Q̄ is the
in-plane fourth-order rotation of Q (:func:`toporlogy.core.tensor.rotate4`), which commutes with the reduction for a
rotation about z. No Voigt transformation matrices are written here.

Backend: :func:`transformed_reduced_stiffness`, :func:`abd_matrix` and :func:`complementary_energy` take NumPy
arrays or Torch tensors (angles / thicknesses); with Torch inputs they are differentiable.

Scope (CLT assumptions): thin laminate, plane sections remain plane and normal (Kirchhoff), plane stress in every
ply, perfect bonding, linear elasticity, small strains. No transverse shear or interlaminar stresses, no free-edge
effect, no progressive damage, no plate / shell element coupling. Ply angles and thicknesses are smooth design
variables; the choice of a stacking sequence is a discrete problem and is not handled here.

Verification: ``tests/toporlogy/test_laminate.py`` (records L1-L11 of ``docs/olcumler/laminat_uq_onkayit.md``).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material import plane_stress

__all__ = ["Ply", "Laminate", "LaminateState", "PlyStresses", "ply_reduced_stiffness", "in_plane_rotation",
           "transformed_reduced_stiffness", "transformed_thermal_expansion", "ply_interfaces", "abd_matrix",
           "complementary_energy", "abd_design_derivatives"]


def _is_torch(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "torch"


def ply_reduced_stiffness(material: ElasticTensor) -> np.ndarray:
    """Plane-stress reduced stiffness Q (3, 3) of a ply material in ply axes [11, 22, 12]."""
    if not isinstance(material, ElasticTensor):
        raise TypeError(f"a ply material is an ElasticTensor, got {type(material).__name__}")
    return plane_stress(material.stiffness())


def in_plane_rotation(angles_deg: Any) -> Any:
    """R (n, 2, 2) with the ply axes 1, 2 as columns in global x, y for angles (n,) in degrees."""
    if _is_torch(angles_deg):
        import torch
        rad = angles_deg.reshape(-1) * (math.pi / 180.0)
        ct, st = torch.cos(rad), torch.sin(rad)
        return torch.stack([torch.stack([ct, -st], dim=-1), torch.stack([st, ct], dim=-1)], dim=-2)
    t = np.deg2rad(np.asarray(angles_deg, float).reshape(-1))
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)


def _per_ply(Q: Any, n: int, like: Any) -> Any:
    """(n, 3, 3) on the backend of ``like`` from one Q (3, 3) or one per ply."""
    if _is_torch(like):
        import torch
        Qt = torch.as_tensor(np.asarray(Q, float) if not _is_torch(Q) else Q, dtype=like.dtype, device=like.device)
        return Qt.expand(n, 3, 3) if Qt.ndim == 2 else Qt
    Qn = np.asarray(Q, float)
    return np.broadcast_to(Qn, (n, 3, 3)) if Qn.ndim == 2 else Qn


def transformed_reduced_stiffness(Q: Any, angles_deg: Any) -> Any:
    """Q̄ (n, 3, 3) in global axes for ply stiffnesses Q (3, 3) or (n, 3, 3) at angles (n,) in degrees."""
    R = in_plane_rotation(angles_deg)
    Qp = _per_ply(Q, R.shape[0], R)
    if Qp.shape[0] != R.shape[0]:
        raise ValueError(f"{Qp.shape[0]} ply stiffnesses for {R.shape[0]} angles")
    return T.tensor_to_stiffness(T.rotate4(T.stiffness_to_tensor(Qp), R))


def transformed_thermal_expansion(alpha: Any, angles_deg: Any) -> np.ndarray:
    """ᾱ (n, 3) = [α_x, α_y, α_xy (engineering)] for ply-axis expansion (α1, α2) (n, 2) at angles (n,)."""
    a = np.asarray(alpha, float).reshape(-1, 2)
    R = in_plane_rotation(np.asarray(angles_deg, float))
    A = np.zeros((len(a), 2, 2))
    A[:, 0, 0], A[:, 1, 1] = a[:, 0], a[:, 1]
    return np.asarray(T.strain_to_voigt(T.rotate2(A, R), rtol=None))


def ply_interfaces(thicknesses: Any) -> Any:
    """z (n + 1,) of the ply interfaces, from −h/2 to +h/2, for ply thicknesses (n,) listed bottom to top."""
    if _is_torch(thicknesses):
        import torch
        flat = thicknesses.reshape(-1)
        upper = torch.cumsum(flat, dim=0)
        return torch.cat([torch.zeros(1, dtype=flat.dtype, device=flat.device), upper]) - 0.5 * upper[-1]
    top = np.cumsum(np.asarray(thicknesses, float).reshape(-1))
    return np.concatenate([[0.0], top]) - 0.5 * top[-1]


def abd_matrix(Qbar: Any, thicknesses: Any) -> Any:
    """(6, 6) [[A, B], [B, D]] of transformed stiffnesses Q̄ (n, 3, 3) and thicknesses (n,)."""
    torch_in = _is_torch(Qbar) or _is_torch(thicknesses)
    Qb: Any
    t: Any
    if torch_in:
        import torch
        like = Qbar if _is_torch(Qbar) else thicknesses
        Qb = torch.as_tensor(Qbar, dtype=like.dtype)
        t = torch.as_tensor(thicknesses, dtype=like.dtype)
    else:
        Qb, t = np.asarray(Qbar, float), np.asarray(thicknesses, float)
    z = ply_interfaces(t)
    if Qb.shape[0] != z.shape[0] - 1:
        raise ValueError(f"{Qb.shape[0]} ply stiffnesses for {z.shape[0] - 1} thicknesses")
    z0, z1 = z[:-1], z[1:]
    A = (Qb * (z1 - z0)[:, None, None]).sum(0)
    B = (Qb * (0.5 * (z1 ** 2 - z0 ** 2))[:, None, None]).sum(0)
    D = (Qb * ((z1 ** 3 - z0 ** 3) / 3.0)[:, None, None]).sum(0)
    if torch_in:
        import torch
        return torch.cat([torch.cat([A, B], dim=1), torch.cat([B, D], dim=1)], dim=0)
    return np.block([[A, B], [B, D]])


def complementary_energy(Q: Any, angles_deg: Any, thicknesses: Any, resultants: Any) -> Any:
    """½ rᵀ ABD⁻¹ r for applied resultants r = [N_x, N_y, N_xy, M_x, M_y, M_xy]: the strain energy per unit area under
    force control (smaller = stiffer for that load). Differentiable in angles and thicknesses with Torch inputs."""
    if _is_torch(angles_deg) or _is_torch(thicknesses):
        import torch
        like = angles_deg if _is_torch(angles_deg) else thicknesses
        ang = torch.as_tensor(angles_deg, dtype=like.dtype)
        K = abd_matrix(transformed_reduced_stiffness(Q, ang), torch.as_tensor(thicknesses, dtype=like.dtype))
        r = torch.as_tensor(np.asarray(resultants, float), dtype=like.dtype)
        return 0.5 * r @ torch.linalg.solve(K, r)
    Kn = abd_matrix(transformed_reduced_stiffness(Q, angles_deg), thicknesses)
    rn = np.asarray(resultants, float)
    return 0.5 * float(rn @ np.linalg.solve(Kn, rn))


def abd_design_derivatives(laminate: Laminate) -> tuple[np.ndarray, np.ndarray]:
    """(dABD/dθ (n, 6, 6) per degree, dABD/dt (n, 6, 6)) of every ply, by automatic differentiation. A thickness change
    keeps the mid-plane as reference, so it moves every ply."""
    import torch

    Q = laminate.Q_plies
    ang = torch.as_tensor(laminate.angles, dtype=torch.float64)
    thk = torch.as_tensor(laminate.thicknesses, dtype=torch.float64)

    def f(a: Any, t: Any) -> Any:
        return abd_matrix(transformed_reduced_stiffness(Q, a), t)

    Ja, Jt = torch.autograd.functional.jacobian(f, (ang, thk), vectorize=True)
    return Ja.permute(2, 0, 1).numpy().copy(), Jt.permute(2, 0, 1).numpy().copy()


@dataclass(frozen=True)
class Ply:
    """One lamina: ``material`` (3-D elastic tensor in ply axes), ``angle`` (degrees), ``thickness`` and the thermal
    expansion ``alpha`` = (α1, α2) in ply axes (None: no thermal strain may be applied)."""

    material: ElasticTensor
    angle: float
    thickness: float
    alpha: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.material, ElasticTensor):
            raise TypeError(f"a ply material is an ElasticTensor, got {type(self.material).__name__}")
        if not (math.isfinite(self.thickness) and self.thickness > 0):
            raise ValueError(f"ply thickness must be positive and finite, got {self.thickness}")
        if not math.isfinite(self.angle):
            raise ValueError("ply angle must be finite")
        if self.alpha is not None:
            a = tuple(float(v) for v in self.alpha)
            if len(a) != 2 or not all(math.isfinite(v) for v in a):
                raise ValueError("alpha is (alpha1, alpha2), finite, in ply axes")
            object.__setattr__(self, "alpha", a)


@dataclass(frozen=True)
class LaminateState:
    """Mid-plane strains ε⁰ (3,), curvatures κ (3,) and the temperature field ΔT0 + z ΔT1 they belong to."""

    eps0: np.ndarray
    kappa: np.ndarray
    dT0: float = 0.0
    dT1: float = 0.0


@dataclass(frozen=True)
class PlyStresses:
    """Stresses and strains of every ply at its bottom (index 0) and top (index 1) surface.

    ``z`` (n, 2); ``stress`` / ``strain`` (n, 2, 3) in global axes; ``stress_material`` / ``strain_material``
    (n, 2, 3) in ply axes [11, 22, 12]. ``strain`` is the total strain ε⁰ + zκ (thermal part included)."""

    z: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    stress_material: np.ndarray
    strain_material: np.ndarray


@dataclass(frozen=True)
class Laminate:
    """A stack of plies listed from the bottom to the top (module notes)."""

    plies: tuple[Ply, ...]

    def __post_init__(self) -> None:
        plies = tuple(self.plies)
        if not plies:
            raise ValueError("a laminate needs at least one ply")
        if not all(isinstance(p, Ply) for p in plies):
            raise TypeError("plies must be Ply objects")
        object.__setattr__(self, "plies", plies)

    @classmethod
    def from_angles(cls, material: ElasticTensor, angles_deg: Sequence[float], thickness: float | Sequence[float],
                    alpha: tuple[float, float] | None = None, symmetric: bool = False) -> Laminate:
        """One material, a list of angles (bottom to top) and a ply thickness (or one per angle). ``symmetric``: the
        list is the lower half and is mirrored (``[0, 90]`` → [0/90]s)."""
        ang = [float(a) for a in angles_deg]
        t = np.broadcast_to(np.asarray(thickness, float), (len(ang),) if np.ndim(thickness) == 0
                            else np.shape(thickness)).tolist()
        if len(t) != len(ang):
            raise ValueError(f"{len(t)} thicknesses for {len(ang)} angles")
        plies = [Ply(material, a, h, alpha) for a, h in zip(ang, t, strict=True)]
        if symmetric:
            plies = plies + plies[::-1]
        return cls(tuple(plies))

    # ------------------------------------------------------------------ stack data
    @property
    def n_plies(self) -> int:
        return len(self.plies)

    @property
    def angles(self) -> np.ndarray:
        return np.array([p.angle for p in self.plies], float)

    @property
    def thicknesses(self) -> np.ndarray:
        return np.array([p.thickness for p in self.plies], float)

    @property
    def thickness(self) -> float:
        return float(self.thicknesses.sum())

    @property
    def z(self) -> np.ndarray:
        return np.asarray(ply_interfaces(self.thicknesses))

    @property
    def Q_plies(self) -> np.ndarray:
        """Reduced stiffness (n, 3, 3) of every ply in its own axes."""
        return np.array([ply_reduced_stiffness(p.material) for p in self.plies])

    @property
    def Qbar(self) -> np.ndarray:
        return np.asarray(transformed_reduced_stiffness(self.Q_plies, self.angles))

    @property
    def alpha_bar(self) -> np.ndarray:
        """Transformed thermal expansion (n, 3); raises when a ply has none."""
        missing = [i for i, p in enumerate(self.plies) if p.alpha is None]
        if missing:
            raise ValueError(f"thermal strain requested but plies {missing} have no alpha")
        return transformed_thermal_expansion([p.alpha for p in self.plies], self.angles)

    # ------------------------------------------------------------------ matrices
    def abd(self) -> np.ndarray:
        """(6, 6) [[A, B], [B, D]]."""
        return np.asarray(abd_matrix(self.Qbar, self.thicknesses))

    @property
    def A(self) -> np.ndarray:
        return self.abd()[:3, :3]

    @property
    def B(self) -> np.ndarray:
        return self.abd()[:3, 3:]

    @property
    def D(self) -> np.ndarray:
        return self.abd()[3:, 3:]

    def classification(self, rtol: float = 1e-10) -> dict[str, bool]:
        """Stacking and matrix properties.

        ``symmetric_stack``: ply k and ply n−1−k have the same material, angle (modulo 180°), thickness and α.
        ``balanced_stack``: for every material and angle other than 0° / 90°, +θ and −θ have the same total thickness.
        ``B_zero``: max|B| ≤ rtol · max|A| · h. ``A16_A26_zero``: |A16|, |A26| ≤ rtol · max|A|. The stack properties
        are sufficient for the matrix ones, not necessary."""
        sym = True
        for p, q in zip(self.plies, self.plies[::-1], strict=True):
            same = (np.array_equal(p.material.stiffness(), q.material.stiffness())
                    and abs((p.angle - q.angle + 90.0) % 180.0 - 90.0) <= 1e-9
                    and math.isclose(p.thickness, q.thickness, rel_tol=1e-12) and p.alpha == q.alpha)
            sym = sym and same
        totals: dict[tuple[bytes, float], float] = {}
        for p in self.plies:
            a = (p.angle + 90.0) % 180.0 - 90.0              # in [−90, 90)
            if abs(a) <= 1e-9 or abs(abs(a) - 90.0) <= 1e-9:
                continue
            key = (p.material.stiffness().tobytes(), round(abs(a), 9))
            totals[key] = totals.get(key, 0.0) + math.copysign(p.thickness, a)
        h = self.thickness
        balanced = all(abs(v) <= 1e-12 * h for v in totals.values())
        K = self.abd()
        amax = float(np.abs(K[:3, :3]).max())
        return {"symmetric_stack": bool(sym), "balanced_stack": bool(balanced),
                "B_zero": bool(np.abs(K[:3, 3:]).max() <= rtol * amax * h),
                "A16_A26_zero": bool(max(abs(K[0, 2]), abs(K[1, 2])) <= rtol * amax)}

    # ------------------------------------------------------------------ loads
    def thermal_resultants(self, dT0: float, dT1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(N_T, M_T) (3,) each for ΔT(z) = ΔT0 + z ΔT1."""
        if dT0 == 0.0 and dT1 == 0.0:
            return np.zeros(3), np.zeros(3)
        qa = np.einsum("kij,kj->ki", self.Qbar, self.alpha_bar)
        z = self.z
        m0, m1, m2 = (z[1:] ** p - z[:-1] ** p for p in (1, 2, 3))
        N = (qa * (dT0 * m0 + dT1 * m1 / 2.0)[:, None]).sum(0)
        M = (qa * (dT0 * m1 / 2.0 + dT1 * m2 / 3.0)[:, None]).sum(0)
        return N, M

    def response(self, N: Any = None, M: Any = None, dT0: float = 0.0, dT1: float = 0.0) -> LaminateState:
        """Mid-plane strains and curvatures for applied resultants N, M (3,) and temperature ΔT0 + z ΔT1."""
        n = np.zeros(3) if N is None else np.asarray(N, float).reshape(3)
        m = np.zeros(3) if M is None else np.asarray(M, float).reshape(3)
        NT, MT = self.thermal_resultants(dT0, dT1)
        e = np.linalg.solve(self.abd(), np.concatenate([n + NT, m + MT]))
        return LaminateState(e[:3], e[3:], float(dT0), float(dT1))

    def stresses_at(self, state: LaminateState, ply: int, z: Any) -> np.ndarray:
        """Global stresses (len(z), 3) of ply ``ply`` at heights z (inside that ply)."""
        zz = np.atleast_1d(np.asarray(z, float))
        strain = state.eps0 + zz[:, None] * state.kappa
        if state.dT0 != 0.0 or state.dT1 != 0.0:
            strain = strain - self.alpha_bar[ply] * (state.dT0 + zz * state.dT1)[:, None]
        return strain @ self.Qbar[ply].T

    def ply_stresses(self, state: LaminateState) -> PlyStresses:
        """Stresses and strains at the bottom and top of every ply (module notes)."""
        z = self.z
        zs = np.stack([z[:-1], z[1:]], axis=1)                                   # (n, 2)
        strain = state.eps0 + zs[:, :, None] * state.kappa                       # (n, 2, 3)
        mech = strain
        if state.dT0 != 0.0 or state.dT1 != 0.0:
            mech = strain - self.alpha_bar[:, None, :] * (state.dT0 + zs * state.dT1)[:, :, None]
        stress = np.einsum("kij,ksj->ksi", self.Qbar, mech)
        R = np.asarray(in_plane_rotation(self.angles))[:, None, :, :]            # (n, 1, 2, 2)
        sm = T.stress_to_voigt(T.rotate2(T.voigt_to_stress(stress), R, inverse=True), rtol=None)
        em = T.strain_to_voigt(T.rotate2(T.voigt_to_strain(strain), R, inverse=True), rtol=None)
        return PlyStresses(zs, stress, strain, np.asarray(sm), np.asarray(em))
