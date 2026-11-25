"""Failure criteria as differentiable failure indices of a 3-D stress state.

Every criterion maps a stress (n, 6) in the library's Voigt order ``[xx, yy, zz, yz, xz, xy]`` to a failure index
``I ≥ 0`` with **I = 1 at the allowable** (strength / safety factor) and **I(s·σ) = s·I(σ)** for s > 0: the index is
the reciprocal of the factor by which the stress may be scaled before failure ("strength ratio"). A constraint
``I ≤ 1`` is therefore a stress limit in the usual sense, and 1/max(I) is the first-failure load factor of a linear
analysis. 2-D stresses are expanded with σzz (0 in plane stress, from the plane-strain row otherwise) and
σyz = σxz = 0 (:func:`expand_plane`).

Criteria (σt, σc: allowable tension / compression magnitudes; I1 = tr σ, J2 = s:s/2, σ_vM = sqrt(3 J2)):

``von_mises``        I = σ_vM / σt
``tresca``           I = (σ_max − σ_min) / σt            (principal stresses; τ_allow = σt / 2)
``drucker_prager``   I = (σ_vM + β I1) / k,  β = (σc − σt)/(σc + σt),  k = 2 σt σc / (σc + σt)
                     (the cone through uniaxial tension σt and uniaxial compression σc)
``raghava``          paraboloid 3 J2 + (σc − σt) I1 = σt σc, as a strength ratio:
                     I = (b + sqrt(b² + 4 a c)) / (2 c),  a = 3 J2, b = (σc − σt) I1, c = σt σc
``tsai_wu``          (orthotropic) F_i σ_i + F_ij σ_i σ_j = 1 in the material axes, as a strength ratio
                     I = (b + sqrt(b² + 4 a)) / 2,  a = σᵀ F σ, b = f · σ;  F_i = 1/X_t − 1/X_c, F_ii = 1/(X_t X_c),
                     F_44,55,66 = 1/S23², 1/S13², 1/S12², F_ij = f12 sqrt(F_ii F_jj)

The two quadratic criteria are solved for the strength ratio R (failure at R·σ: a R² + b R − 1 = 0, Tsai-Wu; the
Raghava form divides by c), whose reciprocal in the rationalised form above is smooth and has no cancellation.
Drucker-Prager and Raghava reduce to von Mises for σc = σt; the paraboloid is the form associated with Raghava,
Caddell & Yeh (1973) for polymers (citation from memory, not checked); Tsai-Wu: Tsai & Wu (1971) (repository
record ``tsai_wu_1971``; text not read). The formulas are written
from the criteria's definitions and verified in ``tests/toporlogy/test_strength.py`` (index 1 on the uniaxial,
shear and biaxial reference states, homogeneity, symmetry under rotation for the isotropic criteria, gradients).

Smoothing: von Mises and the square roots use ``sqrt(x + (eps·scale)²)`` (``eps`` default 1e-6) so the index is
differentiable at zero stress, as :class:`toporlogy.optimization.stress.VonMises` does. Tresca is not
differentiable where two principal stresses coincide (the gradient there is that of one branch); prefer von Mises
or Drucker-Prager inside an optimisation, Tresca for checks.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toporlogy.core.material import OrthotropicStrength, Strength, VOIGT_3D
from toporlogy.references import cite

__all__ = ["CRITERIA", "Criterion", "make_criterion", "expand_plane", "rotate_stress_to_material",
           "von_mises_equivalent", "tresca_equivalent"]

CRITERIA = ("von_mises", "tresca", "drucker_prager", "raghava", "tsai_wu")


def expand_plane(s3: torch.Tensor, szz: torch.Tensor | None = None) -> torch.Tensor:
    """(n, 3) in-plane [xx, yy, xy] → (n, 6) with σzz = ``szz`` (or 0) and σyz = σxz = 0."""
    z = torch.zeros_like(s3[:, 0])
    return torch.stack([s3[:, 0], s3[:, 1], z if szz is None else szz, z, z, s3[:, 2]], dim=1)


def rotate_stress_to_material(s: torch.Tensor, R: Any) -> torch.Tensor:
    """σ' = Rᵀ σ R for stresses (n, 6) in global axes; R's columns are the material axes. ``R`` (3, 3) for all
    rows, or (n, 3, 3) one per row (per-element orientation)."""
    Rt = torch.as_tensor(np.asarray(R, float), dtype=s.dtype)
    T = torch.zeros(s.shape[0], 3, 3, dtype=s.dtype)
    for v, (i, j) in enumerate(VOIGT_3D):
        T[:, i, j] = s[:, v]
        T[:, j, i] = s[:, v]
    if Rt.dim() == 3:
        if Rt.shape[0] != s.shape[0]:
            raise ValueError(f"per-row R has {Rt.shape[0]} rotations for {s.shape[0]} stresses")
        Tm = torch.einsum("npi,npq,nqj->nij", Rt, T, Rt)
    else:
        Tm = torch.einsum("pi,npq,qj->nij", Rt, T, Rt)
    return torch.stack([Tm[:, i, j] for i, j in VOIGT_3D], dim=1)


def _invariants(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """I1 and 3 J2 (= σ_vM²) of stresses (n, 6)."""
    sx, sy, sz, tyz, txz, txy = (s[:, k] for k in range(6))
    I1 = sx + sy + sz
    j2x3 = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
    return I1, j2x3


def von_mises_equivalent(s: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """σ_vM of stresses (n, 6), ``sqrt(3 J2 + eps²)``."""
    return torch.sqrt(_invariants(s)[1] + eps ** 2)


def tresca_equivalent(s: torch.Tensor) -> torch.Tensor:
    """σ_max − σ_min (principal stresses) of stresses (n, 6)."""
    T = torch.zeros(s.shape[0], 3, 3, dtype=s.dtype)
    for v, (i, j) in enumerate(VOIGT_3D):
        T[:, i, j] = s[:, v]
        T[:, j, i] = s[:, v]
    w = torch.linalg.eigvalsh(T)
    return w[:, 2] - w[:, 0]


class Criterion:
    """A failure criterion with its allowables (module notes). ``index(s)`` maps stresses (n, 6) to I (n,).

    Parameters
    ----------
    kind : str
        One of :data:`CRITERIA`.
    strength : Strength | OrthotropicStrength
        Isotropic strengths for the first four, orthotropic for ``tsai_wu``.
    R : array, optional
        Material axes (columns, global coordinates) for ``tsai_wu``; None = aligned with x, y, z. (n, 3, 3): one
        frame per element (fibre orientation field); :class:`~toporlogy.optimization.stress.FailureIndex` then
        expects n = active elements and passes each stress row its element's frame.
    eps : float
        Smoothing at zero stress, relative to the allowable tension (module notes).
    """

    def __init__(self, kind: str, strength: Strength | OrthotropicStrength, R: np.ndarray | None = None,
                 eps: float = 1e-6) -> None:
        if kind not in CRITERIA:
            raise ValueError(f"criterion: one of {CRITERIA}")
        if kind == "tsai_wu":
            if not isinstance(strength, OrthotropicStrength):
                raise ValueError("tsai_wu needs orthotropic strengths (Xt, Xc, Yt, Yc, S12, ...)")
        elif not isinstance(strength, Strength):
            raise ValueError(f"{kind} needs isotropic strengths (tension, compression, ...)")
        if not eps >= 0:
            raise ValueError("eps >= 0 is required")
        self.kind, self.strength, self.eps = kind, strength, float(eps)
        self.R = None if R is None else np.asarray(R, float)
        if isinstance(strength, OrthotropicStrength):
            a = strength.allowables()
            X = [(a["Xt"], a["Xc"]), (a["Yt"], a["Yc"]), (a["Zt"], a["Zc"])]
            self.f = np.array([1 / t - 1 / c for t, c in X] + [0.0, 0.0, 0.0])
            Fd = np.array([1 / (t * c) for t, c in X])
            F = np.zeros((6, 6))
            F[:3, :3] = strength.f12 * np.sqrt(np.outer(Fd, Fd))
            F[np.arange(3), np.arange(3)] = Fd
            F[3, 3], F[4, 4], F[5, 5] = 1 / a["S23"] ** 2, 1 / a["S13"] ** 2, 1 / a["S12"] ** 2
            self.F = F
            self.scale = min(a["Xt"], a["Yt"], a["Zt"])
        else:
            self.st, self.sc = strength.allowable_tension, strength.allowable_compression
            self.scale = self.st

    @property
    def per_element(self) -> bool:
        """True when ``R`` holds one frame per element."""
        return self.R is not None and self.R.ndim == 3

    def index(self, s: torch.Tensor, R: Any = None) -> torch.Tensor:
        """Failure index (n,) of stresses (n, 6); 1 at the allowable, degree-1 homogeneous. ``R`` (n, 3, 3)
        overrides the criterion's frames row by row (per-element orientation)."""
        e = self.eps
        if self.kind == "tsai_wu":
            Rr = self.R if R is None else R
            sm = s if Rr is None else rotate_stress_to_material(s, Rr)
            F = torch.as_tensor(self.F, dtype=s.dtype)
            f = torch.as_tensor(self.f, dtype=s.dtype)
            a = torch.einsum("nv,vw,nw->n", sm, F, sm)
            b = sm @ f
            return 0.5 * (b + torch.sqrt(b * b + 4.0 * a + e ** 2))
        sn = s / self.scale                                            # dimensionless: eps is relative
        st, sc = self.st / self.scale, self.sc / self.scale
        I1, j3 = _invariants(sn)
        if self.kind == "von_mises":
            return torch.sqrt(j3 + e ** 2) / st
        if self.kind == "tresca":
            return tresca_equivalent(sn) / st
        if self.kind == "drucker_prager":
            beta = (sc - st) / (sc + st)
            k = 2.0 * st * sc / (sc + st)
            return (torch.sqrt(j3 + e ** 2) + beta * I1) / k
        b = (sc - st) * I1                                             # raghava
        c = st * sc
        return (b + torch.sqrt(b * b + 4.0 * j3 * c + e ** 2)) / (2.0 * c)


@cite("tsai_wu_1971", note="Tsai-Wu quadratic criterion; record, text not read; written from the definition")
def make_criterion(kind: str, strength: Any, R: np.ndarray | None = None, eps: float = 1e-6) -> Criterion:
    """:class:`Criterion` from a kind and a :class:`Strength` / :class:`OrthotropicStrength`."""
    return Criterion(kind, strength, R, eps)
