"""First-ply failure of a CLT laminate: Tsai-Wu and Hashin (plane stress), failure index and load factor.

First-ply failure only: the load at which the first ply surface reaches a criterion in a linear elastic analysis.
Nothing is degraded afterwards; progressive damage, delamination and ultimate laminate strength are not modelled and
the first-ply load is not the failure load of the laminate.

Strengths are always user input (:class:`toporlogy.core.material.OrthotropicStrength`; X along the fibre, Y across,
S12 in-plane shear, S23 transverse shear). No allowable is built in. Note that its default ``f12 = −0.5`` is a common
choice for the Tsai-Wu interaction, not a measured value.

Criteria, on the ply-axis stresses [σ11, σ22, τ12] (σ33 = τ13 = τ23 = 0):

* ``tsai_wu``: the library's :class:`toporlogy.optimization.strength.Criterion` (the same F_i, F_ij).
* ``hashin``: four modes, written from the Abaqus 2025 reference page "Damage initiation for fiber-reinforced
  composites" (read 26 Nov 2026; Hashin's 1980 paper itself was not read), with α = 1 for the Hashin (1980) model:

      fibre tension       σ11 ≥ 0:  (σ11/X_t)² + α (τ12/S12)²
      fibre compression   σ11 < 0:  (σ11/X_c)²
      matrix tension      σ22 ≥ 0:  (σ22/Y_t)² + (τ12/S12)²
      matrix compression  σ22 < 0:  (σ22/(2 S23))² + [(Y_c/(2 S23))² − 1] σ22/Y_c + (τ12/S12)²

  S23 must be given explicitly (it is not replaced by S12).

The reported **index** I is the reciprocal strength ratio, as in :mod:`toporlogy.optimization.strength`: I = 1 on the
failure surface and I(sσ) = s I(σ). It is not the value of the quadratic expression. The **load factor** λ scales the
applied N, M with the temperature field held fixed: λ is the smallest λ ≥ 0 at which σ_T + λ σ_M reaches a mode.
With residual thermal stress, 1/I is not λ.

Evaluation points are the bottom and top surface of every ply. Within a ply the stress is linear in z and the
Tsai-Wu function is convex, so its maximum is at a surface; for Hashin a ply whose σ11 or σ22 changes sign inside it
can have its maximum at the sign change, which is not searched (``sign_change_inside_ply`` reports it).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from toporlogy.core.laminate import Laminate
from toporlogy.core.material import OrthotropicStrength
from toporlogy.optimization.strength import Criterion, expand_plane

__all__ = ["PLY_CRITERIA", "FailureMode", "FirstPlyFailure", "failure_modes", "ply_failure_index",
           "load_factor_to_failure", "first_ply_failure"]

PLY_CRITERIA = ("tsai_wu", "hashin")


@dataclass(frozen=True)
class FailureMode:
    """One quadratic surface σᵀ P σ + p·σ = 1 on [σ11, σ22, τ12], active where ``sign · σ[component] ≥ 0``
    (``component`` None: everywhere; for a negative sign the boundary σ = 0 belongs to the positive mode)."""

    name: str
    P: np.ndarray
    p: np.ndarray
    component: int | None = None
    sign: int = 1

    def active(self, s: np.ndarray) -> np.ndarray:
        if self.component is None:
            return np.ones(s.shape[:-1], bool)
        v = s[..., self.component]
        return v >= 0 if self.sign > 0 else v < 0

    def value(self, s: np.ndarray) -> np.ndarray:
        """The quadratic expression σᵀPσ + p·σ (failure at 1), regardless of activity."""
        return np.einsum("...i,ij,...j->...", s, self.P, s) + s @ self.p


def _hashin_strengths(strength: OrthotropicStrength) -> dict[str, float]:
    if strength.S23 is None:
        raise ValueError("hashin needs the transverse shear strength S23 (it is not replaced by S12)")
    return strength.allowables()


def failure_modes(criterion: str, strength: OrthotropicStrength, alpha: float = 1.0) -> tuple[FailureMode, ...]:
    """The quadratic surfaces of a criterion (module notes). ``alpha``: Hashin fibre-tension shear contribution."""
    if criterion not in PLY_CRITERIA:
        raise ValueError(f"criterion: one of {PLY_CRITERIA}")
    if not isinstance(strength, OrthotropicStrength):
        raise TypeError("ply strengths are an OrthotropicStrength (user input)")
    if criterion == "tsai_wu":
        c = Criterion("tsai_wu", strength)
        idx = [0, 1, 5]                                   # σ11, σ22, τ12 inside the 6-component Voigt vector
        return (FailureMode("tsai_wu", c.F[np.ix_(idx, idx)], c.f[idx]),)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("0 <= alpha <= 1 is required")
    a = _hashin_strengths(strength)
    Xt, Xc, Yt, Yc, SL, ST = a["Xt"], a["Xc"], a["Yt"], a["Yc"], a["S12"], a["S23"]
    zero = np.zeros(3)
    mc_linear = ((Yc / (2.0 * ST)) ** 2 - 1.0) / Yc
    return (FailureMode("fibre_tension", np.diag([1 / Xt ** 2, 0.0, alpha / SL ** 2]), zero, 0, 1),
            FailureMode("fibre_compression", np.diag([1 / Xc ** 2, 0.0, 0.0]), zero, 0, -1),
            FailureMode("matrix_tension", np.diag([0.0, 1 / Yt ** 2, 1 / SL ** 2]), zero, 1, 1),
            FailureMode("matrix_compression", np.diag([0.0, 1 / (2 * ST) ** 2, 1 / SL ** 2]),
                        np.array([0.0, mc_linear, 0.0]), 1, -1))


def ply_failure_index(stress_material: Any, criterion: str, strength: OrthotropicStrength,
                      alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(index, mode name) for ply-axis stresses (..., 3) = [σ11, σ22, τ12] (module notes)."""
    s = np.asarray(stress_material, float)
    if s.shape[-1] != 3 or not np.isfinite(s).all():
        raise ValueError("stresses must be finite with last axis [s11, s22, t12]")
    flat = s.reshape(-1, 3)
    if criterion == "tsai_wu":
        c = Criterion("tsai_wu", strength, eps=0.0)
        index = c.index(expand_plane(torch.as_tensor(flat, dtype=torch.float64))).numpy()
        names = np.full(len(flat), "tsai_wu", dtype=object)
    else:
        index = np.full(len(flat), -math.inf)
        names = np.full(len(flat), "", dtype=object)
        for m in failure_modes(criterion, strength, alpha):
            a = np.einsum("ni,ij,nj->n", flat, m.P, flat)
            b = flat @ m.p
            val = 0.5 * (b + np.sqrt(b * b + 4.0 * a))
            take = m.active(flat) & (val > index)
            index[take], names[take] = val[take], m.name
    return index.reshape(s.shape[:-1]), names.reshape(s.shape[:-1])


def load_factor_to_failure(stress_mechanical: Any, stress_residual: Any,
                           modes: Sequence[FailureMode]) -> tuple[float, str]:
    """(λ, mode) smallest λ ≥ 0 with σ_r + λ σ_m on an active mode surface, one stress point (3,) each.
    λ = 0 when the residual stress alone has reached a surface; (inf, "") when no λ ≥ 0 does."""
    sm = np.asarray(stress_mechanical, float).reshape(3)
    sr = np.asarray(stress_residual, float).reshape(3)
    best, name = math.inf, ""
    for m in modes:
        if bool(m.active(sr)) and float(m.value(sr)) >= 1.0:
            return 0.0, m.name
        a = float(sm @ m.P @ sm)
        b = float(2.0 * sr @ m.P @ sm + m.p @ sm)
        c = float(m.value(sr)) - 1.0
        roots: list[float] = []
        if a > 0.0:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                q = -0.5 * (b + math.copysign(math.sqrt(disc), b))       # no cancellation
                roots = [q / a] + ([c / q] if q != 0.0 else [])
        elif b != 0.0:
            roots = [-c / b]
        # a mode that switches on along the load path may already be exceeded where it switches on (fibre tension
        # with α > 0 is discontinuous at σ11 = 0): the switching point is then the first failure of that mode
        if m.component is not None and m.sign * sm[m.component] > 0.0:
            lam0 = -sr[m.component] / sm[m.component]
            if lam0 > 0.0 and float(m.value(sr + lam0 * sm)) >= 1.0:
                roots.append(lam0)
        for lam in roots:
            at = sr + lam * sm
            if m.component is not None and abs(at[m.component]) <= 1e-12 * float(np.abs(at).max()):
                at[m.component] = 0.0                      # a switching point belongs to the mode it switches on
            if lam >= 0.0 and lam < best and (bool(m.active(at)) or
                                              (m.component is not None and at[m.component] == 0.0)):
                best, name = float(lam), m.name
    return best, name


@dataclass(frozen=True)
class FirstPlyFailure:
    """Result of :func:`first_ply_failure`.

    ``index`` (n, 2): failure index of the total stress at every ply bottom / top surface; ``mode`` (n, 2) its mode.
    ``load_factor``: smallest multiplier of the applied N, M (thermal field fixed) at which a surface fails;
    ``critical_ply`` / ``critical_surface`` (0 bottom, 1 top) / ``critical_mode`` where it happens.
    """

    criterion: str
    index: np.ndarray
    mode: np.ndarray
    load_factor: float
    load_factors: np.ndarray
    critical_ply: int
    critical_surface: int
    critical_mode: str
    sign_change_inside_ply: bool


def first_ply_failure(laminate: Laminate, strength: OrthotropicStrength | Sequence[OrthotropicStrength],
                      criterion: str = "tsai_wu", N: Any = None, M: Any = None, dT0: float = 0.0,
                      dT1: float = 0.0, alpha: float = 1.0) -> FirstPlyFailure:
    """First-ply failure of ``laminate`` under resultants N, M and temperature ΔT0 + z ΔT1 (module notes).

    ``strength``: one for all plies or one per ply."""
    n = laminate.n_plies
    strengths = [strength] * n if isinstance(strength, OrthotropicStrength) else list(strength)
    if len(strengths) != n:
        raise ValueError(f"{len(strengths)} strengths for {n} plies")
    mech = laminate.ply_stresses(laminate.response(N, M)).stress_material
    resid = laminate.ply_stresses(laminate.response(None, None, dT0, dT1)).stress_material
    total = mech + resid
    index = np.zeros((n, 2))
    mode = np.full((n, 2), "", dtype=object)
    lam = np.full((n, 2), math.inf)
    lam_mode = np.full((n, 2), "", dtype=object)
    for k in range(n):
        modes = failure_modes(criterion, strengths[k], alpha)
        index[k], mode[k] = ply_failure_index(total[k], criterion, strengths[k], alpha)
        for s in range(2):
            lam[k, s], lam_mode[k, s] = load_factor_to_failure(mech[k, s], resid[k, s], modes)
    kc, sc = (int(v) for v in np.unravel_index(int(np.argmin(lam)), lam.shape))
    crosses = bool(np.any(total[:, 0, :2] * total[:, 1, :2] < 0.0))
    return FirstPlyFailure(criterion, index, mode, float(lam[kc, sc]), lam, kc, sc, str(lam_mode[kc, sc]), crosses)
