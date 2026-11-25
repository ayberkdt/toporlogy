"""Several solid phases + void in one design (docs/olcumler/cok_malzeme.md).

Scope
-----
m solid phases that share ONE unit elasticity matrix (same ν, same orientation); they differ only in Young's modulus
E_i, mass density ρ_i and cost c_i. The physics is unchanged: the interpolation gives one effective E per element,
exactly as for one material. Phases with their own ν or anisotropy need an element-wise elasticity matrix (not here).

Interpolation
-------------
Phases are listed in increasing E (E_1 ≤ … ≤ E_m). Every element carries m design fields t_1 … t_m ∈ [0, 1] (box
bounds only, so MMA / GCMMA apply unchanged). The modulus is the recursion

    E^(0) = E_min,   E^(j) = E^(j−1) + t_j^p (E_j − E^(j−1))   (j = 1 … m),   E_e = E^(m) · frac_e

and the volume fractions of void and phases (unpenalised, they enter the volume / mass / cost responses) are

    η_m = t_m,   η_j = t_j Π_{k>j} (1 − t_k),   η_void = Π_k (1 − t_k)          (Σ η = 1 by construction).

This is the recursion of Gangl, Keith, Kim, Lazarov & Surowiec (2026, arXiv:2605.11994, §6.1), written there on the
simplex as E^(j) = E^(j−1) [1 − (η̃_j/S_j)^p] + E_j (η̃_j/S_j)^p with S_j = Σ_{i≤j} η̃_i, void first (E = 1e-6);
t_j = η_j / S_j maps the simplex onto the box and back (S_m = 1, S_{j−1} = S_j (1 − t_j)). Read from the paper's
HTML (v2) by a sub-agent of this session; the starting value E^(1) of the paper's recursion is not stated there, the
void modulus is used. Difference to the paper: the filter and projection act on t (box fields), the paper filters η.

Why increasing E: for two phases E = E_1 + t_2^p (E_2 − E_1) once t_1 = 1; with E_2 > E_1 an intermediate t_2 is
less stiff than the linear mixture while its volume is linear in t_2, so mixtures are uneconomical (the SIMP
argument). With E_2 < E_1 the mixture would be stiffer than linear and favoured, so a decreasing order is rejected.

One phase: E = E_min + t^p (E_1 − E_min), the operations of :class:`~toporlogy.design.interp.SIMP` in the same order
(bit-identical, tests/toporlogy/test_multi_material.py).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.references import cite

__all__ = ["Phase", "MultiPhaseSIMP", "phase_fractions", "phase_labels", "phase_report", "phase_start",
           "is_multiphase", "as_phases"]


@dataclass(frozen=True)
class Phase:
    """One solid phase: Young's modulus ``E`` (force/length²), mass ``density`` (mass/length³), ``cost`` per unit
    volume of fully dense phase (any currency/length³)."""

    name: str
    E: float
    density: float = 1.0
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.E > 0:
            raise ValueError(f"phase {self.name}: E > 0 is required")
        if self.density < 0 or self.cost < 0:
            raise ValueError(f"phase {self.name}: density and cost must be >= 0")


def as_phases(spec: Sequence[Any]) -> tuple[Phase, ...]:
    """Phases from :class:`Phase` objects or mappings ``{name, E, density, cost}`` (density, cost default 1)."""
    out = []
    for i, p in enumerate(spec):
        if isinstance(p, Phase):
            out.append(p)
            continue
        if not isinstance(p, Mapping):
            raise ValueError(f"phase {i}: expected a mapping {{name, E, density, cost}}")
        unknown = set(p) - {"name", "E", "density", "cost"}
        if unknown:
            raise ValueError(f"phase {i}: unknown key(s) {sorted(unknown)}; allowed name, E, density, cost")
        if "E" not in p:
            raise ValueError(f"phase {i}: E is required")
        out.append(Phase(str(p.get("name", f"phase{i + 1}")), float(p["E"]), float(p.get("density", 1.0)),
                         float(p.get("cost", 1.0))))
    names = [p.name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"phase names must be unique: {names}")
    if not out:
        raise ValueError("at least one phase is required")
    return tuple(out)


def phase_fractions(t: torch.Tensor) -> torch.Tensor:
    """Volume fractions ``(m + 1, *shape)`` = [void, phase 1, …, phase m] of the fields ``t`` ``(m, *shape)``
    (module notes); differentiable."""
    rem = torch.ones_like(t[0])
    eta: list[torch.Tensor] = []
    for j in range(t.shape[0] - 1, -1, -1):
        eta.append(t[j] * rem)
        rem = rem * (1.0 - t[j])
    return torch.stack([rem] + eta[::-1])


@cite("gangl_2026", note="recursive multi-phase SIMP of Sec. 6.1, written in box variables t_j = eta_j / S_j")
class MultiPhaseSIMP(torch.nn.Module):
    """Recursive multi-phase SIMP (module notes): fields ``(m, *shape)`` → element modulus ``(*shape)``.

    ``E_min = emin_ratio · E_m`` (the stiffest phase; for one phase the E0 of :class:`SIMP`). ``penal`` may be
    scheduled. ``phases``: :class:`Phase` objects or mappings, in increasing E.
    """

    frac: torch.Tensor

    def __init__(self, domain: Domain, phases: Sequence[Any], penal: float = 3.0, emin_ratio: float = 1e-6) -> None:
        super().__init__()
        self.phases = as_phases(phases)
        Es = [p.E for p in self.phases]
        if any(b < a for a, b in zip(Es, Es[1:], strict=False)):
            raise ValueError(f"phases must be listed in increasing E (got {Es}): the recursion penalises mixtures "
                             "only when the outer phase is the stiffer one (toporlogy.design.phases notes)")
        self.penal, self.emin_ratio, self.dim = float(penal), float(emin_ratio), domain.dim
        self.E0 = Es[-1]
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    @property
    def n_fields(self) -> int:
        return len(self.phases)

    def _fields(self, xphys: torch.Tensor) -> torch.Tensor:
        if xphys.dim() == self.dim:
            if self.n_fields != 1:
                raise ValueError(f"{self.n_fields} phases need fields of shape (m, *domain.shape)")
            return xphys[None]
        if xphys.dim() != self.dim + 1 or xphys.shape[0] != self.n_fields:
            raise ValueError(f"fields of shape {tuple(xphys.shape)}, {self.n_fields} phases need "
                             f"({self.n_fields}, *domain.shape)")
        return xphys

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        t = self._fields(xphys)
        Emin = self.emin_ratio * self.E0
        E: Any = Emin
        for j, ph in enumerate(self.phases):
            E = E + t[j] ** self.penal * (ph.E - E)
        return E * self.frac.to(xphys.dtype)

    def fractions(self, xphys: torch.Tensor) -> torch.Tensor:
        """Volume fractions [void, phase 1 … m] ``(m + 1, *shape)``."""
        return phase_fractions(self._fields(xphys))

    def solid(self, xphys: torch.Tensor) -> torch.Tensor:
        """Solid volume fraction 1 − η_void ``(*shape)``."""
        return 1.0 - self.fractions(xphys)[0]


def is_multiphase(interp: Any) -> bool:
    """True for an interpolation with more than one design field."""
    return int(getattr(interp, "n_fields", 1)) > 1


def phase_labels(interp: MultiPhaseSIMP, xphys: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """Dominant phase of every element (int array of ``domain.shape``): 0 void, i (1 … m) phase i; −1 outside
    ``active``. Ties go to the lower index."""
    with torch.no_grad():
        eta = interp.fractions(torch.as_tensor(np.asarray(xphys, float))).numpy()
    lab = np.argmax(eta, axis=0).astype(np.int64)
    if active is not None:
        lab[~np.asarray(active, bool)] = -1
    return lab


def phase_report(interp: MultiPhaseSIMP, xphys: np.ndarray, domain: Domain, lo: float = 0.1,
                 hi: float = 0.9) -> dict[str, Any]:
    """Per-phase volume fraction (element count, as :class:`~toporlogy.optimization.responses.PhaseVolume`), volume,
    mass and cost (frac-weighted, absolute), element counts of the label field, and the grey share.

    Grey share: active elements whose dominant fraction max_i η_i (void included) is below ``hi``; for one phase
    with ``lo = 1 − hi`` this is ``grey_fraction`` (lo < x̄ < hi).
    """
    with torch.no_grad():
        eta = interp.fractions(torch.as_tensor(np.asarray(xphys, float))).numpy()
    act = domain.active
    ve = domain.frac * domain.h ** domain.dim * (domain.thickness if domain.dim == 2 else 1.0)
    lab = phase_labels(interp, xphys, act)
    out: dict[str, Any] = {"names": ["void"] + [p.name for p in interp.phases], "phases": {}}
    for i, ph in enumerate(interp.phases, start=1):
        vol = float((ve * eta[i]).sum())
        out["phases"][ph.name] = {"E": ph.E, "density": ph.density, "cost": ph.cost,
                                  "volume_fraction": float(eta[i][act].sum() / domain.n_active),
                                  "volume": vol, "mass": ph.density * vol, "cost_total": ph.cost * vol,
                                  "elements": int((lab == i).sum())}
    out["void"] = {"volume_fraction": float(eta[0][act].sum() / domain.n_active), "elements": int((lab == 0).sum())}
    out["mass"] = float(sum(v["mass"] for v in out["phases"].values()))
    out["cost"] = float(sum(v["cost_total"] for v in out["phases"].values()))
    out["grey_fraction"] = float(np.mean(eta.max(axis=0)[act] < hi)) if act.any() else 0.0
    out["grey_rule"] = f"max_i eta_i < {hi:g} (void included)"
    return out


def phase_start(n_fields: int, n_design: int, fractions: Sequence[float | None]) -> np.ndarray:
    """Uniform start ``t`` (field-major, ``n_fields · n_design``) whose phase volume fractions are ``fractions``
    (one per phase; None = 0.5 for that field): t_m = f_m, t_j = f_j / Π_{k>j}(1 − t_k), clipped to [0, 1]."""
    if len(fractions) != n_fields:
        raise ValueError(f"{len(fractions)} fractions for {n_fields} phases")
    t = np.full(n_fields, 0.5)
    rem = 1.0
    for j in range(n_fields - 1, -1, -1):
        f = fractions[j]
        if f is not None:
            t[j] = min(max(float(f) / rem, 0.0), 1.0) if rem > 0 else 0.0
        rem *= 1.0 - t[j]
    return np.repeat(t, n_design)
