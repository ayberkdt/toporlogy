"""Several solid phases + void in one design (docs/olcumler/cok_malzeme.md).

Scope
-----
m solid phases that share ONE unit elasticity matrix (same ν, same orientation); they differ only in Young's modulus
E_i, mass density ρ_i and cost c_i. The physics is unchanged: the interpolation gives one effective E per element,
exactly as for one material. Phases with their own ν, anisotropy or orientation: :class:`TensorPhases` (below), which
mixes the phases' elasticity tensors and needs :class:`~toporlogy.fem.tensor_elastic.TensorElasticity`.

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

from toporlogy.core.domain import Domain, require_grid
from toporlogy.core.material_field import (FIELD_REQUESTS, PLANES, MaterialField, PhaseMaterial,
                                           as_phase_materials, rotate_in_plane)
from toporlogy.references import cite

__all__ = ["Phase", "MultiPhaseSIMP", "TensorPhases", "MIXING_RULES", "phase_fractions", "phase_labels",
           "phase_report", "phase_start", "is_multiphase", "as_phases"]


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
        require_grid(domain, "MultiPhaseSIMP")
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
    require_grid(domain, "phase_report")
    with torch.no_grad():
        eta = interp.fractions(torch.as_tensor(np.asarray(xphys, float))).numpy()
    act = domain.active
    ve = domain.frac * domain.h ** domain.dim * (domain.thickness if domain.dim == 2 else 1.0)
    lab = phase_labels(interp, xphys, act)
    out: dict[str, Any] = {"names": ["void"] + [p.name for p in interp.phases], "phases": {}}
    for i, ph in enumerate(interp.phases, start=1):
        vol = float((ve * eta[i]).sum())
        out["phases"][ph.name] = {"E": getattr(ph, "E", None), "density": ph.density, "cost": ph.cost,
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


# ---------------------------------------------------------------------------------------------- tensor phases
MIXING_RULES = ("simplex", "recursive")


class TensorPhases(torch.nn.Module):
    """Several solid phases + void, each with its own elasticity tensor (record docs/olcumler/malzeme_tasarimi.md).

    Fields ``t`` (m, *shape) ∈ [0, 1] (the box fields of :class:`MultiPhaseSIMP`, same filter / projection chain) give
    the volume fractions η = :func:`phase_fractions` (void + m phases, Σ η = 1). The element elasticity is a convex
    combination of positive definite tensors, hence positive definite (energy positivity by construction):

    * ``rule="simplex"``: C = (1 − Σ_i η_i^p) C_void + Σ_i η_i^p C_i, C_void = ε (1/m) Σ_i C_i (Σ η_i^p ≤ 1, so no
      weight is negative). Symmetric in the phases (a permutation of phases and fractions gives the same C).
    * ``rule="recursive"``: C⁽⁰⁾ = C_void = ε C_m, C⁽ʲ⁾ = C⁽ʲ⁻¹⁾ + t_j^p (C_j − C⁽ʲ⁻¹⁾); with C_i = E_i D it is
      :class:`MultiPhaseSIMP` times D. Depends on the phase order.

    ε = ``emin_ratio``. Both are penalisation rules, not homogenisation: a grey element is not claimed to be a
    physical mixture (isotropic Hashin-Shtrikman bounds do not apply to anisotropic C and are not used). With p = 1
    and identical phases both give the single material at solid fraction 1 − η_void; for p > 1 identical phases
    equal one material at the pure states only.

    2-D: each phase's tensor is reduced to (3, 3) by ``plane`` (stress | strain) first; a designed fibre angle
    (:class:`~toporlogy.design.orientation.OrientedEmbed`, :meth:`attach_angle`) turns the phases with
    ``rotate = True`` element by element (C_void follows the turned phases). ``frac`` is applied once, to C.

    ``forward`` returns the informational scalar grid trace(C_e) / trace(C_ref) (C_ref the phase of largest trace,
    0 outside the part) for ``Study`` and publishes the :class:`~toporlogy.core.material_field.MaterialField`, which
    :class:`~toporlogy.fem.tensor_elastic.TensorElasticity` takes once (:meth:`take_field`). ``request``: the
    fields to compute (:data:`~toporlogy.core.material_field.FIELD_REQUESTS`; conductivity needs every phase's).
    """

    D: torch.Tensor
    frac_active: torch.Tensor
    active_idx: torch.Tensor

    def __init__(self, domain: Domain, phases: Sequence[Any], rule: str = "simplex", penal: float = 3.0,
                 emin_ratio: float = 1e-6, plane: str = "stress", request: Sequence[str] = ("stiffness",)) -> None:
        super().__init__()
        require_grid(domain, "TensorPhases")
        if rule not in MIXING_RULES:
            raise ValueError(f"rule: one of {MIXING_RULES}, got {rule!r}")
        if plane not in PLANES:
            raise ValueError(f"plane: one of {PLANES}, got {plane!r}")
        if not (penal >= 1.0 and 0.0 < emin_ratio < 1.0):
            raise ValueError("penal >= 1 and 0 < emin_ratio < 1 are required")
        req = frozenset(request) | {"stiffness"}
        bad = sorted(req - set(FIELD_REQUESTS))
        if bad:
            raise ValueError(f"request: unknown field(s) {bad}; allowed {list(FIELD_REQUESTS)}")
        self.phases: tuple[PhaseMaterial, ...] = as_phase_materials(phases)
        if "conductivity" in req and any(p.conductivity is None for p in self.phases):
            raise ValueError("request conductivity: every phase needs a conductivity")
        self.rule, self.penal, self.emin_ratio, self.plane = rule, float(penal), float(emin_ratio), plane
        self.dim, self.shape, self.request = domain.dim, tuple(domain.shape), req
        Ds = np.array([p.stiffness(domain.dim, plane) for p in self.phases])
        self.register_buffer("D", torch.as_tensor(Ds, dtype=torch.float64))
        self.register_buffer("active_idx", torch.as_tensor(domain.active_elements, dtype=torch.long))
        self.register_buffer("frac_active", torch.as_tensor(domain.frac.reshape(-1)[domain.active_elements],
                                                            dtype=torch.float64))
        self.ref_trace = float(np.trace(Ds, axis1=1, axis2=2).max())
        self.angle_source: Any = None
        self.field: MaterialField | None = None
        self._fresh = False
        self.n_published = 0

    @property
    def n_fields(self) -> int:
        return len(self.phases)

    # ------------------------------------------------------------------ angle
    def attach_angle(self, source: Any) -> float:
        """Take the angle field from ``source`` (an :class:`~toporlogy.design.orientation.OrientedEmbed`) at every
        forward pass; sets the source's period from the turned phases and returns it."""
        from toporlogy.design.orientation import angle_period
        if self.dim != 2:
            raise ValueError("a designed angle is 2-D only")
        turned = [self.D[i].numpy() for i, p in enumerate(self.phases) if p.rotate]
        if not turned:
            raise ValueError("no phase has rotate = true: the angle block would not change the material")
        period = angle_period(turned)
        given = source.angle_block.period
        if given is not None and abs(given - period) > 1e-12:
            raise ValueError(f"angle block period {given} differs from the material's {period}")
        if given is None:
            source.set_period(period)
        self.angle_source = source
        return period

    # ------------------------------------------------------------------ fractions (PhaseVolume, Mass, Cost)
    def _fields(self, xphys: torch.Tensor) -> torch.Tensor:
        if xphys.dim() == self.dim:
            if self.n_fields != 1:
                raise ValueError(f"{self.n_fields} phases need fields of shape (m, *domain.shape)")
            return xphys[None]
        if xphys.dim() != self.dim + 1 or xphys.shape[0] != self.n_fields:
            raise ValueError(f"fields of shape {tuple(xphys.shape)}, {self.n_fields} phases need "
                             f"({self.n_fields}, *domain.shape)")
        return xphys

    def fractions(self, xphys: torch.Tensor) -> torch.Tensor:
        """Volume fractions [void, phase 1 … m] ``(m + 1, *shape)``."""
        return phase_fractions(self._fields(xphys))

    def solid(self, xphys: torch.Tensor) -> torch.Tensor:
        """Solid volume fraction 1 − η_void ``(*shape)``."""
        return 1.0 - self.fractions(xphys)[0]

    # ------------------------------------------------------------------ mixing
    def phase_stiffness(self, theta: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Element-level stiffness of each phase: (k, k), or (n, k, k) turned by ``theta`` (n,) for the phases
        with rotate = True."""
        return [rotate_in_plane(self.D[i], theta) if (theta is not None and p.rotate) else self.D[i]
                for i, p in enumerate(self.phases)]

    def _mix(self, items: Sequence[torch.Tensor], void: torch.Tensor, t: torch.Tensor,
             eta: torch.Tensor) -> torch.Tensor:
        """The rule (class notes) applied to per-phase values ``items`` and ``void``, all (n, ...) alike."""
        extra = (1,) * (void.dim() - 1)
        p = self.penal
        if self.rule == "simplex":
            wt = eta[1:] ** p
            out = (1.0 - wt.sum(0)).reshape(wt.shape[1:] + extra) * void
            for i, it in enumerate(items):
                out = out + wt[i].reshape(wt.shape[1:] + extra) * it
            return out
        out = void
        for j, it in enumerate(items):
            out = out + (t[j] ** p).reshape(t.shape[1:] + extra) * (it - out)
        return out

    def _void(self, items: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.rule == "simplex":
            return self.emin_ratio * (sum(items[1:], items[0]) / len(items))
        return self.emin_ratio * items[-1]

    def stiffness_of(self, t_active: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        """(n, k, k) mixed stiffness (without ``frac``) of fields ``t_active`` (m, n) and angles ``theta`` (n,)."""
        if t_active.dim() != 2 or t_active.shape[0] != self.n_fields:
            raise ValueError(f"fields of shape {tuple(t_active.shape)}, expected ({self.n_fields}, n)")
        n, k = t_active.shape[1], self.D.shape[-1]
        Cs = [c if c.dim() == 3 else c.expand(n, k, k) for c in self.phase_stiffness(theta)]
        return self._mix(Cs, self._void(Cs), t_active, phase_fractions(t_active))

    def stiffness_from_fractions(self, eta: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        """(n, k, k) ``simplex`` stiffness of volume fractions ``eta`` (m + 1, n) = [void, phases] (for checks of
        the permutation symmetry, which holds on the simplex, not in the box fields)."""
        if self.rule != "simplex":
            raise ValueError("stiffness_from_fractions: rule simplex only")
        n, k = eta.shape[1], self.D.shape[-1]
        Cs = [c if c.dim() == 3 else c.expand(n, k, k) for c in self.phase_stiffness(theta)]
        return self._mix(Cs, self._void(Cs), eta[1:], eta)

    def material_field(self, xphys: torch.Tensor, theta: torch.Tensor | None = None) -> MaterialField:
        """The :class:`MaterialField` of physical fields ``xphys`` and an angle grid ``theta`` (domain.shape)."""
        t = self._fields(xphys).reshape(self.n_fields, -1)[:, self.active_idx]
        th = None
        if theta is not None:
            if self.dim != 2:
                raise ValueError("a designed angle is 2-D only")
            th = theta.reshape(-1)[self.active_idx]
        fr = self.frac_active.to(t.dtype)
        C = self.stiffness_of(t, th) * fr[:, None, None]
        scale = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / self.ref_trace
        mass = cond = None
        if "mass_density" in self.request or "conductivity" in self.request:
            eta = phase_fractions(t)
            if "mass_density" in self.request:
                rho = torch.as_tensor([p.density for p in self.phases], dtype=t.dtype)
                mass = torch.tensordot(rho, eta[1:], dims=1)
            if "conductivity" in self.request:
                ks = [torch.full((t.shape[1],), float(p.conductivity or 0.0), dtype=t.dtype) for p in self.phases]
                cond = self._mix(ks, self._void(ks), t, eta) * fr
        return MaterialField(C, scale, self.dim, self.plane, mass, cond, self.request)

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        theta = self.angle_source.take() if self.angle_source is not None else None
        f = self.material_field(xphys, theta)
        self.field, self._fresh = f, True
        self.n_published += 1
        grid = torch.zeros(int(np.prod(self.shape)), dtype=f.scale.dtype)
        grid = grid.index_put((self.active_idx,), f.scale)
        return grid.reshape(self.shape)

    def take_field(self, E_active: torch.Tensor | None = None) -> MaterialField:
        """The field of the last forward pass, once; ``E_active`` (the grid the study indexed) must match it."""
        from toporlogy.design.orientation import StaleFieldError
        if not self._fresh or self.field is None:
            raise StaleFieldError("the material field was already used or the interpolation did not run for this "
                                  "design")
        if E_active is not None and not torch.equal(E_active.detach().reshape(-1), self.field.scale.detach()):
            raise StaleFieldError("the moduli handed to the physics are not those of the published material field")
        self._fresh = False
        return self.field
