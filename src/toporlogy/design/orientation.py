"""Fibre angle as a design block (2-D), named design blocks and their flat vector (02_MATERIAL_DESIGN §1, §3;
record docs/olcumler/malzeme_tasarimi.md).

Angle variable
--------------
Per design element (``location="element"``) or one for the whole part (``"global"``), a box variable s ∈ [0, 1]
with θ = θ_lo + (θ_hi − θ_lo) s (radians, material axis 1 from global x about z). Elements that are not design
elements (keep, void, outside) carry the fixed ``keep_angle``.

Periodicity
-----------
The 2-D design turns the reduced in-plane (3, 3) matrix, which holds only components with four in-plane indices;
a turn by π about z (diag(−1, −1, 1)) leaves those unchanged, so D(θ + π) = D(θ) for every material. (The full 3-D
tensor is π-periodic about z only when z is a two-fold material axis: :mod:`toporlogy.core.material_field`.) A
material with a four-fold axis along z (tetragonal / cubic in-plane symmetry) has period π/2; an isotropic one
does not depend on θ at all and an angle block is refused for it (zero gradient, an ill-posed design variable).
:func:`angle_period` finds the period numerically.

Filtering and the wrap-around
-----------------------------
θ cannot be filtered linearly: +89° and −89° are nearly the same fibre, their mean 0° is the opposite one. The
filter acts on the periodic representation (cos kθ, sin kθ), k = 2π / period, with the density filter's cone
weights, and θ̃ = atan2(S̃, C̃) / k. For k = 2 in 2-D this is the same as filtering the orientation tensor n ⊗ n
= ½ (I + [[cos 2θ, sin 2θ], [sin 2θ, −cos 2θ]]) and taking its principal axis, so no separate orientation-tensor
path is kept. The filtered vector's length ρ̃ ∈ [0, 1] (:attr:`PeriodicAngleFilter.coherence`) says how well the
neighbourhood agrees; at ρ̃ → 0 the direction is undefined and dθ̃/dx diverges (∝ 1/ρ̃): a known weakness of
every normalising orientation filter, reported, not hidden.

Named blocks
------------
:class:`DesignLayout` holds the blocks of the flat optimisation vector (density, angle, ...) with their location,
size, bounds and units; :meth:`DesignLayout.split` / :meth:`~DesignLayout.join` map between the vector and the
blocks. Element angles are filtered over the part and then reset to ``keep_angle`` outside the design elements
(the angle counterpart of ``Passive``); a global angle applies to every element of the part.
:class:`OrientedEmbed` is the first chain link of a density + angle design: it maps the vector to the
density grid (which the rest of the chain filters / projects as usual) and computes the angle field, which the
tensor interpolation takes (:meth:`take`) once per chain evaluation.

3-D orientation is not implemented (refused).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.core.material_field import rotate_in_plane
from toporlogy.design.chain import DensityFilter, Embed

__all__ = ["angle_period", "wrap_angle", "directional_modulus", "directional_compliance_derivative",
           "PeriodicAngleFilter", "AngleBlock", "DesignBlock", "DesignLayout", "OrientedEmbed", "StaleFieldError"]


class StaleFieldError(RuntimeError):
    """A published field (angle, material field) was read although it does not belong to the current evaluation."""


def angle_period(Ds: Sequence[Any], rtol: float = 1e-10) -> float:
    """Smallest period in {π/2, π} of θ ↦ D(θ) common to the in-plane matrices ``Ds`` (3, 3) (module notes).
    Raises when every D is independent of θ (isotropic in-plane)."""
    rng = np.random.default_rng(0)
    probes = torch.as_tensor(rng.uniform(-math.pi, math.pi, 7), dtype=torch.float64)
    iso, quarter = True, True
    for D in Ds:
        Dt = torch.as_tensor(np.asarray(D, float))
        scale = float(Dt.abs().max())
        R0, Rq = rotate_in_plane(Dt, probes), rotate_in_plane(Dt, probes + math.pi / 2)
        Rx = rotate_in_plane(Dt, probes + 0.37)
        iso = iso and float((Rx - R0).abs().max()) <= rtol * scale
        quarter = quarter and float((Rq - R0).abs().max()) <= rtol * scale
    if iso:
        raise ValueError("the material does not depend on the in-plane angle (isotropic in-plane): an angle design "
                         "variable would have zero gradient")
    return math.pi / 2 if quarter else math.pi


def wrap_angle(theta: Any, period: float = math.pi) -> Any:
    """θ mapped to (−period/2, period/2]."""
    return theta - period * np.round(np.asarray(theta) / period) if not isinstance(theta, torch.Tensor) else \
        theta - period * torch.round(theta / period)


def directional_modulus(E1: float, E2: float, G12: float, nu12: float, theta: Any) -> Any:
    """Young's modulus along x of an orthotropic plane-stress lamina with axis 1 at ``theta`` (radians):
    1/E_x = cos⁴θ/E1 + (1/G12 − 2ν12/E1) sin²θ cos²θ + sin⁴θ/E2 (compliance transformation of the lamina)."""
    c, s = np.cos(theta), np.sin(theta)
    return 1.0 / (c ** 4 / E1 + (1.0 / G12 - 2.0 * nu12 / E1) * s ** 2 * c ** 2 + s ** 4 / E2)


def directional_compliance_derivative(E1: float, E2: float, G12: float, nu12: float, theta: Any) -> Any:
    """d(1/E_x)/dθ of :func:`directional_modulus` (written out)."""
    c, s = np.cos(theta), np.sin(theta)
    a = 1.0 / G12 - 2.0 * nu12 / E1
    return -4 * c ** 3 * s / E1 + a * (2 * s * c ** 3 - 2 * s ** 3 * c) + 4 * s ** 3 * c / E2


class PeriodicAngleFilter(torch.nn.Module):
    """θ (domain.shape) → θ̃ by filtering (cos kθ, sin kθ) with the cone weights of :class:`DensityFilter`
    (module notes). ``coherence`` holds the last ρ̃ (domain.shape, 0 outside the part, no gradient)."""

    def __init__(self, domain: Domain, rmin: float, period: float = math.pi, backend: str = "auto") -> None:
        super().__init__()
        if not period > 0:
            raise ValueError("period > 0 is required")
        self.period, self.k = float(period), 2.0 * math.pi / float(period)
        self.filter = DensityFilter(domain, rmin, backend=backend)
        self.register_buffer("mask", torch.as_tensor(domain.active))
        self.coherence: torch.Tensor | None = None

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        c, s = self.filter(torch.cos(self.k * theta)), self.filter(torch.sin(self.k * theta))
        m = self.mask
        cs, ss = torch.where(m, c, torch.ones_like(c)), torch.where(m, s, torch.zeros_like(s))
        with torch.no_grad():
            self.coherence = torch.where(m, torch.sqrt(cs ** 2 + ss ** 2), torch.zeros_like(cs))
        out = torch.atan2(ss, cs) / self.k
        return torch.where(m, out, theta)


@dataclass(frozen=True)
class AngleBlock:
    """Settings of an angle design block: ``location`` "element" | "global"; ``bounds`` (θ_lo, θ_hi) and ``start``
    and ``keep_angle`` in radians; ``rmin`` (elements) of the periodic filter, None = unfiltered; ``period`` of
    the material (:func:`angle_period`), set by the interpolation when None."""

    location: str = "element"
    bounds: tuple[float, float] = (-math.pi / 2, math.pi / 2)
    start: float = 0.0
    keep_angle: float = 0.0
    rmin: float | None = None
    period: float | None = None

    def __post_init__(self) -> None:
        if self.location not in ("element", "global"):
            raise ValueError(f'angle location: element | global, got "{self.location}"')
        lo, hi = (float(b) for b in self.bounds)
        if not hi > lo:
            raise ValueError("angle bounds: hi > lo is required")
        if not lo <= self.start <= hi:
            raise ValueError("angle start must lie within the bounds")
        if self.location == "global" and self.rmin is not None:
            raise ValueError("a global angle has nothing to filter (rmin must be None)")
        if self.rmin is not None and not self.rmin > 0:
            raise ValueError("angle rmin > 0 is required")

    @classmethod
    def from_degrees(cls, spec: Mapping[str, Any]) -> AngleBlock:
        """From a mapping in degrees ``{location, bounds: [lo, hi], start, keep_angle, rmin}``."""
        allowed = {"location", "bounds", "start", "keep_angle", "rmin"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"angle: unknown key(s) {sorted(unknown)}; allowed {sorted(allowed)}")
        b = spec.get("bounds", (-90.0, 90.0))
        if len(b) != 2:
            raise ValueError("angle bounds: [lo, hi] in degrees")
        r = spec.get("rmin")
        return cls(str(spec.get("location", "element")), (math.radians(float(b[0])), math.radians(float(b[1]))),
                   math.radians(float(spec.get("start", 0.0))), math.radians(float(spec.get("keep_angle", 0.0))),
                   None if r is None else float(r))

    def s_of(self, theta: float) -> float:
        lo, hi = self.bounds
        return (float(theta) - lo) / (hi - lo)


@dataclass(frozen=True)
class DesignBlock:
    """One named block of the flat design vector: ``size`` entries from ``offset``; ``location`` element | global;
    box ``bounds`` of the optimisation variable (always [0, 1] here) and the physical ``units`` it maps to."""

    name: str
    offset: int
    size: int
    location: str
    units: str
    bounds: tuple[float, float] = (0.0, 1.0)


class DesignLayout:
    """Ordered named blocks of the optimisation vector (module notes)."""

    def __init__(self, blocks: Sequence[tuple[str, int, str, str]]) -> None:
        out, off = [], 0
        for name, size, location, units in blocks:
            if int(size) < 0:
                raise ValueError(f"block {name}: size >= 0 is required")
            out.append(DesignBlock(str(name), off, int(size), str(location), str(units)))
            off += int(size)
        names = [b.name for b in out]
        if len(set(names)) != len(names):
            raise ValueError(f"block names must be unique: {names}")
        self.blocks: tuple[DesignBlock, ...] = tuple(out)
        self.size = off

    def __getitem__(self, name: str) -> DesignBlock:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"no design block {name!r}; blocks are {[b.name for b in self.blocks]}")

    def __contains__(self, name: object) -> bool:
        return any(b.name == name for b in self.blocks)

    def split(self, x: Any) -> dict[str, Any]:
        """Block name → its slice of ``x`` (NumPy or torch; views)."""
        if len(x) != self.size:
            raise ValueError(f"design vector has {len(x)} entries, the layout {self.size}")
        return {b.name: x[b.offset:b.offset + b.size] for b in self.blocks}

    def join(self, parts: Mapping[str, Any]) -> np.ndarray:
        """Flat vector from block name → values (every block required)."""
        missing = [b.name for b in self.blocks if b.name not in parts]
        if missing:
            raise ValueError(f"missing block(s) {missing}")
        out = np.empty(self.size)
        for b in self.blocks:
            v = np.asarray(parts[b.name], float).reshape(-1)
            if len(v) != b.size:
                raise ValueError(f"block {b.name}: {len(v)} entries, expected {b.size}")
            out[b.offset:b.offset + b.size] = v
        return out

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": b.name, "offset": b.offset, "size": b.size, "location": b.location, "units": b.units,
                 "bounds": list(b.bounds)} for b in self.blocks]


class OrientedEmbed(Embed):
    """First chain link of a density + angle design: vector [density block (n_design) | angle block (n_design or
    1)] → density grid (keep 1, void 0, outside 0, as :class:`Embed`); the angle field θ̃ (domain.shape, radians) is
    computed in the same call and kept for :meth:`take`. ``density=False``: the density is not designed (every
    active element 1) and the vector is the angle block alone. 2-D only."""

    design_mask: torch.Tensor

    def __init__(self, domain: Domain, angle: AngleBlock, density: bool = True) -> None:
        super().__init__(domain)
        if domain.dim != 2:
            raise ValueError("the angle design block is 2-D only (3-D orientation is not implemented)")
        self.angle_block, self.density = angle, bool(density)
        n_ang = domain.n_design if angle.location == "element" else 1
        if angle.location == "element" and n_ang == 0:
            raise ValueError("an element angle block needs design elements")
        blocks = [("density", domain.n_design if self.density else 0, "element", "1"),
                  ("angle", n_ang, angle.location, "rad")]
        self.layout = DesignLayout(blocks)
        self.n_in = self.layout.size
        self.register_buffer("design_mask", torch.as_tensor(domain.design))
        self.angle_filter: PeriodicAngleFilter | None = None
        self._domain = domain
        self._angle: torch.Tensor | None = None
        self._fresh = False
        self.n_published = 0
        if angle.period is not None:
            self.set_period(angle.period)

    def set_period(self, period: float) -> None:
        """Period of the material's θ dependence (the interpolation calls this with :func:`angle_period`)."""
        self.period = float(period)
        if self.angle_block.rmin is not None:
            self.angle_filter = PeriodicAngleFilter(self._domain, self.angle_block.rmin, self.period)

    def angle_raw(self, s: torch.Tensor) -> torch.Tensor:
        """Unfiltered angle grid (radians) from the angle block ``s``."""
        lo, hi = self.angle_block.bounds
        th = lo + (hi - lo) * s
        if self.angle_block.location == "global":
            return th.reshape(()) * torch.ones(self.shape, dtype=s.dtype)
        g = torch.full(self.shape, float(self.angle_block.keep_angle), dtype=s.dtype).reshape(-1)
        g = g.index_put((self.idx,), th)
        return g.reshape(self.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() != self.n_in:
            raise ValueError(f"design vector has {x.numel()} entries, the layout {self.n_in} "
                             f"({self.layout.describe()})")
        if not hasattr(self, "period"):
            raise ValueError("OrientedEmbed: the material period is not set (build the tensor interpolation with "
                             "this link, or pass AngleBlock(period=...))")
        parts = self.layout.split(x)
        th = self.angle_raw(parts["angle"])
        if self.angle_filter is not None:          # filtered, then non-design elements reset (as Passive does)
            th = torch.where(self.design_mask, self.angle_filter(th),
                             torch.full_like(th, float(self.angle_block.keep_angle)))
        self._angle, self._fresh = th, True
        self.n_published += 1
        if self.density:
            return self._one(parts["density"])
        return torch.where(torch.as_tensor(self._domain.active), torch.ones(self.shape, dtype=x.dtype),
                           torch.zeros(self.shape, dtype=x.dtype)) + 0.0 * x.sum()

    def take(self) -> torch.Tensor:
        """The angle field of the last forward pass; each pass may be taken once (a second read, or a read without
        a new pass, raises :class:`StaleFieldError`: the angle would belong to another design)."""
        if not self._fresh or self._angle is None:
            raise StaleFieldError("the angle field was already used or the chain did not run for this design "
                                  "(evaluate through the chain: Study.evaluate_density is not supported with an "
                                  "angle block)")
        self._fresh = False
        return self._angle

    def reduce(self, x_design: np.ndarray) -> np.ndarray:
        """Start vector from a density start (n_design,): the angle block at ``start``."""
        n_ang = self.layout["angle"].size
        parts = {"angle": np.full(n_ang, self.angle_block.s_of(self.angle_block.start))}
        parts["density"] = np.asarray(x_design, float).reshape(-1) if self.density else np.zeros(0)
        return self.layout.join(parts)
