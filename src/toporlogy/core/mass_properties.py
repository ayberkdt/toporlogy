"""Mass, centre of mass and inertia tensor of a density design on the regular grid (05 §3).

Element e has mass m_e = ρ0 · ρ(x̄_e) · frac_e · V_e with V_e = h^dim (2-D: × thickness), uniform inside the element
(a partial voxel is a full voxel of reduced density). Its own inertia about its centre is that of a box with sides
(h, h, h) in 3-D or (h, h, thickness) in 2-D; the tensor about a point o adds the parallel-axis term:

    I_o = Σ_e [ I_e^c + m_e (|r_e|² 1 − r_e r_eᵀ) ],   r_e = c_e − o.

2-D parts lie in the plane z = 0, symmetric through the thickness, so the 3 × 3 tensor has I_xz = I_yz = 0. Void
elements contribute nothing. NumPy or Torch in, the same out: with Torch every quantity differentiates in x̄.
Pre-registered checks: ``docs/olcumler/kutle_ozellikleri_onkayit.md`` (MP1-MP6).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["MassProperties", "mass_properties"]


@dataclass
class MassProperties:
    mass: Any                       # scalar
    centre: Any                     # (3,)
    inertia: Any                    # (3, 3) about ``about``
    about: np.ndarray               # (3,) the reference point


def _backend(x: Any) -> Any:
    if type(x).__module__.startswith("torch"):
        import torch
        return torch
    return np


def mass_properties(domain: Any, xphys: Any, rho0: float = 1.0, density: Callable[[Any], Any] | None = None,
                    about: Any = "centre") -> MassProperties:
    """Mass properties of ``xphys`` (``domain.shape``); ``density`` maps x̄ to the relative mass density (default
    x̄, linear). ``about``: "centre" (the centre of mass) or a point (2 or 3 coordinates)."""
    xp = _backend(xphys)
    dim = domain.dim
    if tuple(xphys.shape) != tuple(domain.shape):
        raise ValueError(f"xphys must have the domain shape {tuple(domain.shape)}, got {tuple(xphys.shape)}")
    use = np.asarray(domain.active, bool) & ~np.asarray(domain.void, bool)
    idx = np.flatnonzero(use.ravel())
    h = float(domain.h)
    sides = np.array([h, h, h]) if dim == 3 else np.array([h, h, float(domain.thickness)])
    V = float(np.prod(sides))
    c = np.zeros((len(idx), 3))
    c[:, :dim] = np.asarray(domain.origin, float) + h * (np.argwhere(use) + 0.5)      # C order = order of idx
    frac = np.asarray(domain.frac, float).ravel()[idx]
    x = xphys.reshape(-1)[xp.as_tensor(idx) if xp is not np else idx]
    rel = x if density is None else density(x)
    w = (rho0 * V) * frac
    m_e = rel * (xp.as_tensor(w, dtype=x.dtype) if xp is not np else w)
    M = m_e.sum()
    C = xp.as_tensor(c, dtype=x.dtype) if xp is not np else c
    centre = (m_e[:, None] * C).sum(0) / M
    o = centre if isinstance(about, str) and about == "centre" else None
    if o is None:
        p = np.zeros(3)
        p[:len(np.ravel(about))] = np.ravel(np.asarray(about, float))
        o = xp.as_tensor(p, dtype=x.dtype) if xp is not np else p
    r = C - o
    eye = xp.eye(3, dtype=x.dtype) if xp is not np else np.eye(3)
    own = np.diag([sides[1] ** 2 + sides[2] ** 2, sides[0] ** 2 + sides[2] ** 2, sides[0] ** 2 + sides[1] ** 2]) / 12
    own_t = xp.as_tensor(own, dtype=x.dtype) if xp is not np else own
    rr = (r * r).sum(1)
    I = M * own_t + (m_e[:, None, None] * (rr[:, None, None] * eye - r[:, :, None] * r[:, None, :])).sum(0)
    about_np = np.asarray(o if xp is np else o.detach().numpy(), float)  # type: ignore[union-attr]
    return MassProperties(M, centre, I, about_np)
