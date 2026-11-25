"""Topology of a solid: bodies, enclosed voids (cavities), Euler characteristic and tunnels (genus).

Grid version (:func:`grid_topology`): the solid is the union of closed sample cubes (26-connected solid,
6-connected void — the pairing for which the cubical complex is a valid digital topology). The Euler characteristic
of the cubical complex is χ = V − E + F − C, counted with array shifts (vertices, edges, faces belonging to at
least one solid cube). For a solid, χ = bodies − tunnels + cavities, so tunnels = bodies + cavities − χ.

Mesh version (:func:`toporlogy.post.mesh.mesh_topology`): for every closed connected triangle surface
χ = V − E + F = 2 − 2g; the solid bounded by the surfaces has χ = Σ χ_surface / 2. Outer shells and cavity shells
are told apart by the sign of their enclosed volume.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from toporlogy.core.bodies import label_bodies

__all__ = ["euler_cubical", "grid_topology", "same_topology", "enclosed_voids"]

_S6 = ndimage.generate_binary_structure(3, 1)


def euler_cubical(solid: np.ndarray) -> int:
    """χ of the union of closed unit cubes at the True entries of ``solid`` (3-D)."""
    a = np.pad(np.asarray(solid, bool), 1)
    C = int(a.sum())
    F = 0
    E = 0
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
        F += int((a[tuple(sl0)] | a[tuple(sl1)]).sum())
    # edges along axis ax are shared by the 4 cubes around them in the other two axes
    for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        acc = None
        for du in (0, 1):
            for dv in (0, 1):
                sl = [slice(None)] * 3
                sl[u] = slice(du, a.shape[u] - 1 + du)
                sl[v] = slice(dv, a.shape[v] - 1 + dv)
                part = a[tuple(sl)]
                acc = part if acc is None else (acc | part)
        assert acc is not None
        E += int(acc.sum())
    acc = None
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                part = a[dx:a.shape[0] - 1 + dx, dy:a.shape[1] - 1 + dy, dz:a.shape[2] - 1 + dz]
                acc = part if acc is None else (acc | part)
    assert acc is not None
    V = int(acc.sum())
    return V - E + F - C


def enclosed_voids(solid: np.ndarray) -> np.ndarray:
    """Void samples (6-connected components) that do not reach the grid border."""
    void = ~np.asarray(solid, bool)
    lab, n = ndimage.label(np.pad(void, 1, constant_values=True), structure=_S6)
    outer = lab[0, 0, 0]
    inner = (lab != outer) & (lab > 0)
    return inner[1:-1, 1:-1, 1:-1]


def grid_topology(solid: np.ndarray, h: float | None = None) -> dict[str, Any]:
    """Bodies, cavities, χ and tunnels (Σ genus) of a boolean solid; volumes in h³ if ``h`` is given."""
    solid = np.asarray(solid, bool)
    nb = len(label_bodies(solid, "vertex")[1])                # 26-connected solid
    cav = enclosed_voids(solid)
    _, nc = ndimage.label(cav, structure=_S6)
    chi = euler_cubical(solid)
    out: dict[str, Any] = {"bodies": int(nb), "cavities": int(nc), "euler": int(chi), "tunnels": int(nb + nc - chi)}
    if h is not None:
        out["volume_voxels"] = float(solid.sum()) * h ** 3
    return out


def same_topology(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a[k] == b[k] for k in ("bodies", "cavities", "euler"))
