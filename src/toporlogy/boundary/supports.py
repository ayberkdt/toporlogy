"""Supports: fixed (any directions), roller, symmetry, pinned bore, springs.
"""
from __future__ import annotations

from collections.abc import Sequence
from toporlogy.geometry import Bore
from typing import Any, cast

import numpy as np

from toporlogy.boundary.part import PartInfo, _vec
from toporlogy.boundary.selectors import OnCylinder, OnPlane, Selector


# =============================================================================================== supports
class Support:
    """Base class: ``directions(info) -> (nodes (k,), D (k, m, dim))`` zero-displacement directions per node and
    ``springs(info) -> (nodes, K (k, dim, dim))``. ``name`` labels it in reports."""

    name = "support"
    kind = "support"

    def directions(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(0, np.int64), np.zeros((0, 0, info.dim))

    def springs(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(0, np.int64), np.zeros((0, info.dim, info.dim))

    def nodes(self, info: PartInfo) -> np.ndarray:
        a, _ = self.directions(info)
        b, _ = self.springs(info)
        return np.union1d(a, b)


class Fixed(Support):
    """Zero displacement of the selected nodes in ``directions`` (default all; ints = global axes, or vectors)."""

    kind = "fixed"

    def __init__(self, where: Selector, directions: Sequence[int | Sequence[float] | np.ndarray] | None = None,
                 name: str | None = None) -> None:
        self.where, self.dirs = where, directions
        self.name = name or f"Fixed {where!r}"

    def _D(self, dim: int) -> np.ndarray:
        if self.dirs is None:
            return np.eye(dim)
        rows = []
        for d in self.dirs:
            if np.isscalar(d):
                rows.append(np.eye(dim)[int(np.asarray(d).item())])
            else:
                v = _vec(d, dim, "direction")
                rows.append(v / np.linalg.norm(v))
        return np.array(rows)

    def directions(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        n = self.where.select(info)
        D = self._D(info.dim)
        return n, np.broadcast_to(D, (len(n),) + D.shape).copy()


class Roller(Fixed):
    """Zero displacement along ``normal`` (any direction) at the selected nodes; free in the tangent plane."""

    kind = "roller"

    def __init__(self, where: Selector, normal: Sequence[float] | np.ndarray | int, name: str | None = None) -> None:
        super().__init__(where, [normal], name or f"Roller {where!r}")


def SymmetrySupport(point: Sequence[float] | np.ndarray, normal: Sequence[float] | np.ndarray | int,
                    tol: float | None = None, name: str | None = None) -> Roller:
    """Symmetry plane: zero normal displacement of the part's surface nodes on the plane."""
    sel = OnPlane(point, normal, tol=tol, facing=None)
    return Roller(sel, sel.normal, name or f"Symmetry {sel!r}")


class PinnedBore(Support):
    """Pin in a bore: radial displacement zero (local cylindrical frame), rotation about the pin free.

    ``axial=True`` also fixes the axial displacement, ``tangential=True`` the tangential one (= fixed bore).
    ``bore``: a :class:`~toporlogy.geometry.Bore` or an :class:`OnCylinder` selector.
    """

    kind = "pinned"

    def __init__(self, bore: Bore | OnCylinder, axial: bool = False, tangential: bool = False,
                 name: str | None = None, **select_kw: Any) -> None:
        self.sel = bore if isinstance(bore, OnCylinder) else OnCylinder.from_bore(bore, facing="inward", **select_kw)
        self.axial, self.tangential = axial, tangential
        self.name = name or f"Pinned {self.sel!r}"

    def directions(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        n = self.sel.select(info)
        _, _, er = self.sel.coords(info.X[n])
        rows = [er]
        if info.dim == 3:
            a = np.broadcast_to(cast(np.ndarray, self.sel.axis), er.shape)   # 3-D OnCylinder: axis is set
            if self.axial:
                rows.append(a.copy())
            if self.tangential:
                rows.append(np.cross(a, er))
        elif self.tangential:
            rows.append(np.column_stack([-er[:, 1], er[:, 0]]))
        bad = np.linalg.norm(er, axis=1) == 0
        if bad.any():
            raise ValueError(f"{self.name}: {int(bad.sum())} node(s) on the pin axis (radial direction undefined)")
        return n, np.stack(rows, axis=1)


class ElasticSupport(Support):
    """Springs to ground on surface nodes: stiffness per unit area ``k`` × the node's geometric area.

    ``direction``: "normal" (k n̂ n̂ᵀ with the outward normal), "all" (k I) or a vector d (k d̂ d̂ᵀ). Alternatively
    ``total`` = total stiffness, shared in proportion to area (then ``k`` is ignored).
    """

    kind = "spring"

    def __init__(self, where: Selector, k: float = 0.0, direction: str | Sequence[float] = "normal",
                 total: float | None = None, name: str | None = None) -> None:
        self.where, self.k, self.direction, self.total = where, float(k), direction, total
        self.name = name or f"Springs {where!r}"

    def springs(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        n = self.where.select(info)
        _, area = info.patch(n)
        if not area.sum() > 0:
            raise ValueError(f"{self.name}: the selection contains no complete surface face (area 0)")
        kn = (self.total * area / area.sum()) if self.total is not None else self.k * area
        dim = info.dim
        if isinstance(self.direction, str) and self.direction == "normal":
            nn = info.patch_normals(n)
            B = nn[:, :, None] * nn[:, None, :]
        elif isinstance(self.direction, str) and self.direction == "all":
            B = np.broadcast_to(np.eye(dim), (len(n), dim, dim))
        elif isinstance(self.direction, str):
            raise ValueError("direction: 'normal' | 'all' | vector")
        else:
            d = _vec(self.direction, dim, "spring direction")
            d = d / np.linalg.norm(d)
            B = np.broadcast_to(np.outer(d, d), (len(n), dim, dim))
        keep = kn > 0
        return n[keep], (kn[:, None, None] * B)[keep]
