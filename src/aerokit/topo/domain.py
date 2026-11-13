"""Design domain on a regular grid (2-D quadrilaterals or 3-D voxels) — layer "Domain" of docs/mimari_plan.md §3.

The grid always covers the full bounding box. Elements outside the part have ``frac = 0`` (no stiffness, no design
variable); the nodes they alone touch carry no degrees of freedom in the solve. Every element field is an array of
shape :attr:`Domain.shape` whose C-order flattening is the element number.

Numbering (identical to the legacy voxel mesh used as a regression oracle in ``tests/topo``):

* 3-D element e = (ix·ny + iy)·nz + iz; node n = (ix·(ny+1) + iy)·(nz+1) + iz; DOF 3n + d.
  Element corners in the order of :data:`aerokit.topo.fem.elements.H8_CORNER`: (ix, iy, iz), (ix+1, iy, iz),
  (ix+1, iy+1, iz), (ix, iy+1, iz), then the same four at iz+1.
* 2-D element e = ix·ny + iy; node n = ix·(ny+1) + iy; DOF 2n + d. Corners counter-clockwise from lower-left
  (the DOF order of :func:`aerokit.topo.fem.elements.ke_q4`); y points up.

Coordinates: node (ix, iy[, iz]) sits at ``origin + h·(ix, iy[, iz])``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["Domain"]

_CORNERS_3D = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
_CORNERS_2D = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)


class Domain:
    """Regular grid with a part mask, partial volumes and frozen regions.

    Parameters
    ----------
    shape : (nx, ny) or (nx, ny, nz)
        Number of elements per axis.
    h : float
        Element edge length (any consistent unit system).
    origin : sequence of float, optional
        Coordinates of node (0, 0[, 0]).
    frac : array of ``shape``, optional
        Solid fraction of each element in [0, 1]; 0 = outside the part. Default: all 1 (full box).
    keep, void : bool arrays of ``shape``, optional
        Non-design elements forced solid / empty. ``void`` loses against ``keep``; both are restricted to the part.
    thickness : float
        Out-of-plane thickness (2-D only).
    """

    def __init__(self, shape: Sequence[int], h: float = 1.0, origin: Sequence[float] | None = None,
                 frac: Any = None, keep: Any = None, void: Any = None, thickness: float = 1.0) -> None:
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) not in (2, 3) or min(self.shape) < 1:
            raise ValueError("shape must be (nx, ny) or (nx, ny, nz) with positive entries")
        if not h > 0:
            raise ValueError("h > 0 is required")
        self.dim = len(self.shape)
        self.h = float(h)
        self.thickness = float(thickness)
        self.origin = np.zeros(self.dim) if origin is None else np.asarray(origin, float).reshape(self.dim)
        self.frac = np.ones(self.shape) if frac is None else np.asarray(frac, float).reshape(self.shape).copy()
        if (self.frac < 0).any() or (self.frac > 1).any():
            raise ValueError("frac must lie in [0, 1]")
        self.active = self.frac > 0
        if not self.active.any():
            raise ValueError("the domain has no element with frac > 0")
        self.keep = (np.zeros(self.shape, bool) if keep is None else np.asarray(keep, bool).reshape(self.shape)) \
            & self.active
        self.void = (np.zeros(self.shape, bool) if void is None else np.asarray(void, bool).reshape(self.shape)) \
            & self.active & ~self.keep
        self.design = self.active & ~self.keep & ~self.void
        self._edof: np.ndarray | None = None
        self.voxelization_info: dict[str, Any] | None = None   # set by geometry builders (rays used/dropped …)

    # ------------------------------------------------------------------ sizes
    @property
    def node_shape(self) -> tuple[int, ...]:
        return tuple(s + 1 for s in self.shape)

    @property
    def ne(self) -> int:
        """Elements in the box (active or not)."""
        return int(np.prod(self.shape))

    @property
    def nn(self) -> int:
        return int(np.prod(self.node_shape))

    @property
    def ndof(self) -> int:
        """DOFs of the box grid (``dim`` per node)."""
        return self.dim * self.nn

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    @property
    def n_design(self) -> int:
        return int(self.design.sum())

    @property
    def active_elements(self) -> np.ndarray:
        """Element numbers with frac > 0 (ascending)."""
        return np.flatnonzero(self.active.ravel())

    @property
    def design_elements(self) -> np.ndarray:
        """Element numbers that carry a design variable (ascending); the design vector follows this order."""
        return np.flatnonzero(self.design.ravel())

    # ------------------------------------------------------------------ connectivity
    def element_ijk(self, elements: np.ndarray | None = None) -> np.ndarray:
        """(n, dim) grid index of the given elements (default all)."""
        e = np.arange(self.ne) if elements is None else np.asarray(elements, np.int64)
        return np.column_stack(np.unravel_index(e, self.shape)).astype(np.int64)

    def node_number(self, ijk: np.ndarray) -> np.ndarray:
        """Node number of grid node indices ``ijk`` (..., dim)."""
        ijk = np.asarray(ijk, np.int64)
        ns = self.node_shape
        if self.dim == 3:
            return (ijk[..., 0] * ns[1] + ijk[..., 1]) * ns[2] + ijk[..., 2]
        return ijk[..., 0] * ns[1] + ijk[..., 1]

    @property
    def corners(self) -> np.ndarray:
        return _CORNERS_3D if self.dim == 3 else _CORNERS_2D

    @property
    def edof(self) -> np.ndarray:
        """(ne, dim·2^dim) element DOFs of every box element, node-major (u, v[, w] per node)."""
        if self._edof is None:
            ijk = self.element_ijk()
            nodes = np.stack([self.node_number(ijk + c) for c in self.corners], axis=1)
            self._edof = (self.dim * nodes[:, :, None] + np.arange(self.dim)).reshape(self.ne, -1)
        return self._edof

    @property
    def active_dofs(self) -> np.ndarray:
        """DOFs touched by at least one active element (ascending)."""
        return np.unique(self.edof[self.active_elements])

    # ------------------------------------------------------------------ geometry and selections
    def node_coords(self) -> np.ndarray:
        """(nn, dim) node coordinates."""
        idx = np.indices(self.node_shape).reshape(self.dim, -1).T
        return self.origin + self.h * idx

    def element_centers(self) -> np.ndarray:
        """(ne, dim) element centres."""
        return self.origin + self.h * (self.element_ijk() + 0.5)

    def nodes_where(self, predicate: Callable[[np.ndarray], Any], active_only: bool = True) -> np.ndarray:
        """Node numbers where ``predicate(X)`` is true (X: (nn, dim) coordinates); by default only nodes of the part."""
        hit = np.asarray(predicate(self.node_coords()), bool)
        if active_only:
            on = np.zeros(self.nn, bool)
            on[self.active_dofs[::1] // self.dim] = True
            hit &= on
        return np.flatnonzero(hit)

    def dofs(self, nodes: np.ndarray, directions: Sequence[int] | int | None = None) -> np.ndarray:
        """DOF numbers of ``nodes`` in ``directions`` (default all), sorted."""
        d = range(self.dim) if directions is None else ([directions] if np.isscalar(directions) else directions)
        nodes = np.asarray(nodes, np.int64)
        return np.sort(np.concatenate([self.dim * nodes + int(k) for k in d])) if len(nodes) else \
            np.zeros(0, np.int64)

    def load_vector(self, n_cases: int = 1) -> np.ndarray:
        """Zero load array (ndof, n_cases)."""
        return np.zeros((self.ndof, n_cases))

    def uniform_start(self, volume_fraction: float) -> np.ndarray:
        """Design vector that gives mean x̄ = ``volume_fraction`` over the part before filtering (topo3d start rule).

        Keep elements count as 1, void as 0; the free elements share the rest, clipped to [0, 1].
        """
        start = (volume_fraction * self.n_active - self.keep.sum()) / max(self.n_design, 1)
        return np.full(self.n_design, float(min(max(start, 0.0), 1.0)))

    def to_grid(self, design: np.ndarray, fill_keep: float = 1.0, fill_void: float = 0.0) -> np.ndarray:
        """Design vector → element array of ``shape`` (keep / void / outside filled)."""
        g = np.zeros(self.shape)
        g[self.keep] = fill_keep
        g[self.void] = fill_void
        g.ravel()[self.design_elements] = np.asarray(design, float)
        return g

    def __repr__(self) -> str:
        return (f"Domain(shape={self.shape}, h={self.h:g}, active={self.n_active}, design={self.n_design}, "
                f"keep={int(self.keep.sum())}, void={int(self.void.sum())})")
