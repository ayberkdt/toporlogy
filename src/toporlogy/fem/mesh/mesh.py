"""General unstructured mesh: coordinates, cells of one element type, physical tags, boundary facets, DOF map.

A :class:`Mesh` holds one element type (mixed meshes are refused, not silently split). Facet sets are arrays of
``(cell, local facet)`` pairs, so the owning cell (and with it the outward normal) is always known. DOFs are
node-major: DOF ``ncomp · node + component`` (the numbering of :class:`toporlogy.core.domain.Domain`), so the
adapter :meth:`Mesh.from_domain` gives the grid's own DOF numbers.

Generators :func:`rectangle` and :func:`box` build structured meshes of every element (triangles: each square split
along its (0,0)–(1,1) diagonal; tetrahedra: the six-tetrahedron Kuhn split of each cube along the main diagonal,
conforming between neighbours; quadratic nodes on the doubled grid). :func:`perturb` moves interior vertices and
places higher-order nodes on the straight-sided cell (optionally curved).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import permutations
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from toporlogy.fem.mesh.reference import ReferenceElement, get

__all__ = ["Mesh", "rectangle", "box", "perturb", "straighten"]


@dataclass
class Mesh:
    """Nodes, cells and named sets.

    Parameters
    ----------
    points : (n_nodes, dim) float
    cells : (n_cells, n_nodes_per_cell) int
    element : str
        Reference element name (:mod:`toporlogy.fem.mesh.reference`).
    cell_tags : (n_cells,) int, optional
        Physical (material/region) tag per cell; default 0.
    facet_sets : mapping name → (nf, 2) int
        (cell, local facet) pairs.
    node_sets : mapping name → (n,) int
    """

    points: np.ndarray
    cells: np.ndarray
    element: str
    cell_tags: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    facet_sets: dict[str, np.ndarray] = field(default_factory=dict)
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.cells, Mapping):
            raise ValueError("one element type per mesh: mixed cell types are not supported")
        self.points = np.asarray(self.points, float)
        self.cells = np.asarray(self.cells, np.int64)
        ref = get(self.element)
        if self.points.ndim != 2 or self.points.shape[1] != ref.dim:
            raise ValueError(f"{self.element} is a {ref.dim}-D element; points must be (n, {ref.dim}), got "
                             f"{self.points.shape}")
        if self.cells.ndim != 2 or self.cells.shape[1] != ref.n_nodes:
            raise ValueError(f"{self.element} cells have {ref.n_nodes} nodes, got {self.cells.shape}")
        if self.cells.size and (self.cells.min() < 0 or self.cells.max() >= len(self.points)):
            raise ValueError("cell node index out of range")
        if self.cell_tags.size == 0:
            self.cell_tags = np.zeros(len(self.cells), np.int64)
        self.cell_tags = np.asarray(self.cell_tags, np.int64).reshape(len(self.cells))
        self.facet_sets = {k: np.asarray(v, np.int64).reshape(-1, 2) for k, v in self.facet_sets.items()}
        self.node_sets = {k: np.asarray(v, np.int64).ravel() for k, v in self.node_sets.items()}
        self._boundary: np.ndarray | None = None

    # ------------------------------------------------------------------ sizes
    @property
    def ref(self) -> ReferenceElement:
        return get(self.element)

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def used_nodes(self) -> np.ndarray:
        """Nodes of at least one cell (ascending)."""
        return np.unique(self.cells)

    def cell_points(self, cells: np.ndarray | None = None) -> np.ndarray:
        """(nc, n_nodes_per_cell, dim) node coordinates of the cells."""
        c = self.cells if cells is None else self.cells[np.asarray(cells, np.int64)]
        return self.points[c]

    def cell_centroids(self) -> np.ndarray:
        """(n_cells, dim) mean of the vertex coordinates."""
        return self.points[self.cells[:, :self.ref.n_vertices]].mean(axis=1)

    # ------------------------------------------------------------------ DOFs
    def cell_dofs(self, ncomp: int | None = None) -> np.ndarray:
        """(n_cells, n_nodes_per_cell · ncomp) node-major DOFs; ncomp defaults to dim."""
        k = self.dim if ncomp is None else int(ncomp)
        return (k * self.cells[:, :, None] + np.arange(k)).reshape(self.n_cells, -1)

    def dofs(self, nodes: np.ndarray, components: Sequence[int] | int | None = None,
             ncomp: int | None = None) -> np.ndarray:
        """DOFs of ``nodes`` in ``components`` (default all), sorted."""
        k = self.dim if ncomp is None else int(ncomp)
        comps = list(range(k)) if components is None else \
            [int(c) for c in np.atleast_1d(np.asarray(components, np.int64))]
        nodes = np.asarray(nodes, np.int64).ravel()
        if not len(nodes):
            return np.zeros(0, np.int64)
        return np.sort(np.concatenate([k * nodes + int(c) for c in comps]))

    # ------------------------------------------------------------------ facets
    def facet_nodes(self, facets: np.ndarray) -> np.ndarray:
        """(nf, n_facet_nodes) node numbers of (cell, local facet) pairs."""
        facets = np.asarray(facets, np.int64).reshape(-1, 2)
        loc = np.array(self.ref.facets)
        return self.cells[facets[:, 0][:, None], loc[facets[:, 1]]]

    def _all_facets(self) -> tuple[np.ndarray, np.ndarray]:
        ref = self.ref
        nfac = len(ref.facets)
        cf = np.column_stack([np.repeat(np.arange(self.n_cells), nfac), np.tile(np.arange(nfac), self.n_cells)])
        nv = 1 if ref.dim == 1 else get(ref.facet_element).n_vertices
        keys = np.sort(self.facet_nodes(cf)[:, :nv], axis=1)
        return cf, keys

    def boundary_facets(self) -> np.ndarray:
        """(nf, 2) facets that belong to exactly one cell."""
        if self._boundary is None:
            cf, keys = self._all_facets()
            _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            self._boundary = cf[cnt[inv.ravel()] == 1]
        return self._boundary

    def is_boundary(self, facets: np.ndarray) -> np.ndarray:
        """(nf,) True where a (cell, local facet) pair lies on the boundary."""
        b = self.boundary_facets()
        code = b[:, 0] * 64 + b[:, 1]
        f = np.asarray(facets, np.int64).reshape(-1, 2)
        return np.isin(f[:, 0] * 64 + f[:, 1], code)

    def facets_where(self, predicate: Callable[[np.ndarray], Any]) -> np.ndarray:
        """Boundary facets whose every node satisfies ``predicate(X (n, dim)) -> bool (n,)``."""
        b = self.boundary_facets()
        fn = self.facet_nodes(b)
        ok = np.asarray(predicate(self.points[fn.ravel()]), bool).reshape(fn.shape).all(axis=1)
        return b[ok]

    def nodes_where(self, predicate: Callable[[np.ndarray], Any]) -> np.ndarray:
        """Used nodes where ``predicate(X)`` holds."""
        u = self.used_nodes()
        return u[np.asarray(predicate(self.points[u]), bool)]

    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.facet_nodes(self.boundary_facets()))

    # ------------------------------------------------------------------ topology
    def components(self) -> tuple[int, np.ndarray]:
        """(number of connected pieces, piece label per cell); cells are connected through shared nodes."""
        ne, nl = self.cells.shape
        A = sp.csr_matrix((np.ones(ne * nl), (np.repeat(np.arange(ne), nl), self.cells.ravel())),
                          shape=(ne, self.n_nodes))
        C = (A @ A.T).tocsr()
        n, lab = connected_components(C, directed=False)
        return int(n), lab

    # ------------------------------------------------------------------ adapters
    @classmethod
    def from_domain(cls, domain: Any, element: str | None = None) -> Mesh:
        """The active elements of a regular-grid :class:`~toporlogy.core.domain.Domain` as a Q4 / H8 mesh with the
        grid's node (and therefore DOF) numbers; box nodes no active element touches stay in ``points`` unused.
        ``cell_tags`` holds the grid element number of each cell."""
        el = element or ("H8" if domain.dim == 3 else "Q4")
        if el != ("H8" if domain.dim == 3 else "Q4"):
            raise ValueError(f"from_domain: the regular grid is Q4 (2-D) / H8 (3-D), got {el!r} (incompatible-mode "
                             "elements exist on the grid only: toporlogy.fem.elastic)")
        act = domain.active_elements
        cells = domain.edof[act][:, ::domain.dim] // domain.dim
        m = cls(domain.node_coords(), cells, el, cell_tags=act)
        return m


def straighten(mesh: Mesh) -> Mesh:
    """Copy of ``mesh`` with every higher-order node on the straight-sided (affine / bilinear / trilinear) cell of
    its vertices."""
    ref = mesh.ref
    if ref.degree == 1:
        return replace(mesh, points=mesh.points.copy())
    W = ref.vertex_weights()                                  # (n_nodes, n_vertices)
    P = mesh.points.copy()
    Xv = mesh.points[mesh.cells[:, :ref.n_vertices]]          # (ne, nv, dim)
    Xn = np.einsum("nv,evi->eni", W, Xv)
    hi = mesh.cells[:, ref.n_vertices:]
    P[hi.ravel()] = Xn[:, ref.n_vertices:].reshape(-1, mesh.dim)
    return replace(mesh, points=P)


def perturb(mesh: Mesh, amplitude: float, seed: int = 0, curved: float = 0.0) -> Mesh:
    """Move every interior vertex by a uniform random vector in ``[-amplitude, amplitude]^dim`` (boundary nodes
    stay), then place higher-order nodes on the straight-sided cells; ``curved`` > 0 then moves interior
    higher-order nodes by up to ``curved`` as well (curved edges)."""
    rng = np.random.default_rng(seed)
    ref = mesh.ref
    bnd = np.zeros(mesh.n_nodes, bool)
    bnd[mesh.boundary_nodes()] = True
    vert = np.zeros(mesh.n_nodes, bool)
    vert[np.unique(mesh.cells[:, :ref.n_vertices])] = True
    P = mesh.points.copy()
    mv = vert & ~bnd
    P[mv] += rng.uniform(-amplitude, amplitude, size=(int(mv.sum()), mesh.dim))
    out = straighten(replace(mesh, points=P))
    if curved > 0 and ref.degree > 1:
        hi = ~vert & ~bnd
        hi[np.setdiff1d(np.arange(mesh.n_nodes), mesh.used_nodes())] = False
        out.points[hi] += rng.uniform(-curved, curved, size=(int(hi.sum()), mesh.dim))
    return out


# ---------------------------------------------------------------------------------------------- generators
def _grid_points(counts: Sequence[int], lengths: Sequence[float], origin: Sequence[float]) -> np.ndarray:
    axes = [o + np.linspace(0.0, L, n + 1) for n, L, o in zip(counts, lengths, origin, strict=True)]
    G = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in G], axis=1)


def _named_sets(mesh: Mesh, lengths: Sequence[float], origin: Sequence[float]) -> Mesh:
    names = (("left", "right"), ("bottom", "top"), ("back", "front"))
    tol = 1e-9 * max(lengths)
    for d in range(mesh.dim):
        for side, val in zip(names[d], (origin[d], origin[d] + lengths[d]), strict=True):
            sel = (lambda X, d=d, val=val: np.abs(X[:, d] - val) <= tol)
            mesh.facet_sets[side] = mesh.facets_where(sel)
            mesh.node_sets[side] = mesh.nodes_where(sel)
    return mesh


_TRI_SPLIT = (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1)))
_Q4_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))
_H8_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def _kuhn() -> list[tuple[tuple[int, ...], ...]]:
    """Six tetrahedra of the unit cube along (0,0,0)–(1,1,1), each positively oriented."""
    out = []
    for p in permutations(range(3)):
        v = [np.zeros(3, int)]
        for a in p:
            w = v[-1].copy()
            w[a] = 1
            v.append(w)
        if np.linalg.det(np.array([v[1] - v[0], v[2] - v[0], v[3] - v[0]], float)) < 0:
            v[1], v[2] = v[2], v[1]
        out.append(tuple(tuple(int(c) for c in x) for x in v))
    return out


def _structured(counts: Sequence[int], lengths: Sequence[float], origin: Sequence[float] | None,
                element: str) -> Mesh:
    ref = get(element)
    dim = len(counts)
    if ref.dim != dim:
        raise ValueError(f"{element} is a {ref.dim}-D element")
    org = np.zeros(dim) if origin is None else np.asarray(origin, float)
    k = ref.degree                                           # node spacing: doubled grid for quadratic elements
    nn = [k * n + 1 for n in counts]
    pts = _grid_points([k * n for n in counts], lengths, org.tolist())

    def node(ijk: np.ndarray) -> np.ndarray:
        return np.ravel_multi_index(tuple(ijk[..., d] for d in range(dim)), nn)

    base = np.stack(np.meshgrid(*[np.arange(n) for n in counts], indexing="ij"), axis=-1).reshape(-1, dim)
    corners: list[Any]
    if ref.cell in ("quad", "hex"):
        corners = [_Q4_CORNERS if dim == 2 else _H8_CORNERS]
    elif ref.cell == "tri":
        corners = list(_TRI_SPLIT)
    else:
        corners = list(_kuhn())
    cells = []
    for cset in corners:
        verts = np.array(cset)                               # (nv, dim) in cell units
        lnodes = ref.vertex_weights() @ verts                 # reference node → position in cell units
        loc = np.rint(k * lnodes).astype(int)                 # doubled-grid offsets
        cells.append(node(k * base[:, None, :] + loc[None, :, :]))
    C = np.stack(cells, axis=1).reshape(-1, ref.n_nodes)
    m = Mesh(pts, C, element)
    used = m.used_nodes()
    if len(used) != len(pts):                                # compact (no unused node in generated meshes)
        renum = np.full(len(pts), -1, np.int64)
        renum[used] = np.arange(len(used))
        m = Mesh(pts[used], renum[C], element)
    return _named_sets(m, lengths, org.tolist())


def rectangle(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0, element: str = "Q4",
              origin: Sequence[float] | None = None) -> Mesh:
    """Structured mesh of ``[0, lx] × [0, ly]`` (+ origin), nx × ny squares (``Q4``, ``Q9``) or 2 nx ny triangles
    (``T3``, ``T6``); facet and node sets ``left right bottom top``."""
    return _structured((nx, ny), (lx, ly), origin, element)


def box(nx: int, ny: int, nz: int, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0, element: str = "H8",
        origin: Sequence[float] | None = None) -> Mesh:
    """Structured mesh of a box: nx ny nz hexahedra (``H8``) or 6 nx ny nz tetrahedra (``T4``, ``T10``); facet and
    node sets ``left right bottom top back front`` (x, y, z min / max)."""
    return _structured((nx, ny, nz), (lx, ly, lz), origin, element)
