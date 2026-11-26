"""Reference elements: nodes, shape functions N, derivatives ∂N/∂ξ, facets and the polynomial space.

Each element is a separate object with one job: what happens on its reference cell. Mapping to physical space,
Jacobians and the detJ check are in :mod:`toporlogy.fem.mesh.kernels`; quadrature in
:mod:`toporlogy.fem.mesh.quadrature`.

Local node order (counter-clockwise / right-handed, so a positively oriented cell has detJ > 0):

* ``L2`` (-1), (1); ``L3`` (-1), (1), (0).
* ``Q4`` (-1,-1), (1,-1), (1,1), (-1,1) — the order of :data:`toporlogy.fem.elements.Q4_CORNER`.
* ``Q9`` the four corners as Q4, then the edge midpoints (0,-1), (1,0), (0,1), (-1,0), then the centre (0,0)
  (the order of scikit-fem's ``ElementQuad2``).
* ``H8`` the order of :data:`toporlogy.fem.elements.H8_CORNER` (bottom face counter-clockwise, then the top face).
* ``T3`` 0, e₁, e₂; ``T6`` the vertices, then the midpoints of edges 01, 12, 20.
* ``T4`` 0, e₁, e₂, e₃; ``T10`` the vertices, then the midpoints of edges 01, 12, 20, 30, 32, 31 (gmsh order, as in
  :mod:`toporlogy.post.verify`, so a :class:`~toporlogy.post.verify.TetMesh` is a T10 mesh as it stands).

``degree`` is the complete polynomial degree k (P_k is in the span); ``kind`` "Q" elements also contain the tensor
products of degree k per variable. Facets list the local nodes of each facet; the outward orientation is decided
geometrically by the kernels, not by the node order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from toporlogy.references import cite

__all__ = ["ReferenceElement", "ELEMENTS", "get", "monomial_exponents"]


def _lagrange_1d(points: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values (nq, np) and derivatives (nq, np) of the 1-D Lagrange polynomials on ``points`` at ``x`` (nq,)."""
    n = len(points)
    V = np.ones((len(x), n))
    dV = np.zeros((len(x), n))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            V[:, i] *= (x - points[j]) / (points[i] - points[j])
        for k in range(n):
            if k == i:
                continue
            term = np.full(len(x), 1.0 / (points[i] - points[k]))
            for j in range(n):
                if j in (i, k):
                    continue
                term = term * (x - points[j]) / (points[i] - points[j])
            dV[:, i] += term
    return V, dV


@dataclass(frozen=True)
class ReferenceElement:
    """A Lagrange element on a reference cell.

    Attributes
    ----------
    name, cell : str
        Element name ("Q9") and reference cell ("quad").
    dim : int
    kind : {"P", "Q"}
        Complete polynomials of total degree ``degree`` (simplices) or tensor products of degree ``degree`` per
        variable (lines, quadrilaterals, hexahedra).
    degree : int
    nodes : numpy.ndarray
        (n_nodes, dim) reference coordinates of the nodes.
    n_vertices : int
        The first ``n_vertices`` nodes are the cell's vertices.
    vertex_element : str
        The degree-1 element on the same cell (its nodes are this element's vertices).
    facets : tuple of tuple of int
        Local nodes of each facet.
    facet_element : str or None
        Reference element of a facet ("" for lines: a facet of a line is a point).
    """

    name: str
    cell: str
    dim: int
    kind: str
    degree: int
    nodes: np.ndarray
    n_vertices: int
    vertex_element: str
    facets: tuple[tuple[int, ...], ...]
    facet_element: str
    _edges: tuple[tuple[int, int], ...] = field(default=(), repr=False)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------ shape functions
    def shape(self, xi: np.ndarray) -> np.ndarray:
        """N (nq, n_nodes) at reference points ``xi`` (nq, dim)."""
        return self._eval(np.atleast_2d(np.asarray(xi, float)))[0]

    def grad(self, xi: np.ndarray) -> np.ndarray:
        """∂N/∂ξ (nq, n_nodes, dim) at reference points ``xi`` (nq, dim)."""
        return self._eval(np.atleast_2d(np.asarray(xi, float)))[1]

    def _eval(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if xi.shape[1] != self.dim:
            raise ValueError(f"{self.name}: points must be (nq, {self.dim}), got {xi.shape}")
        if self.kind == "Q":
            return self._eval_tensor(xi)
        return self._eval_simplex(xi)

    def _eval_tensor(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.array([-1.0, 1.0]) if self.degree == 1 else np.array([-1.0, 0.0, 1.0])
        idx = np.rint((self.nodes + 1.0) * (len(pts) - 1) / 2.0).astype(int)     # 1-D index per node, direction
        vals = [_lagrange_1d(pts, xi[:, d]) for d in range(self.dim)]
        nq = len(xi)
        N = np.ones((nq, self.n_nodes))
        dN = np.ones((nq, self.n_nodes, self.dim))
        for d in range(self.dim):
            V, dV = vals[d]
            N *= V[:, idx[:, d]]
            for e in range(self.dim):
                dN[:, :, e] *= (dV if e == d else V)[:, idx[:, d]]
        return N, dN

    def _eval_simplex(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nq, d = xi.shape
        L = np.column_stack([1.0 - xi.sum(axis=1), xi])                      # barycentric (nq, d + 1)
        dL = np.vstack([-np.ones(d), np.eye(d)])                              # (d + 1, d)
        if self.degree == 1:
            return L, np.broadcast_to(dL, (nq, d + 1, d)).copy()
        N = np.empty((nq, self.n_nodes))
        dN = np.empty((nq, self.n_nodes, d))
        for i in range(d + 1):
            N[:, i] = L[:, i] * (2 * L[:, i] - 1)
            dN[:, i] = (4 * L[:, i] - 1)[:, None] * dL[i]
        for k, (i, j) in enumerate(self._edges):
            N[:, d + 1 + k] = 4 * L[:, i] * L[:, j]
            dN[:, d + 1 + k] = 4 * (L[:, i, None] * dL[j] + L[:, j, None] * dL[i])
        return N, dN

    def vertex_weights(self) -> np.ndarray:
        """(n_nodes, n_vertices): the degree-1 shape functions of the vertices at every node; ``W @ X_vertices``
        places the higher-order nodes on the straight-sided (affine / bilinear / trilinear) cell."""
        return get(self.vertex_element).shape(self.nodes)


def monomial_exponents(element: ReferenceElement) -> list[tuple[int, ...]]:
    """Exponents of the monomials that span the element's polynomial space (E4 completeness check)."""
    k, d = element.degree, element.dim
    if element.kind == "Q":
        return [e for e in product(range(k + 1), repeat=d)]
    return [e for e in product(range(k + 1), repeat=d) if sum(e) <= k]


_SQ = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
_CUBE = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_TRI = np.array([[0, 0], [1, 0], [0, 1]], float)
_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
_TRI_EDGES = ((0, 1), (1, 2), (2, 0))
_TET_EDGES = ((0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1))       # gmsh 10-node tetrahedron


def _mid(X: np.ndarray, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.array([(X[i] + X[j]) / 2 for i, j in edges])


def _build() -> dict[str, ReferenceElement]:
    q9_nodes = np.vstack([_SQ, [[0, -1], [1, 0], [0, 1], [-1, 0]], [[0, 0]]])
    els = [
        ReferenceElement("L2", "line", 1, "Q", 1, np.array([[-1.0], [1.0]]), 2, "L2", ((0,), (1,)), ""),
        ReferenceElement("L3", "line", 1, "Q", 2, np.array([[-1.0], [1.0], [0.0]]), 2, "L2", ((0,), (1,)), ""),
        ReferenceElement("Q4", "quad", 2, "Q", 1, _SQ, 4, "Q4", ((0, 1), (1, 2), (2, 3), (3, 0)), "L2"),
        ReferenceElement("Q9", "quad", 2, "Q", 2, q9_nodes, 4, "Q4",
                         ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)), "L3"),
        ReferenceElement("H8", "hex", 3, "Q", 1, _CUBE, 8, "H8",
                         ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)), "Q4"),
        ReferenceElement("T3", "tri", 2, "P", 1, _TRI, 3, "T3", ((0, 1), (1, 2), (2, 0)), "L2"),
        ReferenceElement("T6", "tri", 2, "P", 2, np.vstack([_TRI, _mid(_TRI, _TRI_EDGES)]), 3, "T3",
                         ((0, 1, 3), (1, 2, 4), (2, 0, 5)), "L3", _TRI_EDGES),
        ReferenceElement("T4", "tet", 3, "P", 1, _TET, 4, "T4", ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)), "T3"),
        ReferenceElement("T10", "tet", 3, "P", 2, np.vstack([_TET, _mid(_TET, _TET_EDGES)]), 4, "T4",
                         ((0, 2, 1, 6, 5, 4), (0, 1, 3, 4, 9, 7), (1, 2, 3, 5, 8, 9), (0, 3, 2, 7, 8, 6)), "T6",
                         _TET_EDGES),
    ]
    return {e.name: e for e in els}


ELEMENTS: dict[str, ReferenceElement] = _build()


@cite("zienkiewicz_taylor_2013", "cook_2002", note="Lagrange elements on reference cells")
def get(name: str) -> ReferenceElement:
    """The reference element ``name`` (``L2 L3 Q4 Q9 H8 T3 T6 T4 T10``)."""
    try:
        return ELEMENTS[name]
    except KeyError:
        hint = " (incompatible-mode elements exist on the regular grid only: toporlogy.fem.elastic)" \
            if name in ("Q4IM", "H8IM") else ""
        raise ValueError(f"unknown element {name!r}: {' | '.join(ELEMENTS)}{hint}") from None
