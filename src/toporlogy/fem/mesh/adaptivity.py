"""A posteriori error indicators and conforming adaptive refinement of T3 meshes (2-D Poisson).

Scope (WP09, record ``docs/foundations/prereg/A_adaptivity.md``, design ``docs/foundations/ADAPTIVITY.md``): the steady
problem ``−∇·(κ∇u) = s`` on a T3 mesh with Dirichlet data on the **whole** boundary (:class:`HeatProblem`). Natural
conditions, elasticity, T6/Q4 and 3-D are not implemented.

Indicators are **indicators, not bounds**: the coefficients are all 1, no reliability or efficiency constant is
proved, and nothing here is robust against κ contrast.

* :func:`residual_indicators` — squared cell indicators
  ``η_T² = (h_T²/κ_T)‖s‖²_T + ½ Σ_E (h_E/κ̄_E)‖[κ∇u_h·n]‖²_E + Σ_{E⊂Γ_D} h_E‖∂_s(g − I_h g)‖²_E`` (``h_T`` longest
  edge, ``h_E`` edge length, ``κ̄_E`` the larger neighbouring κ; units: energy, ``κ·(gradient)²·area``).
  ``div(κ∇u_h)`` vanishes in a T3 cell with constant κ. The last term needs the exact trace ``g`` and its gradient.
* :func:`zz_indicators` — Zienkiewicz–Zhu: area-weighted patch average of the constant cell gradients, P1
  interpolation, ``η_T² = κ_T ‖G_h − ∇u_h‖²_T``.
* :func:`dorfler_mark` — bulk marking with a fixed fraction.
* :func:`refine` — newest-vertex bisection with closure (no hanging nodes), cell tags, facet sets and node sets
  carried to the children; :class:`Refinement` transfers nodal fields (edge-midpoint average: exact for affine
  fields) and cell fields (inheritance).
* :func:`solve_adaptive_poisson` — solve → estimate → mark → refine loop with caps and a stop reason.

Convention of the bisection: ``cells[:, 0]`` is the newest vertex (peak) and the refinement edge is
``(cells[:, 1], cells[:, 2])``; the children are ``(m, c0, c1)`` and ``(m, c2, c0)`` with ``m`` the edge midpoint
(orientation is preserved). Call :func:`newest_vertex_labelling` once on an unlabelled mesh (it makes the longest
edge the refinement edge; for right isosceles meshes that is the hypotenuse, the classical compatible labelling).
Refinement does not call it again: a labelled mesh stays labelled through :func:`refine`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import Dirichlet, HeatProblem
from toporlogy.fem.mesh.quadrature import rule

__all__ = ["cell_areas", "cell_gradients", "newest_vertex_labelling", "Refinement", "refine", "refine_uniform",
           "residual_indicators", "zz_indicators", "dorfler_mark", "free_dofs", "AdaptiveLevel", "AdaptiveResult",
           "solve_adaptive_poisson", "mesh_state", "mesh_from_state"]


DirichletData = tuple[Callable[[np.ndarray], Any], Callable[[np.ndarray], Any]]
"""``(g, grad_g)``: exact Dirichlet trace ``g(x (n, 2)) -> (n,)`` and the gradient ``grad_g(x) -> (n, 2)`` of an
extension of it (only the tangential part on boundary edges is used)."""


def _require_t3(mesh: Mesh) -> None:
    if mesh.element != "T3":
        raise ValueError(f"adaptivity is implemented for T3 meshes only, got {mesh.element!r}")


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------------------------------------- geometry
def cell_areas(mesh: Mesh) -> np.ndarray:
    """(n_cells,) signed areas of a T3 mesh (positive for counter-clockwise cells)."""
    _require_t3(mesh)
    X = mesh.points[mesh.cells]
    e1, e2 = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def cell_gradients(mesh: Mesh, u: np.ndarray) -> np.ndarray:
    """(n_cells, 2) constant gradient of the P1 field with nodal values ``u`` (n_nodes,)."""
    _require_t3(mesh)
    u = np.asarray(u, float).ravel()
    if len(u) != mesh.n_nodes:
        raise ValueError(f"u must have {mesh.n_nodes} nodal values, got {len(u)}")
    X = mesh.points[mesh.cells]
    e1, e2 = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    g1 = np.stack([e2[:, 1], -e2[:, 0]], axis=1) / det[:, None]
    g2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1) / det[:, None]
    ue = u[mesh.cells]
    return (ue[:, 1] - ue[:, 0])[:, None] * g1 + (ue[:, 2] - ue[:, 0])[:, None] * g2


def free_dofs(mesh: Mesh) -> int:
    """Number of free nodes (nodes minus boundary nodes) of the scalar Dirichlet problem."""
    return int(len(mesh.used_nodes()) - len(mesh.boundary_nodes()))


def _kappa_cells(mesh: Mesh, kappa: Any) -> np.ndarray:
    k = np.asarray(kappa, float)
    if k.ndim == 0:
        k = np.full(mesh.n_cells, float(k))
    elif k.shape != (mesh.n_cells,):
        raise ValueError(f"kappa must be a scalar or ({mesh.n_cells},) (isotropic), got {k.shape}")
    if not (np.isfinite(k).all() and (k > 0).all()):
        raise ValueError("kappa must be finite and positive")
    return k


# ---------------------------------------------------------------------------------------------- indicators
def _edge_table(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per (cell, local edge) instance, in cell-major order: edge id, a, b (node numbers, cell orientation) and
    the number of cells that own the edge."""
    c = mesh.cells
    a = c.ravel()
    b = np.roll(c, -1, axis=1).ravel()
    code = np.minimum(a, b) * mesh.n_nodes + np.maximum(a, b)
    _, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    return inv, a, b, cnt[inv]


def residual_indicators(mesh: Mesh, u: np.ndarray, source: Any = 0.0, kappa: Any = 1.0,
                        dirichlet_data: DirichletData | None = None,
                        jump: bool = True, degree: int = 5) -> np.ndarray:
    """(n_cells,) squared residual indicators η_T² (see the module notes).

    Parameters
    ----------
    u : (n_nodes,) nodal values of the P1 solution.
    source : constant or callable ``s(x (n, 2)) -> (n,)`` (the right-hand side of ``−∇·(κ∇u) = s``).
    kappa : scalar or (n_cells,) isotropic conductivity.
    dirichlet_data : ``(g, grad_g)`` exact trace ``g(x) -> (n,)`` and its gradient ``grad_g(x) -> (n, 2)`` (a gradient
        of any extension; only its tangential part is used). Adds the data term on boundary edges.
    jump : False drops the flux-jump term. Diagnostic for the negative control only (the result is then not an
        error indicator of the discrete solution).
    degree : quadrature degree of ‖s‖² on the cells.

    The whole boundary is treated as Dirichlet; natural boundary data are not supported.
    """
    _require_t3(mesh)
    ne = mesh.n_cells
    kap = _kappa_cells(mesh, kappa)
    grad = cell_gradients(mesh, u)
    X = mesh.points[mesh.cells]
    eta2 = np.zeros(ne)
    # element residual: (h_T² / κ_T) ‖s‖²
    el = np.stack([np.linalg.norm(X[:, 1] - X[:, 2], axis=1), np.linalg.norm(X[:, 2] - X[:, 0], axis=1),
                   np.linalg.norm(X[:, 0] - X[:, 1], axis=1)], axis=1)
    hT = el.max(axis=1)
    if callable(source) or float(np.asarray(source, float).ravel()[0]) != 0.0:
        g = kn.geometry(mesh.ref, mesh.cell_points(), rule("tri", degree))
        if callable(source):
            s = np.asarray(source(g.x.reshape(-1, 2)), float).reshape(g.x.shape[:2])
        else:
            s = np.full(g.x.shape[:2], float(np.asarray(source, float).ravel()[0]))
        eta2 += hT ** 2 / kap * np.einsum("eq,eq->e", s * s, g.dV)
    inv, a, b, cnt = _edge_table(mesh)
    cell_of = np.repeat(np.arange(ne), 3)
    t = mesh.points[b] - mesh.points[a]
    L = np.linalg.norm(t, axis=1)
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1) / L[:, None]            # outward for counter-clockwise cells
    flux = kap[cell_of] * np.einsum("ij,ij->i", grad[cell_of], nrm)     # κ ∇u·n_out of the owning cell
    if jump:
        J = np.bincount(inv, weights=flux)[inv]                         # sum of the two outward fluxes = [κ∇u·n]
        kbar = np.zeros(inv.max() + 1)
        np.maximum.at(kbar, inv, kap[cell_of])
        interior = cnt == 2
        contrib = np.where(interior, 0.5 * L * L * J * J / kbar[inv], 0.0)
        eta2 += np.bincount(cell_of, weights=contrib, minlength=ne)
    if dirichlet_data is not None:
        g_fun, grad_fun = dirichlet_data
        bd = np.flatnonzero(cnt == 1)
        if len(bd):
            x5, w5 = np.polynomial.legendre.leggauss(5)
            xi, wi = 0.5 * (x5 + 1.0), 0.5 * w5
            pa, pb = mesh.points[a[bd]], mesh.points[b[bd]]
            tau = t[bd] / L[bd, None]
            ga = np.asarray(g_fun(pa), float).ravel()
            gb = np.asarray(g_fun(pb), float).ravel()
            slope = (gb - ga) / L[bd]
            acc = np.zeros(len(bd))
            for xq, wq in zip(xi, wi, strict=True):
                x = pa + xq * (pb - pa)
                ds = np.einsum("ij,ij->i", np.asarray(grad_fun(x), float).reshape(-1, 2), tau)
                acc += wq * (ds - slope) ** 2
            eta2 += np.bincount(cell_of[bd], weights=L[bd] * L[bd] * acc, minlength=ne)
    return eta2


def zz_indicators(mesh: Mesh, u: np.ndarray, kappa: Any = 1.0) -> np.ndarray:
    """(n_cells,) squared Zienkiewicz–Zhu indicators ``κ_T ‖G_h − ∇u_h‖²_T`` (see the module notes)."""
    _require_t3(mesh)
    kap = _kappa_cells(mesh, kappa)
    grad = cell_gradients(mesh, u)
    A = np.abs(cell_areas(mesh))
    n = mesh.n_nodes
    num = np.zeros((n, 2))
    den = np.zeros(n)
    for j in range(3):
        np.add.at(num, mesh.cells[:, j], A[:, None] * grad)
        np.add.at(den, mesh.cells[:, j], A)
    used = den > 0
    G = np.zeros((n, 2))
    G[used] = num[used] / den[used, None]
    Gc = G[mesh.cells] - grad[:, None, :]                              # (ne, 3, 2) nodal differences
    # ∫_T |q|² for P1 q with vertex values q_i: A/3 Σ_edges |q(mid)|² (exact for quadratics)
    mids = 0.5 * (Gc + np.roll(Gc, -1, axis=1))
    return kap * A / 3.0 * np.einsum("eij,eij->e", mids, mids)


def dorfler_mark(eta2: np.ndarray, theta: float) -> np.ndarray:
    """Cell numbers (ascending) of the smallest set M with ``Σ_M η_T² ≥ θ Σ η_T²``.

    Cells are taken in descending η² with ties broken by cell number (stable sort), so the result is deterministic.
    Returns an empty array when the total is zero."""
    e = np.asarray(eta2, float).ravel()
    if not 0.0 < theta <= 1.0:
        raise ValueError("Dörfler fraction theta must lie in (0, 1]")
    if (e < 0).any() or not np.isfinite(e).all():
        raise ValueError("indicators must be finite and non-negative")
    tot = e.sum()
    if tot <= 0.0:
        return np.zeros(0, np.int64)
    order = np.argsort(-e, kind="stable")
    csum = np.cumsum(e[order])
    k = int(np.searchsorted(csum, theta * tot, side="left")) + 1
    return np.sort(order[:min(k, len(e))]).astype(np.int64)


# ---------------------------------------------------------------------------------------------- bisection
def newest_vertex_labelling(mesh: Mesh) -> Mesh:
    """Copy of a T3 mesh whose cells are rotated cyclically (orientation kept) so that ``cells[:, 0]`` is the vertex
    opposite the longest edge (ties: the edge with the lexicographically smallest node pair). Facet sets are
    renumbered to the rotated local edges; tags and node sets are unchanged."""
    _require_t3(mesh)
    if (cell_areas(mesh) <= 0).any():
        raise ValueError("cells must be counter-clockwise with positive area")
    c = mesh.cells
    X = mesh.points[c]
    L = np.stack([np.linalg.norm(X[:, 1] - X[:, 2], axis=1), np.linalg.norm(X[:, 2] - X[:, 0], axis=1),
                  np.linalg.norm(X[:, 0] - X[:, 1], axis=1)], axis=1)       # L[:, i] opposite local vertex i
    tie = L.max(axis=1, keepdims=True) * (1.0 - 1e-12) <= L
    n = mesh.n_nodes
    code = np.stack([np.minimum(c[:, 1], c[:, 2]) * n + np.maximum(c[:, 1], c[:, 2]),
                     np.minimum(c[:, 2], c[:, 0]) * n + np.maximum(c[:, 2], c[:, 0]),
                     np.minimum(c[:, 0], c[:, 1]) * n + np.maximum(c[:, 0], c[:, 1])], axis=1)
    big = np.iinfo(np.int64).max
    i = np.argmin(np.where(tie, code, big), axis=1)
    rot = (i[:, None] + np.arange(3)) % 3
    new_cells = np.take_along_axis(c, rot, axis=1)
    fs = {k: np.column_stack([v[:, 0], (v[:, 1] - i[v[:, 0]]) % 3]) for k, v in mesh.facet_sets.items()}
    return Mesh(mesh.points.copy(), new_cells, "T3", cell_tags=mesh.cell_tags.copy(), facet_sets=fs,
                node_sets={k: v.copy() for k, v in mesh.node_sets.items()})


@dataclass(frozen=True)
class Refinement:
    """Result of :func:`refine`: the new mesh and the maps to the old one.

    ``parent_cell[j]`` is the old cell that contains new cell ``j``; new nodes are numbered after the old ones and
    ``node_parents[k]`` are the two nodes (old or earlier-new) whose edge midpoint is node ``n_old_nodes + k``.
    """

    mesh: Mesh
    parent_cell: np.ndarray
    n_old_nodes: int
    node_parents: np.ndarray
    n_bisections: int

    def transfer_nodal(self, u: np.ndarray) -> np.ndarray:
        """Nodal values (n_old, ...) → (n_new, ...) of the P1 interpolant: each new node takes the mean of its two
        edge ends (so the P1 interpolant is preserved: exact for affine fields, an interpolation for the rest)."""
        u = np.asarray(u, float)
        if u.shape[0] != self.n_old_nodes:
            raise ValueError(f"expected {self.n_old_nodes} nodal values, got {u.shape[0]}")
        out = np.empty((self.mesh.n_nodes, *u.shape[1:]))
        out[:self.n_old_nodes] = u
        for k, (p, q) in enumerate(self.node_parents.tolist()):
            out[self.n_old_nodes + k] = 0.5 * (out[p] + out[q])
        return out

    def inherit_cell(self, values: np.ndarray) -> np.ndarray:
        """Cell field (n_old_cells, ...) → children take the value of the parent."""
        v = np.asarray(values)
        return v[self.parent_cell]


def refine(mesh: Mesh, marked: np.ndarray) -> Refinement:
    """Bisect the ``marked`` cells (and whatever the closure requires) by newest-vertex bisection.

    The mesh must be labelled (:func:`newest_vertex_labelling`, or the output of an earlier :func:`refine`). A cell
    whose refinement edge is not the refinement edge of its neighbour forces the neighbour to be bisected first, so
    every bisection splits the two cells that share the edge and no hanging node arises. Deterministic: marked cells
    are processed in ascending number. Raises ``RuntimeError`` when the closure does not terminate (an unlabelled
    input).

    Cell tags are inherited; every facet set is split into the child facets; a node set receives the new midpoint
    when both ends are in the set and the bisected edge lies on the boundary. Facet sets must consist of boundary
    facets (``ValueError`` otherwise). Midpoints are straight-line (curved boundaries are not tracked).
    """
    _require_t3(mesh)
    ne0 = mesh.n_cells
    mk = np.unique(np.asarray(marked, np.int64).ravel())
    if len(mk) and (mk[0] < 0 or mk[-1] >= ne0):
        raise ValueError("marked cell number out of range")
    if (cell_areas(mesh) <= 0).any():
        raise ValueError("cells must be counter-clockwise with positive area")
    C: list[tuple[int, int, int]] = [tuple(r) for r in mesh.cells.tolist()]
    P: list[list[float]] = mesh.points.tolist()
    n_old = len(P)
    tags = mesh.cell_tags.tolist()
    root = list(range(ne0))
    alive = [True] * ne0
    emap: dict[tuple[int, int], list[int]] = {}

    def add_edges(t: int) -> None:
        a, b, c = C[t]
        for e in (_key(a, b), _key(b, c), _key(c, a)):
            emap.setdefault(e, []).append(t)

    def del_edges(t: int) -> None:
        a, b, c = C[t]
        for e in (_key(a, b), _key(b, c), _key(c, a)):
            lst = emap[e]
            lst.remove(t)
            if not lst:
                del emap[e]

    for t in range(ne0):
        add_edges(t)
    edge_sets: dict[tuple[int, int], set[str]] = {}
    for name, fs in mesh.facet_sets.items():
        for cell, f in fs.tolist():
            a, b, c = C[cell]
            e = _key(*((a, b), (b, c), (c, a))[f])
            if len(emap[e]) != 1:
                raise ValueError(f"facet set {name!r} contains an interior facet (cell {cell}, facet {f})")
            edge_sets.setdefault(e, set()).add(name)
    node_sets = {name: set(v.tolist()) for name, v in mesh.node_sets.items()}
    parents: list[tuple[int, int]] = []

    def split(t: int, m: int) -> None:
        v0, v1, v2 = C[t]
        del_edges(t)
        alive[t] = False
        for kid in ((m, v0, v1), (m, v2, v0)):
            C.append(kid)
            tags.append(tags[t])
            root.append(root[t])
            alive.append(True)
            add_edges(len(C) - 1)

    def bisect_pair(t: int, nb: int | None) -> None:
        a, b = C[t][1], C[t][2]
        e = _key(a, b)
        m = len(P)
        P.append([0.5 * (P[a][0] + P[b][0]), 0.5 * (P[a][1] + P[b][1])])
        parents.append((a, b))
        for name in edge_sets.pop(e, ()):
            for half in (_key(a, m), _key(m, b)):
                edge_sets.setdefault(half, set()).add(name)
        if nb is None:
            for s in node_sets.values():
                if a in s and b in s:
                    s.add(m)
        split(t, m)
        if nb is not None:
            split(nb, m)

    guard = 4 * ne0 + 1000
    for t0 in mk.tolist():
        stack = [t0]
        while stack:
            t = stack[-1]
            if not alive[t]:
                stack.pop()
                continue
            e = _key(C[t][1], C[t][2])
            others = [c for c in emap[e] if c != t]
            if others:
                nb = others[0]
                if _key(C[nb][1], C[nb][2]) != e:
                    stack.append(nb)
                    if len(stack) > guard:
                        raise RuntimeError("bisection closure did not terminate: the mesh is not labelled "
                                           "(call newest_vertex_labelling)")
                    continue
                bisect_pair(t, nb)
            else:
                bisect_pair(t, None)
            stack.pop()
    keep = [i for i, ok in enumerate(alive) if ok]
    cells = np.array([C[i] for i in keep], np.int64)
    fsets: dict[str, list[tuple[int, int]]] = {name: [] for name in mesh.facet_sets}
    for j, (a, b, c) in enumerate(cells.tolist()):
        for f, e in enumerate((_key(a, b), _key(b, c), _key(c, a))):
            for name in edge_sets.get(e, ()):
                fsets[name].append((j, f))
    new = Mesh(np.array(P, float), cells, "T3", cell_tags=np.array([tags[i] for i in keep], np.int64),
               facet_sets={k: np.array(v, np.int64).reshape(-1, 2) for k, v in fsets.items()},
               node_sets={k: np.array(sorted(v), np.int64) for k, v in node_sets.items()})
    return Refinement(new, np.array([root[i] for i in keep], np.int64), n_old,
                      np.array(parents, np.int64).reshape(-1, 2), len(parents))


def refine_uniform(mesh: Mesh, rounds: int = 1) -> Refinement:
    """``rounds`` times bisect every cell (one bisection per cell and round; two rounds halve h for right isosceles
    meshes). The returned maps refer to the **input** mesh (composed over the rounds)."""
    if rounds < 1:
        raise ValueError("rounds >= 1")
    cur = refine(mesh, np.arange(mesh.n_cells))
    for _ in range(rounds - 1):
        nxt = refine(cur.mesh, np.arange(cur.mesh.n_cells))
        cur = Refinement(nxt.mesh, cur.parent_cell[nxt.parent_cell], cur.n_old_nodes,
                         np.vstack([cur.node_parents, nxt.node_parents]), cur.n_bisections + nxt.n_bisections)
    return cur


# ---------------------------------------------------------------------------------------------- state
def mesh_state(mesh: Mesh) -> dict[str, np.ndarray]:
    """Checkpoint of a mesh as plain arrays (``np.savez``-able): points, cells, cell tags, and one array per facet and
    node set (keys ``facet:<name>``, ``node:<name>``)."""
    st = {"points": mesh.points.copy(), "cells": mesh.cells.copy(), "cell_tags": mesh.cell_tags.copy()}
    st.update({f"facet:{k}": v.copy() for k, v in mesh.facet_sets.items()})
    st.update({f"node:{k}": v.copy() for k, v in mesh.node_sets.items()})
    return st


def mesh_from_state(state: dict[str, np.ndarray]) -> Mesh:
    """Inverse of :func:`mesh_state`."""
    return Mesh(state["points"], state["cells"], "T3", cell_tags=state["cell_tags"],
                facet_sets={k[6:]: v for k, v in state.items() if k.startswith("facet:")},
                node_sets={k[5:]: v for k, v in state.items() if k.startswith("node:")})


# ---------------------------------------------------------------------------------------------- workflow
@dataclass(frozen=True)
class AdaptiveLevel:
    """One solved level: the mesh, the nodal solution, free DOFs, the estimator total ``η`` and the cell indicators
    (squared)."""

    iteration: int
    mesh: Mesh
    u: np.ndarray
    n_dofs: int
    eta: float
    eta2_cells: np.ndarray


@dataclass
class AdaptiveResult:
    """All solved levels and why the loop stopped (``target``, ``max_iterations``, ``max_dofs``,
    ``zero_estimator``)."""

    levels: list[AdaptiveLevel] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def mesh(self) -> Mesh:
        return self.levels[-1].mesh

    @property
    def u(self) -> np.ndarray:
        return self.levels[-1].u


def solve_adaptive_poisson(mesh: Mesh, source: Any = 0.0, kappa: Any = 1.0,
                           dirichlet_data: DirichletData | None = None,
                           theta: float = 0.5, estimator: str = "residual", max_dofs: int = 20000,
                           max_iterations: int = 30, target: float | None = None, load_degree: int = 5,
                           on_level: Callable[[AdaptiveLevel], None] | None = None) -> AdaptiveResult:
    """Solve → estimate → mark (Dörfler ``theta``) → refine, on a labelled T3 mesh with Dirichlet data on the whole
    boundary (zero unless ``dirichlet_data = (g, grad_g)`` is given; ``g`` at the boundary nodes).

    ``estimator``: ``"residual"``, ``"zz"`` or ``"residual_no_jump"`` (negative control: the flux-jump term is
    dropped). Stops, with ``stop_reason``, when η ≤ ``target``, after ``max_iterations`` refinements, when the next
    mesh would exceed ``max_dofs`` free nodes (it is not solved; the last solved level is returned) or when the
    estimator is identically zero. The first mesh must not exceed ``max_dofs``. The solve is a fresh direct solve per
    level (the transferred field of :meth:`Refinement.transfer_nodal` is not needed as a start).
    ``on_level`` is called after each solve (checkpoint hook; :func:`mesh_state` of ``level.mesh``).
    """
    if estimator not in ("residual", "zz", "residual_no_jump"):
        raise ValueError(f"unknown estimator {estimator!r}")
    if free_dofs(mesh) > max_dofs:
        raise ValueError(f"the initial mesh has {free_dofs(mesh)} free DOFs, over the cap {max_dofs}")
    res = AdaptiveResult()
    cur = mesh
    for it in range(max_iterations + 1):
        prob = HeatProblem(cur, kappa)
        bn = cur.boundary_nodes()
        g = 0.0 if dirichlet_data is None else np.asarray(dirichlet_data[0](cur.points[bn]), float).ravel()
        f = prob.body_force(source, degree=load_degree)
        u = prob.solve(f, Dirichlet([(bn, g)])).u[:, 0]
        if estimator == "zz":
            eta2 = zz_indicators(cur, u, kappa)
        else:
            eta2 = residual_indicators(cur, u, source, kappa, dirichlet_data, jump=estimator == "residual",
                                       degree=load_degree)
        lev = AdaptiveLevel(it, cur, u, free_dofs(cur), float(np.sqrt(eta2.sum())), eta2)
        res.levels.append(lev)
        if on_level is not None:
            on_level(lev)
        if target is not None and lev.eta <= target:
            res.stop_reason = "target"
            break
        if it == max_iterations:
            res.stop_reason = "max_iterations"
            break
        marked = dorfler_mark(eta2, theta)
        if not len(marked):
            res.stop_reason = "zero_estimator"
            break
        nxt = refine(cur, marked).mesh
        if free_dofs(nxt) > max_dofs:
            res.stop_reason = "max_dofs"
            break
        cur = nxt
    return res
