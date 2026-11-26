"""Stress and strain recovery on the regular grid (Q4 / Q4IM in 2-D, H8 / H8IM in 3-D) as separate output types.

A displacement FE solution has no single stress field. Each way of reading one is a different quantity with its own
accuracy, so each is returned as its own type and none is called *the* stress:

=========================  ==========================  =======================================================
:class:`StressRecovery`     result (``kind``)           what it is
=========================  ==========================  =======================================================
``integration_points()``    :class:`PointField`         σ = E D (B u + B_α α) at the 2 × 2 (× 2) Gauss points
``centroids()``             :class:`ElementField`       the same at the element centre
``ip_maximum(measure)``     :class:`ElementMaximum`     largest value of a scalar measure over the Gauss points
``extrapolated()``          :class:`CornerField`        bi/trilinear extrapolation of the Gauss values to the
                                                        element's own corners (discontinuous between elements)
``nodal_averaged()``        :class:`NodalField`         arithmetic mean of the corner values at each node
``patch_recovered()``       :class:`NodalField`         bi/trilinear fit through the 2^dim element centres around
                                                        an interior node; boundary nodes from interior patches
=========================  ==========================  =======================================================

``patch_recovered`` is the superconvergent patch recovery idea of Zienkiewicz & Zhu (1992) restricted to the regular
grid, written from the textbook description; the paper was not read. A node that no complete interior patch reaches
(a member one element thick) falls back to the nodal average and is marked in :attr:`NodalField.fallback`.

Interfaces: nodal values are never averaged across elements of different material (``separate="material"``: same
unit elasticity D and same modulus E), because the in-plane stress components jump there. A node on an interface
therefore has one entry per material.

Stress is the stress of the modulus ``E`` that is passed: ``E = E0`` (a number) is the *solid* stress used by
:mod:`toporlogy.optimization.stress`; the interpolated moduli of a grey design give the stress of the interpolated
material. Voigt order and engineering shear strains as in :mod:`toporlogy.fem.elements`. 2-D fields hold the in-plane
components; von Mises assumes plane stress (σzz = 0) and refuses a 2-D elasticity that is not known to be one.

These are diagnostics in NumPy: no gradients.

Measured orders and exactness: ``docs/olcumler/gerilme.md`` (pre-registered in ``gerilme_onkayit.md``).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.core.domain import require_grid
from toporlogy.fem.elements import H8_CORNER, Q4_CORNER, _B_unit_h8, _B_unit_q4, mode_recovery_from_D
from toporlogy.optimization.stress import von_mises_matrix

__all__ = ["StressRecovery", "PointField", "ElementField", "ElementMaximum", "CornerField", "NodalField"]

Measure = Callable[[np.ndarray], Any]        # (N, n_voigt) stresses -> (N,) values


@dataclass(frozen=True)
class _Field:
    """A stress array with the Voigt axis second to last and the load case axis last."""

    stress: np.ndarray
    dim: int
    plane_stress: bool

    def measure(self, fn: Measure) -> np.ndarray:
        """``fn`` of every stress vector: array of ``stress.shape`` without the Voigt axis. ``fn`` maps (N, n_voigt)
        to (N,) (for example a failure criterion after its plane expansion)."""
        s = np.moveaxis(self.stress, -2, -1)
        out = np.asarray(fn(s.reshape(-1, s.shape[-1])), float)
        if out.shape != (int(np.prod(s.shape[:-1])),):
            raise ValueError(f"the measure returned shape {out.shape}, expected one value per stress vector")
        return out.reshape(s.shape[:-1])

    def von_mises(self) -> np.ndarray:
        """von Mises stress: array of ``stress.shape`` without the Voigt axis (2-D: plane stress)."""
        if self.dim == 2 and not self.plane_stress:
            raise ValueError("von Mises in 2-D needs plane stress (σzz = 0): this elasticity was given as a D matrix, "
                             "so pass plane_stress=True to StressRecovery if it is one (plane strain is not covered)")
        V = von_mises_matrix(self.dim)
        return self.measure(lambda s: np.sqrt(np.maximum(np.einsum("nv,vw,nw->n", s, V, s), 0.0)))


@dataclass(frozen=True)
class PointField(_Field):
    """Gauss point values: ``stress`` and ``strain`` (n_el, n_points, n_voigt, n_cases)."""

    strain: np.ndarray
    points: np.ndarray            # (n_el, n_points, dim) physical coordinates
    local: np.ndarray             # (n_points, dim) natural coordinates in [-1, 1]
    weights: np.ndarray           # (n_points,) quadrature volume of each point (thickness included in 2-D)
    elements: np.ndarray
    kind: str = "integration_point"

    def energy_density(self) -> np.ndarray:
        """½ σ : ε at every point, (n_el, n_points, n_cases)."""
        return 0.5 * np.einsum("egvc,egvc->egc", self.stress, self.strain)


@dataclass(frozen=True)
class ElementField(_Field):
    """Element centre values: ``stress`` and ``strain`` (n_el, n_voigt, n_cases)."""

    strain: np.ndarray
    points: np.ndarray            # (n_el, dim) centres
    elements: np.ndarray
    kind: str = "centroid"


@dataclass(frozen=True)
class ElementMaximum:
    """Largest value of a scalar measure over the Gauss points of each element."""

    value: np.ndarray             # (n_el, n_cases)
    point: np.ndarray             # (n_el, n_cases) index of the Gauss point that attains it
    points: np.ndarray            # (n_el, n_cases, dim) its physical coordinates
    elements: np.ndarray
    kind: str = "integration_point_max"


@dataclass(frozen=True)
class CornerField(_Field):
    """Extrapolated values at each element's corners: ``stress`` (n_el, n_corners, n_voigt, n_cases)."""

    points: np.ndarray            # (n_el, n_corners, dim)
    nodes: np.ndarray             # (n_el, n_corners) node numbers
    elements: np.ndarray
    kind: str = "extrapolated"


@dataclass(frozen=True)
class NodalField(_Field):
    """Recovered nodal values, one entry per (node, material group): ``stress`` (n_entries, n_voigt, n_cases).

    ``corner_entry[e, a]`` is the entry used by corner a of element ``elements[e]``, so ``stress[corner_entry]`` is
    the field element by element, continuous inside a material group.
    """

    node: np.ndarray              # (n_entries,)
    group: np.ndarray             # (n_entries,)
    points: np.ndarray            # (n_entries, dim)
    corner_entry: np.ndarray      # (n_el, n_corners)
    fallback: np.ndarray          # (n_entries,) True: no interior patch, nodal average used
    elements: np.ndarray
    kind: str = "nodal_averaged"


def _gauss_local(dim: int) -> np.ndarray:
    """Gauss points in [-1, 1]^dim, point i next to corner i of the element."""
    return np.asarray(H8_CORNER if dim == 3 else Q4_CORNER, float) / np.sqrt(3.0)


class StressRecovery:
    """Recovery of one displacement solution.

    Parameters
    ----------
    physics : Elasticity
        Gives the domain, the element type and the unit elasticity (one D or one per active element).
    U : array
        Box displacements (ndof,) or (ndof, n_cases); a torch tensor is accepted and detached.
    E : float or array (n_active,)
        Modulus of the stress (module notes). Default 1.
    elements : array of int, optional
        Box element numbers recovered (a subset of the active ones, e.g. the solid elements of a 0/1 design). Nodal
        averaging and patches use only these.
    plane_stress : bool, optional
        2-D only. Default: True for the isotropic plane-stress elasticity built from ``nu``, False for a D matrix.
    """

    def __init__(self, physics: Any, U: Any, E: Any = 1.0, elements: Any = None,
                 plane_stress: bool | None = None) -> None:
        d = getattr(physics, "domain", None)
        if d is None:
            raise ValueError(f"StressRecovery needs the physics of a structured element grid (Elasticity on a Domain), "
                             f"not a {type(physics).__name__}; it has no mesh or shell counterpart yet")
        require_grid(d, "StressRecovery")
        self.domain, self.dim = d, int(d.dim)
        if physics.element not in ("Q4", "Q4IM", "H8", "H8IM"):
            raise ValueError(f"stress recovery covers Q4, Q4IM, H8 and H8IM, not {physics.element!r}")
        self.element = str(physics.element)
        U_arr = np.asarray(U.detach().cpu() if hasattr(U, "detach") else U, float)
        if U_arr.ndim == 1:
            U_arr = U_arr[:, None]
        if U_arr.ndim != 2 or U_arr.shape[0] != d.ndof:
            raise ValueError(f"U must be (ndof,) or (ndof, n_cases) with ndof = {d.ndof}, got {U_arr.shape}")
        active = d.active_elements
        el = active if elements is None else np.unique(np.asarray(elements, np.int64))
        if el.size == 0:
            raise ValueError("no element to recover")
        rows = np.searchsorted(active, el)
        if (rows >= active.size).any() or (active[np.minimum(rows, active.size - 1)] != el).any():
            raise ValueError("elements must be active box element numbers")
        self.elements, self._rows = el, rows
        self.plane_stress = bool(self.dim == 2 and (physics.nu is not None if plane_stress is None
                                                    else plane_stress))
        E_arr = np.asarray(E.detach().cpu() if hasattr(E, "detach") else E, float)
        if E_arr.ndim == 0:
            E_arr = np.full(el.size, float(E_arr))
        elif E_arr.shape == (active.size,):
            E_arr = E_arr[rows]
        else:
            raise ValueError(f"E must be a number or ({active.size},) over the active elements, got {E_arr.shape}")
        self.E = E_arr
        D = np.asarray(physics.D, float)
        self._D = D[rows] if D.ndim == 3 else D                              # (k, k) or (n_el, k, k)
        ue = U_arr[d.edof[el]]                                               # (n_el, n_loc, n_cases)
        if not (np.isfinite(ue).all() and np.isfinite(E_arr).all()):
            raise ValueError("non-finite displacement or modulus: no stress is recovered from it")
        self._ue = ue
        self._alpha = self._modes(physics) if self.element.endswith("IM") else None
        self._points: PointField | None = None
        self._centres: ElementField | None = None

    # ------------------------------------------------------------------ strain and stress
    def _modes(self, physics: Any) -> np.ndarray:
        """(n_el, n_modes, n_cases) amplitudes α = R u of the condensed incompatible modes."""
        if physics.ke_index is None:
            return np.einsum("mk,ekc->emc", mode_recovery_from_D(self.dim, np.asarray(physics.D, float)), self._ue)
        R = np.stack([mode_recovery_from_D(self.dim, D) for D in physics.D_distinct])
        return np.einsum("emk,ekc->emc", R[physics.ke_index[self._rows]], self._ue)

    def _evaluate(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(strain, stress), each (n_el, n_points, n_voigt, n_cases), at natural points ``local``."""
        unit = _B_unit_h8 if self.dim == 3 else _B_unit_q4
        pairs = [unit(p) for p in local]
        Bu = np.stack([p[0] for p in pairs]) / self.domain.h                 # (g, v, n_loc)
        eps = np.einsum("gvk,ekc->egvc", Bu, self._ue)
        if self._alpha is not None:
            Ba = np.stack([p[1] for p in pairs]) / self.domain.h
            eps = eps + np.einsum("gvm,emc->egvc", Ba, self._alpha)
        sig = np.einsum("vw,egwc->egvc", self._D, eps) if self._D.ndim == 2 else \
            np.einsum("evw,egwc->egvc", self._D, eps)
        return eps, sig * self.E[:, None, None, None]

    def _coords(self, local: np.ndarray) -> np.ndarray:
        """(n_el, n_points, dim) physical coordinates of natural points."""
        ijk = self.domain.element_ijk(self.elements)
        return self.domain.origin + self.domain.h * (ijk[:, None, :] + 0.5 * (local[None, :, :] + 1.0))

    # ------------------------------------------------------------------ output types
    def integration_points(self) -> PointField:
        if self._points is None:
            local = _gauss_local(self.dim)
            eps, sig = self._evaluate(local)
            vol = self.domain.h ** self.dim * (self.domain.thickness if self.dim == 2 else 1.0)
            self._points = PointField(sig, self.dim, self.plane_stress, eps, self._coords(local), local,
                                      np.full(len(local), vol / len(local)), self.elements)
        return self._points

    def centroids(self) -> ElementField:
        if self._centres is None:
            local = np.zeros((1, self.dim))
            eps, sig = self._evaluate(local)
            self._centres = ElementField(sig[:, 0], self.dim, self.plane_stress, eps[:, 0], self._coords(local)[:, 0],
                                         self.elements)
        return self._centres

    def ip_maximum(self, measure: Measure | None = None) -> ElementMaximum:
        """Element maximum over the Gauss points of ``measure`` (default von Mises). A maximum is defined for a scalar
        measure only, never component by component."""
        pf = self.integration_points()
        v = pf.von_mises() if measure is None else pf.measure(measure)       # (e, g, c)
        arg = v.argmax(axis=1)
        e = np.arange(len(self.elements))[:, None]
        return ElementMaximum(np.take_along_axis(v, arg[:, None, :], 1)[:, 0], arg, pf.points[e, arg], self.elements)

    def extrapolated(self) -> CornerField:
        pf = self.integration_points()
        corners = np.sqrt(3.0) * _gauss_local(self.dim)                      # element corners
        # the Gauss values are the nodal values of a bi/trilinear field in s = sqrt(3) * xi; the element corner
        # xi = c is s = sqrt(3) c
        s = np.sqrt(3.0) * corners
        N = np.prod(1.0 + s[:, None, :] * corners[None, :, :], axis=2) / 2 ** self.dim     # (corner, gauss)
        sig = np.einsum("ag,egvc->eavc", N, pf.stress)
        nodes = self.domain.edof[self.elements][:, ::self.dim] // self.dim
        return CornerField(sig, self.dim, self.plane_stress, self._coords(corners), nodes, self.elements)

    def _groups(self, separate: Any) -> np.ndarray:
        """(n_el,) material group of each recovered element."""
        n = len(self.elements)
        if isinstance(separate, str):
            if separate == "none":
                return np.zeros(n, np.int64)
            if separate != "material":
                raise ValueError('separate: "material", "none" or an integer array of domain.shape')
            D = self._D.reshape(n, -1) if self._D.ndim == 3 else np.zeros((n, 1))
            key = np.column_stack([self.E, D])
            return np.unique(key, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)
        g = np.asarray(separate)
        if g.shape != tuple(self.domain.shape) or not np.issubdtype(g.dtype, np.integer):
            raise ValueError(f"separate as an array must hold integers of shape {tuple(self.domain.shape)}")
        return np.unique(g.ravel()[self.elements], return_inverse=True)[1].reshape(-1).astype(np.int64)

    def _entries(self, corner: CornerField, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """(entry keys = node * n_groups + group, corner_entry, nodal average of the corner values, n_groups)."""
        ng = int(groups.max()) + 1
        keys = corner.nodes * ng + groups[:, None]
        ukey, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(keys.shape)
        total = np.zeros((len(ukey),) + corner.stress.shape[2:])
        np.add.at(total, inv.ravel(), corner.stress.reshape((-1,) + corner.stress.shape[2:]))
        count = np.bincount(inv.ravel(), minlength=len(ukey))
        return ukey, inv, total / count[:, None, None], ng

    def _nodal(self, stress: np.ndarray, ukey: np.ndarray, ng: int, corner_entry: np.ndarray, fallback: np.ndarray,
               kind: str) -> NodalField:
        node = ukey // ng
        ijk = np.column_stack(np.unravel_index(node, self.domain.node_shape))
        return NodalField(stress, self.dim, self.plane_stress, node, ukey % ng,
                          self.domain.origin + self.domain.h * ijk, corner_entry, fallback, self.elements, kind)

    def nodal_averaged(self, separate: Any = "material") -> NodalField:
        """Mean of the extrapolated corner values at each node, per material group (``separate="none"`` averages
        across materials: the discontinuous components are then smeared over the interface)."""
        corner = self.extrapolated()
        ukey, inv, mean, ng = self._entries(corner, self._groups(separate))
        return self._nodal(mean, ukey, ng, inv, np.zeros(len(ukey), bool), "nodal_averaged")

    def patch_recovered(self, separate: Any = "material") -> NodalField:
        """Superconvergent patch recovery from the element centre values (module notes)."""
        dim, shape = self.dim, self.domain.shape
        corner, centre = self.extrapolated(), self.centroids().stress
        groups = self._groups(separate)
        ukey, inv, average, ng = self._entries(corner, groups)
        row = np.full(self.domain.ne, -1, np.int64)
        row[self.elements] = np.arange(len(self.elements))
        # interior grid nodes and the 2^dim elements around each of them
        offsets = np.array(list(itertools.product((-1, 0), repeat=dim)))
        stress = average.copy()
        fallback = np.ones(len(ukey), bool)
        if all(s >= 2 for s in shape):
            nodes = np.indices([s - 1 for s in shape]).reshape(dim, -1).T + 1
            around = nodes[:, None, :] + offsets[None, :, :]
            rows = row[np.ravel_multi_index(tuple(np.moveaxis(around, -1, 0)), shape)]       # (n, 2^dim)
            g = np.where(rows >= 0, groups[rows], -1)
            complete = (rows >= 0).all(1) & (g == g[:, :1]).all(1)
            nodes, rows, g = nodes[complete], rows[complete], g[complete, 0]
        else:
            nodes = np.zeros((0, dim), np.int64)
        if len(nodes):
            centres_local = offsets + 0.5                                    # element centres around the node, in h
            subsets = list(itertools.product((0, 1), repeat=dim))            # monomials 1, x, y, xy, ...
            basis = lambda X: np.stack([np.prod(X ** np.array(p), axis=1) for p in subsets], axis=1)   # noqa: E731
            where = np.array(list(itertools.product((-1, 0, 1), repeat=dim)))
            W = basis(where.astype(float)) @ np.linalg.inv(basis(centres_local))             # (3^dim, 2^dim)
            values = np.einsum("pq,nqvc->npvc", W, centre[rows])
            target = np.ravel_multi_index(tuple(np.moveaxis(nodes[:, None, :] + where[None, :, :], -1, 0)),
                                          self.domain.node_shape)
            entry = np.searchsorted(ukey, target * ng + g[:, None])          # every patch node is an element corner
            own = (where == 0).all(1)
            total = np.zeros_like(stress)
            count = np.zeros(len(ukey))
            np.add.at(total, entry[:, ~own].ravel(), values[:, ~own].reshape((-1,) + stress.shape[1:]))
            np.add.at(count, entry[:, ~own].ravel(), 1.0)
            seen = count > 0
            stress[seen] = total[seen] / count[seen][:, None, None]
            stress[entry[:, own].ravel()] = values[:, own].reshape((-1,) + stress.shape[1:])   # own patch wins
            fallback[seen] = False
            fallback[entry[:, own].ravel()] = False
        return self._nodal(stress, ukey, ng, inv, fallback, "patch_recovered")
