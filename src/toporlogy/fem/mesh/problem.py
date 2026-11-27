"""Linear static elasticity and heat conduction on a general :class:`~toporlogy.fem.mesh.mesh.Mesh`.

``K u = f`` with essential conditions given as (DOF, value) pairs (:class:`Dirichlet`): nonzero values, one column
per load case, conflicting values on one DOF refused. The constrained DOFs are eliminated,
``K_ff u_f = f_f − K_fc u_c``; reactions ``r = K u − f`` at the constrained DOFs. DOFs that no cell touches are not
unknowns. Before the first factorisation the supports are checked: in every connected piece of the mesh the rigid
motions (2-D: 3, 3-D: 6; heat: the constant) restricted to the constrained DOFs must have full rank, else
:class:`~toporlogy.fem.mechanisms.MechanismError` names the piece and the number of free rigid motions. Internal
mechanisms that are not rigid motions of a piece are caught by the residual check of every solve.

Not supported, refused with an error: multi-point constraints, periodicity and oblique (rotated) supports
(:meth:`ElasticProblem.constrain` raises :class:`UnsupportedBoundaryCondition`; the grid physics has
``ConstrainedElasticity`` for those), loads on interior facets, D that is not symmetric positive definite (ν = 0.5
with a displacement-only element: use :mod:`toporlogy.fem.mesh.mixed`).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from toporlogy.core.linear_elastic import NotPositiveDefiniteError, check_positive_definite
from toporlogy.fem.mechanisms import MechanismError
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.assembly import Pattern, assemble_vector
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.quadrature import rule
from toporlogy.fem.mesh.reference import get

__all__ = ["Dirichlet", "UnsupportedBoundaryCondition", "Solution", "ElasticProblem", "HeatProblem",
           "rigid_modes", "stiffness_nullity"]


class UnsupportedBoundaryCondition(ValueError):
    """A boundary-condition kind or combination the general mesh does not implement."""

    code = "unsupported_bc"


class Dirichlet:
    """Essential conditions: DOFs and their values (n, n_cases).

    ``Dirichlet([(dofs, value), ...], n_cases=1)``: each value is a scalar, an (n,) array (one value per DOF), or
    (n, n_cases). A DOF listed twice must get the same value in every case, else ``ValueError``.
    """

    def __init__(self, pairs: Sequence[tuple[Any, Any]] = (), n_cases: int = 1) -> None:
        self.n_cases = int(n_cases)
        vals: dict[int, np.ndarray] = {}
        for dofs, value in pairs:
            d = np.asarray(dofs, np.int64).ravel()
            v = np.asarray(value, float)
            if v.ndim == 0:
                v = np.full((len(d), self.n_cases), float(v))
            elif v.ndim == 1:
                if len(v) != len(d):
                    raise ValueError(f"{len(v)} values for {len(d)} DOFs")
                v = np.repeat(v[:, None], self.n_cases, axis=1)
            elif v.shape != (len(d), self.n_cases):
                raise ValueError(f"values must be ({len(d)}, {self.n_cases}) for {self.n_cases} load case(s), "
                                 f"got {v.shape}")
            for k, dof in enumerate(d.tolist()):
                if dof in vals and not np.array_equal(vals[dof], v[k]):
                    raise ValueError(f"DOF {dof} is constrained twice with different values "
                                     f"({vals[dof].tolist()} and {v[k].tolist()})")
                vals[dof] = v[k]
        order = sorted(vals)
        self.dofs = np.array(order, np.int64)
        self.values = np.array([vals[k] for k in order], float).reshape(len(order), self.n_cases)

    @classmethod
    def fixed(cls, dofs: Any, n_cases: int = 1) -> Dirichlet:
        return cls([(dofs, 0.0)], n_cases)


@dataclass
class Solution:
    """u (ndof, n_cases), reactions (ndof, n_cases) nonzero only at constrained DOFs, the constrained DOFs."""

    u: np.ndarray
    reactions: np.ndarray
    constrained: np.ndarray


def rigid_modes(X: np.ndarray, ncomp: int) -> np.ndarray:
    """(n · ncomp, n_modes) rigid motions of nodes X (n, dim), node-major, coordinates centred and scaled to unit
    size: translations, then rotations (2-D: 1, 3-D: 3). ncomp = 1: the constant."""
    n, dim = X.shape
    if ncomp == 1:
        return np.ones((n, 1))
    Y = X - X.mean(axis=0)
    Y = Y / max(float(np.abs(Y).max()), 1e-300)
    cols = []
    for d in range(dim):
        c = np.zeros((n, dim))
        c[:, d] = 1.0
        cols.append(c.ravel())
    if dim == 2:
        cols.append(np.column_stack([-Y[:, 1], Y[:, 0]]).ravel())
    else:
        x, y, z = Y.T
        o = np.zeros(n)
        cols += [np.column_stack([o, -z, y]).ravel(), np.column_stack([z, o, -x]).ravel(),
                 np.column_stack([-y, x, o]).ravel()]
    return np.column_stack(cols)


def stiffness_nullity(K: np.ndarray, rtol: float = 1e-10) -> int:
    """Number of eigenvalues of a small dense symmetric K with |λ| ≤ rtol · max |λ| (oracle use only)."""
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    return int((np.abs(w) <= rtol * np.abs(w).max()).sum())


class _Linear:
    """Common machinery: element matrices, pattern, loads, support check, elimination and solve."""

    ncomp: int

    def __init__(self, mesh: Mesh, ncomp: int, thickness: float, degree: int | None) -> None:
        self.mesh = mesh
        self.ref = mesh.ref
        self.ncomp = int(ncomp)
        self.thickness = float(thickness)
        if mesh.dim == 3 and thickness != 1.0:
            raise ValueError("thickness applies to 2-D meshes only")
        self.degree = kn.default_degree(self.ref, mesh.cell_points()) if degree is None else int(degree)
        self.geom = kn.geometry(self.ref, mesh.cell_points(), rule(self.ref.cell, self.degree))
        self.cell_dofs = mesh.cell_dofs(self.ncomp)
        self.ndof = mesh.n_nodes * self.ncomp
        self.active = np.unique(self.cell_dofs)
        self.pattern = Pattern(self.cell_dofs, (self.ndof, self.ndof))
        self._K: sp.csr_matrix | None = None
        self._fact: tuple[tuple[int, bytes], Factorization] | None = None
        self._version = 0
        self._checked: set[bytes] = set()
        self.n_factorizations = 0

    # ------------------------------------------------------------------ matrices
    def _element_matrices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def K(self) -> sp.csr_matrix:
        """Full global matrix (CSR)."""
        if self._K is None:
            self._K = self.pattern.assemble(self._element_matrices())
        return self._K

    def _invalidate(self) -> None:
        self._K = None
        self._fact = None
        self._version += 1

    # ------------------------------------------------------------------ loads
    def _check_facets(self, facets: np.ndarray) -> np.ndarray:
        f = np.asarray(facets, np.int64).reshape(-1, 2)
        if len(f) and ((f[:, 0] < 0).any() or (f[:, 0] >= self.mesh.n_cells).any()
                       or (f[:, 1] < 0).any() or (f[:, 1] >= len(self.ref.facets)).any()):
            raise ValueError("facet (cell, local facet) out of range")
        inner = ~self.mesh.is_boundary(f)
        if inner.any():
            raise UnsupportedBoundaryCondition(f"{int(inner.sum())} facet(s) are not on the boundary (a load on an "
                                               f"interior facet), first: {f[inner][:5].tolist()}")
        return f

    def facet_geometry(self, facets: np.ndarray, degree: int | None = None) -> tuple[kn.FacetGeometry, np.ndarray]:
        f = self._check_facets(facets)
        fref = get(self.ref.facet_element)
        Xf = self.mesh.points[self.mesh.facet_nodes(f)]
        cen = self.mesh.cell_centroids()[f[:, 0]]
        deg = (2 * fref.degree + 2) if degree is None else int(degree)
        return kn.facet_geometry(fref, Xf, cen, deg), f

    def facet_dofs(self, facets: np.ndarray) -> np.ndarray:
        nodes = self.mesh.facet_nodes(facets)
        return (self.ncomp * nodes[:, :, None] + np.arange(self.ncomp)).reshape(len(nodes), -1)

    def body_force(self, b: Any, degree: int | None = None) -> np.ndarray:
        """Consistent global load of a body force b (vector per unit volume; constant or callable of x)."""
        g = self.geom if degree is None else kn.geometry(self.ref, self.mesh.cell_points(), rule(self.ref.cell, degree))
        fe = kn.body_load(g, b, self.ncomp, self.thickness)
        return assemble_vector(self.ndof, self.cell_dofs, fe)

    # ------------------------------------------------------------------ supports and solve
    def _check_supports(self, constrained: np.ndarray) -> None:
        key = constrained.tobytes()
        if key in self._checked:
            return
        n, lab = self.mesh.components()
        cmask = np.zeros(self.ndof, bool)
        cmask[constrained] = True
        problems = []
        for c in range(n):
            nodes = np.unique(self.mesh.cells[lab == c])
            R = rigid_modes(self.mesh.points[nodes], self.ncomp)
            dofs = (self.ncomp * nodes[:, None] + np.arange(self.ncomp)).ravel()
            Rc = R[cmask[dofs]]
            rank = 0 if not len(Rc) else int((np.linalg.svd(Rc, compute_uv=False) > 1e-8).sum())
            if rank < R.shape[1]:
                problems.append((c, R.shape[1] - rank, len(nodes)))
        if problems:
            txt = "; ".join(f"piece {c} ({nn} nodes): {m} free rigid motion(s)" for c, m, nn in problems)
            raise MechanismError(f"the essential conditions leave rigid motions, the system is singular: {txt}")
        self._checked.add(key)

    def solve(self, f: np.ndarray, bc: Dirichlet | None = None) -> Solution:
        """Solve for load(s) f (ndof,) or (ndof, n_cases) with essential conditions ``bc``."""
        F = np.asarray(f, float)
        F = F[:, None] if F.ndim == 1 else F
        if F.shape[0] != self.ndof:
            raise ValueError(f"load must have {self.ndof} rows, got {F.shape[0]}")
        bc = Dirichlet((), F.shape[1]) if bc is None else bc
        if bc.n_cases not in (1, F.shape[1]):
            raise ValueError(f"the Dirichlet values have {bc.n_cases} load case(s), the load {F.shape[1]}")
        c = bc.dofs
        if len(c) and (c.min() < 0 or c.max() >= self.ndof):
            raise ValueError("constrained DOF out of range")
        if not np.isin(c, self.active).all():
            raise ValueError("a constrained DOF belongs to no cell")
        self._check_supports(c)
        free = np.setdiff1d(self.active, c)
        K = self.K
        key = (self._version, c.tobytes())
        if self._fact is None or self._fact[0] != key:
            self._fact = (key, Factorization(K[free][:, free], "spd"))
            self.n_factorizations += 1
        fact = self._fact[1]
        Uc = np.broadcast_to(bc.values, (len(c), F.shape[1]))
        rhs = F[free] - K[free][:, c] @ Uc
        U = np.zeros_like(F)
        U[c] = Uc
        U[free] = fact.solve(rhs).reshape(rhs.shape)
        R = np.zeros_like(F)
        R[c] = (K[c] @ U) - F[c]
        return Solution(U, R, c)

    def factorization(self) -> Factorization | None:
        return None if self._fact is None else self._fact[1]

    def element_values(self, u: np.ndarray) -> np.ndarray:
        """(ne, n_nodes, ncomp) nodal values of a global vector (one load case)."""
        return np.asarray(u, float).reshape(-1, self.ncomp)[self.mesh.cells]


class ElasticProblem(_Linear):
    """Linear static elasticity on a general mesh.

    Parameters
    ----------
    mesh : Mesh
    D : array
        Physical elasticity (Voigt, engineering shear): (k, k) or one per cell (n_cells, k, k); k = 3 in 2-D (plane
        stress or plane strain as the caller builds it, e.g.
        :func:`toporlogy.core.linear_elastic.isotropic_plane_strain`), 6 in 3-D. Must be symmetric positive definite.
    thickness : float
        2-D only.
    degree : int, optional
        Quadrature degree (default: exact for Bᵀ D B on affine simplices / parallelograms).
    """

    def __init__(self, mesh: Mesh, D: np.ndarray, thickness: float = 1.0, degree: int | None = None) -> None:
        super().__init__(mesh, mesh.dim, thickness, degree)
        self.k = 3 if mesh.dim == 2 else 6
        self.set_D(D)

    def set_D(self, D: np.ndarray) -> None:
        """New elasticity (same pattern; the factorisation is dropped)."""
        D = np.asarray(D, float)
        if D.shape not in ((self.k, self.k), (self.mesh.n_cells, self.k, self.k)):
            raise ValueError(f"D must be ({self.k}, {self.k}) or ({self.mesh.n_cells}, {self.k}, {self.k}), "
                             f"got {D.shape}")
        Ds = D.reshape(-1, self.k, self.k)
        for i in np.unique(Ds.reshape(len(Ds), -1), axis=0, return_index=True)[1]:
            try:
                check_positive_definite(Ds[i], "stiffness", "D")
            except NotPositiveDefiniteError as e:
                raise ValueError(f"D must be positive definite ({e}); a nearly incompressible or incompressible "
                                 "material needs the mixed u-p formulation (toporlogy.fem.mesh.mixed)") from None
        self.D = D
        self._invalidate()

    def _element_matrices(self) -> np.ndarray:
        return kn.stiffness(self.geom, self.D, self.thickness)

    def element_stiffness(self) -> np.ndarray:
        return self._element_matrices()

    def traction(self, facets: np.ndarray, t: Any, degree: int | None = None) -> np.ndarray:
        """Consistent global load of a traction vector t (force per area; constant or callable of x)."""
        fg, f = self.facet_geometry(facets, degree)
        return assemble_vector(self.ndof, self.facet_dofs(f), kn.facet_load(fg, traction=t, thickness=self.thickness))

    def pressure(self, facets: np.ndarray, p: Any, degree: int | None = None) -> np.ndarray:
        """Consistent global load of a pressure p (t = −p n, n the outward normal)."""
        fg, f = self.facet_geometry(facets, degree)
        return assemble_vector(self.ndof, self.facet_dofs(f), kn.facet_load(fg, pressure=p, thickness=self.thickness))

    def constrain(self, kind: str, *args: Any, **kwargs: Any) -> None:
        """Placeholder of the constraint kinds the general mesh does not implement; always raises."""
        raise UnsupportedBoundaryCondition(f"{kind!r} constraints are not implemented on the general mesh (only "
                                           "Dirichlet DOF values); multi-point, periodic and oblique supports exist "
                                           "on the regular grid (toporlogy.boundary ConstrainedElasticity)")

    def mass(self, rho: float | np.ndarray = 1.0, lumped: bool = False) -> sp.csr_matrix:
        """Global consistent (or HRZ-lumped, diagonal) mass matrix."""
        g = kn.geometry(self.ref, self.mesh.cell_points(), rule(self.ref.cell, 2 * self.ref.degree))
        M = kn.mass(g, rho, self.ncomp, self.thickness)
        if lumped:
            return sp.diags(assemble_vector(self.ndof, self.cell_dofs, kn.lumped_mass_hrz(M))).tocsr()
        return self.pattern.assemble(M)

    def strain(self, u: np.ndarray, degree: int | None = None) -> tuple[np.ndarray, kn.Geometry]:
        """Voigt strain (ne, nq, k) at the points of a rule (default the stiffness rule) and that geometry."""
        g = self.geom if degree is None else kn.geometry(self.ref, self.mesh.cell_points(), rule(self.ref.cell, degree))
        B = kn.strain_matrix(g.dNdx)
        ue = np.asarray(u, float)[self.cell_dofs]
        return np.einsum("eqkj,ej->eqk", B, ue), g

    def stress(self, u: np.ndarray, degree: int | None = None) -> np.ndarray:
        """Voigt stress (ne, nq, k) = D ε at the rule points."""
        eps, _ = self.strain(u, degree)
        D = np.broadcast_to(self.D, (self.mesh.n_cells, self.k, self.k))
        return np.einsum("ekl,eql->eqk", D, eps)

    def errors(self, u: np.ndarray, exact: Any, exact_grad: Any, degree: int | None = None,
               chunk_points: int = 400_000) -> dict[str, float]:
        """Relative L2 and energy errors against an exact field: ``exact(x (n, dim)) -> (n, dim)``,
        ``exact_grad(x) -> (n, dim, dim)`` with [i, j] = ∂u_i/∂x_j. Rule degree default 2 · degree + 4; cells are
        processed in chunks of about ``chunk_points`` quadrature points."""
        deg = 2 * self.ref.degree + 4 if degree is None else int(degree)
        q = rule(self.ref.cell, deg)
        dim = self.mesh.dim
        ue_all = self.element_values(u)
        D_all = np.broadcast_to(self.D, (self.mesh.n_cells, self.k, self.k))
        acc = np.zeros(4)                                     # ∫|e|², ∫|u|², ∫ε_e:C:ε_e, ∫ε:C:ε
        step = max(1, chunk_points // len(q.weights))
        for s0 in range(0, self.mesh.n_cells, step):
            cells = np.arange(s0, min(s0 + step, self.mesh.n_cells))
            g = kn.geometry(self.ref, self.mesh.cell_points(cells), q, cells=cells)
            ne, nq, _ = g.x.shape
            X = g.x.reshape(-1, dim)
            ue = ue_all[cells]
            uh = np.einsum("qn,eni->eqi", g.N, ue)
            Gh = kn.gradients(g, ue)
            ux = np.asarray(exact(X)).reshape(ne, nq, dim)
            Gx = np.asarray(exact_grad(X)).reshape(ne, nq, dim, dim)
            w = g.dV * (self.thickness if dim == 2 else 1.0)
            D = D_all[cells]
            de, ex = _voigt(Gh - Gx), _voigt(Gx)
            acc += [np.einsum("eqi,eq->", (uh - ux) ** 2, w), np.einsum("eqi,eq->", ux ** 2, w),
                    np.einsum("eqk,ekl,eql,eq->", de, D, de, w), np.einsum("eqk,ekl,eql,eq->", ex, D, ex, w)]
        return {"l2": float(np.sqrt(acc[0] / acc[1])), "energy": float(np.sqrt(acc[2] / acc[3]))}


def _voigt(G: np.ndarray) -> np.ndarray:
    """Engineering Voigt strain of displacement gradients (..., dim, dim)."""
    if G.shape[-1] == 2:
        return np.stack([G[..., 0, 0], G[..., 1, 1], G[..., 0, 1] + G[..., 1, 0]], axis=-1)
    return np.stack([G[..., 0, 0], G[..., 1, 1], G[..., 2, 2], G[..., 1, 2] + G[..., 2, 1],
                     G[..., 0, 2] + G[..., 2, 0], G[..., 0, 1] + G[..., 1, 0]], axis=-1)


class HeatProblem(_Linear):
    """Steady heat conduction ``−∇·(κ∇T) = s`` with Dirichlet temperatures and Robin (convection) facets.

    κ: scalar, one per cell, or a tensor (dim, dim) / (n_cells, dim, dim). Robin facets add ``h ∫ T v dS`` to K and
    ``h T∞ ∫ v dS`` to the load (:meth:`robin`).
    """

    def __init__(self, mesh: Mesh, kappa: Any = 1.0, thickness: float = 1.0, degree: int | None = None) -> None:
        super().__init__(mesh, 1, thickness, degree)
        self.kappa = kappa
        self._robin: list[tuple[np.ndarray, np.ndarray]] = []

    def _element_matrices(self) -> np.ndarray:
        return kn.conduction(self.geom, self.kappa, self.thickness)

    @property
    def K(self) -> sp.csr_matrix:
        if self._K is None:
            K = self.pattern.assemble(self._element_matrices())
            for dofs, Kf in self._robin:
                K = K + Pattern(dofs, (self.ndof, self.ndof)).assemble(Kf)
            self._K = sp.csr_matrix(K)
        return self._K

    def robin(self, facets: np.ndarray, h: float, T_inf: float) -> np.ndarray:
        """Add the convection term h ∫ T v dS on ``facets`` to K (h > 0); returns its load h T∞ ∫ v dS."""
        if not h > 0:
            raise ValueError("Robin coefficient h > 0 is required")
        fg, f = self.facet_geometry(facets)
        dofs = self.facet_dofs(f)
        self._robin.append((dofs, kn.facet_mass(fg, h, self.thickness)))
        self._robin_cells = np.union1d(getattr(self, "_robin_cells", np.zeros(0, np.int64)), f[:, 0])
        self._invalidate()
        self._checked.clear()
        fe = kn.facet_load(fg, traction=h * T_inf, ncomp=1, thickness=self.thickness)
        return assemble_vector(self.ndof, dofs, fe)

    def flux(self, facets: np.ndarray, q: Any) -> np.ndarray:
        """Load of a prescribed inward heat flux q (per area) on ``facets``."""
        fg, f = self.facet_geometry(facets)
        fe = kn.facet_load(fg, traction=q, ncomp=1, thickness=self.thickness)
        return assemble_vector(self.ndof, self.facet_dofs(f), fe)

    def _check_supports(self, constrained: np.ndarray) -> None:
        """Every connected piece needs a Dirichlet temperature or a convection facet (else T is free up to a
        constant)."""
        n, lab = self.mesh.components()
        held = np.zeros(n, bool)
        if len(constrained):
            node_piece = np.full(self.mesh.n_nodes, -1)
            node_piece[self.mesh.cells.ravel()] = np.repeat(lab, self.mesh.cells.shape[1])
            held[np.unique(node_piece[constrained])] = True
        rc = getattr(self, "_robin_cells", np.zeros(0, np.int64))
        held[np.unique(lab[rc])] = True
        if not held.all():
            raise MechanismError(f"pieces {np.flatnonzero(~held).tolist()} have neither a Dirichlet temperature nor "
                                 "a convection facet: the temperature is free up to a constant")
