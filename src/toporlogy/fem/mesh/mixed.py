"""Mixed displacement–pressure (u-p) formulation for nearly incompressible and incompressible isotropic elasticity.

Sign convention: ``σ = 2μ ε(u) − p I`` with ``p = −λ div u`` (positive in compression; equal to the mean pressure
−tr σ / 3 when div u = 0). Weak form, for all (v, q)::

    ∫ 2μ ε(u):ε(v) − ∫ p div v = f(v)
    −∫ q div u − ∫ q p / λ      = 0

so the system is ``[[A, Bᵀ], [B, −M/λ]]`` with ``A = ∫ 2μ ε:ε``, ``B = −∫ ψ div φ``, ``M = ∫ ψ ψ``; symmetric
indefinite, solved by LU with partial pivoting (:class:`~toporlogy.fem.mesh.linear.Factorization`,
``"symmetric_indefinite"``). λ = ∞ (ν = 0.5) is exact: the pressure block vanishes. When then the constant pressure
is in the null space (every boundary DOF constrained: ``Bᵀ 1 = 0`` on the free DOFs), the gauge ``∫ p = 0`` is added
with a Lagrange multiplier; the solution is unique and the reported pressure has zero mean.

Pressure spaces: ``"vertex"`` — continuous, degree 1 on the vertices of the displacement cells (with T6, T10, Q9:
Taylor–Hood P2–P1 / Q2–Q1, inf-sup stable — Brezzi & Fortin 1991, not read; with Q4/T3/T4: equal order, unstable);
``"P0"`` — one constant per cell (with Q4: Q1–P0, which equals the mean-dilatation / B-bar element after static
condensation and has the checkerboard pressure mode; kept as the unstable comparison).
:class:`MixedKG` is the K/G alternative with the mean pressure ``p = −K div u`` and ``A = ∫ 2G dev ε:dev ε``; unlike
the λ-based class it is valid at λ = 0 (ν = 0) and ν < 0 (derivation: ``docs/foundations/MIXED_FEM.md``).
Plane stress is refused: its effective modulus λ* = 2λμ/(λ + 2μ) stays bounded, there is no volumetric constraint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.assembly import Pattern
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import Dirichlet, ElasticProblem
from toporlogy.fem.mesh.reference import get

__all__ = ["MixedElasticity", "MixedKG", "MixedSolution", "deviatoric_D"]


def deviatoric_D(dim: int, mu: float) -> np.ndarray:
    """Voigt matrix of 2μ ε (engineering shear): diag(2μ, …, μ, …)."""
    n = 2 if dim == 2 else 3
    k = 3 if dim == 2 else 6
    return np.diag([2 * mu] * n + [mu] * (k - n))


@dataclass
class MixedSolution:
    u: np.ndarray            # (ndof_u, n_cases)
    p: np.ndarray            # (n_p, n_cases) pressure DOF values
    reactions: np.ndarray    # (ndof_u, n_cases)
    gauge: bool              # the ∫p = 0 multiplier was used


class MixedElasticity:
    """u-p elasticity (plane strain in 2-D, or 3-D) with shear modulus μ and Lamé λ (``math.inf`` allowed).

    Parameters
    ----------
    mesh : Mesh
    mu, lam : float
    pressure : {"vertex", "P0"}
    plane : {"strain", "stress"}
        2-D only; "stress" is refused.
    """

    def __init__(self, mesh: Mesh, mu: float, lam: float, pressure: str = "vertex", plane: str = "strain") -> None:
        if mesh.dim == 2 and plane != "strain":
            raise ValueError("the mixed u-p formulation is for plane strain and 3-D; plane stress has no "
                             "incompressibility constraint (λ* = 2λμ/(λ+2μ) is bounded): use ElasticProblem")
        if not mu > 0 or not lam > -2 * mu / 3:
            raise ValueError("μ > 0 and λ > −2μ/3 (positive bulk modulus) are required")
        if lam == 0:
            raise ValueError("λ = 0 (ν = 0): the pressure p = −λ div u vanishes identically and the pressure block "
                             "−M/λ is undefined; use ElasticProblem")
        self.lam = float(lam)
        self._build(mesh, mu, pressure)

    def _build(self, mesh: Mesh, mu: float, pressure: str) -> None:
        """Pressure space, B, M and the displacement block A = ∫ 2μ ε:ε (shared by :class:`MixedKG`)."""
        if pressure not in ("vertex", "P0"):
            raise ValueError("pressure: vertex | P0")
        self.mesh = mesh
        self.mu = float(mu)
        self.pressure_space = pressure
        self.disp = ElasticProblem(mesh, deviatoric_D(mesh.dim, mu))
        ref = mesh.ref
        g = self.disp.geom
        if pressure == "vertex":
            pref = get(ref.vertex_element)
            psi = pref.shape(g.rule.points)                                  # (nq, nv)
            vnodes, inv = np.unique(mesh.cells[:, :ref.n_vertices], return_inverse=True)
            self.pressure_nodes = vnodes
            self.cell_pdofs = inv.reshape(mesh.n_cells, ref.n_vertices)
        else:
            psi = np.ones((len(g.rule.weights), 1))
            self.pressure_nodes = np.zeros(0, np.int64)
            self.cell_pdofs = np.arange(mesh.n_cells)[:, None]
        self.psi = psi
        self.n_p = int(self.cell_pdofs.max()) + 1
        self.ndof_u = self.disp.ndof
        div = g.dNdx.reshape(*g.dNdx.shape[:2], -1)                          # node-major: ∂N_n/∂x_i at n·dim + i
        Be = -np.einsum("qa,eqj,eq->eaj", psi, div, g.dV)
        Me = np.einsum("qa,qb,eq->eab", psi, psi, g.dV)
        self.B = Pattern(self.cell_pdofs, (self.n_p, self.ndof_u), self.disp.cell_dofs).assemble(Be)
        self.M = Pattern(self.cell_pdofs, (self.n_p, self.n_p)).assemble(Me)
        self.A = self.disp.K
        self._fact: tuple[bytes, Factorization, bool, np.ndarray] | None = None

    @property
    def nu(self) -> float:
        return 0.5 if math.isinf(self.lam) else self.lam / (2 * (self.lam + self.mu))

    def _block_modulus(self) -> float:
        """The modulus c of the pressure block −M/c: λ here (p = −λ div u), K in :class:`MixedKG`."""
        return self.lam

    def _system(self, c: np.ndarray) -> tuple[Factorization, bool, np.ndarray]:
        key = c.tobytes()
        if self._fact is not None and self._fact[0] == key:
            return self._fact[1], self._fact[2], self._fact[3]
        self.disp._check_supports(c)
        free = np.setdiff1d(self.disp.active, c)
        A, B = self.A[free][:, free], self.B[:, free]
        mod = self._block_modulus()
        C = sp.csr_matrix((self.n_p, self.n_p)) if math.isinf(mod) else -self.M / mod
        gauge = False
        if math.isinf(mod):
            r = B.T @ np.ones(self.n_p)
            gauge = bool(np.linalg.norm(r) <= 1e-12 * sp.linalg.norm(B) * math.sqrt(self.n_p))
        if gauge:
            gv = sp.csr_matrix(self.M @ np.ones(self.n_p))                   # ∫ ψ_k
            K = sp.bmat([[A, B.T, None], [B, C, gv.T], [None, gv, None]], format="csr")
        else:
            K = sp.bmat([[A, B.T], [B, C]], format="csr")
        f = Factorization(K, "symmetric_indefinite")
        self._fact = (key, f, gauge, free)
        return f, gauge, free

    def system(self, bc: Dirichlet) -> tuple[sp.csr_matrix, Factorization, bool]:
        """(saddle-point matrix on the free DOFs (+ gauge row), its factorisation, gauge used) for ``bc``."""
        fact, gauge, _ = self._system(bc.dofs)
        return fact.A, fact, gauge

    def solve(self, f: np.ndarray, bc: Dirichlet | None = None) -> MixedSolution:
        """Displacement and pressure for load(s) f (ndof_u,) or (ndof_u, n_cases). Raises ValueError for an
        incompressible body (λ or K = ∞) with every boundary DOF prescribed when the prescribed displacement has a net
        volume change (∫ div u ≠ 0: no solution exists; ``docs/foundations/prereg/XF_mixed_compatibility.md``)."""
        F = np.asarray(f, float)
        F = F[:, None] if F.ndim == 1 else F
        bc = Dirichlet((), F.shape[1]) if bc is None else bc
        _, gauge, _ = self._system(bc.dofs)
        if gauge and len(bc.dofs):
            Uc = np.broadcast_to(bc.values, (len(bc.dofs), F.shape[1]))
            Bc = self.B[:, bc.dofs]
            net = np.abs(np.ones(self.n_p) @ (Bc @ Uc))
            scale = np.abs(np.ones(self.n_p) @ abs(Bc)) @ np.abs(Uc)
            if (net > 1e-10 * np.maximum(scale, 1e-300)).any():
                raise ValueError("incompressible body with every boundary DOF prescribed: the data have a net volume "
                                 f"change (∮ u·n ≠ 0, relative {float((net / np.maximum(scale, 1e-300)).max()):.2e}); "
                                 "no solution exists")
        c = bc.dofs
        fact, gauge, free = self._system(c)
        Uc = np.broadcast_to(bc.values, (len(c), F.shape[1]))
        ru = F[free] - self.A[free][:, c] @ Uc
        rp = -(self.B[:, c] @ Uc)
        rhs = np.vstack([ru, rp] + ([np.zeros((1, F.shape[1]))] if gauge else []))
        X = fact.solve(rhs).reshape(rhs.shape)
        U = np.zeros_like(F)
        U[c] = Uc
        U[free] = X[:len(free)]
        P = X[len(free):len(free) + self.n_p]
        R = np.zeros_like(F)
        R[c] = self.A[c] @ U + self.B[:, c].T @ P - F[c]
        return MixedSolution(U, P, R, gauge)

    # ------------------------------------------------------------------ errors
    def pressure_error(self, p: np.ndarray, exact: Any, degree: int | None = None, zero_mean: bool = False) -> float:
        """Relative L2 pressure error; ``exact(x (n, dim)) -> (n,)``. ``zero_mean``: compare mean-free parts."""
        ref = self.mesh.ref
        deg = 2 * ref.degree + 4 if degree is None else int(degree)
        from toporlogy.fem.mesh.quadrature import rule
        g = kn.geometry(ref, self.mesh.cell_points(), rule(ref.cell, deg))
        psi = get(ref.vertex_element).shape(g.rule.points) if self.pressure_space == "vertex" else \
            np.ones((len(g.rule.weights), 1))
        ph = np.einsum("qa,ea->eq", psi, np.asarray(p, float).ravel()[self.cell_pdofs])
        px = np.asarray(exact(g.x.reshape(-1, self.mesh.dim))).reshape(ph.shape)
        if zero_mean:
            V = g.dV.sum()
            ph = ph - (ph * g.dV).sum() / V
            px = px - (px * g.dV).sum() / V
        return float(np.sqrt(((ph - px) ** 2 * g.dV).sum() / (px ** 2 * g.dV).sum()))

    def inf_sup(self, bc: Dirichlet, n_eigs: int | None = None) -> np.ndarray:
        """Eigenvalues β² of B A⁻¹ Bᵀ q = β² M q on the free DOFs (dense, small meshes only), ascending."""
        c = bc.dofs
        self.disp._check_supports(c)
        free = np.setdiff1d(self.disp.active, c)
        A = self.A[free][:, free].toarray()
        B = self.B[:, free].toarray()
        S = B @ sla.solve(A, B.T, assume_a="pos")
        w = sla.eigh(0.5 * (S + S.T), self.M.toarray(), eigvals_only=True)
        return np.sort(w)[:n_eigs] if n_eigs else np.sort(w)


class MixedKG(MixedElasticity):
    """Volumetric–deviatoric u-p elasticity with shear modulus G and bulk modulus K (``math.inf`` allowed).

    ``σ = 2G dev ε − p I`` with the **mean pressure** ``p = −tr σ / 3 = −K div u`` (the λ-based
    :class:`MixedElasticity` carries ``p_λ = −λ div u = p_K + (2G/3) div u`` instead). 3-D physical strain; plane
    strain embeds ε_zz = 0, so ``dev ε = ε − (div u / 3) I₃``. Weak form, all (v, q)::

        ∫ 2G dev ε(u):dev ε(v) − ∫ p div v = f(v),     −∫ q div u − ∫ q p / K = 0

    The system is ``[[A_dev, Bᵀ], [B, −M/K]]`` with the same ``B``, ``M`` as the λ-based class and
    ``A_dev = ∫ 2G ε:ε − (2G/3) ∫ div u div v``. Only K > 0 and G > 0 are required: ν = 0 (λ = 0) and ν < 0 are valid.
    K = ∞ is the exact constraint div u = 0 (zero pressure block, gauge ``∫ p = 0`` when the constant pressure is in
    the null space). On a fully constrained boundary with K = ∞ the boundary data must satisfy ∮ u·n = 0:
    :meth:`solve` raises ``ValueError`` otherwise. Scalar G and K only (a heterogeneous or anisotropic material needs
    its own volumetric split, not this scalar formula). Plane stress is refused, as in the parent class.

    Parameters
    ----------
    mesh : Mesh
    G, K : float
    pressure : {"vertex", "P0"}
    plane : {"strain", "stress"}
    """

    def __init__(self, mesh: Mesh, G: float, K: float, pressure: str = "vertex", plane: str = "strain") -> None:
        if mesh.dim == 2 and plane != "strain":
            raise ValueError("the mixed u-p formulation is for plane strain and 3-D; plane stress has no "
                             "incompressibility constraint: use ElasticProblem")
        if not (G > 0 and K > 0):
            raise ValueError("G > 0 and K > 0 are required")
        self._bulk = float(K)
        self._build(mesh, G, pressure)
        g = self.disp.geom
        div = g.dNdx.reshape(*g.dNdx.shape[:2], -1)                          # node-major: ∂N_n/∂x_i at n·dim + i
        Ge = np.einsum("eqi,eqj,eq->eij", div, div, g.dV)
        self.G_div = self.disp.pattern.assemble(Ge)                          # ∫ div φ_i div φ_j
        self.A = (self.disp.K - (2.0 * self.mu / 3.0) * self.G_div).tocsr()

    @property
    def G(self) -> float:
        return self.mu

    @property
    def K(self) -> float:
        return self._bulk

    @property
    def lam(self) -> float:  # type: ignore[override]
        """Lamé λ = K − 2G/3 (derived; may be zero or negative)."""
        return math.inf if math.isinf(self._bulk) else self._bulk - 2.0 * self.mu / 3.0

    def _block_modulus(self) -> float:
        return self._bulk
