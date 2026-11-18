"""Free vibration on the regular grid: consistent mass, generalised eigenproblem, differentiable eigenvalues.

``K(E) φ = λ M(ρ) φ`` on the free DOFs of an :class:`~toporlogy.fem.elastic.Elasticity` model (same element
stiffness, same fixed DOFs, same reduced numbering ``free``), with

* ``K = Σ_e E_e · s_K · KE`` — ``s_K = h`` in 3-D, ``thickness`` in 2-D (exactly as :class:`Elasticity`);
* ``M = Σ_e ρ_e · s_M · ME + M_ns`` — ``ME`` the consistent mass of the unit cube / unit square for ρ = 1
  (:func:`element_mass`), ``s_M = h³`` in 3-D and ``thickness · h²`` in 2-D, ``M_ns`` a diagonal of non-structural
  point masses (:meth:`Modal.add_point_mass`).

``ρ_e`` comes from a mass interpolation module (default :class:`toporlogy.optimization.dynamics.LinearMass`) and already
contains the element's solid fraction. Units: λ = ω² in (model E unit) / ((model ρ unit) · (model length unit)²);
with SI E [Pa], ρ [kg/m³] and h [m], λ is in rad²/s². Point masses are in (ρ unit) · (length unit)³.

Eigen solver: :func:`scipy.sparse.linalg.eigsh` in shift-invert mode about ``sigma`` (default 0; K is positive
definite when the fixed DOFs remove the rigid-body modes), with the operator ``(K − σ M)⁻¹`` applied by a
:class:`~toporlogy.solvers.DirectSolver` factorisation (PARDISO or SuperLU). The returned Ritz vectors are
cleaned by one Rayleigh–Ritz step in their own span, so ``Φᵀ M Φ = I`` and ``Φᵀ K Φ = diag(λ)`` to round-off.

Sensitivities (:meth:`Modal.eigenvalues`, a ``torch.autograd.Function``) for a *simple* eigenvalue with
M-normalised φ:

    ∂λ/∂E_e = s_K · φ_eᵀ KE φ_e,        ∂λ/∂ρ_e = −λ · s_M · φ_eᵀ ME φ_e .

No adjoint solve is needed. For a repeated eigenvalue the individual λ_i are not differentiable (only directionally);
the formula above summed over a whole cluster with *equal* weights is the derivative of the cluster sum, which is
invariant to the basis chosen inside the eigenspace and therefore correct. The backward pass detects clusters
(relative gap ≤ ``cluster_tol``, one extra mode is always computed so a cluster cut at the last requested mode is
seen) and warns (or raises) when the incoming weights differ inside a cluster. Smooth symmetric aggregates such as
:class:`toporlogy.optimization.dynamics.MinEigenvalueKS` over all modes of a cluster give equal weights and are safe.

H8IM: the incompatible modes are condensed statically from K; they carry no mass (the consistent mass of the
compatible trilinear field is used). This is the usual practice; it is not a consistent mass of the enhanced field.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.solvers.direct import DirectSolver

__all__ = ["Modal", "element_mass", "find_clusters", "RepeatedEigenvalueWarning", "SPLU_MAX_DOF"]


SPLU_MAX_DOF = 30_000
"""Free-DOF count up to which the default eigen solver factorises with SuperLU instead of PARDISO."""


class _NoSolver:
    """Placeholder until the solver is chosen (the free-DOF count is known only after Elasticity set-up)."""

    name = "none"


class RepeatedEigenvalueWarning(UserWarning):
    """An individual (near-)repeated eigenvalue was differentiated with unequal weights inside its cluster."""


def element_mass(dim: int) -> np.ndarray:
    """Consistent mass ``∫ Nᵀ N dV`` of the unit cube (3-D, 24×24) or unit square (2-D, 8×8, unit thickness), ρ = 1.

    Bilinear / trilinear shape functions, 2-point Gauss rule per axis (exact for the products N_a N_b). Node order
    of :class:`toporlogy.core.domain.Domain` (corners in :data:`Domain.corners` order), DOFs node-major (u, v[, w]).
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    corners = (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
               if dim == 3 else np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    nat = 2.0 * corners - 1.0                                   # natural coordinates ±1
    g = 1.0 / np.sqrt(3.0)
    nn = len(corners)
    Ms = np.zeros((nn, nn))
    detJ = 0.5 ** dim                                           # unit element mapped from [−1, 1]^dim
    for p in product((-g, g), repeat=dim):
        N = np.prod(1.0 + nat * np.asarray(p)[None, :], axis=1) / 2.0 ** dim
        Ms += np.outer(N, N) * detJ                             # Gauss weights are 1
    return np.kron(Ms, np.eye(dim))


def find_clusters(lam: np.ndarray, rel_tol: float) -> list[np.ndarray]:
    """Groups of consecutive sorted eigenvalues with ``λ_{i+1} − λ_i ≤ rel_tol · max(|λ_{i+1}|, tiny)``.

    Returns the index arrays of groups with more than one member.
    """
    lam = np.asarray(lam, float)
    out, cur = [], [0]
    for i in range(1, lam.size):
        if lam[i] - lam[i - 1] <= rel_tol * max(abs(lam[i]), 1e-300):
            cur.append(i)
        else:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = [i]
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


class Modal:
    """Generalised eigenproblem ``K φ = λ M φ`` on a :class:`Domain` with fixed DOFs.

    Parameters
    ----------
    domain : Domain
    fixed : sequence of int
        Box DOF numbers held at zero (as for :class:`Elasticity`).
    nu : float
        Poisson's ratio.
    element : str, optional
        "H8" (3-D default), "H8IM", or "Q4" (2-D).
    mass : torch module x̄ → ρ (element grid), optional
        Mass interpolation; default :class:`toporlogy.optimization.dynamics.LinearMass` with ρ0 = 1.
    n_modes : int
        Modes computed per evaluation at least (responses may ask for more).
    sigma : float
        Shift of the shift-invert Lanczos iteration; must lie below λ_1 (use σ < 0 for free structures).
    tol : float
        ARPACK tolerance (0 = machine precision; gradients by finite differences need it).
    cluster_tol : float
        Relative gap below which consecutive eigenvalues are treated as one (near-)repeated cluster.
    repeated : {"warn", "raise", "ignore"}
        What the backward pass does when the weights differ inside a cluster.
    solver : object with ``factor(A_upper)`` / ``solve(B)``, optional
        Default :class:`DirectSolver`: "splu" up to :data:`SPLU_MAX_DOF` free DOFs, else "auto" (PARDISO).

    Attributes
    ----------
    elastic : Elasticity
        The stiffness model (``free``, ``KE``, ``geom``, ``elements`` are shared with it).
    last_lambda, last_modes : numpy.ndarray
        Eigenvalues (k+1,) and M-orthonormal modes (ndof, k+1) of the last solve (box DOF numbering, zeros on fixed
        DOFs); ``last_clusters`` the detected clusters.
    """

    def __init__(self, domain: Domain, fixed: Sequence[int], nu: float = 0.3, element: str | None = None,
                 mass: Any = None, n_modes: int = 6, sigma: float = 0.0, tol: float = 0.0,
                 cluster_tol: float = 1e-3, repeated: str = "warn", solver: Any = None) -> None:
        if repeated not in ("warn", "raise", "ignore"):
            raise ValueError("repeated: warn | raise | ignore")
        if n_modes < 1:
            raise ValueError("n_modes ≥ 1 is required")
        self.elastic = Elasticity(domain, fixed, nu=nu, element=element, solver=solver or _NoSolver())
        if solver is None:
            # ARPACK applies (K − σM)⁻¹ to one vector at a time (≈ 20-140 solves per eigen solve). Measured on this
            # machine (13 Nov 2026, 2-D 56×8, 918 free DOFs): PARDISO 13 ms per single-RHS call (thread start-up),
            # SuperLU < 1 ms. SuperLU is therefore the default below ``SPLU_MAX_DOF`` free DOFs.
            solver = DirectSolver("splu" if len(self.elastic.free) <= SPLU_MAX_DOF else "auto")
            self.elastic.solver = solver
        self.solver = solver
        self.domain = domain
        self.nu, self.element = self.elastic.nu, self.elastic.element
        self.KE, self.geom = self.elastic.KE, self.elastic.geom
        self.ME = element_mass(domain.dim)
        self.mgeom = domain.h ** 3 if domain.dim == 3 else domain.thickness * domain.h ** 2
        self.elements, self.edof = self.elastic.elements, self.elastic.edof
        self.free, self.fixed = self.elastic.free, self.elastic.fixed
        if mass is None:
            from toporlogy.optimization.dynamics import LinearMass
            mass = LinearMass(domain)
        self.mass = mass
        self.n_modes, self.sigma, self.tol = int(n_modes), float(sigma), float(tol)
        self.cluster_tol, self.repeated = float(cluster_tol), repeated
        nf, nl = len(self.free), self.KE.shape[0]
        gmap = np.full(domain.ndof, -1, np.int64)
        gmap[self.free] = np.arange(nf)
        self._gmap = gmap
        r = gmap[np.repeat(self.edof, nl, axis=1).ravel()]
        c = gmap[np.tile(self.edof, (1, nl)).ravel()]
        keep = (r >= 0) & (c >= 0)
        self._sel = np.flatnonzero(keep)
        key = r[keep] * nf + c[keep]
        ukey, inv = np.unique(key, return_inverse=True)
        self._inv = inv.astype(np.int64)
        self._P = sp.csr_matrix((np.zeros(len(ukey)), (ukey // nf, ukey % nf)), shape=(nf, nf))
        self._P.sort_indices()
        self.point_mass = np.zeros(nf)                          # non-structural diagonal mass on free DOFs
        self.last_lambda = np.zeros(0)
        self.last_modes = np.zeros((domain.ndof, 0))
        self.last_clusters: list[np.ndarray] = []
        self.n_eigen_solves = 0

    # ------------------------------------------------------------------ set-up
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_free(self) -> int:
        return len(self.free)

    def add_point_mass(self, nodes: Sequence[int], mass: float, directions: Sequence[int] | None = None) -> None:
        """Add a lumped non-structural mass ``mass`` to *each* of ``nodes`` (all translations by default).

        DOFs that are fixed (or not part of the model) are skipped. Units: (ρ unit)·(length unit)³.
        """
        d = range(self.domain.dim) if directions is None else directions
        dofs = self.domain.dofs(np.asarray(nodes, np.int64), list(d))
        idx = self._gmap[dofs]
        np.add.at(self.point_mass, idx[idx >= 0], float(mass))

    # ------------------------------------------------------------------ assembly (NumPy)
    def _assemble(self, values: np.ndarray, Ke: np.ndarray, scale: float) -> sp.csr_matrix:
        s = np.broadcast_to(np.asarray(values, float) * scale, (self.n_elements,))
        w = (s[:, None] * Ke.ravel()[None, :]).ravel()[self._sel]
        A = self._P.copy()
        A.data = np.bincount(self._inv, weights=w, minlength=self._P.nnz)
        return A

    def stiffness(self, E_active: np.ndarray) -> sp.csr_matrix:
        """Full symmetric reduced stiffness (free DOFs) for element moduli ``E_active`` (na,)."""
        return self._assemble(E_active, self.KE, self.geom)

    def mass_matrix(self, rho_active: np.ndarray) -> sp.csr_matrix:
        """Full symmetric reduced mass (free DOFs) for element densities ``rho_active`` (na,), point masses included."""
        M = self._assemble(rho_active, self.ME, self.mgeom)
        if np.any(self.point_mass):
            M = (M + sp.diags(self.point_mass)).tocsr()
        return M

    # ------------------------------------------------------------------ eigen solve (NumPy)
    def solve_numpy(self, E_active: np.ndarray, rho_active: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Lowest ``k`` eigenpairs: λ (k,) ascending and M-orthonormal Φ (n_free, k)."""
        nf = self.n_free
        if not 1 <= k < nf:
            raise ValueError(f"k must lie in [1, {nf - 1}]")
        K = self.stiffness(E_active)
        M = self.mass_matrix(rho_active)
        A = K - self.sigma * M if self.sigma != 0.0 else K
        Au = sp.triu(A).tocsr()
        Au.sort_indices()
        self.solver.factor(Au)
        op = spla.LinearOperator((nf, nf), matvec=lambda b: self.solver.solve(b), dtype=float)
        v0 = np.random.default_rng(0).standard_normal(nf)
        ncv = min(nf, max(2 * k + 1, k + 16))
        _, V = spla.eigsh(K, k=k, M=M, sigma=self.sigma, which="LM", OPinv=op, tol=self.tol, ncv=ncv, v0=v0)
        # Rayleigh–Ritz in span(V): exact M-orthonormality and refined eigenvalues.
        Kr, Mr = V.T @ (K @ V), V.T @ (M @ V)
        lam, Q = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
        self.n_eigen_solves += 1
        return lam, V @ Q

    def full_modes(self, Phi: np.ndarray) -> np.ndarray:
        """Free-DOF vectors (n_free, k) → box numbering (ndof, k), zeros on fixed / unused DOFs."""
        out = np.zeros((self.domain.ndof, Phi.shape[1]))
        out[self.free] = Phi
        return out

    def modes_numpy(self, E_active: np.ndarray, rho_active: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues (k,) and box-numbered modes (ndof, k); no autograd."""
        lam, Phi = self.solve_numpy(E_active, rho_active, k)
        return lam, self.full_modes(Phi)

    def element_modal_energies(self, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit-coefficient element quadratic forms (na, k): ``s_K φ_eᵀ KE φ_e`` and ``s_M φ_eᵀ ME φ_e``."""
        Pe = modes[self.edof]                                      # (na, nloc, k)
        kq = self.geom * np.einsum("eak,ab,ebk->ek", Pe, self.KE, Pe)
        mq = self.mgeom * np.einsum("eak,ab,ebk->ek", Pe, self.ME, Pe)
        return kq, mq

    # ------------------------------------------------------------------ autograd
    def density(self, xphys: torch.Tensor) -> torch.Tensor:
        """Element mass densities on the grid from the mass interpolation."""
        return self.mass(xphys)

    def eigenvalues(self, E_active: torch.Tensor, rho_active: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Differentiable lowest ``k`` eigenvalues (k,) of ``K(E) φ = λ M(ρ) φ``; default ``k = n_modes``."""
        k = self.n_modes if k is None else int(k)
        return _Eigen.apply(E_active, rho_active, self, k)

    def low_density_share(self, xphys: np.ndarray, rho_active: np.ndarray, modes: np.ndarray | None = None,
                          threshold: float = 0.1) -> np.ndarray:
        """Share of each mode's kinetic energy (φᵀMφ = 1) carried by elements with x̄ < ``threshold``.

        A value near 1 flags a spurious, localised mode in the low-density region. ``modes`` default
        :attr:`last_modes`.
        """
        modes = self.last_modes if modes is None else modes
        _, mq = self.element_modal_energies(modes)
        rho = np.asarray(rho_active, float)
        low = np.asarray(xphys, float).reshape(-1)[self.elements] < threshold
        return (rho[low, None] * mq[low]).sum(axis=0)

    def check_cluster_weights(self, g: np.ndarray, lam: np.ndarray) -> None:
        """Warn / raise if the weights ``g`` (len(lam),) differ inside a cluster of ``lam``."""
        if self.repeated == "ignore":
            return
        for cl in find_clusters(lam, self.cluster_tol):
            gc = g[cl]
            scale = max(float(np.max(np.abs(g))), 1e-300)
            if np.max(gc) - np.min(gc) > 1e-8 * scale:
                msg = (f"eigenvalues {list(cl + 1)} are (near-)repeated (relative gap ≤ {self.cluster_tol:g}, "
                       f"λ = {lam[cl]}) but are weighted differently; an individual repeated eigenvalue is not "
                       "differentiable — aggregate all modes of the cluster (MinEigenvalueKS) instead")
                if self.repeated == "raise":
                    raise FloatingPointError(msg)
                warnings.warn(msg, RepeatedEigenvalueWarning, stacklevel=3)


class _Eigen(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E_active: torch.Tensor, rho_active: torch.Tensor, model: Modal,
                k: int) -> torch.Tensor:
        E_np = E_active.detach().cpu().double().numpy()
        r_np = rho_active.detach().cpu().double().numpy()
        kk = min(k + 1, model.n_free - 1)                         # one extra mode: see a cluster cut at mode k
        lam, Phi = model.solve_numpy(E_np, r_np, kk)
        modes = model.full_modes(Phi)
        model.last_lambda, model.last_modes = lam, modes
        model.last_clusters = find_clusters(lam, model.cluster_tol)
        ctx.model, ctx.lam, ctx.modes, ctx.k = model, lam, modes, k
        ctx.device, ctx.dtypes = E_active.device, (E_active.dtype, rho_active.dtype)
        return torch.from_numpy(lam[:k].copy()).to(device=E_active.device, dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, glam: torch.Tensor) -> tuple[Any, Any, None, None]:
        model: Modal = ctx.model
        g = np.zeros(ctx.lam.size)
        g[:ctx.k] = glam.detach().cpu().double().numpy()
        model.check_cluster_weights(g, ctx.lam)
        kq, mq = model.element_modal_energies(ctx.modes)
        dE = kq @ g
        drho = -(mq * ctx.lam[None, :]) @ g
        return (torch.from_numpy(dE).to(device=ctx.device, dtype=ctx.dtypes[0]),
                torch.from_numpy(drho).to(device=ctx.device, dtype=ctx.dtypes[1]), None, None)
