"""Linearised buckling on the regular grid: stress (geometric) stiffness and the buckling eigenproblem.

For a reference load F with displacements u (``K u = F``) the structure buckles at the load λ F, where λ is the
smallest positive eigenvalue of

    (K + λ K_G(u)) φ = 0 ,          K_G = Σ_e ∫ Gᵀ S(σ_e) G dV ,

with G the gradient operator of the displacement components and S the stress tensor of the reference state (positive
in tension, so compression makes K_G negative along the buckling mode). The eigenproblem is solved in the form

    (−K_G) φ = μ K φ ,              μ = 1/λ ,

for the largest μ: K is positive definite (its factorisation from the static solve is reused as the inverse), −K_G is
indefinite, and the modes of interest are the largest μ, which is what a Lanczos method finds first. Modes are
returned K-normalised (φᵀ K φ = 1), so μ = −φᵀ K_G φ.

Element level (unit element, unit modulus; ``B`` strain matrix, ``D`` elasticity matrix, ``dN`` shape-function
gradients, 2-point Gauss rule per axis):

    K_G,e = s_G · E_G,e · Σ_gp w · (dNᵀ S_gp dN) ⊗ I_dim ,       S_gp = tensor of σ = D B_gp u_e ,

with ``s_G = 1`` in 3-D (h³ · h⁻² · h⁻¹) and ``thickness / h`` in 2-D. ``E_G,e`` is the modulus used for the stress
of element e; it is a separate argument because the standard remedy against spurious buckling modes in void regions
is to interpolate it WITHOUT the lower bound E_min of the stiffness (void elements then carry no stress stiffness).
The element matrix is linear in u_e. Incompatible-mode elements (H8IM, Q4IM): by default (``modes_in_kg``) the
recovered modes α = R u_e enter K_G consistently, in the stress σ = D (B_u + B_α R) u_e and in the displacement
gradient of the mode, K_G,e = Tᵀ K_G,ext T with T = [I; R]. Measured on the column of Ferrari & Sigmund (2019)
with a scratch code: the error of λ1 against the converged 2-D value falls by about a quarter on coarse meshes
(20 × 2: 1.9e-3 → 1.4e-3; ``docs/olcumler/malzeme_modelleri.md``). ``modes_in_kg=False`` uses the compatible
field only (the behaviour before 25 Nov 2026, as the mass matrix in :mod:`toporlogy.fem.modal`).

Per-element material (fibre orientation; ``model.ke_index`` set): σ = D_e B u_e with the element's own D and, for
incompatible-mode elements, its own mode recovery R_e (T_e = [I; R_e]); the quantities are built once per distinct
D and gathered per element in chunks. With one material the original arrays and code path are used unchanged.

Preload. With a load P0 that stays fixed and a load P1 that is scaled, buckling is
``(K + K_G(u0) + λ K_G(u1)) φ = 0``; :meth:`LinearBuckling.modes` takes ``KG0 = K_G(u0)`` and solves
``(−K_G(u1)) φ = μ (K + K_G(u0)) φ``. It requires K + K_G(u0) positive definite, i.e. the preload alone must not
buckle (λ0 > 1), and checks that first (one extra eigenvalue).

Singular stresses. At the end of a partly clamped edge, at a re-entrant corner or at a point load the linear
elastic stress is unbounded; its compressive part gives K_G an unbounded local term, and the first buckling mode
then localises there with λ → 0 as the mesh is refined: such a λ is a property of the mesh, not of the structure
(measured on the wall of Ferrari, Sigmund & Guest 2021: λ1 2.85, 2.11, 1.60, 1.20 at 40², 80², 160², 320²;
``docs/olcumler/malzeme_modelleri.md``). :meth:`LinearBuckling.localization` measures it (energy share of the most
loaded elements; rising under refinement = singular), and the responses accept ``exclude``, elements whose stress
stiffness is dropped (E_G = 0) around such points. Excluding moves the localisation to the edge of the excluded
zone rather than removing it (measured on the same wall), so λ then depends on the zone size: report it.

Diagnostics. :meth:`LinearBuckling.low_density_share` is the share of each mode's strain energy φᵀKφ held by
elements below a density threshold; a share near 1 flags a spurious mode localised in grey or void material (the
same measure as :meth:`toporlogy.fem.modal.Modal.low_density_share`). :class:`BucklingClusterWarning` is raised by
the responses when the last aggregated mode and the next one form a (near-)repeated cluster, because aggregating a
cut cluster is not differentiable.

What the model does not contain: geometric non-linearity, imperfections and post-buckling (λ is the bifurcation
load of the linearised problem); in 2-D, buckling out of the plane (the plane-stress model has in-plane DOFs only,
so a thin plate's usually critical out-of-plane mode is not found); follower loads (the load is dead); material
non-linearity — compare λ with the first-failure load factor of a strength criterion
(:func:`toporlogy.optimization.buckling.first_failure_load_factor`).

Verification (``tests/toporlogy/test_buckling.py``): the Gauss-point strain matrices reproduce the element
stiffness; K_G vanishes for rigid-body displacement fields and scales linearly with the load; Euler columns in 2-D and
3-D against π² E I / (K L)².

Literature. The formulation is the classical one (Cook et al. 2002, ch. 18). The practice of a separate stress
interpolation and of aggregating μ = 1/λ follows what is reported for Ferrari & Sigmund (2019) and Ferrari, Sigmund &
Guest (2021); only records and abstracts of those were seen (``docs/literatur_2026-11-21.md``).
"""
from __future__ import annotations

from itertools import product
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.core.tensor import voigt_pairs
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import H8_CORNER, Q4_CORNER
from toporlogy.references import cite
from toporlogy.solvers.direct import DirectSolver

__all__ = ["gauss_operators", "LinearBuckling", "BucklingClusterWarning"]


class BucklingClusterWarning(UserWarning):
    """The aggregated buckling modes end inside a (near-)repeated cluster: compute more modes."""

_G = 1.0 / np.sqrt(3.0)
_CHUNK = 2048                      # elements per gathered block of per-element arrays


def _chunks(n: int) -> list[slice]:
    return [slice(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)]


def gauss_operators(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-element operators at the 2^dim Gauss points: ``dN`` (ngp, dim, nn) shape-function gradients, ``B``
    (ngp, nvoigt, dim·nn) strain matrices (node-major DOFs), ``w`` (ngp,) weights including det J (they sum to the
    unit element volume 1)."""
    corners = Q4_CORNER if dim == 2 else H8_CORNER
    nn = len(corners)
    pairs = voigt_pairs(dim)
    dNs, Bs = [], []
    for p in product((-_G, _G), repeat=dim):
        a = 1.0 + corners * np.asarray(p)
        dN = np.empty((dim, nn))
        for i in range(dim):
            others = np.prod(np.delete(a, i, axis=1), axis=1)
            dN[i] = corners[:, i] * others / 2.0 ** dim * 2.0          # d/dx = 2 d/dxi on the unit element
        B = np.zeros((len(pairs), dim * nn))
        for v, (i, j) in enumerate(pairs):
            if i == j:
                B[v, i::dim] = dN[i]
            else:
                B[v, i::dim] = dN[j]
                B[v, j::dim] = dN[i]
        dNs.append(dN)
        Bs.append(B)
    w = np.full(2 ** dim, 1.0 / 2 ** dim)
    return np.array(dNs), np.array(Bs), w


@cite("cook_2002", "ferrari_sigmund_2019", "ferrari_sigmund_guest_2021",
      note="stress stiffness (Cook ch. 18); eigenproblem in μ = 1/λ and separate stress modulus as reported for "
           "Ferrari et al. (records seen, texts not read)")
class LinearBuckling:
    """Buckling analysis on top of an :class:`~toporlogy.fem.elastic.Elasticity` model (same mesh, supports and
    reduced numbering).

    Parameters
    ----------
    model : Elasticity
        Plain elasticity with axis-aligned fixed DOFs (the direct-solver model).
    modes_in_kg : bool
        Incompatible-mode elements: include the condensed modes in K_G (module notes). Ignored for H8 / Q4.
    """

    def __init__(self, model: Elasticity, modes_in_kg: bool = True) -> None:
        if not isinstance(model, Elasticity):
            raise TypeError("LinearBuckling needs an Elasticity model (direct solver, axis-aligned fixed DOFs)")
        self.model = model
        d = model.domain
        self.dim = d.dim
        self.dN, self.B, self.w = gauss_operators(d.dim)
        self.D = np.asarray(model.D, float)                                # unit-modulus D: (k, k) or (na, k, k)
        self.modes_in_kg = bool(modes_in_kg) and model.element in ("H8IM", "Q4IM")
        self.ke_index = model.ke_index
        self._DBd: np.ndarray | None = None                                  # per-element material (module notes)
        self._Td: np.ndarray | None = None
        nloc = self.B.shape[2]
        if self.modes_in_kg:
            nm = d.dim * d.dim                                               # dim modes per displacement component
            dP = np.array([np.diag(-4.0 * np.asarray(p)) for p in product((-_G, _G), repeat=d.dim)])
            Ba = np.zeros((len(dP), self.B.shape[1], nm))
            for g, P in enumerate(dP):                                       # strain of the modes, node-major
                for v, (i, j) in enumerate(voigt_pairs(d.dim)):
                    if i == j:
                        Ba[g, v, i::d.dim] = P[i]
                    else:
                        Ba[g, v, i::d.dim] = P[j]
                        Ba[g, v, j::d.dim] = P[i]
            self.dN = np.concatenate([self.dN, dP], axis=2)                  # (ngp, dim, nn + n_modes)
        if self.ke_index is None:                                            # one material: the original path
            if self.modes_in_kg:
                from toporlogy.fem.elements import ke_h8_from_D, ke_q4im_from_D
                R = ke_h8_from_D(self.D, "H8IM")[1] if d.dim == 3 else ke_q4im_from_D(self.D)[1]
                assert R is not None
                self.B = self.B + np.einsum("gvm,mk->gvk", Ba, R)            # enhanced strain per unit u_e
                self.T = np.vstack([np.eye(nloc), R])                        # u_e -> [u_e; α]
            else:
                self.T = np.eye(nloc)
            self.DB = np.einsum("vw,gwk->gvk", self.D, self.B)             # (ngp, nvoigt, nloc): σ per unit u_e
        else:                                                                # per distinct D, gathered per element
            from toporlogy.fem.elements import mode_recovery_from_D
            Dd = model.D_distinct
            assert Dd is not None
            if self.modes_in_kg:
                Rd = np.array([mode_recovery_from_D(d.dim, Dk) for Dk in Dd])           # (m, n_modes, nloc)
                Bd = self.B[None] + np.einsum("gvm,jmk->jgvk", Ba, Rd)                 # (m, ngp, nvoigt, nloc)
                self._Td = np.concatenate([np.broadcast_to(np.eye(nloc), (len(Rd), nloc, nloc)), Rd], axis=1)
            else:
                Bd = np.broadcast_to(self.B, (len(Dd),) + self.B.shape)
            self._DBd = np.einsum("jvw,jgwk->jgvk", Dd, Bd)                  # σ per unit u_e, per distinct D
            self.T = np.eye(nloc) if not self.modes_in_kg else np.zeros((0, nloc))   # per element: _Td
            self.DB = np.zeros((0, 0, nloc))                                 # per element: _DBd
        self.scale = 1.0 if d.dim == 3 else d.thickness / d.h
        self.h = d.h
        nf = len(model.free)
        gmap = np.full(d.ndof, -1, np.int64)
        gmap[model.free] = np.arange(nf)
        nl = model.edof.shape[1]
        r = gmap[np.repeat(model.edof, nl, axis=1).ravel()]
        c = gmap[np.tile(model.edof, (1, nl)).ravel()]
        self._keep = (r >= 0) & (c >= 0)
        self._rows, self._cols, self._nf = r[self._keep], c[self._keep], nf

    # ------------------------------------------------------------------ element level
    def element_stress(self, EG_active: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Gauss-point stresses (na, ngp, nvoigt) of the active elements for displacements ``u`` (ndof,), with the
        stress modulus ``EG_active`` (na,)."""
        ue = np.asarray(u, float)[self.model.edof]                           # (na, nloc)
        if self._DBd is None:
            return np.einsum("gvk,ek->egv", self.DB, ue) * (np.asarray(EG_active, float)[:, None, None] / self.h)
        idx = self.ke_index
        assert idx is not None
        out = np.empty((len(ue),) + self._DBd.shape[1:3])
        for c in _chunks(len(ue)):
            out[c] = np.einsum("egvk,ek->egv", self._DBd[idx[c]], ue[c])
        return out * (np.asarray(EG_active, float)[:, None, None] / self.h)

    def element_matrices(self, EG_active: np.ndarray, u: np.ndarray) -> np.ndarray:
        """(na, nloc, nloc) stress stiffness of every active element (physical scale)."""
        s = self.element_stress(EG_active, u) * self.h                       # unit-element stress per unit h
        dim, nn = self.dim, self.dN.shape[2]
        S = np.zeros(s.shape[:2] + (dim, dim))
        for v, (i, j) in enumerate(voigt_pairs(dim)):
            S[..., i, j] = s[..., v]
            S[..., j, i] = s[..., v]
        kn = np.einsum("g,gia,egij,gjb->eab", self.w, self.dN, S, self.dN) * self.scale      # (na, nn, nn)
        out = np.zeros((len(kn), dim * nn, dim * nn))
        for c in range(dim):
            out[:, c::dim, c::dim] = kn
        if not self.modes_in_kg:
            return out
        if self._Td is None:
            return np.einsum("ai,eab,bj->eij", self.T, out, self.T)
        idx = self.ke_index
        assert idx is not None
        nloc = self._Td.shape[2]
        res = np.empty((len(out), nloc, nloc))
        for ch in _chunks(len(out)):
            Tc = self._Td[idx[ch]]
            res[ch] = np.einsum("eai,eab,ebj->eij", Tc, out[ch], Tc)
        return res

    def geometric_matrix(self, EG_active: np.ndarray, u: np.ndarray) -> sp.csr_matrix:
        """K_G on the free DOFs (symmetric, full storage) for the reference displacements ``u`` (ndof,)."""
        ke = self.element_matrices(EG_active, u).ravel()[self._keep]
        K = sp.coo_matrix((ke, (self._rows, self._cols)), shape=(self._nf, self._nf)).tocsr()
        return ((K + K.T) * 0.5).tocsr()

    # ------------------------------------------------------------------ eigenproblem
    def modes(self, E_active: np.ndarray, EG_active: np.ndarray, u: np.ndarray, k: int = 6,
              tol: float = 1e-10, KG0: sp.spmatrix | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` largest μ = 1/λ (descending) and their modes (ndof, k), K-normalised.

        ``u``: reference displacements (ndof,) of the SAME moduli ``E_active``. The model's factorisation is reused
        when it belongs to ``E_active``; otherwise K is factorised again. μ ≤ 0 means that mode does not buckle under
        the reference load direction (tension). ``KG0``: stress stiffness of a fixed preload on the free DOFs
        (:meth:`geometric_matrix` of its displacements); the modes are then (K + KG0)-normalised and μ refers to the
        scaled load only (module notes). Raises ValueError if the preload alone buckles.
        """
        m = self.model
        E = np.asarray(E_active, float)
        if m._factored_E is None or m._A is None or not np.array_equal(m._factored_E, E):
            m.factor(E)
        A = m._A
        assert A is not None
        n = self._nf
        if not 1 <= k < n - 1:
            raise ValueError(f"need 1 <= k < {n - 1} (free DOFs − 1), got {k}")
        Kfull = (A + A.T - sp.diags(A.diagonal())).tocsr()
        KG = self.geometric_matrix(EG_active, u)
        Minv = spla.LinearOperator((n, n), matvec=lambda b: np.asarray(m.solver.solve(b)).reshape(-1),
                                   dtype=float)
        v0 = np.random.default_rng(0).standard_normal(n)                     # fixed start: reproducible runs
        if KG0 is not None:
            KG0 = sp.csr_matrix(KG0)
            if KG0.shape != (n, n):
                raise ValueError(f"KG0 must be {n} x {n} (free DOFs)")
            mu0 = spla.eigsh(-KG0, k=1, M=Kfull, Minv=Minv, which="LA", tol=tol, v0=v0,
                             return_eigenvectors=False)[0]
            if mu0 >= 1.0:
                raise ValueError(f"the preload alone buckles (load factor 1/mu0 = {1 / mu0:.4g} <= 1)")
            Kfull = (Kfull + KG0).tocsr()
            if m.ordering == "nested_dissection":                            # SPD (mu0 < 1): diagonal pivots
                ps = DirectSolver("splu")
                ps.factor(sp.triu(Kfull).tocsr(), order=m.elimination_order())
                Minv = spla.LinearOperator((n, n), matvec=lambda b: ps.solve(b).reshape(-1), dtype=float)
            else:
                lu = spla.splu(Kfull.tocsc())
                Minv = spla.LinearOperator((n, n), matvec=lu.solve, dtype=float)
        mu, Phi = spla.eigsh(-KG, k=k, M=Kfull, Minv=Minv, which="LA", tol=tol, v0=v0,
                             ncv=min(n - 1, max(2 * k + 2, 24)), maxiter=20 * n)
        # Rayleigh–Ritz in the returned span: K-orthonormal modes and consistent μ to round-off
        Kr = Phi.T @ (Kfull @ Phi)
        Gr = -(Phi.T @ (KG @ Phi))
        Lc = np.linalg.cholesky(0.5 * (Kr + Kr.T))
        Li = np.linalg.inv(Lc)
        mu, V = np.linalg.eigh(Li @ (0.5 * (Gr + Gr.T)) @ Li.T)
        order = np.argsort(-mu)
        mu, Phi = mu[order], Phi @ (Li.T @ V[:, order])
        full = np.zeros((m.domain.ndof, k))
        full[m.free] = Phi
        return mu, full

    def low_density_share(self, E_active: np.ndarray, modes: np.ndarray, xphys_active: np.ndarray,
                          threshold: float = 0.5) -> np.ndarray:
        """(k,) share of each mode's strain energy φᵀKφ in active elements with x̄ < ``threshold`` (module notes)."""
        _, kE = self.mode_forms(modes)
        w = np.asarray(E_active, float)[:, None] * kE
        low = np.asarray(xphys_active, float) < threshold
        tot = w.sum(axis=0)
        return np.where(tot > 0, w[low].sum(axis=0) / np.where(tot > 0, tot, 1.0), 0.0)

    def localization(self, E_active: np.ndarray, modes: np.ndarray, fraction: float = 0.01) -> np.ndarray:
        """(k,) share of each mode's strain energy φᵀKφ held by the ``fraction`` of active elements that carry the
        most (at least one element). A global mode spreads its energy (share ≈ a few × fraction); a mode driven by a
        stress singularity concentrates it, and the share grows as the mesh is refined (module notes)."""
        if not 0 < fraction <= 1:
            raise ValueError("0 < fraction <= 1 is required")
        _, kE = self.mode_forms(modes)
        w = np.abs(np.asarray(E_active, float)[:, None] * kE)
        n_top = max(1, int(np.ceil(fraction * w.shape[0])))
        top = np.sort(w, axis=0)[-n_top:].sum(axis=0)
        tot = w.sum(axis=0)
        return np.where(tot > 0, top / np.where(tot > 0, tot, 1.0), 0.0)

    # ------------------------------------------------------------------ pieces of the Rayleigh quotient
    def mode_forms(self, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per element and mode, the two quadratic forms of the Rayleigh quotient as functions of the design:

        ``C`` (na, k, nloc) with ``φ_iᵀ K_G φ_i = Σ_e E_G,e · (C[e, i] · u_e)`` and ``kE`` (na, k) with
        ``φ_iᵀ K φ_i = Σ_e E_e · kE[e, i]`` (both include the geometric scale factors).
        """
        m = self.model
        pe = modes[m.edof]                                                   # (na, nloc, k)
        dim, nn = self.dim, self.dN.shape[2]
        idx = self.ke_index
        if not self.modes_in_kg:
            pg = pe
        elif self._Td is None:
            pg = np.einsum("ai,eik->eak", self.T, pe)                        # with the recovered modes
        else:
            assert idx is not None
            pg = np.empty((len(pe), self._Td.shape[1], pe.shape[2]))
            for c in _chunks(len(pe)):
                pg[c] = np.einsum("eai,eik->eak", self._Td[idx[c]], pe[c])
        pc = pg.reshape(len(pe), nn, dim, -1)                                # (na, node, comp, k)
        G = np.einsum("gia,eack->egick", self.dN, pc)                        # ∂φ_c/∂x_i at the Gauss points
        M = np.einsum("egick,egjck->egijk", G, G)                            # Σ_c ∂_i φ_c ∂_j φ_c
        pairs = voigt_pairs(dim)
        mv = np.stack([M[:, :, i, j] if i == j else 2.0 * M[:, :, i, j] for i, j in pairs], axis=2)   # (na,g,v,k)
        if self._DBd is None:
            C = np.einsum("g,egvk,gvl->ekl", self.w, mv, self.DB) * self.scale
            kE = np.einsum("eak,ab,ebk->ek", pe, m.KE, pe) * m.geom
            return C, kE
        assert idx is not None
        C = np.empty((len(pe), pe.shape[2], self._DBd.shape[3]))
        for c in _chunks(len(pe)):
            C[c] = np.einsum("g,egvk,egvl->ekl", self.w, mv[c], self._DBd[idx[c]])
        return C * self.scale, m.element_energy(modes) * m.geom

