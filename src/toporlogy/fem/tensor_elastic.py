"""Linear elasticity with a designed elasticity tensor in every element (record docs/olcumler/malzeme_tasarimi.md).

``K(C) = Σ_e s · KE(C_e)`` over the active elements (s = thickness in 2-D, h in 3-D, as :class:`Elasticity`); C_e
in physical units, ``frac`` already applied (:class:`~toporlogy.core.material_field.MaterialField`).

Why a separate physics
----------------------
:class:`~toporlogy.fem.elastic.Elasticity` differentiates with respect to one scalar modulus per element. A design
that changes the tensor itself (phases with their own ν or anisotropy, a fibre angle) needs dJ/dC_e. For the
fully integrated bilinear / trilinear elements (Q4, H8) the element matrix is linear in D,

    KE(C) = Σ_a c_a KE(S_a),     C = Σ_a c_a S_a   (S_a: :func:`~toporlogy.core.material_field.symmetric_basis`),

so with the adjoint λ (K λ = ∂J/∂U; λ = a U for compliance-like J, no solve) dJ/dc_{e,a} = −s λ_eᵀ KE(S_a) u_e.
KE(S_a) = 2 (KE(I + ½ S_a) − KE(I)) with :func:`~toporlogy.fem.elements.element_matrix_from_D`, which accepts
positive definite D only (I + ½ S_a is; the identity holds by linearity, rounding ~1e-16 relative).
The incompatible-mode elements (Q4IM, H8IM) condense their modes with Kaa(D)⁻¹: not linear in D, refused.

This class is :class:`Elasticity` in its per-element-matrix mode (``KE`` (n_active, nloc, nloc), ``ke_index`` =
identity), refilled for every new stiffness field; assembly, factorisation, support check, residual checks and
the solver are the parent's. Nothing in ``fem/elastic.py`` changes.

Study adapter
-------------
``Study`` hands ``physics.solve`` the grid its interpolation returned. A
:class:`~toporlogy.design.phases.TensorPhases` interpolation returns an informational scalar grid and publishes the
material field; :meth:`solve` takes that field (once, after checking that the grid is the field's) from the
interpolation bound with :meth:`bind`. Outside a study: :meth:`solve_stiffness`.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain, require_grid
from toporlogy.core.material_field import MaterialField, stiffness_coefficients, symmetric_basis
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import element_matrix_from_D

__all__ = ["TensorElasticity", "TENSOR_ELEMENTS", "basis_element_matrices"]

TENSOR_ELEMENTS = {2: ("Q4",), 3: ("H8",)}


def basis_element_matrices(dim: int, element: str) -> np.ndarray:
    """(n_basis, nloc, nloc) unit element matrices KE(S_a) of the symmetric basis (module notes)."""
    if element not in TENSOR_ELEMENTS.get(dim, ()):
        raise ValueError(f"tensor elasticity needs an element linear in D: {' | '.join(TENSOR_ELEMENTS[dim])} in "
                         f"{dim}-D, got {element!r} (incompatible modes are condensed with Kaa(D)^-1)")
    k = 6 if dim == 3 else 3
    eye = np.eye(k)
    K0 = element_matrix_from_D(dim, element, eye)
    return np.array([2.0 * (element_matrix_from_D(dim, element, eye + 0.5 * S) - K0) for S in symmetric_basis(k)])


class TensorElasticity(Elasticity):
    """Static linear elasticity with a per-element elasticity tensor (module notes).

    Parameters as :class:`Elasticity` without ``nu`` / ``elasticity``; ``element`` Q4 (2-D) | H8 (3-D).
    Attributes: ``KE_basis`` (n_basis, nloc, nloc); ``source`` the bound interpolation (or None); ``D`` the last
    stiffness field (n_active, k, k).
    """

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, element: str | None = None,
                 solver: Any = None, ordering: str = "nested_dissection") -> None:
        require_grid(domain, "TensorElasticity")
        el = element or ("H8" if domain.dim == 3 else "Q4")
        self.KE_basis = basis_element_matrices(domain.dim, el)
        super().__init__(domain, fixed, element=el, solver=solver, ordering=ordering)
        self.nu = None
        self.source: Any = None
        self._coef: np.ndarray | None = None

    def bind(self, source: Any) -> None:
        """Take the material field from ``source`` (``take_field(E_active)``) at every :meth:`solve`."""
        if not callable(getattr(source, "take_field", None)):
            raise TypeError("the source must publish a material field (take_field), e.g. TensorPhases")
        self.source = source

    # ------------------------------------------------------------------ the stiffness field
    def set_coefficients(self, coef: np.ndarray) -> None:
        """Element matrices from basis coefficients ``coef`` (n_active, n_basis) (physical units)."""
        c = np.asarray(coef, float)
        if c.shape != (self.n_elements, len(self.KE_basis)):
            raise ValueError(f"coefficients of shape {c.shape}, expected ({self.n_elements}, {len(self.KE_basis)})")
        self._coef = c.copy()
        self.KE = np.einsum("ea,anm->enm", c, self.KE_basis)
        self.ke_index = np.arange(self.n_elements)
        self._group_cache = None
        k = 6 if self.domain.dim == 3 else 3
        S = symmetric_basis(k)
        self.D = np.einsum("ea,aij->eij", c, S)
        self.D_distinct = self.D

    def set_stiffness(self, C: np.ndarray) -> None:
        """Element matrices of a stiffness field ``C`` (n_active, k, k) (physical units, symmetric)."""
        self.set_coefficients(stiffness_coefficients(torch.as_tensor(np.asarray(C, float))).numpy())

    # ------------------------------------------------------------------ solves
    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        """Study entry point: the bound interpolation's material field (``E_active`` is the grid it returned)."""
        if self.source is None:
            raise ValueError("TensorElasticity has no bound material field source: build it with a tensor "
                             "interpolation (TensorPhases) and call bind()")
        f: MaterialField = self.source.take_field(E_active)
        return self.solve_field(f, F)

    def solve_field(self, field: MaterialField, F: np.ndarray) -> torch.Tensor:
        """Differentiable U (ndof, n_cases) of a material field."""
        if field.n_elements != self.n_elements:
            raise ValueError(f"the material field has {field.n_elements} elements, the physics {self.n_elements}")
        return self.solve_stiffness(field.stiffness, F)

    def solve_stiffness(self, C: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        """Differentiable U (ndof, n_cases) of a stiffness field ``C`` (n_active, k, k) tensor."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        return _TensorSolve.apply(stiffness_coefficients(C), self, F2)

    def coefficient_gradient(self, U: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """(n_active, n_basis) −s Σ_c λ_e,cᵀ KE(S_a) u_e,c (module notes)."""
        U = U[:, None] if U.ndim == 1 else U
        lam = lam[:, None] if lam.ndim == 1 else lam
        Ue, Le = U[self.edof], lam[self.edof]                       # (na, nloc, nc)
        G = np.einsum("eic,ejc->eij", Le, Ue)
        return -self.geom * np.einsum("eij,aij->ea", G, self.KE_basis)


class _TensorSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, coef: torch.Tensor, model: TensorElasticity, F: np.ndarray) -> torch.Tensor:
        c = coef.detach().cpu().double().numpy()
        model.set_coefficients(c)
        U = model.solve_numpy(np.ones(model.n_elements), F)
        ctx.model, ctx.F, ctx.U, ctx.coef, ctx.token = model, F, U, c, model._token
        ctx.device, ctx.dtype = coef.device, coef.dtype
        return torch.from_numpy(U).to(device=coef.device, dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, None, None]:
        model: TensorElasticity = ctx.model
        G = gU.detach().cpu().double().numpy()
        U, F = ctx.U, ctx.F
        lam = np.zeros_like(U)
        need = []
        fr = model.free
        for c in range(U.shape[1]):
            g, f = G[fr, c], F[fr, c]
            ff = float(f @ f)
            if ff > 0:
                a = float(g @ f) / ff
                if np.linalg.norm(g - a * f) <= 1e-12 * max(float(np.linalg.norm(g)), 1e-300):
                    lam[:, c] = a * U[:, c]                     # self-adjoint: K λ = a F  →  λ = a U
                    continue
            if np.any(g):
                need.append(c)
        if need:
            if model._token != ctx.token:                      # another forward refactorised: restore ours
                model.set_coefficients(ctx.coef)
                model.factor(np.ones(model.n_elements))
            lam[:, need] = model._solve_free(G[:, need])
            model.n_adjoint_solves += 1
        dc = model.coefficient_gradient(U, lam)
        return torch.from_numpy(dc).to(device=ctx.device, dtype=ctx.dtype), None, None
