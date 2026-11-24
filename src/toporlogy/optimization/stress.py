"""Stress constraints on the regular grid, 2-D (Q4, plane stress) and 3-D (H8 / H8IM).

Stress measure
--------------
For every active element e and load case c the *solid* stress at the element centroid,

    σ_e = E0 · D(ν) · (B_c / h) · u_e ,

i.e. the stress the fully solid material would carry under the element's strain (often called microscopic stress),
and its von Mises equivalent σ_vM. ``B_c`` is the centroid strain matrix of the unit element
(:func:`toporlogy.fem.elements.b_centroid_h8`, exact for H8IM too), ``D`` the unit-modulus elasticity matrix.

Relaxation
----------
An element that vanishes (x̄ → 0) keeps a finite strain, so an unrelaxed constraint σ_vM ≤ σ_lim stays active on
void: the "singularity" problem. Three forms of the local value are available (``form``):

``"ratio"``       v = x̄^q · σ_vM / σ_lim ≥ 0                     (qp relaxation: feasible when v ≤ 1; for aggregation)
``"margin"``      g = x̄^q · σ_vM / σ_lim − 1                     (feasible when g ≤ 0)
``"polynomial"``  g = x̄^p · Λ (Λ² + 1), Λ = σ_vM / σ_lim − 1    (vanishing constraint; for the augmented Lagrangian)

With stiffness penalty p (SIMP) the qp relaxation corresponds to a stress exponent p − q applied to the solid stress;
q = 0.5 with p = 3 is the choice reported for Le et al. (2010). The polynomial form is the one reported for
PolyStress (Giraldo-Londoño & Paulino 2021): Λ³ dominates for large violations, Λ near the limit. Both statements
come from records, abstracts and citing papers; the source texts were not read (``docs/literatur_2026-11-21.md``).
The formulas themselves are implemented from first principles and verified here: constant-stress patches, a bending
stress and finite differences (``tests/toporlogy/test_stress.py``).

Use
---
* global: ``Constraint(Aggregated(VonMises(...), "pnorm", 8, adaptive=0.5), upper=1.0)`` with MMA / GCMMA;
* local: ``AugmentedLagrangian(Volume(), VonMises(..., form="polynomial"))`` as objective plus
  :class:`~toporlogy.optimization.local.MultiplierUpdate` as schedule.

Known limits (not solved here, see the literature notes): the centroid stress of a voxel model is inaccurate at
jagged boundaries and under-reported in grey interfaces; elements at point loads and re-entrant corners are singular
and should be excluded (``exclude``) or the load distributed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toporlogy.fem.elements import D_iso, D_plane_stress, b_centroid_h8, b_centroid_q4
from toporlogy.optimization.local import LocalField
from toporlogy.optimization.responses import EvalContext
from toporlogy.references import cite

__all__ = ["von_mises_matrix", "centroid_stress", "VonMises", "FORMS"]

FORMS = ("ratio", "margin", "polynomial")


def von_mises_matrix(dim: int) -> np.ndarray:
    """Symmetric V with ``σ_vM² = σᵀ V σ`` in the Voigt order of :mod:`toporlogy.fem.elements` (plane stress in
    2-D)."""
    if dim == 2:
        return np.array([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
    if dim == 3:
        V = np.zeros((6, 6))
        V[:3, :3] = -0.5
        np.fill_diagonal(V[:3, :3], 1.0)
        V[3:, 3:] = 3.0 * np.eye(3)
        return V
    raise ValueError("dim must be 2 or 3")


def _stress_operator(dim: int, nu: float, h: float, E0: float) -> np.ndarray:
    """S (n_voigt, n_local) with solid stress σ = S u_e at the centroid of an element of edge h."""
    if dim == 3:
        return E0 * D_iso(nu) @ b_centroid_h8() / h
    return E0 * D_plane_stress(nu) @ b_centroid_q4() / h


def centroid_stress(domain: Any, U: torch.Tensor, nu: float, E0: float = 1.0,
                    elements: np.ndarray | None = None) -> torch.Tensor:
    """Solid centroid stress (n_elements, n_voigt, n_cases) of the given box elements (default: active ones) for
    displacements ``U`` (ndof,) or (ndof, n_cases)."""
    el = domain.active_elements if elements is None else np.asarray(elements, np.int64)
    U2 = U[:, None] if U.dim() == 1 else U
    S = torch.as_tensor(_stress_operator(domain.dim, nu, domain.h, E0), dtype=U2.dtype, device=U2.device)
    ue = U2[torch.as_tensor(domain.edof[el], dtype=torch.long, device=U2.device)]       # (n, nloc, nc)
    return torch.einsum("vk,ekc->evc", S, ue)


@cite("le_2010", "bruggi_2008", "giraldo_londono_paulino_2021",
      note="qp-relaxed and polynomial vanishing forms; records seen, texts not read")
class VonMises(LocalField):
    """Relaxed von Mises stress of every active element and load case (module notes).

    Parameters
    ----------
    domain : Domain
    sigma_lim : float
        Stress limit (same unit as E0).
    nu : float
        Poisson's ratio (of the physics).
    E0 : float, optional
        Solid modulus; None = ``ctx.interp.E0``.
    form : {"ratio", "margin", "polynomial"}
    q : float
        Relaxation exponent of "ratio" / "margin" (0 = no relaxation).
    penal : float
        Exponent p of the "polynomial" form's x̄^p.
    exclude : bool array of ``domain.shape``, optional
        Elements left out (singular points: point loads, supports, re-entrant corners).
    cases : sequence of int, optional
        Load cases used (default all).
    eps : float
        σ_vM = sqrt(σᵀVσ + (eps·σ_lim)²) keeps the derivative finite at zero stress.

    The value has shape (n_selected_elements, n_cases). :attr:`last_solid_ratio` holds max σ_vM/σ_lim over the
    elements with x̄ > 0.5 at the last evaluation (the number a designer reads; unrelaxed).
    """

    name = "von_mises"
    needs = frozenset({"u"})

    def __init__(self, domain: Any, sigma_lim: float, nu: float = 0.3, E0: float | None = None,
                 form: str = "ratio", q: float = 0.5, penal: float = 3.0, exclude: Any = None,
                 cases: Any = None, eps: float = 1e-6) -> None:
        if not sigma_lim > 0:
            raise ValueError("sigma_lim > 0 is required")
        if form not in FORMS:
            raise ValueError(f"form: one of {FORMS}")
        if q < 0 or penal < 0:
            raise ValueError("q >= 0 and penal >= 0 are required")
        self.domain, self.sigma_lim, self.nu = domain, float(sigma_lim), float(nu)
        self.E0 = None if E0 is None else float(E0)
        self.form, self.q, self.penal, self.eps = form, float(q), float(penal), float(eps)
        sel = domain.active.copy()
        if exclude is not None:
            ex = np.asarray(exclude, bool)
            if ex.shape != tuple(domain.shape):
                raise ValueError(f"exclude has shape {ex.shape}, the domain {tuple(domain.shape)}")
            sel &= ~ex
        if not sel.any():
            raise ValueError("the stress field has no element (everything excluded)")
        self.elements = np.flatnonzero(sel.ravel())
        self.cases = None if cases is None else [int(c) for c in cases]
        self._V = von_mises_matrix(domain.dim)
        self.last_solid_ratio = float("nan")

    def _modulus(self, ctx: EvalContext) -> float:
        if self.E0 is not None:
            return self.E0
        E0 = getattr(ctx.interp, "E0", None)
        if E0 is None:
            raise ValueError("VonMises: give E0 (the interpolation has no attribute E0)")
        return float(E0)

    def solid_ratio(self, ctx: EvalContext) -> torch.Tensor:
        """σ_vM / σ_lim of the solid material, (n_elements, n_cases), unrelaxed."""
        if ctx.U is None:
            raise ValueError("VonMises needs U")
        U = ctx.U if self.cases is None else ctx.U[:, self.cases]
        s = centroid_stress(self.domain, U, self.nu, self._modulus(ctx), self.elements)
        V = torch.as_tensor(self._V, dtype=s.dtype, device=s.device)
        vm2 = torch.einsum("evc,vw,ewc->ec", s, V, s)
        return torch.sqrt(vm2 + (self.eps * self.sigma_lim) ** 2) / self.sigma_lim

    def density(self, ctx: EvalContext) -> torch.Tensor:
        idx = torch.as_tensor(self.elements, dtype=torch.long, device=ctx.xphys.device)
        return ctx.xphys.reshape(-1)[idx]

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        r = self.solid_ratio(ctx)
        x = self.density(ctx)[:, None]
        solid = x.detach() > 0.5
        self.last_solid_ratio = float((r.detach() * solid).max()) if bool(solid.any()) else 0.0
        if self.form == "polynomial":
            lam = r - 1.0
            return x ** self.penal * lam * (lam * lam + 1.0)
        v = r if self.q == 0 else x.clamp_min(0.0) ** self.q * r
        return v if self.form == "ratio" else v - 1.0

    def weights(self, ctx: EvalContext) -> torch.Tensor:
        """Solid fraction of each element, repeated per load case."""
        fr = torch.as_tensor(self.domain.frac.ravel()[self.elements], dtype=ctx.xphys.dtype)
        n_cases = (ctx.U.shape[1] if ctx.U is not None else 1) if self.cases is None else len(self.cases)
        return fr[:, None].expand(-1, n_cases)
