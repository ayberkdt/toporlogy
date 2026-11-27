"""History-dependent small-strain solid on a general mesh (WP17 minimal slice;
``docs/foundations/prereg/H_history.md``).

One material state per quadrature point, physical time steps, Newton with the material's algorithmic tangent and
commit-after-convergence:

* the model is used as given — physical parameters, no element-modulus scaling (the design-density coupling of
  physical history models is not part of this slice);
* every Newton iterate evaluates the model on the committed state; the trial state of the converged iterate is
  committed once; a failed step (no convergence or a ``ConstitutiveFailure``) commits nothing, restores the last
  accepted displacement and raises :class:`StepRejected`, so the caller can retry with a smaller step;
* reactions are taken from the original residual f_int − f_ext at the prescribed DOFs;
* the tangent is factorised as a general matrix (a history tangent need not be symmetric).

2-D meshes are plane strain (the model's embedding). Finite strain, plane stress, dynamics and design sensitivities
are not implemented here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toporlogy.core.constitutive import (ConstitutiveFailure, Kinematics, MaterialRequest, StateStore, TimeStep)
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.quadrature import rule

__all__ = ["HistoryProblem", "StepResult", "StepRejected"]


class StepRejected(ArithmeticError):
    """A time step that did not converge; nothing was committed."""


@dataclass
class StepResult:
    u: np.ndarray                    # displacement (ndof,)
    stress: np.ndarray               # Voigt stress (ne, nq, k)
    reactions: np.ndarray            # f_int - f_ext at the prescribed DOFs
    iterations: int
    residual: float
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


class HistoryProblem:
    """Small-strain solid with a history material on ``mesh`` (module notes)."""

    def __init__(self, mesh: Mesh, model: Any, thickness: float = 1.0, plane: str = "strain",
                 degree: int | None = None) -> None:
        if getattr(model, "kinematics", None) != "small":
            raise ValueError("HistoryProblem: small-strain models only (model.kinematics == 'small')")
        if not callable(getattr(model, "evaluate", None)) or not hasattr(model, "state_fields"):
            raise ValueError("HistoryProblem: the model must implement the constitutive contract (evaluate, "
                             "state_fields, initial_state)")
        if mesh.dim == 2 and plane != "strain":
            raise ValueError("HistoryProblem: 2-D meshes are plane strain only (plane stress of a history material "
                             "needs a local condensation that is not implemented)")
        if mesh.dim == 3 and thickness != 1.0:
            raise ValueError("thickness applies to 2-D meshes only")
        self.mesh, self.model = mesh, model
        self.dim = int(mesh.dim)
        self.k = 3 if self.dim == 2 else 6
        self.plane = "3d" if self.dim == 3 else "strain"
        self.thickness = float(thickness)
        deg = kn.default_degree(mesh.ref, mesh.cell_points()) if degree is None else int(degree)
        self.geom = kn.geometry(mesh.ref, mesh.cell_points(), rule(mesh.ref.cell, deg))
        self.B = kn.strain_matrix(self.geom.dNdx)                          # (ne, nq, k, nloc)
        self.dV = self.geom.dV * (self.thickness if self.dim == 2 else 1.0)
        self.cell_dofs = mesh.cell_dofs(self.dim)
        self.ndof = mesh.n_nodes * self.dim
        self.ne, self.nq = self.dV.shape
        from toporlogy.fem.mesh.assembly import Pattern
        self.pattern = Pattern(self.cell_dofs, (self.ndof, self.ndof))
        self.store = StateStore(model, self.ne * self.nq)
        self.u = np.zeros(self.ndof)
        self.times: list[float] = []
        self.n_rejected = 0

    # ------------------------------------------------------------------ kinematics
    def _grad(self, u: np.ndarray) -> np.ndarray:
        """Symmetric small-strain tensors as displacement gradients (n_points, dim, dim) (sym part only)."""
        eps = np.einsum("eqkj,ej->eqk", self.B, u[self.cell_dofs]).reshape(-1, self.k)
        H = np.zeros((len(eps), self.dim, self.dim))
        if self.dim == 2:
            H[:, 0, 0], H[:, 1, 1] = eps[:, 0], eps[:, 1]
            H[:, 0, 1] = H[:, 1, 0] = 0.5 * eps[:, 2]
        else:
            for a in range(3):
                H[:, a, a] = eps[:, a]
            for v, (i, j) in zip((3, 4, 5), ((1, 2), (0, 2), (0, 1)), strict=True):
                H[:, i, j] = H[:, j, i] = 0.5 * eps[:, v]
        return H

    def _evaluate(self, u: np.ndarray, step: TimeStep) -> Any:
        return self.model.evaluate(Kinematics(self._grad(u), "small", self.plane), None, self.store.committed,
                                   MaterialRequest(step=step))

    def _internal(self, r: Any) -> tuple[np.ndarray, np.ndarray]:
        sig = r.stress.reshape(self.ne, self.nq, self.k)
        C = r.tangent.reshape(self.ne, self.nq, self.k, self.k)
        fe = np.einsum("eqkj,eqk,eq->ej", self.B, sig, self.dV)
        Ke = np.einsum("eqki,eqkl,eqlj,eq->eij", self.B, C, self.B, self.dV)
        f = np.zeros(self.ndof)
        np.add.at(f, self.cell_dofs, fe)
        return f, Ke

    # ------------------------------------------------------------------ step
    def step(self, step: TimeStep, f_ext: np.ndarray | None = None,
             prescribed: tuple[Sequence[int], Sequence[float]] | None = None, tol: float = 1e-10,
             max_iter: int = 25) -> StepResult:
        """One time step to t + dt with external nodal forces ``f_ext`` (ndof,) and prescribed displacements
        ``prescribed = (dofs, values)`` at t + dt. Commits the material state on convergence; raises
        :class:`StepRejected` (nothing committed, displacement restored) otherwise."""
        f = np.zeros(self.ndof) if f_ext is None else np.asarray(f_ext, float).reshape(self.ndof)
        pd = np.zeros(0, np.int64) if prescribed is None else np.asarray(prescribed[0], np.int64).reshape(-1)
        pv = np.zeros(0) if prescribed is None else np.asarray(prescribed[1], float).reshape(-1)
        if pd.shape != pv.shape or (len(pd) and (pd.min() < 0 or pd.max() >= self.ndof)):
            raise ValueError(f"prescribed dofs must be within 0..{self.ndof - 1} and match the values")
        used = np.unique(self.cell_dofs)
        free = np.setdiff1d(used, pd)
        u0 = self.u.copy()
        u = u0.copy()
        u[pd] = pv
        scale_f = max(float(np.abs(f).max()), 1e-300)
        try:
            for it in range(max_iter + 1):
                r = self._evaluate(u, step)
                fint, Ke = self._internal(r)
                res = fint - f
                rmax = float(np.abs(res[free]).max()) if len(free) else 0.0
                if rmax <= tol * max(scale_f, float(np.abs(fint).max()), 1e-300):
                    break
                if it == max_iter:
                    raise StepRejected(f"no convergence in {max_iter} Newton iterations (residual {rmax:.3e})")
                K = self.pattern.assemble(Ke).tocsr()
                du = Factorization(K[free][:, free], "general").solve(-res[free])
                u[free] += du
        except (StepRejected, ConstitutiveFailure) as e:
            self.u = u0
            self.n_rejected += 1
            raise StepRejected(f"step to t = {step.t_next}: {e}") from e
        self.store.set_trial(r.trial_state)
        self.store.commit()
        self.u = u
        self.times.append(step.t_next)
        return StepResult(u.copy(), r.stress.reshape(self.ne, self.nq, self.k).copy(), res[pd].copy(), it, rmax,
                          {k: np.asarray(v).reshape(self.ne, self.nq) for k, v in r.diagnostics.items()
                           if np.asarray(v).size == self.ne * self.nq})
