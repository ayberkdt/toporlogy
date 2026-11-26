"""The flat-facet shell of :mod:`toporlogy.fem.shell` inside :class:`~toporlogy.optimization.study.Study`
(``docs/olcumler/kabuk_onkayit.md``, KS1-KS5).

As with :class:`~toporlogy.optimization.mesh_study.MeshDomain`, the facets are presented to an unchanged ``Study`` as a
1-D "grid" of ``ne`` (all active and full), so :class:`~toporlogy.design.chain.Embed` and
:class:`~toporlogy.design.chain.Passive` work on it, plus:

* :class:`ShellDomain`: the facets of a :class:`~toporlogy.fem.shell.ShellMesh`, with ``centroids`` (mean of the four
  corners) and ``cell_volume`` (facet area a b), so :class:`~toporlogy.optimization.mesh_study.MeshFilterLink` (cone
  filter on the 3-D chord distance between facet centroids, not the geodesic one) and
  :class:`~toporlogy.optimization.mesh_study.MeshVolume` (area-weighted fraction; the volume fraction for a constant
  thickness) apply to it;
* :class:`ShellPhysics`: ``solve(E, F)`` → U (6 nn, n_cases) of a :class:`~toporlogy.fem.shell.FlatShell`,
  differentiable in E, with ``state_dofs`` = 6 nn.

Grid-only links and responses refuse a ``ShellDomain`` with a ``ValueError``
(:func:`toporlogy.core.domain.require_grid`; ``docs/olcumler/ag_topoloji_onkayit.md``, RF1-RF3). YAML: ``domain:
{type: shell}`` (:mod:`toporlogy.configuration.mesh_build`, SY1-SY6 in ``docs/olcumler/kabuk_onkayit.md``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toporlogy.fem.shell import FlatShell, ShellMesh, facet_frame

__all__ = ["ShellDomain", "ShellPhysics"]


class ShellDomain:
    """The facets of ``mesh`` as the design domain of a Study (a 1-D grid of ne facets, every facet active and full).
    ``cell_volume``: facet areas (ne,); ``centroids``: facet centres (ne, 3); ``ndof`` = 6 nn."""

    def __init__(self, mesh: ShellMesh) -> None:
        n = int(mesh.ne)
        self.mesh, self.dim = mesh, 3                                   # dim: the embedding space
        self.shape: tuple[int, ...] = (n,)
        self.ne = self.n_active = self.n_design = n
        self.active = np.ones(n, bool)
        self.design = np.ones(n, bool)
        self.keep = np.zeros(n, bool)
        self.void = np.zeros(n, bool)
        self.frac = np.ones(n)
        self.active_elements = np.arange(n, dtype=np.int64)
        self.design_elements = np.arange(n, dtype=np.int64)
        self.ndof = 6 * int(mesh.nn)
        sizes = []
        for k, f in enumerate(mesh.facets):
            try:
                _, a, b = facet_frame(mesh.nodes[f])
            except ValueError as err:
                raise ValueError(f"facet {k}: {err}") from None
            sizes.append(a * b)
        self.cell_volume = np.asarray(sizes, float)
        self.centroids = np.asarray(mesh.nodes[mesh.facets].mean(axis=1), float)

    def uniform_start(self, volume_fraction: float, weighted: bool = False) -> np.ndarray:
        return np.full(self.n_design, float(min(max(volume_fraction, 0.0), 1.0)))

    def __repr__(self) -> str:
        return f"ShellDomain(facets={self.n_design}, nodes={self.mesh.nn})"


class ShellPhysics:
    """Study physics of a :class:`~toporlogy.fem.shell.FlatShell`: ``solve(E, F)`` → U (6 nn, n_cases) for facet
    moduli E (ne,), differentiable in E. ``state_dofs`` tells :class:`~toporlogy.optimization.study.Study` the length
    of a load vector (6 per node: u, v, w, θx, θy, θz); ``free`` its free DOFs and ``element`` a label (both read by
    the CLI summary)."""

    name = "shell"
    element = "flat-facet"

    def __init__(self, shell: FlatShell) -> None:
        self.shell = shell
        self.state_dofs = shell.ndof
        self.free = shell.free
        self.n_solves = 0

    def solve(self, E: torch.Tensor, F: Any) -> torch.Tensor:
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.ndim != 2 or F2.shape[0] != self.state_dofs:
            raise ValueError(f"shell loads have {F2.shape[0]} rows, the shell {self.state_dofs} state DOFs "
                             "(6 per node)")
        self.n_solves += 1
        return torch.stack([self.shell.solve(E, F2[:, c]) for c in range(F2.shape[1])], dim=1)
