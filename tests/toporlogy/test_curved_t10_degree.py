"""Default quadrature degree of curved T10 cells (docs/foundations/prereg/F2fix_curved_t10_degree.md, FX1-FX2)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.fem.mesh import ElasticProblem
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.reference import get

from test_mesh_assembly import _D, _mesh, _patch


def test_fx1_curved_t10_patch_exact_with_default_rule() -> None:
    m = _mesh("T10", curved=0.1)
    assert kn.is_curved(m.ref, m.cell_points())
    assert ElasticProblem(m, _D(3)).degree == 3
    eu, es, _, _ = _patch(m, _D(3))
    print(f"FX1 curved T10 default rule: u {eu:.3e}, stress {es:.3e}")
    assert eu <= 1e-12 and es <= 1e-12


@pytest.mark.parametrize("name", ["T3", "T6", "Q4", "Q9", "T4", "T10", "H8"])
def test_fx2_affine_and_other_elements_unchanged(name) -> None:
    m = _mesh(name)                                   # distorted vertices, straight edges
    D = _D(get(name).dim)
    if name == "T10":
        assert not kn.is_curved(m.ref, m.cell_points())
    P = ElasticProblem(m, D)
    assert P.degree == kn.stiffness_degree(m.ref)
    Q = ElasticProblem(m, D, degree=kn.stiffness_degree(m.ref))
    assert np.array_equal(P.element_stiffness(), Q.element_stiffness())
