"""Stiffness (incl. incompatible modes), consistent mass, modal eigenvalues and conduction against the separately
written implementation of ``_independent_fe`` (docs/olcumler/dogrulama_turu.md §3, K1-K5, pre-registered
2026-11-25 22:20 +0300 before the first run).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import _independent_fe as ind
from toporlogy.core.domain import Domain
from toporlogy.core.material import Orthotropic, rotation_from_axes, rotations_about_z, unit_elasticity_field
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.heat import assemble_conduction
from toporlogy.fem.modal import Modal

ENG = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
MAT = Orthotropic(**ENG)
NU = 0.3
TOL = 1e-12            # pre-registered K1-K3, K5
TOL_EIG = 1e-9         # pre-registered K4


def _rel(a, b) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def _domain(dim):
    d = Domain((5, 3), h=0.7, thickness=0.6) if dim == 2 else Domain((3, 3, 2), h=0.7)
    assert np.array_equal(np.sort(d.edof, 1), np.sort(ind.element_dofs(d.shape), 1))
    X = d.node_coords()
    return d, d.dofs(np.flatnonzero(X[:, 0] < 1e-9))


def _materials(d, field, rng):
    """(library D or None, own unit D per element)."""
    dim, n = d.dim, d.n_active
    if not field:
        D1 = ind.plane_stress_isotropic(1.0, NU) if dim == 2 else ind.isotropic_3d(1.0, NU)
        return None, np.broadcast_to(D1, (n,) + D1.shape)
    if dim == 2:
        ang = rng.uniform(-90, 90, n)
        R = rotations_about_z(ang)
    else:
        R = np.array([rotation_from_axes(rng.standard_normal(3), rng.standard_normal(3)) for _ in range(n)])
    C = np.array([ind.rotate(ind.orthotropic_3d(**ENG), r) / ENG["E1"] for r in R])
    own = C if dim == 3 else np.array([ind.plane_stress(c) for c in C])
    return unit_elasticity_field(MAT, dim, "stress", R), own


def _own_K(d, E, own, incompatible):
    t = d.thickness if d.dim == 2 else 1.0
    kes = np.array([E[e] * ind.element_stiffness(d.dim, d.h, t, own[e], incompatible) for e in range(d.n_active)])
    return ind.assemble(d.shape, kes)


def _lib_full(A):
    return (A + A.T).toarray() - np.diag(A.diagonal())


CASES = [(2, "Q4", False), (3, "H8", False), (2, "Q4", True), (3, "H8", True),
         (2, "Q4IM", False), (3, "H8IM", False), (2, "Q4IM", True), (3, "H8IM", True)]
IDS = ["K1-Q4", "K1-H8", "K1-Q4-field", "K1-H8-field", "K2-Q4IM", "K2-H8IM", "K2-Q4IM-field", "K2-H8IM-field"]


@pytest.mark.parametrize("dim,element,field", CASES, ids=IDS)
def test_k1_k2_reduced_stiffness(dim, element, field):
    rng = np.random.default_rng(0)
    d, fixed = _domain(dim)
    lib_D, own = _materials(d, field, rng)
    E = 0.2 + 0.8 * rng.random(d.n_active)
    phys = Elasticity(d, fixed, nu=NU, element=element, elasticity=lib_D)
    got = _lib_full(phys.matrix(E))
    want = _own_K(d, E, own, element.endswith("IM"))[np.ix_(phys.free, phys.free)]
    print(f"\n  {element} field={field}: {_rel(got, want):.2e}")
    assert _rel(got, want) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_k3_consistent_mass(dim):
    d, fixed = _domain(dim)
    rho = 0.1 + 0.9 * np.random.default_rng(1).random(d.n_active)
    mo = Modal(d, fixed, nu=NU)
    got = mo.mass_matrix(rho).toarray()
    t = d.thickness if dim == 2 else 1.0
    want = ind.assemble(d.shape, np.array([ind.element_mass(dim, d.h, t, r) for r in rho]))[np.ix_(mo.free, mo.free)]
    print(f"\n  mass {dim}-D: {_rel(got, want):.2e}")
    assert _rel(got, want) <= TOL


@pytest.mark.parametrize("dim,element", [(2, "Q4"), (3, "H8IM")])
def test_k4_modal_eigenvalues(dim, element):
    rng = np.random.default_rng(2)
    d, fixed = _domain(dim)
    E = 0.2 + 0.8 * rng.random(d.n_active)
    rho = 0.1 + 0.9 * rng.random(d.n_active)
    mo = Modal(d, fixed, nu=NU, element=element)
    lam, _ = mo.solve_numpy(E, rho, 6)
    _, own = _materials(d, False, rng)
    t = d.thickness if dim == 2 else 1.0
    K = _own_K(d, E, own, element.endswith("IM"))[np.ix_(mo.free, mo.free)]
    M = ind.assemble(d.shape, np.array([ind.element_mass(dim, d.h, t, r) for r in rho]))[np.ix_(mo.free, mo.free)]
    want = sla.eigh(K, M, eigvals_only=True)[:6]
    err = float(np.abs(np.asarray(lam) - want).max() / want.max())
    print(f"\n  modal {element}: lambda {want}, {err:.2e}")
    assert err <= TOL_EIG


@pytest.mark.parametrize("dim", [2, 3])
def test_k5_conduction_matrix(dim):
    d, _ = _domain(dim)
    k = 0.1 + 0.9 * np.random.default_rng(3).random(d.n_active)
    got = assemble_conduction(d, k).toarray()
    t = d.thickness if dim == 2 else 1.0
    want = ind.assemble(d.shape, np.array([ind.element_conduction(dim, d.h, t, v) for v in k]))
    print(f"\n  conduction {dim}-D: {_rel(got, want):.2e}")
    assert _rel(got, want) <= TOL
