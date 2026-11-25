"""Stress and strain recovery output types (docs/olcumler/gerilme_onkayit.md §2-§3, pre-registered
2026-11-26 00:24 +0300 before the first run): closed forms T1, B0, I1, I2; independent checks O1, O2; F1, F2; the
convergence orders C1-C6 of pure bending. The expected Gauss point and centre stresses of O1 come from the separately
written ``_independent_fe``."""
from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import torch

import _independent_fe as ind
import _stress_cases as S
from toporlogy.core.domain import Domain
from toporlogy.core.material import Strength
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import D_plane_stress
from toporlogy.optimization.strength import expand_plane, make_criterion
from toporlogy.optimization.stress import centroid_stress
from toporlogy.optimization.stress_recovery import StressRecovery

TOL_TENSION, TOL_BENDING, TOL_INTERFACE, TOL_ORACLE = 1e-9, 1e-8, 1e-9, 1e-12


# ----------------------------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize("element", ["Q4", "Q4IM", "H8", "H8IM"])
def test_t1_uniaxial_tension_every_output_type(element):
    err = S.errors(S.tension(element), S.tension_exact, S.SIGMA0)
    print(f"\n  T1 {element}: " + ", ".join(f"{k} {v:.1e}" for k, v in err.items()))
    assert set(err) == set(S.TYPES)
    assert max(err.values()) <= TOL_TENSION


@pytest.mark.parametrize("element,n", [("Q4IM", 8), ("H8IM", 4)])
def test_b0_pure_bending_is_exact_with_incompatible_modes(element, n):
    err = S.errors(S.bending(element, n), S.bending_exact, S.SMAX)
    print(f"\n  B0 {element}: " + ", ".join(f"{k} {v:.1e}" for k, v in err.items()))
    assert max(err.values()) <= TOL_BENDING


@pytest.mark.parametrize("kind", ["nodal_averaged", "patch_recovered"])
def test_i1_i2_material_interface(kind):
    case, E = S.bimaterial()
    rec = case.recovery()
    separated = getattr(rec, kind)()
    got = separated.stress[separated.corner_entry]                      # (e, corners, 3, 1), element by element
    want = np.zeros_like(got)
    want[:, :, 0, 0] = E[:, None]
    err = float(np.abs(got - want).max() / E.max())
    assert err <= TOL_INTERFACE                                         # I1
    assert not separated.fallback.any()
    iface = np.isclose(separated.points[:, 1], 1.0)
    assert iface.sum() == 2 * (case.domain.shape[0] + 1)                # one entry per material on the interface
    merged = getattr(rec, kind)("none")
    at = merged.stress[merged.corner_entry]
    on = np.isclose(merged.points[merged.corner_entry][..., 1], 1.0)
    jump = float(np.abs(at - want)[:, :, :, 0][on].max() / E.max())
    print(f"\n  I {kind}: separated {err:.1e}, averaged across the interface {jump:.12f}")
    assert abs(jump - 3.0 / 8.0) <= TOL_INTERFACE                       # I2: |E2 - E1| / (2 E2)


# ----------------------------------------------------------------------------------------------- independent
def _rotated_orthotropic(dim: int, n: int, rng) -> np.ndarray:
    C = ind.orthotropic_3d(40.0, 9.0, 9.0, 4.0, 4.0, 3.2, 0.28, 0.28, 0.4)
    out = []
    for _ in range(n):
        if dim == 2:
            R = ind.rot_z(rng.uniform(-90.0, 90.0))
        else:
            R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
            R[:, 0] *= np.sign(np.linalg.det(R))
        D = ind.rotate(C, R)
        D = ind.plane_stress(D) if dim == 2 else D
        out.append(0.5 * (D + D.T) / 40.0)
    return np.array(out)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("material", ["isotropic", "rotated_orthotropic"])
def test_o1_gauss_and_centre_stresses_against_the_independent_implementation(dim, material):
    rng = np.random.default_rng(11)
    d = Domain((5, 4), h=0.7, thickness=0.6) if dim == 2 else Domain((4, 3, 3), h=0.7)
    assert np.array_equal(np.sort(d.edof, 1), np.sort(ind.element_dofs(d.shape), 1))
    n = d.n_active
    E = rng.uniform(1.0, 40.0, n)
    if material == "isotropic":
        D1 = ind.plane_stress_isotropic(1.0, 0.3) if dim == 2 else ind.isotropic_3d(1.0, 0.3)
        Ds = np.broadcast_to(D1, (n,) + D1.shape)
        ph = Elasticity(d, np.zeros(0, int), 0.3)
    else:
        Ds = _rotated_orthotropic(dim, n, rng)
        ph = Elasticity(d, np.zeros(0, int), elasticity=Ds)
    u = rng.standard_normal(d.ndof)
    rec = StressRecovery(ph, u, E)
    pf = rec.integration_points()
    pts, _ = ind.gauss(dim)                                             # in [0, 1]^dim
    mine = (pf.local + 1.0) / 2.0
    order = [int(np.flatnonzero(np.isclose(mine, p).all(1))[0]) for p in pts]
    want_gp = ind.gauss_stresses(d.shape, d.h, Ds, E, u)
    want_c = ind.centroid_stresses(d.shape, d.h, Ds, E, u)
    e_gp = np.abs(pf.stress[..., 0][:, order] - want_gp).max() / np.abs(want_gp).max()
    e_c = np.abs(rec.centroids().stress[:, :, 0] - want_c).max() / np.abs(want_c).max()
    print(f"\n  O1 {dim}-D {material}: Gauss {e_gp:.2e}, centre {e_c:.2e}")
    assert e_gp <= TOL_ORACLE and e_c <= TOL_ORACLE


@pytest.mark.parametrize("element", ["Q4", "Q4IM", "H8", "H8IM"])
def test_o2_gauss_point_energy_equals_the_stiffness_energy(element):
    rng = np.random.default_rng(12)
    d = Domain((5, 4), h=0.7, thickness=0.6) if element.startswith("Q4") else Domain((3, 3, 2), h=0.7)
    ph = Elasticity(d, np.zeros(0, int), 0.3, element)
    E = rng.uniform(0.2, 5.0, d.n_active)
    u = rng.standard_normal(d.ndof)
    pf = StressRecovery(ph, u, E).integration_points()
    got = float((pf.energy_density()[:, :, 0] * pf.weights).sum())
    ue = u[ph.edof][:, :, None]
    want = 0.5 * float((E[:, None] * ph.geom * ue[:, :, 0] * ph.element_forces(ue)[:, :, 0]).sum())
    print(f"\n  O2 {element}: {abs(got - want) / want:.2e}")
    assert abs(got - want) <= TOL_ORACLE * want


def test_centre_field_equals_the_constraint_module_stress():
    """The recovery and ``optimization.stress.centroid_stress`` (used by the stress constraints) are the same
    quantity."""
    case = S.l_corner(8)
    a = case.recovery().centroids().stress
    b = centroid_stress(case.domain, torch.as_tensor(case.U), S.NU, 1.0).numpy()
    assert np.abs(a - b).max() <= 1e-12 * np.abs(b).max()


def test_f1_failure_measure_von_mises_criterion():
    case = S.l_corner(8)
    rec = case.recovery()
    crit = make_criterion("von_mises", Strength(2.0), eps=0.0)
    pf = rec.integration_points()
    index = pf.measure(lambda s: crit.index(expand_plane(torch.as_tensor(s))).numpy())
    assert np.abs(index - pf.von_mises() / 2.0).max() <= TOL_ORACLE * index.max()
    top = rec.ip_maximum(lambda s: crit.index(expand_plane(torch.as_tensor(s))).numpy())
    assert np.allclose(top.value, index.max(1), rtol=1e-12)


def test_f2_fail_closed():
    d = Domain((3, 2))
    u = np.zeros(d.ndof)
    custom = Elasticity(d, np.zeros(0, int), elasticity=D_plane_stress(0.3))
    with pytest.raises(ValueError, match="plane stress"):
        StressRecovery(custom, u).centroids().von_mises()
    StressRecovery(custom, u, plane_stress=True).centroids().von_mises()
    iso = Elasticity(d, np.zeros(0, int), 0.3)
    bad = u.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        StressRecovery(iso, bad)
    with pytest.raises(ValueError, match="active box element"):
        StressRecovery(iso, u, elements=np.array([99]))
    with pytest.raises(ValueError, match="separate"):
        StressRecovery(iso, u).nodal_averaged("phase")
    with pytest.raises(ValueError, match="one value per stress vector"):
        StressRecovery(iso, u).centroids().measure(lambda s: s)


def test_one_element_thick_member_has_no_patch_and_says_so():
    d = Domain((5, 1))
    ph = Elasticity(d, np.zeros(0, int), 0.3)
    rec = StressRecovery(ph, np.random.default_rng(0).standard_normal(d.ndof))
    spr, avg = rec.patch_recovered(), rec.nodal_averaged()
    assert spr.fallback.all()
    assert np.array_equal(spr.stress, avg.stress)


def test_recovery_over_a_subset_uses_only_those_elements():
    case = S.l_corner(8)
    el = case.domain.active_elements[::2]
    full, part = case.recovery(), case.recovery(elements=el)
    assert np.array_equal(part.centroids().stress, full.centroids().stress[::2])
    assert part.nodal_averaged().corner_entry.shape == (el.size, 4)


# ----------------------------------------------------------------------------------------------- convergence
@cache
def _bending_errors(dim: int):
    element = "Q4" if dim == 2 else "H8"
    rows = [S.errors(S.bending(element, n), S.bending_exact, S.SMAX, S.REGION_R) for n in S.MESHES[dim]]
    strict = [S.errors(S.bending(element, n), S.bending_exact, S.SMAX, S.REGION_R_STRICT) for n in S.MESHES[dim]] \
        if dim == 2 else None
    return rows, strict


# (check, output type, region, (low, high) of the order in 2-D, in 3-D)
ORDERS = [("C1", "centroid", "R", (1.8, np.inf), (1.6, np.inf)),
          ("C2", "integration_point", "R", (0.8, 1.2), (0.8, 1.25)),
          ("C3", "extrapolated", "R", (0.8, 1.2), (0.8, 1.25)),
          ("C4", "nodal_averaged", "R", (0.8, 1.2), (0.8, 1.25)),
          ("C5", "nodal_averaged", "R'", (1.8, np.inf), None),
          ("C6", "patch_recovered", "R", (1.8, np.inf), (1.6, np.inf))]


ORDER_CASES = [pytest.param(dim, check, kind, region, band, id=f"{check}-{dim}D")
               for check, kind, region, band2, band3 in ORDERS
               for dim, band in ((2, band2), (3, band3)) if band is not None]       # C5 is pre-registered in 2-D only


@pytest.mark.parametrize("dim,check,kind,region,band", ORDER_CASES)
def test_pure_bending_convergence_order(dim, check, kind, region, band):
    rows, strict = _bending_errors(dim)
    e = [r[kind] for r in (rows if region == "R" else strict)]
    p = S.orders(e)
    print(f"\n  {check} {dim}-D {kind} {region}: errors " + " ".join(f"{v:.3e}" for v in e)
          + " orders " + " ".join(f"{v:.2f}" for v in p))
    assert band[0] <= p[-1] <= band[1]
