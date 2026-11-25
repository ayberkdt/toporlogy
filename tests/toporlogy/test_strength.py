"""Failure criteria (optimization/strength.py) and the FailureIndex stress field (optimization/stress.py).

References: the definitions themselves at the reference states (index = 1 at uniaxial tension σt, uniaxial
compression σc, pure shear, the Tsai-Wu strengths along the material axes); degree-1 homogeneity; invariance of the
isotropic criteria under rotation; reduction of Drucker-Prager and Raghava to von Mises when σc = σt; the existing
VonMises field; plane-strain σzz = ν(σxx + σyy) (isotropic); finite differences for the gradient (pre-set 1e-6 as
in test_stress.py). Tolerances: rounding (1e-12) for identities; 1e-6 relative for the smoothing ``eps``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.core.material import (Orthotropic, OrthotropicStrength, Strength, out_of_plane_row,
                                     rotation_about_z, rotation_from_axes, unit_elasticity)
from toporlogy.optimization.diagnostics import study_gradcheck
from toporlogy.optimization.local import Aggregated
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.strength import CRITERIA, make_criterion, rotate_stress_to_material
from toporlogy.optimization.stress import FailureIndex, VonMises

ST, SC, SF = 300.0, 450.0, 1.5


def _s(*v: float) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.float64)


def _iso(kind: str, sc: float = SC) -> object:
    return make_criterion(kind, Strength(ST, compression=sc, safety_factor=SF), eps=0.0)


@pytest.mark.parametrize("kind", ["von_mises", "tresca", "drucker_prager", "raghava"])
def test_isotropic_reference_states(kind):
    at, ac = ST / SF, SC / SF
    c = _iso(kind)
    assert abs(float(c.index(_s(at, 0, 0, 0, 0, 0))) - 1) <= 1e-12             # uniaxial tension
    if kind in ("drucker_prager", "raghava"):
        assert abs(float(c.index(_s(0, -ac, 0, 0, 0, 0))) - 1) <= 1e-12        # uniaxial compression
    sym = _iso(kind, ST)                                                        # σc = σt
    assert abs(float(sym.index(_s(0, 0, -at, 0, 0, 0))) - 1) <= 1e-12
    shear = at / np.sqrt(3) if kind != "tresca" else at / 2                     # pure shear at the limit
    assert abs(float(sym.index(_s(0, 0, 0, 0, 0, shear))) - 1) <= 1e-12
    rng = np.random.default_rng(4)
    s = torch.as_tensor(rng.standard_normal((20, 6)) * 100)
    assert torch.allclose(c.index(2.5 * s), 2.5 * c.index(s), rtol=1e-12)       # degree-1 homogeneous
    R = rotation_from_axes(rng.standard_normal(3), rng.standard_normal(3))
    assert torch.allclose(c.index(rotate_stress_to_material(s, R)), c.index(s), rtol=1e-10)   # isotropic


def test_asymmetric_criteria_reduce_to_von_mises():
    rng = np.random.default_rng(5)
    s = torch.as_tensor(rng.standard_normal((50, 6)) * 100)
    vm = _iso("von_mises", ST).index(s)
    for kind in ("drucker_prager", "raghava"):
        assert torch.allclose(_iso(kind, ST).index(s), vm, rtol=1e-12)
    # compression-strong material: hydrostatic compression lowers the index, tension raises it
    dp, ra = _iso("drucker_prager"), _iso("raghava")
    base = _s(0, 0, 0, 0, 0, 50)
    for c in (dp, ra):
        assert float(c.index(base + _s(-30, -30, -30, 0, 0, 0))) < float(c.index(base)) < \
            float(c.index(base + _s(30, 30, 30, 0, 0, 0)))


def test_tsai_wu_reference_states_and_rotation():
    st = OrthotropicStrength(Xt=1500.0, Xc=1200.0, Yt=50.0, Yc=200.0, S12=70.0, safety_factor=1.25)
    a = st.allowables()
    c = make_criterion("tsai_wu", st, eps=0.0)
    for s in (_s(a["Xt"], 0, 0, 0, 0, 0), _s(-a["Xc"], 0, 0, 0, 0, 0), _s(0, a["Yt"], 0, 0, 0, 0),
              _s(0, -a["Yc"], 0, 0, 0, 0), _s(0, 0, 0, 0, 0, a["S12"]), _s(0, 0, 0, 0, 0, -a["S12"]),
              _s(0, 0, a["Zt"], 0, 0, 0), _s(0, 0, 0, a["S23"], 0, 0)):
        assert abs(float(c.index(s)) - 1) <= 1e-12
    rng = np.random.default_rng(6)
    s = torch.as_tensor(rng.standard_normal((10, 6)) * 30)
    assert torch.allclose(c.index(3.0 * s), 3.0 * c.index(s), rtol=1e-12)
    c90 = make_criterion("tsai_wu", st, R=rotation_about_z(90.0), eps=0.0)       # fibres along global y
    assert abs(float(c90.index(_s(0, a["Xt"], 0, 0, 0, 0))) - 1) <= 1e-12
    assert abs(float(c90.index(_s(a["Yt"], 0, 0, 0, 0, 0))) - 1) <= 1e-12


def test_criterion_input_checks():
    with pytest.raises(ValueError):
        make_criterion("tsai_wu", Strength(1.0))
    with pytest.raises(ValueError):
        make_criterion("von_mises", OrthotropicStrength(1, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        make_criterion("hill", Strength(1.0))
    assert set(CRITERIA) == {"von_mises", "tresca", "drucker_prager", "raghava", "tsai_wu"}


# ================================================================================ the stress field
def _beam(dim: int, D=None):
    d = T.Domain((10, 4)) if dim == 2 else T.Domain((8, 3, 2))
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    F = d.load_vector()
    tip = np.flatnonzero(np.isclose(X[:, 0], X[:, 0].max()))
    F[dim * tip + 1, 0] = -1.0 / len(tip)
    m = T.Elasticity(d, fixed, 0.3, elasticity=D)
    return d, m, F


def _ctx(d, m, F, x=0.7, E0=20.0):
    U = torch.as_tensor(m.solve_numpy(np.full(d.n_active, E0), F))
    return EvalContext(domain=d, xphys=torch.full(d.shape, x, dtype=torch.float64),
                       E=torch.full(d.shape, E0, dtype=torch.float64), U=U)


@pytest.mark.parametrize("dim", [2, 3])
def test_failure_index_von_mises_equals_von_mises_field(dim):
    d, m, F = _beam(dim)
    ctx = _ctx(d, m, F)
    a = VonMises(d, 2.0, 0.3, E0=20.0, form="ratio", q=0.5)(ctx)
    b = FailureIndex(d, make_criterion("von_mises", Strength(3.0, safety_factor=1.5)), E0=20.0, q=0.5)(ctx)
    assert float((a - b).abs().max()) <= 1e-12 * float(a.abs().max())


def test_plane_strain_out_of_plane_stress():
    """Isotropic plane strain: σzz = ν(σxx + σyy); the field's von Mises must then equal the 3-D von Mises of
    (σxx, σyy, ν(σxx + σyy), 0, 0, τxy) computed by hand."""
    nu = 0.3
    D = unit_elasticity(Orthotropic.isotropic(1.0, nu), 2, "strain")
    row = out_of_plane_row(Orthotropic.isotropic(1.0, nu).stiffness())
    d, m, F = _beam(2, D)
    ctx = _ctx(d, m, F, x=1.0, E0=1.0)
    fi = FailureIndex(d, make_criterion("von_mises", Strength(1.0), eps=0.0), D=D, szz_row=row, E0=1.0, q=0.0)
    got = fi.solid_ratio(ctx)[:, 0]
    from toporlogy.fem.elements import b_centroid_q4
    eps = (ctx.U[:, 0][torch.as_tensor(d.edof)] @ torch.as_tensor(b_centroid_q4()).T).numpy()
    sig = eps @ D.T
    sz = nu * (sig[:, 0] + sig[:, 1])
    vm = np.sqrt(0.5 * ((sig[:, 0] - sig[:, 1]) ** 2 + (sig[:, 1] - sz) ** 2 + (sz - sig[:, 0]) ** 2)
                 + 3 * sig[:, 2] ** 2)
    assert np.max(np.abs(got.numpy() - vm)) <= 1e-12 * vm.max()


@pytest.mark.parametrize("kind", ["drucker_prager", "raghava", "tsai_wu"])
def test_failure_index_gradient(kind):
    """d(p-norm of the relaxed index)/dx by finite differences (Richardson, as test_stress.py): ≤ 1e-6."""
    d = T.Domain((10, 4))
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    F = d.load_vector()
    F[2 * np.flatnonzero(np.isclose(X[:, 0], 10) & (X[:, 1] == 0)) + 1, 0] = -1.0
    F[2 * np.flatnonzero(np.isclose(X[:, 0], 10) & (X[:, 1] == 4)), 0] = 0.3
    if kind == "tsai_wu":
        mat = Orthotropic(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
        R = rotation_about_z(30.0)
        D = unit_elasticity(mat, 2, "stress", R)
        crit = make_criterion(kind, OrthotropicStrength(Xt=15.0, Xc=12.0, Yt=0.5, Yc=2.0, S12=0.7), R=R)
        E0 = 40.0
    else:
        D, crit, E0 = None, make_criterion(kind, Strength(3.0, compression=5.0)), 1.0
    field = FailureIndex(d, crit, D=D, E0=E0, q=0.5)
    resp = Aggregated(field, "pnorm", 8.0)
    st = T.Study(d, T.Elasticity(d, fixed, 0.3, elasticity=D), F, T.standard_chain(d, 1.5, beta=2.0),
                 T.SIMP(d, E0=E0), resp, [T.Constraint(T.Volume(), 0.6)], T.MMA())
    x0 = 0.4 + 0.5 * np.random.default_rng(8).random(d.n_design)

    def fd4(h):
        r1, r2 = study_gradcheck(st, x0, "objective", n_dirs=4, h=h, seed=1), \
            study_gradcheck(st, x0, "objective", n_dirs=4, h=2 * h, seed=1)
        return max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
                   for a, b in zip(r1["directions"], r2["directions"], strict=True))
    err = fd4(1e-4)
    print(f"\n  {kind} p-norm gradient: {err:.2e}")
    assert err <= 1e-6
