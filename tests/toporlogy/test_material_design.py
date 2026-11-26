"""Material design: tensor phase mixing, fibre angle design block, material field (02_MATERIAL_DESIGN §1, §3, §4).

Checks and thresholds pre-registered in docs/olcumler/malzeme_tasarimi_onkayit.md (26 Nov 2026 02:42 +0300, before
the first run); results in docs/olcumler/malzeme_tasarimi.md. Test names carry the check ids. Relative errors are
max-norm: max|a − b| / max|b| over the whole array unless the check says otherwise.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import _independent_fe as ife
from toporlogy.configuration.build import build_setup
from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material import (Orthotropic, plane_stress, rotate_stiffness, rotation_about_z,
                                     rotations_about_z, unit_elasticity_field)
from toporlogy.core.material_field import (MaterialField, PhaseMaterial, rotate_in_plane,
                                           rotate_in_plane_derivative, symmetric_basis)
from toporlogy.core.tensor import rotation_from_rotvec
from toporlogy.design.chain import Chain, DensityFilter, Embed, HeavisideProjection, Passive
from toporlogy.design.interp import SIMP
from toporlogy.design.orientation import (AngleBlock, DesignLayout, OrientedEmbed, PeriodicAngleFilter,
                                          StaleFieldError, angle_period, directional_compliance_derivative,
                                          directional_modulus, wrap_angle)
from toporlogy.design.phases import MultiPhaseSIMP, TensorPhases, phase_fractions
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import element_matrix_from_D
from toporlogy.fem.tensor_elastic import TensorElasticity, basis_element_matrices
from toporlogy.optimization.responses import Compliance, Cost, EvalContext, Mass, Volume
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

ROOT = Path(__file__).resolve().parents[2]
RULES = ("simplex", "recursive")
ORTHO = dict(E1=1.0, E2=0.2, E3=0.2, G12=0.1, G13=0.1, G23=0.07, nu12=0.3, nu13=0.3, nu23=0.35)
FIBRE = dict(E1=1.0, E2=0.1, E3=0.1, G12=0.05, G13=0.05, G23=0.04, nu12=0.3, nu13=0.3, nu23=0.3)


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / np.abs(b).max())


def phases(dim: int, R3: np.ndarray | None = None) -> list[PhaseMaterial]:
    """P1, P2, P3 of the record (P3 turned 30° about z, or by R3 in 3-D)."""
    R = rotation_about_z(30.0) if R3 is None else R3
    return [PhaseMaterial("P1", ElasticTensor.isotropic(0.3, 0.2), density=0.4, cost=1.0),
            PhaseMaterial("P2", ElasticTensor.isotropic(0.6, 0.4), density=0.7, cost=2.0),
            PhaseMaterial("P3", ElasticTensor.orthotropic(**ORTHO, R=R), density=1.0, cost=5.0)]


R3D = rotation_from_rotvec(np.array([0.3, -0.5, 0.7]))


def left_fixed(d: Domain) -> np.ndarray:
    """All DOFs of the nodes on x = 0."""
    ny = d.shape[1] + 1
    nodes = np.arange(ny) if d.dim == 2 else np.arange(ny * (d.shape[2] + 1))
    return (nodes[:, None] * d.dim + np.arange(d.dim)).ravel()


def node(d: Domain, *ijk: int) -> int:
    return int(np.ravel_multi_index(ijk, tuple(n + 1 for n in d.shape)))


def tip_load(d: Domain) -> np.ndarray:
    F = np.zeros(d.ndof)
    F[node(d, d.shape[0], d.shape[1] // 2) * d.dim + 1] = -1.0
    return F


def mandel(C: np.ndarray) -> np.ndarray:
    k = C.shape[-1]
    w = np.ones(k)
    w[(3 if k == 6 else 2):] = math.sqrt(2.0)
    return C * w[:, None] * w[None, :]


# ============================================================================================== K: mixing
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rule", RULES)
def test_k1_pure_phase_recovery(rule: str, dim: int) -> None:
    d = Domain((2, 2) if dim == 2 else (2, 2, 2))
    tp = TensorPhases(d, phases(dim, R3D if dim == 3 else None), rule=rule)
    verts = {"void": [0, 0, 0], "P1": [1, 0, 0], "P2": [0, 1, 0], "P3": [0, 0, 1]}
    t = torch.as_tensor(np.array(list(verts.values()), float).T)              # (3, 4)
    C = tp.stiffness_of(t).numpy()
    Ds = tp.D.numpy()
    void = 1e-6 * (Ds.mean(0) if rule == "simplex" else Ds[-1])
    assert rel(C[0], void) <= 1e-14
    for i in range(3):
        assert rel(C[i + 1], Ds[i]) <= 1e-14


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rule", RULES)
def test_k2_positive_definite(rule: str, dim: int) -> None:
    d = Domain((2, 2) if dim == 2 else (2, 2, 2))
    tp = TensorPhases(d, phases(dim, R3D if dim == 3 else None), rule=rule)
    t = torch.as_tensor(np.random.default_rng(0).uniform(0, 1, (3, 2000)))
    C = tp.stiffness_of(t).numpy()
    Ds = tp.D.numpy()
    void = 1e-6 * (Ds.mean(0) if rule == "simplex" else Ds[-1])
    bound = min(np.linalg.eigvalsh(mandel(A)).min() for A in [void, *Ds])
    lam = np.linalg.eigvalsh(mandel(C)).min()
    assert lam >= (1 - 1e-12) * bound


@pytest.mark.parametrize("rule", RULES)
def test_k3a_identical_phases_p1_single_material(rule: str) -> None:
    d = Domain((2, 2))
    P = phases(2)[2]
    tp = TensorPhases(d, [PhaseMaterial(n, P.tensor) for n in "abc"], rule=rule, penal=1.0)
    t = torch.as_tensor(np.random.default_rng(3).uniform(0, 1, (3, 500)))
    C = tp.stiffness_of(t).numpy()
    D = tp.D[0].numpy()
    s = 1.0 - phase_fractions(t)[0].numpy()
    void = 1e-6 * D
    assert rel(C, void[None] + s[:, None, None] * (D - void)[None]) <= 1e-14


def test_k3b_recursive_scalar_multiples_equal_multiphase_simp() -> None:
    d = Domain((50, 40))
    Es = (0.3, 0.6, 1.0)
    tp = TensorPhases(d, [PhaseMaterial(f"p{i}", ElasticTensor.isotropic(E, 0.3)) for i, E in enumerate(Es)],
                      rule="recursive", penal=3.0)
    mp = MultiPhaseSIMP(d, [{"name": f"p{i}", "E": E} for i, E in enumerate(Es)], penal=3.0)
    t = torch.as_tensor(np.random.default_rng(4).uniform(0, 1, (3, 50, 40)))
    C = tp.stiffness_of(t.reshape(3, -1)).numpy()
    D = tp.D[2].numpy()                                         # E = 1: the unit plane-stress D(0.3)
    E = mp(t).reshape(-1).numpy()
    assert rel(C, E[:, None, None] * D[None]) <= 1e-14


def _dirichlet(m: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(m + 1), n).T          # (m + 1, n), [void, phases]


def test_k4_permutation_simplex() -> None:
    d = Domain((2, 2))
    ph = phases(2)
    perm = [2, 0, 1]
    a = TensorPhases(d, ph, rule="simplex")
    b = TensorPhases(d, [ph[i] for i in perm], rule="simplex")
    eta = torch.as_tensor(_dirichlet(3, 500, 5))
    eta_b = eta[[0] + [1 + i for i in perm]]
    assert rel(b.stiffness_from_fractions(eta_b).numpy(), a.stiffness_from_fractions(eta).numpy()) <= 1e-14


def test_k5_simplex_conservation_through_the_chain() -> None:
    d = Domain((16, 10))
    chain = Chain([Embed(d, 3), DensityFilter(d, 2.5), HeavisideProjection(8.0), Passive(d)])
    x = torch.as_tensor(np.random.default_rng(6).uniform(0, 1, 3 * d.n_design))
    t = chain(x)
    eta = phase_fractions(t)
    assert float(eta.min()) >= -1e-15
    assert float((eta.sum(0) - 1).abs().max()) <= 1e-14
    assert float(t.min()) >= -1e-15 and float(t.max()) <= 1 + 1e-15


# ---------------------------------------------------------------------------------------------- K6 bars
def _bar(layers: str, EA: float, nuA: float, EB: float, nuB: float, rule: str) -> tuple[Any, ...]:
    d = Domain((8, 4))
    A = PhaseMaterial("A", ElasticTensor.isotropic(EA, nuA))
    B = PhaseMaterial("B", ElasticTensor.isotropic(EB, nuB))
    tp = TensorPhases(d, [A, B], rule=rule)
    ii, jj = np.indices(d.shape)
    inA = (jj < 2) if layers == "parallel" else (ii < 4)
    t = torch.as_tensor(np.stack([inA, ~inA]).astype(float))
    fixed = np.concatenate([left_fixed(d)[0::2], [node(d, 0, 0) * 2 + 1]])
    te = TensorElasticity(d, fixed)
    return d, tp, t, te, inA


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("case", ["K6a", "K6b"])
def test_k6_series_parallel_bar(case: str, rule: str) -> None:
    """Parallel bar (K6a ν = 0, K6b ν = 0.3): nodal forces of the uniform-strain solution, written from the layer
    stresses; series bar: test_k6_series_bar."""
    nu = 0.0 if case == "K6a" else 0.3
    EA, EB = 1.0, 0.25
    d, tp, t, te, inA = _bar("parallel", EA, nu, EB, nu, rule)
    eps, L, H = 1e-3, 8.0, 4.0
    F = np.zeros(d.ndof)
    for j in range(d.shape[1]):                                   # right-edge element edges, h/2 to each node
        sig = (EA if inA[-1, j] else EB) * eps
        for jj in (j, j + 1):
            F[node(d, 8, jj) * 2] += sig * 0.5
    U = te.solve_field(tp.material_field(t), F).detach().numpy()[:, 0]
    X = np.indices((9, 5)).reshape(2, -1).T.astype(float)
    ustar = np.stack([eps * X[:, 0], -nu * eps * X[:, 1]], 1).ravel()
    assert rel(U, ustar) <= 1e-12
    c = float(F @ U)
    E_voigt = 0.5 * EA + 0.5 * EB
    assert abs(c / (eps ** 2 * L * H) / E_voigt - 1) <= 1e-12


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("case", ["K6c", "K6d"])
def test_k6_series_bar(case: str, rule: str) -> None:
    EA, nuA, EB, nuB = (1.0, 0.0, 0.25, 0.0) if case == "K6c" else (1.0, 0.4, 0.25, 0.1)
    d, tp, t, te, _ = _bar("series", EA, nuA, EB, nuB, rule)
    sig, L, H = 1.0, 8.0, 4.0
    F = np.zeros(d.ndof)
    for j in range(d.shape[1]):
        for jj in (j, j + 1):
            F[node(d, 8, jj) * 2] += sig * 0.5
    U = te.solve_field(tp.material_field(t), F).detach().numpy()[:, 0]
    c = float(F @ U)
    E_reuss = L / (4.0 / EA + 4.0 / EB)
    assert abs(sig ** 2 * L * H / c / E_reuss - 1) <= 1e-12


# ---------------------------------------------------------------------------------------------- K7
def _independent_D(dim: int, R3: np.ndarray) -> list[np.ndarray]:
    o = ife.orthotropic_3d(*[ORTHO[k] for k in ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")])
    if dim == 2:
        return [ife.plane_stress_isotropic(0.3, 0.2), ife.plane_stress_isotropic(0.6, 0.4),
                ife.plane_stress(ife.rotate(o, ife.rot_z(30.0)))]
    return [ife.isotropic_3d(0.3, 0.2), ife.isotropic_3d(0.6, 0.4), ife.rotate(o, R3)]


def _independent_mix(rule: str, Ds: list[np.ndarray], t: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    out = []
    for e in range(t.shape[1]):
        t1, t2, t3 = t[:, e]
        if rule == "simplex":
            eta = [t1 * (1 - t2) * (1 - t3), t2 * (1 - t3), t3]
            w = [v ** p for v in eta]
            C = (1 - sum(w)) * eps * (Ds[0] + Ds[1] + Ds[2]) / 3 + sum(wi * Di for wi, Di in zip(w, Ds, strict=True))
        else:
            C = eps * Ds[2]
            for tj, Dj in zip((t1, t2, t3), Ds, strict=True):
                C = C + tj ** p * (Dj - C)
        out.append(C)
    return np.array(out)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rule", RULES)
def test_k7_independent_assembly(rule: str, dim: int) -> None:
    shape, h, th = ((6, 4), 0.5, 0.7) if dim == 2 else ((3, 2, 2), 0.5, 1.0)
    d = Domain(shape, h=h, thickness=th)
    tp = TensorPhases(d, phases(dim, R3D if dim == 3 else None), rule=rule)
    tn = np.random.default_rng(1).uniform(0.05, 1.0, (3, d.n_active))
    C = tp.stiffness_of(torch.as_tensor(tn)).numpy()
    Cind = _independent_mix(rule, _independent_D(dim, R3D), tn)
    te = TensorElasticity(d, left_fixed(d))
    te.set_stiffness(C)
    A = te.matrix(np.ones(te.n_elements)).toarray()
    Klib = A + A.T - np.diag(np.diag(A))
    Kfull = ife.stiffness(shape, h, th, Cind, np.ones(d.n_active))
    Kind = Kfull[np.ix_(te.free, te.free)]
    assert rel(Klib, Kind) <= 1e-13
    F = np.zeros(d.ndof)
    F[-1] = -1.0
    F[-2] = 0.3
    U = te.solve_stiffness(torch.as_tensor(C), F).numpy()[:, 0]
    Uind = np.zeros(d.ndof)
    Uind[te.free] = np.linalg.solve(Kind, F[te.free])
    assert rel(U, Uind) <= 1e-10


# ---------------------------------------------------------------------------------------------- K8
def test_k8a_frac_applied_once() -> None:
    frac = np.ones((4, 3))
    frac[1, 2] = 0.5
    d = Domain((4, 3), frac=frac)
    tp = TensorPhases(d, phases(2))
    t = torch.as_tensor(np.random.default_rng(7).uniform(0, 1, (3, 4, 3)))
    f = tp.material_field(t)
    e = int(np.flatnonzero(d.active_elements == np.ravel_multi_index((1, 2), (4, 3)))[0])
    ref = tp.stiffness_of(t.reshape(3, -1)[:, d.active_elements])[e]
    assert rel(f.stiffness[e].numpy(), 0.5 * ref.numpy()) <= 1e-15


def _compliance_and_grad(physics: Any, interp: Any, t: torch.Tensor, F: np.ndarray,
                         tensor: bool) -> tuple[float, np.ndarray, np.ndarray]:
    tt = t.clone().requires_grad_(True)
    E = interp(tt)
    U = physics.solve(E.reshape(-1)[torch.as_tensor(physics.elements)], F)
    c = (torch.as_tensor(F)[:, None] * U).sum()
    (g,) = torch.autograd.grad(c, tt)
    return float(c), U.detach().numpy(), g.numpy()


def test_k8b_scalar_limit_matches_multiphase_simp() -> None:
    d = Domain((24, 12))
    F = tip_load(d)
    Es = (0.3, 0.6, 1.0)
    t = torch.as_tensor(np.random.default_rng(8).uniform(0.2, 1.0, (3, 24, 12)))
    mp = MultiPhaseSIMP(d, [{"name": f"p{i}", "E": E} for i, E in enumerate(Es)])
    c1, U1, g1 = _compliance_and_grad(Elasticity(d, left_fixed(d)), mp, t, F, False)
    tp = TensorPhases(d, [PhaseMaterial(f"p{i}", ElasticTensor.isotropic(E, 0.3)) for i, E in enumerate(Es)],
                      rule="recursive")
    te = TensorElasticity(d, left_fixed(d))
    te.bind(tp)
    c2, U2, g2 = _compliance_and_grad(te, tp, t, F, True)
    assert abs(c2 / c1 - 1) <= 1e-10
    assert rel(U2, U1) <= 1e-9
    assert rel(g2, g1) <= 1e-8


# ============================================================================================== A: angle
P3_BASE = ElasticTensor.orthotropic(**ORTHO)


def test_a1_rotation_matches_core_and_independent() -> None:
    C6 = P3_BASE.stiffness()
    D = plane_stress(C6)
    o = ife.orthotropic_3d(*[ORTHO[k] for k in ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")])
    for th in np.linspace(-math.pi, math.pi, 13):
        mine = rotate_in_plane(D, th).numpy()
        assert rel(mine, plane_stress(rotate_stiffness(C6, rotation_about_z(math.degrees(th))))) <= 1e-13
        assert rel(mine, ife.plane_stress(ife.rotate(o, ife.rot_z(math.degrees(th))))) <= 1e-13


def test_a2_periodicity() -> None:
    A = np.random.default_rng(2).normal(size=(6, 6))
    Ctri = A @ A.T + 6 * np.eye(6)
    Dtri = plane_stress(Ctri)
    th = torch.as_tensor(np.linspace(-3, 3, 11))
    assert rel(rotate_in_plane(Dtri, th + math.pi).numpy(), rotate_in_plane(Dtri, th).numpy()) <= 1e-14
    Dcub = plane_stress(ElasticTensor.cubic(1.0, 0.4, 0.2).stiffness())
    assert rel(rotate_in_plane(Dcub, th + math.pi / 2).numpy(), rotate_in_plane(Dcub, th).numpy()) <= 1e-14
    D3 = plane_stress(P3_BASE.stiffness())
    t10 = math.radians(10.0)
    diff = (rotate_in_plane(D3, t10 + math.pi / 2) - rotate_in_plane(D3, t10)).abs().max() / np.abs(D3).max()
    assert float(diff) >= 1e-3
    assert angle_period([D3]) == math.pi and angle_period([Dtri]) == math.pi
    assert angle_period([Dcub]) == math.pi / 2
    with pytest.raises(ValueError, match="isotropic"):
        angle_period([plane_stress(ElasticTensor.isotropic(1.0, 0.3).stiffness())])


@pytest.mark.xfail(strict=True, reason="A2 (6 x 6 part) FAIL, measured 0.206: a turn by pi about z is "
                   "diag(-1, -1, 1), not -I; the 3-D tensor of a triclinic material is not pi-periodic about z (the "
                   "pre-registration claimed it was). The reduced 3 x 3 matrix is (test_a2_periodicity).")
def test_a2_full_tensor_pi_periodicity() -> None:
    A = np.random.default_rng(2).normal(size=(6, 6))
    Ctri = A @ A.T + 6 * np.eye(6)
    for a in np.linspace(-170, 170, 11):
        assert rel(rotate_stiffness(Ctri, rotation_about_z(a + 180.0)), rotate_stiffness(Ctri, rotation_about_z(a))) \
            <= 1e-14


def test_a2_posthoc_full_tensor_period_needs_a_two_fold_z_axis() -> None:
    """POST HOC (after A2 failed): the 3-D tensor is pi-periodic about z when z is a two-fold axis (orthotropic
    with axis 3 = z), and the components with an odd number of z indices are those that change sign."""
    C = ElasticTensor.orthotropic(**ORTHO).stiffness()
    for a in np.linspace(-170, 170, 11):
        assert rel(rotate_stiffness(C, rotation_about_z(a + 180.0)), rotate_stiffness(C, rotation_about_z(a))) <= 1e-14
    A = np.random.default_rng(2).normal(size=(6, 6))
    Ctri = A @ A.T + 6 * np.eye(6)
    flip = rotate_stiffness(Ctri, rotation_about_z(180.0))
    odd = np.array([[sum(int(i in (3, 4)) for i in (a, b)) % 2 for b in range(6)] for a in range(6)], bool)
    assert rel(flip[~odd], Ctri[~odd]) <= 1e-14 and rel(flip[odd], -Ctri[odd]) <= 1e-14


def test_a3_rotation_derivative() -> None:
    D = plane_stress(P3_BASE.stiffness())
    for th0 in np.linspace(-2.5, 2.5, 7):
        th = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
        R = rotate_in_plane(D, th)
        jac = torch.stack([torch.autograd.grad(R[i, j], th, retain_graph=True)[0] for i in range(3)
                           for j in range(3)]).reshape(3, 3).numpy()
        ana = rotate_in_plane_derivative(D, th0).numpy()
        assert rel(jac, ana) <= 1e-13
        h = 1e-6
        fd = ((rotate_in_plane(D, th0 + h) - rotate_in_plane(D, th0 - h)) / (2 * h)).numpy()
        assert rel(fd, ana) <= 1e-8


def test_a4_filter_wrap_around() -> None:
    d = Domain((10, 10))
    ii, jj = np.indices(d.shape)
    th = np.where((ii + jj) % 2 == 0, math.radians(85.0), math.radians(-85.0))
    f = PeriodicAngleFilter(d, 2.5, math.pi)
    out = f(torch.as_tensor(th)).numpy()
    dev = np.abs(wrap_angle(out - math.pi / 2, math.pi))
    assert dev.max() <= math.radians(5.0) + 1e-12
    for a in (-80.0, 0.0, 45.0, 89.9):
        u = f(torch.full(d.shape, math.radians(a), dtype=torch.float64)).numpy()
        assert np.abs(wrap_angle(u - math.radians(a), math.pi)).max() <= 1e-12


def _plate_compliance(theta: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Off-axis tension of the 8 x 4 plate (A5): compliance for a uniform angle and σ² V."""
    d = Domain((8, 4))
    tp = TensorPhases(d, [PhaseMaterial("P3", P3_BASE)])
    fixed = np.concatenate([left_fixed(d)[0::2], [node(d, 0, 0) * 2 + 1]])
    te = TensorElasticity(d, fixed)
    F = np.zeros(d.ndof)
    for j in range(4):
        for jj in (j, j + 1):
            F[node(d, 8, jj) * 2] += 0.5
    f = tp.material_field(torch.ones(d.shape, dtype=torch.float64), theta * torch.ones(d.shape, dtype=torch.float64))
    U = te.solve_field(f, F)
    return (torch.as_tensor(F)[:, None] * U).sum(), 1.0 ** 2 * 8.0 * 4.0


def test_a5_off_axis_tension_and_gradient() -> None:
    for a in range(0, 91, 15):
        th = torch.tensor(math.radians(a), dtype=torch.float64, requires_grad=True)
        c, s2v = _plate_compliance(th)
        Ex = directional_modulus(1.0, 0.2, 0.1, 0.3, math.radians(a))
        assert abs(s2v / float(c) / Ex - 1) <= 1e-12
        (g,) = torch.autograd.grad(c, th)
        ref = s2v * directional_compliance_derivative(1.0, 0.2, 0.1, 0.3, math.radians(a))
        if a in (0, 90):
            assert abs(float(g)) <= 1e-10 * float(c)
        else:
            assert abs(float(g) / ref - 1) <= 1e-8


def _cantilever_angle_study(location: str, rmin: float | None, start_deg: float,
                            max_iter: int) -> tuple[Study, OrientedEmbed]:
    d = Domain((40, 20))
    blk = AngleBlock(location, start=math.radians(start_deg), rmin=rmin)
    emb = OrientedEmbed(d, blk, density=False)
    tp = TensorPhases(d, [PhaseMaterial("fibre", ElasticTensor.orthotropic(**FIBRE))])
    tp.attach_angle(emb)
    te = TensorElasticity(d, left_fixed(d))
    te.bind(tp)
    st = Study(d, te, tip_load(d), Chain([emb]), tp, Compliance(), [Constraint(Volume(), 2.0)], MMA(move=0.1))
    return st, emb


def sweep_cantilever() -> tuple[np.ndarray, np.ndarray]:
    """A6 reference: compliance of the uniform angle θ ∈ [−90°, 90°) in 1° steps."""
    st, emb = _cantilever_angle_study("global", None, 0.0, 1)
    angles = np.arange(-90.0, 90.0, 1.0)
    cs = []
    for a in angles:
        x = np.array([emb.angle_block.s_of(math.radians(a))])
        cs.append(st.evaluate(x, gradients=False).f0)
    return angles, np.array(cs)


def _local_minima(c: np.ndarray) -> int:
    return int(np.sum((c < np.roll(c, 1)) & (c < np.roll(c, -1))))


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="A6 FAIL: the angle criterion holds for every start (error <= 1 deg), the "
                   "compliance criterion does not: MMA oscillates +-1.6 deg about the optimum 0 deg with a constant "
                   "step (change 9e-3), c_opt / c_sweep_min = 1.00084 (start 0), 1.00029 (starts +-45) > 1 + 1e-9")
def test_a6_global_angle_matches_the_sweep() -> None:
    angles, cs = sweep_cantilever()
    k = int(np.argmin(cs))
    for start in (0.0, 45.0, -45.0):
        st, emb = _cantilever_angle_study("global", None, start, 30)
        res = st.run(x0=emb.reduce(np.zeros(0)), max_iter=30, tol=1e-9)
        lo, hi = emb.angle_block.bounds
        th = math.degrees(lo + (hi - lo) * float(res.x[0]))
        assert abs(float(wrap_angle(th - angles[k], 180.0))) <= 1.0
        assert res.objective <= cs[k] * (1 + 1e-9)


@pytest.mark.slow
def test_a6_posthoc_gcmma_reaches_the_sweep_minimum() -> None:
    """Post hoc (after A6 failed): the plain-MMA 2-cycle of an interior optimum with no active constraint is the
    documented MMA limitation (``optimizers/mma.py``); GCMMA on the same problem meets both A6 criteria."""
    from toporlogy.optimizers import GCMMA
    angles, cs = sweep_cantilever()
    k = int(np.argmin(cs))
    for start in (0.0, 45.0, -45.0):
        st, emb = _cantilever_angle_study("global", None, start, 30)
        st.optimizer = GCMMA(move=0.1)
        res = st.run(x0=emb.reduce(np.zeros(0)), max_iter=30, tol=1e-9)
        lo, hi = emb.angle_block.bounds
        th = math.degrees(lo + (hi - lo) * float(res.x[0]))
        assert abs(float(wrap_angle(th - angles[k], 180.0))) <= 1.0
        assert res.objective <= cs[k] * (1 + 1e-9)


@pytest.mark.slow
def test_a7_element_angles_improve_on_the_best_uniform_angle() -> None:
    angles, cs = sweep_cantilever()
    k = int(np.argmin(cs))
    st, emb = _cantilever_angle_study("element", 2.0, float(angles[k]), 40)
    res = st.run(x0=emb.reduce(np.zeros(0)), max_iter=40, tol=1e-9)
    assert res.objective <= cs[k]


def test_a8_yaml_example_runs() -> None:
    raw = yaml.safe_load((ROOT / "examples" / "research" / "fiber_angle_cantilever2d.yaml").read_text(encoding="utf-8"))
    setup = build_setup(raw, overrides=["run.max_iter=30"])
    res = setup.study.run(max_iter=30)
    assert res.objective < res.history["f0"][0]
    assert res.constraint_values["volume"] <= 0.4 * (1 + 1e-3)


# ============================================================================================== G: gradients
def _taylor(f: Any, x: np.ndarray, g: np.ndarray, dvec: np.ndarray) -> float:
    hs = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2])
    f0 = f(x)
    r = [abs(f(x + h * dvec) - f0 - h * float(g @ dvec)) for h in hs]
    return float(np.polyfit(np.log(hs), np.log(r), 1)[0])


def _fd(f: Any, x: np.ndarray, dvec: np.ndarray, h: float = 1e-6) -> float:
    return (f(x + h * dvec) - f(x - h * dvec)) / (2 * h)


def _phase_problem(rule: str, dim: int) -> tuple[Any, int]:
    shape = (12, 6) if dim == 2 else (4, 3, 3)
    d = Domain(shape)
    chain = Chain([Embed(d, 3), DensityFilter(d, 1.5), Passive(d)])
    tp = TensorPhases(d, phases(dim, R3D if dim == 3 else None), rule=rule)
    te = TensorElasticity(d, left_fixed(d))
    F = np.zeros(d.ndof)
    F[-1] = -1.0

    def c(x: Any, grad: bool = False) -> Any:
        xt = torch.as_tensor(np.asarray(x, float)).requires_grad_(grad)
        U = te.solve_field(tp.material_field(chain(xt)), F)
        cc = (torch.as_tensor(F)[:, None] * U).sum()
        if grad:
            return float(cc), torch.autograd.grad(cc, xt)[0].numpy()
        return float(cc)
    return c, 3 * d.n_design


def _angle_problem(density: bool) -> tuple[Any, int, OrientedEmbed]:
    d = Domain((12, 6))
    emb = OrientedEmbed(d, AngleBlock("element", rmin=1.5), density=density)
    tp = TensorPhases(d, [PhaseMaterial("P3", P3_BASE)])
    tp.attach_angle(emb)
    rest = Chain([DensityFilter(d, 1.5), Passive(d)])
    te = TensorElasticity(d, left_fixed(d))
    F = tip_load(d)

    def c(x: Any, grad: bool = False) -> Any:
        xt = torch.as_tensor(np.asarray(x, float)).requires_grad_(grad)
        rho = rest(emb(xt))
        U = te.solve_field(tp.material_field(rho, emb.take()), F)
        cc = (torch.as_tensor(F)[:, None] * U).sum()
        if grad:
            return float(cc), torch.autograd.grad(cc, xt)[0].numpy()
        return float(cc)
    return c, emb.n_in, emb


G1_CASES = ["a-simplex", "a-recursive", "b-angle", "c-density+angle", "d-3d"]


@pytest.mark.parametrize("case", G1_CASES)
def test_g1_compliance_gradient_finite_difference(case: str) -> None:
    rng = np.random.default_rng(11)
    if case.startswith("a"):
        c, n = _phase_problem(case.split("-")[1], 2)
        x = rng.uniform(0.2, 0.9, n)
    elif case == "d-3d":
        c, n = _phase_problem("simplex", 3)
        x = rng.uniform(0.2, 0.9, n)
    else:
        c, n, emb = _angle_problem(case.startswith("c"))
        x = rng.uniform(0.2, 0.9, n)
    dvec = rng.normal(size=n)
    if case == "b-angle":
        dvec[: emb.layout["angle"].offset] = 0.0
    _, g = c(x, grad=True)
    fd = _fd(c, x, dvec)
    assert abs(fd - float(g @ dvec)) <= 1e-6 * abs(fd)


def _g_case(case: str) -> tuple[Any, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    if case.startswith("a"):
        c, n = _phase_problem(case.split("-")[1], 2)
        x = rng.uniform(0.2, 0.9, n)
        dvec = rng.normal(size=n)
    else:
        c, n, emb = _angle_problem(case.startswith("c"))
        x = rng.uniform(0.2, 0.9, n)
        dvec = rng.normal(size=n)
        if case == "b-angle":
            dvec[: emb.layout["angle"].offset] = 0.0
    return c, x, dvec


_G2_FAIL = "G2 FAIL: slope over the pre-registered h in [1.25e-2, 1e-1] is {} (< 1.9; the steps are outside the " \
           "asymptotic range, the slope is 2.00 for h <= 1e-3: test_g2_posthoc_small_steps)"


@pytest.mark.parametrize("case", [
    "a-simplex",
    pytest.param("a-recursive", marks=pytest.mark.xfail(strict=True, reason=_G2_FAIL.format("1.31"))),
    "b-angle",
    pytest.param("c-density+angle", marks=pytest.mark.xfail(strict=True, reason=_G2_FAIL.format("1.55")))])
def test_g2_taylor_remainder(case: str) -> None:
    c, x, dvec = _g_case(case)
    _, g = c(x, grad=True)
    assert _taylor(c, x, g, dvec) >= 1.9


@pytest.mark.parametrize("case", ["a-simplex", "a-recursive", "b-angle", "c-density+angle"])
def test_g2_posthoc_small_steps(case: str) -> None:
    """POST HOC (after G2 failed for two cases): the same remainder with h in {1e-3, 5e-4, 2.5e-4, 1.25e-4}."""
    c, x, dvec = _g_case(case)
    _, g = c(x, grad=True)
    hs = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])
    f0 = c(x)
    r = [abs(c(x + h * dvec) - f0 - h * float(g @ dvec)) for h in hs]
    assert float(np.polyfit(np.log(hs), np.log(r), 1)[0]) >= 1.9


def test_g3_zero_derivatives() -> None:
    d = Domain((6, 4))
    keep = np.zeros(d.shape, bool)
    keep[-1, :] = True
    d = Domain((6, 4), keep=keep)
    ph = [PhaseMaterial("P1", ElasticTensor.isotropic(0.3, 0.2)), PhaseMaterial("P3", P3_BASE)]
    tp = TensorPhases(d, ph, rule="simplex")
    te = TensorElasticity(d, left_fixed(d))
    F = tip_load(d)
    tn = np.random.default_rng(12).uniform(0.2, 0.9, (2, 6, 4))
    pure = np.zeros(d.shape, bool)
    pure[1:3, :] = True                                     # P1-pure elements: t1 = 1, t2 = 0
    tn[0][pure], tn[1][pure] = 1.0, 0.0
    th = torch.as_tensor(np.random.default_rng(13).uniform(-1, 1, d.shape)).requires_grad_(True)
    U = te.solve_field(tp.material_field(torch.as_tensor(tn), th), F)
    (g,) = torch.autograd.grad((torch.as_tensor(F)[:, None] * U).sum(), th)
    g = g.numpy()
    assert np.abs(g[pure]).max() <= 1e-13 * np.abs(g).max()
    emb = OrientedEmbed(d, AngleBlock("element", rmin=2.0, period=math.pi))
    x = torch.as_tensor(np.random.default_rng(14).uniform(0, 1, emb.n_in)).requires_grad_(True)
    emb(x)
    (gk,) = torch.autograd.grad(emb.take()[torch.as_tensor(keep)].sum(), x, allow_unused=True)
    assert gk is None or float(gk.abs().max()) == 0.0


@pytest.mark.parametrize("dim", [2, 3])
def test_g4_basis_linearity(dim: int) -> None:
    el = "Q4" if dim == 2 else "H8"
    KB = basis_element_matrices(dim, el)
    A = np.random.default_rng(15).normal(size=(6, 6))
    for C6 in (P3_BASE.stiffness(), A @ A.T + 6 * np.eye(6)):
        C = plane_stress(C6) if dim == 2 else C6
        k = C.shape[0]
        S = symmetric_basis(k)
        coef = np.array([C[np.nonzero(Sa)][0] for Sa in S])
        assert rel(np.einsum("a,anm->nm", coef, KB), element_matrix_from_D(dim, el, C)) <= 1e-14


def test_g5_mass_and_cost_gradients() -> None:
    d = Domain((5, 4), h=0.5, thickness=0.3)
    tp = TensorPhases(d, phases(2))
    rng = np.random.default_rng(16)
    t0 = rng.uniform(0.1, 0.9, (3, 5, 4))
    dvec = rng.normal(size=t0.shape)
    for resp in (Mass(), Cost()):
        def f(t: np.ndarray, r: Any = resp) -> float:
            tt = torch.as_tensor(t)
            return float(r(EvalContext(d, tt, tt[0], interp=tp)))
        tt = torch.as_tensor(t0).requires_grad_(True)
        (g,) = torch.autograd.grad(resp(EvalContext(d, tt, tt[0], interp=tp)), tt)
        fd = _fd(f, t0, dvec)
        assert abs(fd - float((g.numpy() * dvec).sum())) <= 1e-8 * abs(fd)


# ============================================================================================== E: old path
MBB = {"name": "mbb_small", "domain": {"type": "box", "shape": [30, 10], "h": 1.0}, "material": {"E0": 1.0, "nu": 0.3},
       "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}, "dofs": "x"},
                    {"type": "fixed", "where": {"type": "point", "at": ["max", "min"]}, "dofs": "y"}],
       "load_cases": [{"name": "c", "loads": [{"type": "force", "where": {"type": "point", "at": ["min", "max"]},
                                               "force": [0, -1]}]}],
       "chain": [{"type": "density_filter", "rmin": 2.0}],
       "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-6}, "objective": "compliance",
       "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": {"type": "mma", "move": 0.2},
       "run": {"max_iter": 10}}


def test_e3_single_isotropic_phase_matches_simp() -> None:
    old = build_setup(MBB).study
    cfg = copy.deepcopy(MBB)
    cfg["interpolation"] = {"type": "tensor_phases", "penal": 3.0, "emin_ratio": 1e-6,
                            "phases": [{"name": "solid", "isotropic": {"E": 1.0, "nu": 0.3}}]}
    cfg["physics"] = {"type": "tensor_elasticity"}
    new = build_setup(cfg).study
    x = np.random.default_rng(17).uniform(0.2, 1.0, old.n_var)
    a, b = old.evaluate(x), new.evaluate(x)
    assert abs(b.f0 / a.f0 - 1) <= 1e-10
    assert rel(b.df0, a.df0) <= 1e-8
    ra, rb = old.run(max_iter=10), new.run(max_iter=10)
    assert rel(rb.history["f0"], ra.history["f0"]) <= 1e-6


def test_e4_fixed_angle_field_matches_orthotropic_angle_field_path() -> None:
    d = Domain((16, 8))
    F = tip_load(d)
    ang = np.random.default_rng(18).uniform(-90, 90, d.shape)
    mat = Orthotropic(**ORTHO)
    Dfield = unit_elasticity_field(mat, 2, "stress", rotations_about_z(ang.reshape(-1)[d.active_elements]))
    old = Elasticity(d, left_fixed(d), elasticity=Dfield)
    x = torch.as_tensor(np.random.default_rng(19).uniform(0.2, 1.0, d.shape))
    U1 = old.solve(SIMP(d, E0=mat.reference_modulus)(x).reshape(-1)[old.elements], F).numpy()
    tp = TensorPhases(d, [PhaseMaterial("o", ElasticTensor.orthotropic(**ORTHO))])
    te = TensorElasticity(d, left_fixed(d))
    U2 = te.solve_field(tp.material_field(x, torch.as_tensor(np.radians(ang))), F).numpy()
    assert abs(float(F @ U2[:, 0]) / float(F @ U1[:, 0]) - 1) <= 1e-10
    assert rel(U2, U1) <= 1e-9


# ============================================================================================== M, adapters
def test_m1_field_holds_active_elements_only() -> None:
    frac = np.ones((10, 6))
    frac[:4, :3] = 0.0
    d = Domain((10, 6), frac=frac)
    t = torch.as_tensor(np.random.default_rng(20).uniform(0, 1, (3, 10, 6)))
    f = TensorPhases(d, phases(2)).material_field(t)
    assert isinstance(f, MaterialField)
    assert f.stiffness.shape == (d.n_active, 3, 3) and d.n_active < int(np.prod(d.shape))
    assert f.mass_density is None and f.conductivity is None
    g = TensorPhases(d, phases(2), request=("mass_density",)).material_field(t)
    assert g.mass_density is not None and g.mass_density.shape == (d.n_active,) and g.conductivity is None
    with pytest.raises(ValueError, match="conductivity"):
        TensorPhases(d, phases(2), request=("conductivity",))


def test_stale_fields_are_refused() -> None:
    d = Domain((6, 4))
    emb = OrientedEmbed(d, AngleBlock("element", period=math.pi))
    tp = TensorPhases(d, [PhaseMaterial("P3", P3_BASE)])
    tp.attach_angle(emb)
    te = TensorElasticity(d, left_fixed(d))
    te.bind(tp)
    with pytest.raises(StaleFieldError):
        tp(torch.ones(d.shape, dtype=torch.float64))            # the chain did not run
    x = torch.as_tensor(np.full(emb.n_in, 0.5))
    E = tp(emb(x))
    te.solve(E.reshape(-1), tip_load(d))
    with pytest.raises(StaleFieldError):
        te.solve(E.reshape(-1), tip_load(d))                     # the field was already used
    E = tp(emb(x))
    with pytest.raises(StaleFieldError):
        te.solve(E.reshape(-1) * 2.0, tip_load(d))               # not the published grid


def test_design_layout_round_trip() -> None:
    lay = DesignLayout([("density", 5, "element", "1"), ("angle", 1, "global", "rad")])
    x = lay.join({"density": np.arange(5.0), "angle": [0.25]})
    parts = lay.split(x)
    assert lay.size == 6 and list(parts["density"]) == [0, 1, 2, 3, 4] and float(parts["angle"][0]) == 0.25
    with pytest.raises(ValueError, match="missing"):
        lay.join({"density": np.zeros(5)})


def test_incompatible_mode_elements_are_refused() -> None:
    with pytest.raises(ValueError, match="linear in D"):
        TensorElasticity(Domain((4, 2)), np.arange(6), element="Q4IM")
