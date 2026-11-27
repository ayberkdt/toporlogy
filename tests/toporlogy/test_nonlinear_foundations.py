"""WP10 verification slice N1-N5 (``docs/foundations/prereg/N_nonlinear.md``, preregistered 2026-11-27 02:53 +0300).

References are closed forms written here from lambda, mu, kappa (numpy), torch autograd of an energy written here,
``scipy.linalg.logm`` and ``scipy.optimize.brentq``; none calls the production stress or tangent. The checks that
already existed (P1, P2, V1-V3, J1-J6, A1-A4, T1-T3) are listed in the record and not repeated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.linalg import logm
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from toporlogy.core import spectral
from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (InadmissibleState, J2Plasticity, Kinematics, MaterialRequest, NeoHooke,
                                         SaintVenantKirchhoff)
from toporlogy.core.domain import Domain
from toporlogy.fem.elements import D_iso
from toporlogy.fem.nonlinear import NonlinearElasticity
from toporlogy.solvers.continuation import (ArcLengthSettings, ContinuationError, TangentSpectrum, arc_length,
                                            find_critical, locate_critical)
from toporlogy.solvers.newton import LinearSolveFailure, Linearization

sys.path.insert(0, str(Path(__file__).parent))
from _truss import TwoBarTruss  # noqa: E402

PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]          # Voigt order [xx, yy, zz, yz, xz, xy]
NUS = (0.3, 0.45)
MATS = ("svk", "neo_hooke")
FIN = "finite"


# ============================================================================================== references
def lame(nu: float) -> tuple[float, float, float]:
    """(lambda, mu, kappa) for E = 1."""
    return nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu)), 1 / (3 * (1 - 2 * nu))


def ref_svk(F: np.ndarray, nu: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(psi, S, P) of SVK, psi = lam/2 tr(E)^2 + mu E:E."""
    lam, mu, _ = lame(nu)
    E = 0.5 * (F.T @ F - np.eye(3))
    S = lam * np.trace(E) * np.eye(3) + 2 * mu * E
    return 0.5 * lam * np.trace(E) ** 2 + mu * np.sum(E * E), S, F @ S


def ref_nh(F: np.ndarray, nu: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(psi, S, P) of the compressible neo-Hooke psi = kappa/2 (J-1)^2 + mu/2 (J^(-2/3) tr C - 3)
    (P = kappa J (J-1) F^-T + mu J^(-2/3) (F - tr C / 3 F^-T), derived in the record)."""
    _, mu, kap = lame(nu)
    J = np.linalg.det(F)
    Fit = np.linalg.inv(F).T
    trC = float(np.sum(F * F))
    P = kap * J * (J - 1) * Fit + mu * J ** (-2 / 3) * (F - trC / 3 * Fit)
    psi = 0.5 * kap * (J - 1) ** 2 + 0.5 * mu * (J ** (-2 / 3) * trC - 3)
    return psi, np.linalg.inv(F) @ P, P


def ref(material: str, F: np.ndarray, nu: float):
    return ref_svk(F, nu) if material == "svk" else ref_nh(F, nu)


def model_of(material: str, nu: float):
    return SaintVenantKirchhoff(D_iso(nu)) if material == "svk" else NeoHooke(nu)


def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """(..., 6) stress Voigt (physical shear) -> (..., 3, 3)."""
    out = np.zeros(v.shape[:-1] + (3, 3))
    for k, (a, b) in enumerate(PAIRS):
        out[..., a, b] = out[..., b, a] = v[..., k]
    return out


def rand_F(rng: np.random.Generator, n: int, dim: int = 3, rotate: bool = True) -> np.ndarray:
    out = []
    while len(out) < n:
        H = 0.3 * (2 * rng.random((dim, dim)) - 1)
        if np.linalg.norm(H, 2) <= 0.3 and np.linalg.det(np.eye(dim) + H) >= 0.5:
            F = np.eye(dim) + H
            if rotate:
                F = (Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix() if dim == 3
                     else Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix()[:2, :2]) @ F
            out.append(F)
    return np.array(out)


def evaluate(model, F: np.ndarray, energy: bool = True, plane: str = "3d", H: np.ndarray | None = None, **kw):
    """Production evaluation at F (H = F - I) or, with ``H`` given, at that displacement gradient exactly (a small H
    must not be formed as (I + H) - I: it loses log10(1/|H|) digits)."""
    H = F - np.eye(F.shape[-1]) if H is None else H
    return model.evaluate(Kinematics(H, FIN, plane), None, {}, MaterialRequest(energy=energy, **kw))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


# ============================================================================================== N1a-N1d
@pytest.mark.parametrize("material", MATS)
def test_n1a_identity_is_stress_free_with_the_small_strain_tangent(material):
    """N1a: F = I -> psi = 0, S = 0, P = 0 exactly (abs 1e-15); C(I) = isotropic small-strain C (rel 1e-12)."""
    nu = 0.3
    lam, mu, _ = lame(nu)
    r = evaluate(model_of(material, nu), np.eye(3)[None])
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    print(f"N1a {material}: |psi| {abs(r.energy[0]):.1e}, |S| {np.abs(r.stress).max():.1e}, C rel {rel(r.tangent[0], C):.1e}")
    assert abs(r.energy[0]) <= 1e-15 and np.abs(r.stress).max() <= 1e-15
    assert rel(r.tangent[0], C) <= 1e-12


@pytest.mark.parametrize("material", MATS)
def test_n1b_pure_rotation_is_stress_free(material):
    """N1b: F = Q (angles up to pi): psi = 0, S = 0, P = 0, Cauchy = 0 (abs 1e-13)."""
    rng = np.random.default_rng(1)
    rots = [Rotation.from_rotvec(a * v / np.linalg.norm(v)).as_matrix()
            for a, v in zip(np.linspace(0.3, np.pi, 20), rng.standard_normal((20, 3)), strict=True)]
    m = model_of(material, 0.3)
    worst = 0.0
    for Q in rots:
        r = evaluate(m, Q[None])
        S = voigt_to_tensor(r.stress)[0]
        worst = max(worst, abs(r.energy[0]), np.abs(S).max(), np.abs(Q @ S).max(), np.abs(Q @ S @ Q.T).max())
    print(f"N1b {material}: worst {worst:.1e}")
    assert worst <= 1e-13


@pytest.mark.parametrize("nu", NUS)
@pytest.mark.parametrize("material", MATS)
def test_n1c_simple_shear_closed_forms(material, nu):
    """N1c: simple shear gamma in {0.1, 0.5, 1, 2}: psi, S, P equal the references (rel 1e-12); NeoHooke Cauchy
    sigma_12 = mu g, sigma_11 = 2 mu g^2 / 3, sigma_22 = sigma_33 = -mu g^2 / 3 (derived in the record)."""
    lam, mu, _ = lame(nu)
    m = model_of(material, nu)
    worst = 0.0
    for g in (0.1, 0.5, 1.0, 2.0):
        F = np.eye(3)
        F[0, 1] = g
        r = evaluate(m, F[None])
        S = voigt_to_tensor(r.stress)[0]
        psi, S_ref, P_ref = ref(material, F, nu)
        worst = max(worst, rel(S, S_ref), rel(F @ S, P_ref), abs(r.energy[0] - psi) / abs(psi))
        sig = F @ S @ F.T                                   # J = 1
        if material == "neo_hooke":
            hand = np.array([[2 * mu * g * g / 3, mu * g, 0], [mu * g, -mu * g * g / 3, 0], [0, 0, -mu * g * g / 3]])
            worst = max(worst, rel(sig, hand))
        else:                                               # SVK: E = 1/2 [[0, g, 0], [g, g^2, 0], [0, 0, 0]]
            E = 0.5 * np.array([[0, g, 0], [g, g * g, 0], [0, 0, 0]])
            worst = max(worst, rel(S, lam * np.trace(E) * np.eye(3) + 2 * mu * E))
    print(f"N1c {material} nu={nu}: worst {worst:.1e}")
    assert worst <= 1e-12


@pytest.mark.parametrize("nu", NUS)
@pytest.mark.parametrize("material", MATS)
def test_n1d_uniaxial_biaxial_spherical_stretch(material, nu):
    """N1d: diag(l,1,1), diag(l,l,1), sI: P and psi equal the diagonal closed forms (rel 1e-12); NeoHooke spherical
    P = kappa s^2 (s^3 - 1) I."""
    lam, mu, kap = lame(nu)
    m = model_of(material, nu)
    cases = [np.diag([x, 1, 1]) for x in (0.6, 0.8, 1.5, 2.5)] + [np.diag([x, x, 1]) for x in (0.7, 1.4, 2.0)] \
        + [s * np.eye(3) for s in (0.8, 1.3)]
    worst = 0.0
    for F in cases:
        l = np.diag(F).astype(float)
        r = evaluate(m, F[None])
        P = F @ voigt_to_tensor(r.stress)[0]
        if material == "neo_hooke":
            J, trC = l.prod(), (l ** 2).sum()
            Pd = kap * J * (J - 1) / l + mu * J ** (-2 / 3) * (l - trC / (3 * l))
            psi = 0.5 * kap * (J - 1) ** 2 + 0.5 * mu * (J ** (-2 / 3) * trC - 3)
        else:
            Ed = 0.5 * (l ** 2 - 1)
            Sd = lam * Ed.sum() + 2 * mu * Ed
            Pd = l * Sd
            psi = 0.5 * lam * Ed.sum() ** 2 + mu * (Ed ** 2).sum()
        worst = max(worst, rel(np.diag(P), Pd), np.abs(P - np.diag(np.diag(P))).max() / np.abs(Pd).max(),
                    abs(r.energy[0] - psi) / abs(psi))
        if material == "neo_hooke" and np.allclose(l, l[0]):
            s = l[0]
            worst = max(worst, rel(np.diag(P), np.full(3, kap * s * s * (s ** 3 - 1))))
    print(f"N1d {material} nu={nu}: worst {worst:.1e}")
    assert worst <= 1e-12


# ============================================================================================== N1e: objectivity
@pytest.mark.parametrize("material", MATS)
def test_n1e_objectivity_energy_pk1_cauchy(material):
    """N1e: psi(QF) = psi(F), S(QF) = S(F), P(QF) = Q P(F), sigma(QF) = Q sigma(F) Q^T (rel 1e-12, 50 random pairs);
    negative control: the linear-strain energy of a 30 degree rotation is not zero."""
    rng = np.random.default_rng(5)
    m = model_of(material, 0.3)
    Fs = rand_F(rng, 50, rotate=False)
    Qs = Rotation.random(50, random_state=7).as_matrix()
    worst = 0.0
    for F, Q in zip(Fs, Qs, strict=True):
        r0, r1 = evaluate(m, F[None]), evaluate(m, (Q @ F)[None])
        S0, S1 = voigt_to_tensor(r0.stress)[0], voigt_to_tensor(r1.stress)[0]
        P0, P1 = F @ S0, (Q @ F) @ S1
        J = np.linalg.det(F)
        s0, s1 = F @ S0 @ F.T / J, (Q @ F) @ S1 @ (Q @ F).T / J
        worst = max(worst, abs(r1.energy[0] - r0.energy[0]) / abs(r0.energy[0]), rel(S1, S0), rel(P1, Q @ P0),
                    rel(s1, Q @ s0 @ Q.T))
    Q30 = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    lam, mu, _ = lame(0.3)
    eps = 0.5 * ((Q30 - np.eye(3)) + (Q30 - np.eye(3)).T)
    psi_lin = 0.5 * lam * np.trace(eps) ** 2 + mu * np.sum(eps * eps)
    print(f"N1e {material}: worst {worst:.1e}; linear-strain energy of a 30 degree rotation {psi_lin:.3e}")
    assert worst <= 1e-12 and psi_lin >= 1e-3


# ============================================================================================== N1f: tangent
def _tensor_psi_torch(material: str, nu: float):
    lam, mu, kap = lame(nu)

    def psi(F):
        C = F.T @ F
        if material == "svk":
            E = 0.5 * (C - torch.eye(3, dtype=F.dtype))
            return 0.5 * lam * torch.trace(E) ** 2 + mu * (E * E).sum()
        J = torch.det(F)
        return 0.5 * kap * (J - 1) ** 2 + 0.5 * mu * (J ** (-2 / 3) * torch.trace(C) - 3)
    return psi


def _voigt6_to_D(C6: np.ndarray) -> np.ndarray:
    """C6[I, J] = D[a, b, c, d] for I = (ab), J = (cd) (record: the engineering-shear column absorbs the factor 2)."""
    D = np.zeros((3, 3, 3, 3))
    for I, (a, b) in enumerate(PAIRS):
        for K, (c, d) in enumerate(PAIRS):
            for (p, q) in {(a, b), (b, a)}:
                for (s, t) in {(c, d), (d, c)}:
                    D[p, q, s, t] = C6[I, K]
    return D


def _first_piola_tangent(F: np.ndarray, S: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.einsum("ik,JL->iJkL", np.eye(3), S) + np.einsum("iA,kB,AJBL->iJkL", F, F, D)


@pytest.mark.parametrize("nu", NUS)
@pytest.mark.parametrize("material", MATS)
def test_n1f_tangent_vs_fd_and_autograd(material, nu):
    """N1f: C symmetric; A = d(FS)/dF built from the production C equals the central FD of the reference P (h = 1e-6,
    rel 1e-6) and the torch Hessian of the reference psi(F) (rel 1e-10); major symmetry of A; production psi, S equal
    the reference (rel 1e-11)."""
    rng = np.random.default_rng(11 + int(100 * nu))
    m = model_of(material, nu)
    psi_t = _tensor_psi_torch(material, nu)
    e_sym = e_fd = e_ad = e_maj = e_val = 0.0
    for F in rand_F(rng, 10):
        r = evaluate(m, F[None])
        S = voigt_to_tensor(r.stress)[0]
        C6 = r.tangent[0]
        psi, S_ref, P_ref = ref(material, F, nu)
        e_val = max(e_val, rel(S, S_ref), abs(r.energy[0] - psi) / abs(psi))
        e_sym = max(e_sym, rel(C6, C6.T))
        A = _first_piola_tangent(F, S, _voigt6_to_D(C6))
        A_fd = np.zeros((3, 3, 3, 3))
        h = 1e-6
        for k in range(3):
            for L in range(3):
                Fp, Fm = F.copy(), F.copy()
                Fp[k, L] += h
                Fm[k, L] -= h
                A_fd[:, :, k, L] = (ref(material, Fp, nu)[2] - ref(material, Fm, nu)[2]) / (2 * h)
        Ft = torch.tensor(F, dtype=torch.float64)
        A_ad = torch.autograd.functional.hessian(psi_t, Ft).numpy()
        e_fd = max(e_fd, rel(A, A_fd))
        e_ad = max(e_ad, rel(A, A_ad))
        e_maj = max(e_maj, rel(A, A.transpose(2, 3, 0, 1)))
    print(f"N1f {material} nu={nu}: values {e_val:.1e}, C sym {e_sym:.1e}, A vs FD {e_fd:.1e}, A vs autograd {e_ad:.1e}, "
          f"major sym {e_maj:.1e}")
    assert e_val <= 1e-11 and e_sym <= 1e-12 and e_maj <= 1e-12 and e_fd <= 1e-6 and e_ad <= 1e-10


# ============================================================================================== N1g: small-strain limit
@pytest.mark.parametrize("nu", NUS)
@pytest.mark.parametrize("material", MATS)
def test_n1g_small_strain_limit_rate(material, nu):
    """N1g: H = delta H0: r(delta) = |S - sigma_lin(eps)| / |sigma_lin(eps)| has log-log slope in [0.9, 1.1] over
    delta in [1e-5, 1e-2], r(1e-5) <= 1e-3, r decreasing in delta in {1e-1 ... 1e-5}; NeoHooke tangent deviation from
    the isotropic C has the same slope; SVK: S = sigma_lin(E) exactly (rel 1e-13) and its C is the constant D."""
    lam, mu, _ = lame(nu)
    H0 = np.random.default_rng(3).uniform(-1, 1, (3, 3))
    H0 /= np.abs(H0).max()
    m = model_of(material, nu)
    Clin = np.zeros((6, 6))
    Clin[:3, :3] = lam
    Clin[np.arange(3), np.arange(3)] += 2 * mu
    Clin[np.arange(3, 6), np.arange(3, 6)] = mu
    deltas = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    rs, rc = [], []
    for d in deltas:
        H = d * H0
        r = evaluate(m, None, H=H[None])
        S = voigt_to_tensor(r.stress)[0]
        eps = 0.5 * (H + H.T)
        sl = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        rs.append(np.linalg.norm(S - sl) / np.linalg.norm(sl))
        rc.append(np.linalg.norm(r.tangent[0] - Clin) / np.linalg.norm(Clin))
        if material == "svk":
            E = T.green_lagrange(H[None])[0]
            assert rel(S, lam * np.trace(E) * np.eye(3) + 2 * mu * E) <= 1e-13
    rs, rc = np.array(rs), np.array(rc)
    slope = np.polyfit(np.log(deltas[1:]), np.log(rs[1:]), 1)[0]
    print(f"N1g {material} nu={nu}: r(S) {rs}, slope {slope:.3f}; r(C) {rc}")
    assert 0.9 <= slope <= 1.1 and rs[-1] <= 1e-3 and np.all(np.diff(rs) < 0)
    if material == "neo_hooke":
        sc = np.polyfit(np.log(deltas[1:]), np.log(rc[1:]), 1)[0]
        print(f"N1g neo_hooke nu={nu}: tangent slope {sc:.3f}")
        assert 0.9 <= sc <= 1.1
    else:
        assert rc.max() <= 1e-13                       # constant tangent: the slope of a zero deviation is undefined


# ============================================================================================== N1h, N1i: Hencky, conjugacy
def _stretch_F(rng: np.random.Generator) -> np.ndarray:
    while True:
        F = (Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix() @ np.diag(rng.uniform(0.4, 3.0, 3))
             @ Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix())
        if np.linalg.det(F) >= 0.2:
            return F


def _hencky_ref(F: np.ndarray, frame: str = "material") -> np.ndarray:
    A = F.T @ F if frame == "material" else F @ F.T
    L = logm(A)
    assert np.abs(L.imag).max() <= 1e-12
    return 0.5 * L.real


def test_n1h_hencky_via_spectral_log_and_its_derivative():
    """N1h: spectral log(C)/2 = logm(C)/2 = tensor.hencky_strain (rel 1e-11, material and spatial); the derivative
    (1/2) spectral_frechet(C, Cdot, "log") equals the central FD of logm(C(F + t Fdot))/2 (h = 1e-6, rel 1e-7)."""
    rng = np.random.default_rng(21)
    e_val = e_th = e_der = 0.0
    for _ in range(20):
        F = _stretch_F(rng)
        Fd = rng.standard_normal((3, 3))
        for frame, A in (("material", F.T @ F), ("spatial", F @ F.T)):
            Hs = 0.5 * spectral.spectral_function(0.5 * (A + A.T), "log")
            e_val = max(e_val, rel(Hs, _hencky_ref(F, frame)))
            e_th = max(e_th, rel(T.hencky_strain(F, frame), _hencky_ref(F, frame)))
        C = F.T @ F
        Cd = Fd.T @ F + F.T @ Fd
        J = 0.5 * spectral.spectral_frechet(0.5 * (C + C.T), 0.5 * (Cd + Cd.T), "log")
        h = 1e-6
        fd = (_hencky_ref(F + h * Fd) - _hencky_ref(F - h * Fd)) / (2 * h)
        e_der = max(e_der, rel(J, fd))
    print(f"N1h: spectral {e_val:.1e}, tensor.hencky_strain {e_th:.1e}, derivative {e_der:.1e}")
    assert e_val <= 1e-11 and e_th <= 1e-11 and e_der <= 1e-7


@pytest.mark.parametrize("material", MATS)
def test_n1i_work_conjugacy_and_hencky_rate(material):
    """N1i: S:Edot = P:Fdot = tau:d (rel 1e-12); coaxial F(t) = Q diag(l(t)) Q^T: tau:Hdot = tau:d = S:Edot (rel
    1e-11) with Hdot = (1/2) spectral_frechet; negative control: simple shear gamma = 1, Fdot = e1 (x) e2: tau:Hdot
    differs from tau:d by at least 1e-3 relative."""
    nu = 0.3
    m = model_of(material, nu)
    rng = np.random.default_rng(31)
    e_gen = e_co = 0.0
    for F in rand_F(rng, 20):
        Fd = rng.standard_normal((3, 3))
        S = voigt_to_tensor(evaluate(m, F[None], energy=False).stress)[0]
        Ed = 0.5 * (F.T @ Fd + Fd.T @ F)
        d = 0.5 * (Fd @ np.linalg.inv(F) + (Fd @ np.linalg.inv(F)).T)
        tau = F @ S @ F.T
        w = np.sum(S * Ed)
        scale = np.linalg.norm(S) * np.linalg.norm(Ed)
        e_gen = max(e_gen, abs(w - np.sum((F @ S) * Fd)) / scale, abs(w - np.sum(tau * d)) / scale)
    for _ in range(20):
        Q = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        l, ld = rng.uniform(0.6, 1.8, 3), rng.standard_normal(3)
        F, Fd = Q @ np.diag(l) @ Q.T, Q @ np.diag(ld) @ Q.T
        S = voigt_to_tensor(evaluate(m, F[None], energy=False).stress)[0]
        tau = F @ S @ F.T
        C = F.T @ F
        Cd = Fd.T @ F + F.T @ Fd
        Hd = 0.5 * spectral.spectral_frechet(0.5 * (C + C.T), 0.5 * (Cd + Cd.T), "log")
        d = 0.5 * (Fd @ np.linalg.inv(F) + (Fd @ np.linalg.inv(F)).T)
        Ed = 0.5 * Cd
        scale = np.linalg.norm(tau) * np.linalg.norm(d)
        e_co = max(e_co, abs(np.sum(tau * Hd) - np.sum(tau * d)) / scale, abs(np.sum(S * Ed) - np.sum(tau * d)) / scale)
    F = np.eye(3)
    F[0, 1] = 1.0
    Fd = np.zeros((3, 3))
    Fd[0, 1] = 1.0
    S = voigt_to_tensor(evaluate(m, F[None], energy=False).stress)[0]
    tau = F @ S @ F.T
    C = F.T @ F
    Cd = Fd.T @ F + F.T @ Fd
    Hd = 0.5 * spectral.spectral_frechet(0.5 * (C + C.T), 0.5 * (Cd + Cd.T), "log")
    d = 0.5 * (Fd @ np.linalg.inv(F) + (Fd @ np.linalg.inv(F)).T)
    ctrl = abs(np.sum(tau * Hd) - np.sum(tau * d)) / abs(np.sum(tau * d))
    print(f"N1i {material}: general {e_gen:.1e}, coaxial {e_co:.1e}; non-coaxial control {ctrl:.3f}")
    assert e_gen <= 1e-12 and e_co <= 1e-11 and ctrl >= 1e-3


# ============================================================================================== N2: plane stress
def _ps_reference(F2: np.ndarray, nu: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(lambda_3, in-plane S [S11, S22, S12], 3-D P) of neo-Hooke at in-plane F2 with P33 = 0 (brentq)."""
    def p33(l3: float) -> float:
        F = np.zeros((3, 3))
        F[:2, :2] = F2
        F[2, 2] = l3
        return ref_nh(F, nu)[2][2, 2]
    lo, hi = 0.05, 5.0
    assert p33(lo) < 0 < p33(hi), "set-up: no sign change in the bracket"
    l3 = brentq(p33, lo, hi, xtol=1e-15, rtol=1e-15)
    F = np.zeros((3, 3))
    F[:2, :2] = F2
    F[2, 2] = l3
    _, S, P = ref_nh(F, nu)
    return l3, np.array([S[0, 0], S[1, 1], S[0, 1]]), P


@pytest.mark.parametrize("nu", NUS)
def test_n2a_plane_stress_local_solution(nu):
    """N2a: E33 equals 1/2 (l3^2 - 1) of the brentq root (rel 1e-10), in-plane S equals the reference (rel 1e-10), and
    the reference P33 at the production E33 is zero (abs 1e-12 max|P|); 30 random F2 and simple shear 0.5, 1."""
    rng = np.random.default_rng(41)
    F2s = list(rand_F(rng, 30, dim=2)) + [np.array([[1.0, g], [0.0, 1.0]]) for g in (0.5, 1.0)]
    m = NeoHooke(nu)
    E33, S_prod, E33_ref, S_ref, p33 = [], [], [], [], []
    for F2 in F2s:
        r = m.evaluate(Kinematics((F2 - np.eye(2))[None], FIN, "stress"), None, {}, MaterialRequest(energy=True))
        l3, S2, P = _ps_reference(F2, nu)
        E33.append(r.diagnostics["E33"][0])
        S_prod.append(r.stress[0])
        E33_ref.append(0.5 * (l3 ** 2 - 1))
        S_ref.append(S2)
        F3 = np.zeros((3, 3))
        F3[:2, :2] = F2
        F3[2, 2] = np.sqrt(1 + 2 * r.diagnostics["E33"][0])
        p33.append(abs(ref_nh(F3, nu)[2][2, 2]) / np.abs(P).max())
    e33 = rel(np.array(E33), np.array(E33_ref))
    eS = rel(np.array(S_prod), np.array(S_ref))
    print(f"N2a nu={nu}: E33 {e33:.1e}, S {eS:.1e}, max |P33|/|P| {max(p33):.1e}")
    assert e33 <= 1e-10 and eS <= 1e-10 and max(p33) <= 1e-12


@pytest.mark.parametrize("nu", NUS)
def test_n2b_condensed_tangent_vs_finite_difference(nu):
    """N2b: C_ps (3 x 3) dE = central FD of the reference constrained S_ps (rel 1e-6, dE the exact difference of the
    Green strain of F2 +- h dF); C_ps symmetric (rel 1e-12)."""
    rng = np.random.default_rng(51)
    m = NeoHooke(nu)
    worst = sym = 0.0
    h = 1e-6
    for F2 in rand_F(rng, 15, dim=2):
        r = m.evaluate(Kinematics((F2 - np.eye(2))[None], FIN, "stress"), None, {}, MaterialRequest())
        C = r.tangent[0]
        sym = max(sym, rel(C, C.T))
        for _ in range(3):
            dF = rng.standard_normal((2, 2))
            Fp, Fm = F2 + h * dF, F2 - h * dF
            dS = _ps_reference(Fp, nu)[1] - _ps_reference(Fm, nu)[1]
            Ep, Em = 0.5 * (Fp.T @ Fp - np.eye(2)), 0.5 * (Fm.T @ Fm - np.eye(2))
            dE = np.array([Ep[0, 0] - Em[0, 0], Ep[1, 1] - Em[1, 1], 2 * (Ep[0, 1] - Em[0, 1])])
            worst = max(worst, float(np.abs(C @ dE - dS).max() / np.abs(dS).max()))
    print(f"N2b nu={nu}: tangent vs FD {worst:.1e}, symmetry {sym:.1e}")
    assert worst <= 1e-6 and sym <= 1e-12


@pytest.mark.parametrize("plane", ["3d", "strain", "stress"])
def test_n2c_inadmissible_trial_raises_and_changes_nothing(plane):
    """N2c: det F = 0 (H = -I) and det F < 0 (H = diag(-1.2, 0, 0)) raise InadmissibleState (code invalid_deformation);
    the read-only input and the committed mapping are bytewise unchanged."""
    dim = 3 if plane == "3d" else 2
    committed = {"alpha": np.arange(3.0)}
    committed["alpha"].setflags(write=False)
    for H in (-np.eye(dim), np.diag([-1.2] + [0.0] * (dim - 1))):
        H = H[None].copy()
        H.setflags(write=False)
        before = (H.tobytes(), committed["alpha"].tobytes())
        with pytest.raises(InadmissibleState) as ei:
            NeoHooke(0.3).evaluate(Kinematics(H, FIN, plane), None, committed, MaterialRequest(energy=True))
        assert ei.value.code == "invalid_deformation"
        assert (H.tobytes(), committed["alpha"].tobytes()) == before


def test_n2c_fe_step_with_an_inadmissible_trial_rolls_back():
    """N2c (FE): Newton on a neo-Hooke bar under a huge compression returns converged False, code invalid_deformation,
    rolled_back, and the start displacement bytewise unchanged."""
    d = Domain((4, 2), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
    F = np.zeros(d.ndof)
    F[2 * tip] = -5.0 / len(tip)
    nl = NonlinearElasticity(d, fixed, material="neo_hooke", load_steps=1)
    nl.check_kinematics()
    u0 = np.zeros(d.ndof)
    before = u0.tobytes()
    ok, u, rec, *_ = nl._newton(np.ones(nl.n_elements), F, u0, None, 1.0)
    print(f"N2c FE: ok {ok}, code {rec.code}, rolled back {rec.rolled_back}")
    assert not ok and rec.code == "invalid_deformation" and rec.rolled_back
    assert u.tobytes() == before and u0.tobytes() == before


# ============================================================================================== N3: grid patch
CONFIGS = [("svk", 2, "stress"), ("svk", 2, "strain"), ("svk", 3, "stress"),
           ("neo_hooke", 2, "stress"), ("neo_hooke", 2, "strain"), ("neo_hooke", 3, "stress")]
IDS = [f"{m}-{d}D-{p}" for m, d, p in CONFIGS]


def _patch(material: str, dim: int, plane: str, nu: float = 0.3):
    d = Domain((4, 3) if dim == 2 else (3, 2, 2), h=0.5, thickness=0.8)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    nl = NonlinearElasticity(d, fixed, nu=nu, material=material, plane=plane)
    return d, X, nl


def _p_reference(material: str, dim: int, plane: str, F: np.ndarray, nu: float) -> np.ndarray:
    """In-plane (or 3-D) first Piola stress of the patch's constitutive law at the affine F."""
    if dim == 3:
        return ref(material, F, nu)[2]
    F3 = np.eye(3)
    F3[:2, :2] = F
    if plane == "strain":
        return ref(material, F3, nu)[2][:2, :2]
    if material == "neo_hooke":
        return _ps_reference(F, nu)[2][:2, :2]
    E = 0.5 * (F.T @ F - np.eye(2))                                  # SVK plane stress: the 2-D law D_ps (E = 1)
    c = 1 / (1 - nu ** 2)
    S = c * np.array([[E[0, 0] + nu * E[1, 1], (1 - nu) * E[0, 1]], [(1 - nu) * E[0, 1], E[1, 1] + nu * E[0, 0]]])
    return F @ S


@pytest.mark.parametrize("material,dim,plane", CONFIGS, ids=IDS)
def test_n3a_affine_patch_interior_equilibrium_and_face_resultant(material, dim, plane):
    """N3a: affine u = (F - I) X: interior nodal forces zero (abs 1e-11 max|f|), sum f = 0, resultant on the X1 = 0
    face equals -t A P e1 (rel 1e-10), t = 0.8 in 2-D; rotation included in F; grid element only (no distorted mesh)."""
    nu = 0.3
    d, X, nl = _patch(material, dim, plane, nu)
    rng = np.random.default_rng(61)
    t = d.thickness if dim == 2 else 1.0
    ext = np.array(d.shape) * d.h
    area = t * float(np.prod(ext[1:]))
    interior = np.all((X.min(axis=0) + 1e-9 < X) & (X.max(axis=0) - 1e-9 > X), axis=1)
    assert interior.any()
    worst_int = worst_sum = worst_res = 0.0
    for F in rand_F(rng, 3, dim=dim):
        u = (X @ (F - np.eye(dim)).T).ravel()
        f = nl.internal_force(u, np.ones(nl.n_elements)).reshape(-1, dim)
        P = _p_reference(material, dim, plane, F, nu)
        scale = np.abs(f).max()
        worst_int = max(worst_int, np.abs(f[interior]).max() / scale)
        worst_sum = max(worst_sum, np.abs(f.sum(axis=0)).max() / scale)
        face = X[:, 0] < 1e-9
        res = f[face].sum(axis=0)
        worst_res = max(worst_res, float(np.abs(res + area * P[:, 0]).max() / np.abs(area * P[:, 0]).max()))
    print(f"N3a {material} {dim}D {plane}: interior {worst_int:.1e}, sum {worst_sum:.1e}, resultant {worst_res:.1e}")
    assert worst_int <= 1e-11 and worst_sum <= 1e-11 and worst_res <= 1e-10


@pytest.mark.parametrize("material,dim,plane", CONFIGS, ids=IDS)
def test_n3b_rigid_rotation_patch(material, dim, plane):
    """N3b: u = (Q - I) X for rotations up to pi: every nodal force <= 1e-12 abs, min det F = 1 (1e-12)."""
    d, X, nl = _patch(material, dim, plane)
    worst = dj = 0.0
    for ang in (0.3, 1.0, 2.0, np.pi):
        Q = (Rotation.from_rotvec(ang * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])).as_matrix()
             if dim == 3 else Rotation.from_euler("z", ang).as_matrix()[:2, :2])
        u = (X @ (Q - np.eye(dim)).T).ravel()
        worst = max(worst, np.abs(nl.internal_force(u, np.ones(nl.n_elements))).max())
        dj = max(dj, abs(nl.min_det_f(u) - 1.0))
    print(f"N3b {material} {dim}D {plane}: max |f| {worst:.1e}, |min det F - 1| {dj:.1e}")
    assert worst <= 1e-12 and dj <= 1e-12


# ============================================================================================== N4: path following
DENSE = TangentSpectrum()


@pytest.mark.parametrize("ds", [0.02, 0.01, 0.005, 0.0025])
def test_n4a_truss_step_refinement(ds):
    """N4a: two-bar truss, ds in {0.02, 0.01, 0.005, 0.0025}: path on the closed form (|lam - P(w)/P_ref| <= 1e-9,
    |u| <= 1e-12), two critical intervals, limit points |lam - (+-1)| <= 1e-8, |w - w_1,2| <= 1e-6 h."""
    tr = TwoBarTruss()
    st = ArcLengthSettings(ds=ds, rtol=1e-12, eta_tol=0.0, max_steps=4000)
    path = arc_length(tr, np.zeros(2), 0.0, st, lambda x, lam: -x[1] >= 2.5 * tr.h, spectrum=DENSE)
    w = -path.component(1)
    e_curve = float(np.abs(path.lam - tr.P(w) / tr.P_ref).max())
    e_u = float(np.abs(path.component(0)).max())
    crit = find_critical(path)
    cps = [locate_critical(tr, path, i, DENSE) for i in crit]
    e_lam = [abs(cps[0].lam - 1), abs(cps[1].lam + 1)] if len(cps) == 2 else [np.inf, np.inf]
    e_w = [abs(-cps[0].x[1] - tr.w1) / tr.h, abs(-cps[1].x[1] - tr.w2) / tr.h] if len(cps) == 2 else [np.inf] * 2
    print(f"N4a ds={ds}: {len(path.lam)} points, curve {e_curve:.1e}, |u| {e_u:.1e}, intervals {crit}, "
          f"lam errors {e_lam[0]:.1e} {e_lam[1]:.1e}, w errors/h {e_w[0]:.1e} {e_w[1]:.1e}")
    assert len(crit) == 2 and all(c.kind == "limit" for c in cps)
    assert e_curve <= 1e-9 and e_u <= 1e-12 and max(e_lam) <= 1e-8 and max(e_w) <= 1e-6


class _PolynomialModel:
    """Model A of test_bifurcation (re-derived here): U = ka e^2 / 2 + kr u^2 / 2 + beta u^3 / 6, e = v + u^2 / (2L);
    R = grad U - lam q, q = (0, -1)."""

    conservative = True
    load = np.array([0.0, -1.0])

    def __init__(self, L: float = 1.0, ka: float = 100.0, kr: float = 1.0, beta: float = 3.0) -> None:
        self.L, self.ka, self.kr, self.beta = L, ka, kr, beta

    def linearize(self, x: np.ndarray, lam: float) -> Linearization:
        u, v = x
        L, ka, kr, b = self.L, self.ka, self.kr, self.beta
        e = v + u * u / (2 * L)
        g = np.array([ka * e * u / L + kr * u + 0.5 * b * u * u, ka * e])
        K = np.array([[ka * (u / L) ** 2 + ka * e / L + kr + b * u, ka * u / L], [ka * u / L, ka]])
        return Linearization(g - lam * self.load, K, float(np.abs(K).sum(axis=0).max()))

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        try:
            out = np.linalg.solve(lin.matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise LinearSolveFailure(f"LinAlgError: {e}") from e
        return out


def test_n4b_r07_exactly_singular_accepted_point_still_breaks_the_predictor():
    """N4b (R07 re-check, measured): model A with ds = 0.01 accepts the point (0, -0.01) at lam = 1 where K11 = 0
    exactly; the next predictor K^-1 q fails. Asserts the measured failure (ContinuationError step_limit, cause
    singular_tangent, last accepted lam = 1 within 1e-12): a characterisation of the open defect, not the desired
    behaviour (that stays in the strict xfails test_a1/test_a2 of test_bifurcation.py)."""
    m = _PolynomialModel()
    st = ArcLengthSettings(ds=0.01, rtol=1e-12, eta_tol=0.0, max_steps=500)
    with pytest.raises(ContinuationError) as ei:
        arc_length(m, np.zeros(2), 0.0, st, lambda x, lam: lam >= 1.5)
    pts = ei.value.path.points
    K = m.linearize(pts[-1].x, pts[-1].lam).matrix
    print(f"N4b: code {ei.value.code}, cause {ei.value.cause}, accepted {len(pts)} points, last lam {pts[-1].lam:.15f}, "
          f"K11 {K[0, 0]:.3e}, det K {np.linalg.det(K):.3e}")
    assert ei.value.code == "step_limit" and ei.value.cause == "singular_tangent"
    assert abs(pts[-1].lam - 1.0) <= 1e-12


# ============================================================================================== N5: J2 additions
NU_J2, SY_J2 = 0.3, 1e-3


@pytest.mark.parametrize("k", [1.0, 10.0, 100.0])
def test_n5a_hydrostatic_strain_never_yields(k):
    """N5a: eps = c I, c = k eps_y: elastic, alpha = 0 and eps_p = 0 exactly, sigma = 3 K c I, tangent = elastic C,
    psi = 9/2 K c^2 (rel 1e-13), dissipation 0."""
    lam, mu, K = lame(NU_J2)
    m = J2Plasticity(NU_J2, SY_J2, 0.05)
    c = k * SY_J2
    eps = c * np.eye(3)[None]
    r = m.evaluate(Kinematics(eps, "small"), None, m.initial_state(1), MaterialRequest(energy=True))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    ref_s = np.array([3 * K * c] * 3 + [0, 0, 0])
    print(f"N5a k={k}: plastic {r.diagnostics['plastic'][0]}, sigma rel {rel(r.stress[0], ref_s):.1e}, "
          f"C rel {rel(r.tangent[0], C):.1e}, psi rel {abs(r.energy[0] - 4.5 * K * c * c) / (4.5 * K * c * c):.1e}")
    assert r.diagnostics["plastic"][0] == 0.0
    assert np.all(r.trial_state["alpha"] == 0.0) and np.all(r.trial_state["plastic_strain"] == 0.0)
    assert rel(r.stress[0], ref_s) <= 1e-13 and rel(r.tangent[0], C) <= 1e-13
    assert abs(r.energy[0] - 4.5 * K * c * c) <= 1e-13 * 4.5 * K * c * c and np.all(r.dissipation == 0.0)


@pytest.mark.parametrize("H", [0.05, 0.0])
def test_n5b_tangent_kink_at_the_yield_point(H):
    """N5b: eps = e diag(1, -1/2, -1/2), e = e_y (1 -+ 1e-7), e_y = sigma_y / (3 mu): below elastic flag 0 and the
    elastic C; above flag 1 and C_xxxx jumps by -2 mu 3mu/(3mu + H) (2/3) (rel 1e-5); stress and energy continuous;
    the continuum tangent above has the same jump exactly (rel 1e-12)."""
    _, mu, _ = lame(NU_J2)
    m = J2Plasticity(NU_J2, SY_J2, H)
    e_y = SY_J2 / (3 * mu)
    pred = -2 * mu * 3 * mu / (3 * mu + H) * 2 / 3
    out = {}
    for tag, e in (("below", e_y * (1 - 1e-7)), ("above", e_y * (1 + 1e-7))):
        eps = (e * np.diag([1.0, -0.5, -0.5]))[None]
        out[tag] = m.evaluate(Kinematics(eps, "small"), None, m.initial_state(1), MaterialRequest(energy=True))
    cont = m.evaluate(Kinematics((e_y * (1 + 1e-7) * np.diag([1.0, -0.5, -0.5]))[None], "small"), None,
                      m.initial_state(1), MaterialRequest(tangent_kind="continuum"))
    below, above = out["below"], out["above"]
    jump = above.tangent[0][0, 0] - below.tangent[0][0, 0]
    jump_c = cont.tangent[0][0, 0] - below.tangent[0][0, 0]
    dsig = abs(above.stress[0][0] - below.stress[0][0])
    dpsi = abs(above.energy[0] - below.energy[0])
    print(f"N5b H={H}: flags {below.diagnostics['plastic'][0]}/{above.diagnostics['plastic'][0]}, jump {jump:.6e} "
          f"(predicted {pred:.6e}), continuum jump {jump_c:.6e}, d sigma_xx {dsig:.1e}, d psi {dpsi:.1e}")
    assert below.diagnostics["plastic"][0] == 0.0 and above.diagnostics["plastic"][0] == 1.0
    assert abs(jump - pred) <= 1e-5 * abs(pred) and abs(jump_c - pred) <= 1e-12 * abs(pred)
    assert dsig <= 1e-6 * SY_J2 and dpsi <= 1e-6 * SY_J2 * (SY_J2 / 1.0)
