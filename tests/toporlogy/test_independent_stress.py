"""Centroid stress and failure indices against a separately written implementation (docs/olcumler/dogrulama_turu.md
S1-S4, pre-registered 2026-11-25 21:05 +0300 before the first run).

The expected stresses come from ``_independent_fe`` (own centre strain matrix, own isotropic and orthotropic
stiffness, rotation as a fourth-order tensor, plane stress / plane strain from the 3-D law). The stress is rotated to
the material axes as a tensor, Rᵀ σ R, and the criteria are written from their definitions: von Mises from the
deviatoric invariant, Tsai–Wu as the load factor s that puts the stress on the surface (a s² + b s = 1, index 1/s).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import _independent_fe as ind
from toporlogy.core.domain import Domain
from toporlogy.core.material import (Orthotropic, OrthotropicStrength, Strength, out_of_plane_rows,
                                     rotation_from_axes, unit_elasticity_field)
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.strength import make_criterion
from toporlogy.optimization.stress import FailureIndex, centroid_stress

ENG = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
MAT = Orthotropic(**ENG)
STR = dict(Xt=1500.0, Xc=1200.0, Yt=50.0, Yc=200.0, S12=70.0)
TOL = 1e-12            # pre-registered S1-S4
E0 = 40.0


def _rel(got, want) -> float:
    return float(np.abs(np.asarray(got) - want).max() / np.abs(want).max())


def _domain(dim: int) -> Domain:
    d = Domain((5, 4), h=0.7, thickness=0.6) if dim == 2 else Domain((4, 3, 3), h=0.7)
    assert np.array_equal(np.sort(d.edof, 1), np.sort(ind.element_dofs(d.shape), 1))
    return d


def _ctx(d: Domain, U: np.ndarray) -> EvalContext:
    return EvalContext(domain=d, xphys=torch.ones(d.shape, dtype=torch.float64),
                       E=torch.full(d.shape, E0, dtype=torch.float64), U=torch.as_tensor(U))


def _rotations(dim: int, n: int, rng) -> np.ndarray:
    if dim == 2:
        return np.array([ind.rot_z(a) for a in rng.uniform(-90.0, 90.0, n)])
    return np.array([rotation_from_axes(rng.standard_normal(3), rng.standard_normal(3)) for _ in range(n)])


def _tensor(v: np.ndarray) -> np.ndarray:
    """3 x 3 stress from Voigt (3-D order, or 2-D [xx, yy, xy] with zz = 0 unless given separately)."""
    S = np.zeros((3, 3))
    for k, (i, j) in enumerate(ind.VOIGT[3]):
        S[i, j] = S[j, i] = v[k]
    return S


def _to3(s2: np.ndarray, szz: float = 0.0) -> np.ndarray:
    return np.array([s2[0], s2[1], szz, 0.0, 0.0, s2[2]])


def _von_mises(s6: np.ndarray) -> float:
    dev = _tensor(s6) - np.trace(_tensor(s6)) / 3 * np.eye(3)
    return float(np.sqrt(1.5 * np.sum(dev * dev)))


def _tsai_wu(s6: np.ndarray, R: np.ndarray) -> float:
    m = R.T @ _tensor(s6) @ R                                  # stress in the material axes
    s = [m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]]
    Xt, Xc, Yt, Yc, S12 = (STR[k] for k in ("Xt", "Xc", "Yt", "Yc", "S12"))
    Zt, Zc, S13, S23 = Yt, Yc, S12, S12                       # documented defaults
    F = [1 / Xt - 1 / Xc, 1 / Yt - 1 / Yc, 1 / Zt - 1 / Zc]
    Fd = [1 / (Xt * Xc), 1 / (Yt * Yc), 1 / (Zt * Zc)]
    a = sum(Fd[i] * s[i] ** 2 for i in range(3)) + s[3] ** 2 / S23 ** 2 + s[4] ** 2 / S13 ** 2 + s[5] ** 2 / S12 ** 2
    for i in range(3):
        for j in range(i + 1, 3):
            a += 2 * (-0.5) * np.sqrt(Fd[i] * Fd[j]) * s[i] * s[j]
    b = sum(F[i] * s[i] for i in range(3))
    load = (-b + np.sqrt(b * b + 4 * a)) / (2 * a)            # a load² + b load = 1
    return float(1.0 / load)


@pytest.mark.parametrize("dim", [2, 3])
def test_s1_centroid_stress(dim):
    d = _domain(dim)
    U = np.random.default_rng(0).standard_normal((d.ndof, 2))
    got = centroid_stress(d, torch.as_tensor(U), 0.3, E0).numpy()              # (n, nv, nc)
    D1 = ind.plane_stress_isotropic(E0, 0.3) if dim == 2 else ind.isotropic_3d(E0, 0.3)
    Ds = np.broadcast_to(D1, (d.n_active,) + D1.shape)
    want = np.stack([ind.centroid_stresses(d.shape, d.h, Ds, np.ones(d.n_active), U[:, c]) for c in range(2)], -1)
    print(f"\n  S1 {dim}-D: {_rel(got, want):.2e}")
    assert _rel(got, want) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_s2_von_mises_index(dim):
    d = _domain(dim)
    U = np.random.default_rng(1).standard_normal((d.ndof, 1))
    fi = FailureIndex(d, make_criterion("von_mises", Strength(2.0), eps=0.0), nu=0.3, E0=E0, q=0.0)
    got = fi.solid_ratio(_ctx(d, U))[:, 0].numpy()
    D1 = ind.plane_stress_isotropic(E0, 0.3) if dim == 2 else ind.isotropic_3d(E0, 0.3)
    sig = ind.centroid_stresses(d.shape, d.h, np.broadcast_to(D1, (d.n_active,) + D1.shape), np.ones(d.n_active),
                                U[:, 0])
    want = np.array([_von_mises(_to3(s) if dim == 2 else s) / 2.0 for s in sig])
    print(f"\n  S2 {dim}-D: {_rel(got, want):.2e}")
    assert _rel(got, want) <= TOL


@pytest.mark.parametrize("dim,plane", [(2, "stress"), (3, "stress"), (2, "strain")],
                         ids=["S3-Q4", "S3-H8", "S4-Q4-plane-strain"])
def test_s3_s4_tsai_wu_with_an_orientation_field(dim, plane):
    d = _domain(dim)
    rng = np.random.default_rng(2)
    R = _rotations(dim, d.n_active, rng)
    U = 0.05 * rng.standard_normal((d.ndof, 1))
    D_lib = unit_elasticity_field(MAT, dim, plane, R)
    row = out_of_plane_rows(MAT, R) if plane == "strain" else None
    crit = make_criterion("tsai_wu", OrthotropicStrength(**STR), R, eps=0.0)
    got = FailureIndex(d, crit, D=D_lib, szz_row=row, E0=E0, q=0.0).solid_ratio(_ctx(d, U))[:, 0].numpy()
    C3 = np.array([ind.rotate(ind.orthotropic_3d(**ENG), r) for r in R])
    ip = [0, 1, 5]
    if dim == 3:
        Ds = C3
    elif plane == "stress":
        Ds = np.array([ind.plane_stress(c) for c in C3])
    else:
        Ds = C3[:, ip][:, :, ip]
    B = ind.strain_matrix(ind.shape_gradients(np.full(dim, 0.5), d.h))
    want = []
    for e, ed in enumerate(ind.element_dofs(d.shape)):
        eps = B @ U[ed, 0]
        s = Ds[e] @ eps
        s6 = s if dim == 3 else _to3(s, float(C3[e][2, ip] @ eps) if plane == "strain" else 0.0)
        want.append(_tsai_wu(s6, R[e]))
    want = np.array(want)
    print(f"\n  {dim}-D {plane}: {_rel(got, want):.2e} (index range {want.min():.3g} .. {want.max():.3g})")
    assert _rel(got, want) <= TOL
