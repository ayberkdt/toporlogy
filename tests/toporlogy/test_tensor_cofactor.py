"""Cofactor, determinant derivative and the rotation logarithm near pi (docs/foundations/prereg/TK_cofactor_rotation.md,
TK1-TK5)."""
from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from toporlogy.core import tensor as T


def _random_F(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    while len(out) < n:
        F = rng.normal(size=(d, d))
        if np.linalg.cond(F) <= 1e3:
            out.append(F)
    return np.array(out)


@pytest.mark.parametrize("d", [2, 3])
def test_tk1_cofactor_equals_det_inverse_transpose(d) -> None:
    F = _random_F(50, d, np.random.default_rng(31))
    ref = np.linalg.det(F)[:, None, None] * np.swapaxes(np.linalg.inv(F), 1, 2)
    got = T.cofactor(F)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


def _singular(rank: int, rng: np.random.Generator) -> np.ndarray:
    U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s = np.array([2.0, 0.7, 0.0]) if rank == 2 else np.array([1.5, 0.0, 0.0])
    return (U * s) @ V.T


def _mp_adjugate_T(F: np.ndarray) -> np.ndarray:
    with mpmath.workdps(50):
        M = mpmath.matrix(F.tolist())
        C = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                r = [k for k in range(3) if k != i]
                q = [k for k in range(3) if k != j]
                minor = M[r[0], q[0]] * M[r[1], q[1]] - M[r[0], q[1]] * M[r[1], q[0]]
                C[i, j] = float((-1) ** (i + j) * minor)
        return C


@pytest.mark.parametrize("rank", [2, 1])
def test_tk2_singular_cofactor_against_mpmath(rank) -> None:
    rng = np.random.default_rng(31)
    for _ in range(10):
        F = _singular(rank, rng)
        got = T.cofactor(F)
        ref = _mp_adjugate_T(F)
        assert np.abs(got - ref).max() <= 1e-14 * np.linalg.norm(F) ** 2
        if rank == 1:
            assert np.abs(got).max() <= 1e-14 * np.linalg.norm(F) ** 2


@pytest.mark.parametrize("kind", ["invertible", "rank2", "rank1"])
def test_tk3_det_derivative_fd(kind) -> None:
    rng = np.random.default_rng(33)
    F = _random_F(1, 3, rng)[0] if kind == "invertible" else _singular(2 if kind == "rank2" else 1, rng)
    H = rng.normal(size=(3, 3))
    g = float(T.det_derivative(F, H))
    errs = []
    for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        fd = (np.linalg.det(F + h * H) - np.linalg.det(F - h * H)) / (2 * h)
        errs.append(abs(fd - g))
    if kind == "rank1":
        assert abs(g) <= 1e-12 * np.linalg.norm(F) ** 2
        assert min(errs) <= 1e-12 * np.linalg.norm(F) ** 2 + 1e-12
    else:
        assert min(errs) <= 1e-8 * abs(g), (errs, g)


def test_tk4_rotation_log_small_and_near_pi() -> None:
    rng = np.random.default_rng(32)
    for ang in [1e-10, 1e-4, 0.8, math.pi - 1e-6]:
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        w = ang * a
        R = T.rotation_from_rotvec(w)
        wl = T.rotvec_from_rotation(R)
        assert np.abs(T.rotation_from_rotvec(wl) - R).max() <= 1e-12
        if ang <= 0.8:
            assert np.linalg.norm(wl - w) <= 1e-10
        else:
            th = np.linalg.norm(wl)
            assert abs(th - ang) <= 1e-6
            axis = wl / th
            assert min(np.linalg.norm(axis - a), np.linalg.norm(axis + a)) <= 1e-6


def test_tk5_negative_controls() -> None:
    with pytest.raises(ValueError):
        T.rotvec_from_rotation(np.diag([1.0, 1.0, -1.0]))
    for bad in [np.ones((3, 2)), np.ones((4, 4)), np.ones(3)]:
        with pytest.raises(ValueError):
            T.cofactor(bad)
