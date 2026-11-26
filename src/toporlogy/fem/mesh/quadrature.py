"""Quadrature rules on the reference cells of :mod:`toporlogy.fem.mesh.reference`.

Reference cells: line, quadrilateral and hexahedron are ``[-1, 1]^d``; triangle and tetrahedron are the unit simplex
(vertices 0, e₁, e₂(, e₃)). The weights of a rule sum to the measure of its cell (2, 4, 8; 1/2, 1/6).

* Hypercubes: tensor products of Gauss–Legendre points (:func:`numpy.polynomial.legendre.leggauss`); n points per
  direction integrate every monomial of degree ≤ 2n − 1 in each variable.
* Simplices: the collapsed (Duffy 1982) tensor product of Gauss–Legendre rules with the Jacobian of the collapse
  written out, ``x = r(1 − s)``, ``y = s`` (triangle) and ``x = r(1 − s)(1 − t)``, ``y = s(1 − t)``, ``z = t``
  (tetrahedron), on ``[0, 1]`` per direction. A monomial of total degree p becomes a polynomial of degree p, p + 1
  (, p + 2) in r, s (, t), so the number of points per direction is chosen for exactness up to the requested degree.
  The classical centroid rule (degree 1), the 3-point triangle rule at (1/6, 1/6) … (degree 2) and the 4-point Keast
  rule (degree 2, the constants of :mod:`toporlogy.post.verify`) are used where they suffice.

Exactness of every rule up to its declared degree is a test (``tests/toporlogy/test_mesh_elements.py``, E5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

from toporlogy.references import cite

__all__ = ["Rule", "gauss_legendre", "rule", "MEASURE"]

MEASURE = {"line": 2.0, "quad": 4.0, "hex": 8.0, "tri": 0.5, "tet": 1.0 / 6.0}
_DIM = {"line": 1, "quad": 2, "hex": 3, "tri": 2, "tet": 3}


@dataclass(frozen=True)
class Rule:
    """Points (nq, dim) on a reference cell, weights (nq,), and the polynomial degree integrated exactly."""

    cell: str
    points: np.ndarray
    weights: np.ndarray
    degree: int
    name: str


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n-point Gauss–Legendre rule on [-1, 1] (exact for degree 2n − 1)."""
    if n < 1:
        raise ValueError("n >= 1 points")
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def _tensor(n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = gauss_legendre(n)
    grids = np.meshgrid(*([x] * dim), indexing="ij")
    wgrids = np.meshgrid(*([w] * dim), indexing="ij")
    P = np.stack([g.ravel() for g in grids], axis=1)
    W = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    return P, W


def _gl01(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = gauss_legendre(n)
    return 0.5 * (x + 1.0), 0.5 * w


@cite("duffy_1982", note="collapsed tensor-product rules on simplices")
def _collapsed(cell: str, degree: int) -> tuple[np.ndarray, np.ndarray]:
    if cell == "tri":
        rr, wr = _gl01(max(1, math.ceil((degree + 1) / 2)))
        ss, ws = _gl01(max(1, math.ceil((degree + 2) / 2)))
        R, S = np.meshgrid(rr, ss, indexing="ij")
        WR, WS = np.meshgrid(wr, ws, indexing="ij")
        x, y = R * (1 - S), S
        W = WR * WS * (1 - S)
        return np.column_stack([x.ravel(), y.ravel()]), W.ravel()
    rr, wr = _gl01(max(1, math.ceil((degree + 1) / 2)))
    ss, ws = _gl01(max(1, math.ceil((degree + 2) / 2)))
    tt, wt = _gl01(max(1, math.ceil((degree + 3) / 2)))
    R, S, T = np.meshgrid(rr, ss, tt, indexing="ij")
    WR, WS, WT = np.meshgrid(wr, ws, wt, indexing="ij")
    x, y, z = R * (1 - S) * (1 - T), S * (1 - T), T
    W = WR * WS * WT * (1 - S) * (1 - T) ** 2
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]), W.ravel()


_KA, _KB = 0.5854101966249685, 0.1381966011250105        # Keast (1986) 4-point rule, as in post/verify.py


@cache
def _rule_cached(cell: str, degree: int) -> Rule:
    if cell not in MEASURE:
        raise ValueError(f"unknown reference cell {cell!r}: {' | '.join(MEASURE)}")
    if degree < 0:
        raise ValueError("degree >= 0")
    dim = _DIM[cell]
    if cell in ("line", "quad", "hex"):
        n = max(1, math.ceil((degree + 1) / 2))
        P, W = _tensor(n, dim)
        return Rule(cell, P, W, 2 * n - 1, f"gauss-legendre {'x'.join([str(n)] * dim)}")
    if degree <= 1:
        c = np.full((1, dim), 1.0 / (dim + 1))
        return Rule(cell, c, np.array([MEASURE[cell]]), 1, "centroid")
    if degree == 2 and cell == "tri":
        P = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        return Rule(cell, P, np.full(3, 1 / 6), 2, "triangle 3-point")
    if degree == 2 and cell == "tet":
        P = np.array([[_KB, _KB, _KB], [_KA, _KB, _KB], [_KB, _KA, _KB], [_KB, _KB, _KA]])
        return Rule(cell, P, np.full(4, 1 / 24), 2, "keast 4-point")
    P, W = _collapsed(cell, degree)
    return Rule(cell, P, W, degree, f"collapsed gauss-legendre ({len(W)} points)")


def rule(cell: str, degree: int) -> Rule:
    """A rule on ``cell`` exact for every polynomial of total degree ≤ ``degree`` (hypercubes: in each variable)."""
    r = _rule_cached(cell, int(degree))
    r.points.setflags(write=False)
    r.weights.setflags(write=False)
    return r
