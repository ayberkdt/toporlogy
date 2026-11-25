"""Published finite-element buckling errors of a column: Ferrari & Sigmund (2019), "Revisiting topology optimization
with buckling constraints", arXiv:1809.05300v2, §2.1 and the table of Fig. 1 (read in full text, 2026-11-25).

Problem as printed: cantilever column Lx = 10, Ly = 1, thickness t = 0.005, E = 2·10¹¹, ν = 0 (plane stress, so the
2-D model equals the beam model), "uniform compression at the tip", reference load f0 = 2.5·10⁵; closed form
Pc = π² E I / (4 Lx²) ≈ 2.05617·10⁶ with I = t Ly³ / 12. The table gives the relative error e = (λ1 f0 − Pc)/Pc of
Q4 and Q6 (= Wilson's incompatible-mode element, toporlogy ``Q4IM``) for seven meshes; only 20 × 2 has square
elements, which is all toporlogy's grid can represent.

Pre-registered (session notes, 2026-11-25 16:58 +03:00, before the first run): clamp = every DOF at x = 0; tip
load = consistent nodal loads (end nodes half);
R1a Q4: |e − 0.119| ≤ 5e-4 (+1e-6);  R1b Q4IM: |e − (−6.6e-3)| ≤ 5e-5 (+1e-6) — the printed rounding;
R1c (no published value) Q4IM |e| ≤ 1e-2 on the square meshes 20×2 … 160×16.
Why they can differ: the authors' code is not available (load distribution, the stress used in K_G — Gauss point
or centroid — and how K_G treats the incompatible modes are not printed).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import toporlogy as T
from toporlogy.fem.buckling import LinearBuckling

PC_PUBLISHED = 2.05617e6
F0, E, LX, LY, THICK = 2.5e5, 2e11, 10.0, 1.0, 0.005


def _error(nx: int, ny: int, element: str) -> float:
    h = LX / nx
    assert abs(h - LY / ny) < 1e-12
    d = T.Domain((nx, ny), h, thickness=THICK)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    tip = np.flatnonzero(np.isclose(X[:, 0], LX))
    w = np.ones(len(tip))
    w[np.isclose(X[tip, 1], 0) | np.isclose(X[tip, 1], LY)] = 0.5
    F = d.load_vector()
    F[2 * tip, 0] = -F0 * w / w.sum()
    m = T.Elasticity(d, fixed, 0.0, element=element)
    Ev = np.full(d.n_active, E)
    U = m.solve_numpy(Ev, F)[:, 0]
    mu, _ = LinearBuckling(m).modes(Ev, Ev, U, 2)
    return (F0 / mu[0] - PC_PUBLISHED) / PC_PUBLISHED


def test_closed_form_value():
    pc = math.pi ** 2 * E * THICK * LY ** 3 / 12 / (4 * LX ** 2)
    assert abs(pc - PC_PUBLISHED) <= 5.0                   # printed to 6 digits


def test_published_q4_error_20x2():
    """R1a. First run 2026-11-25: e = 0.118684 against 0.119, PASS."""
    e = _error(20, 2, "Q4")
    print(f"\n  FS2019 column 20 x 2, Q4: e = {e:.6g} (published 0.119)")
    assert abs(e - 0.119) <= 5e-4 + 1e-6


@pytest.mark.xfail(strict=True, reason="pre-registered R1b FAILED: Q4IM gives -0.00511 (now -0.00561 with modes in K_G), published Q6 -0.0066")
def test_published_q6_error_20x2():
    """R1b. First run 2026-11-25: e = −0.005107 against −0.0066, FAILED (K_G from the compatible field). With the
    condensed modes in K_G (LinearBuckling default since the same day) e = −0.005614: still outside the bound.

    Post hoc (scratch code with rectangular elements, so every published mesh could be computed; not adopted as a
    pass criterion): that code reproduces the five coarse Q4 rows to the printed digits (0.4939, 0.4933, 0.1187,
    0.1181, 0.0243), but not the two fine ones (80 × 16: 8.5e-4 against 5.0e-3); Wilson's Q6 gives −0.00205 …
    −0.00688 (K_G from the compatible field) or −0.00408 … −0.00691 (modes in K_G) against the published −0.0069 …
    −0.0026. Our Q4 and Q6 converge to one limit (≈ −0.0070, the 2-D continuum value; Timoshenko's shear correction
    alone gives −0.49 %); the published Q4 and Q6 still differ by 3e-3 at 160 × 32. The published Q6 column is
    therefore not reproduced by any variant tried; the coarse Q4 agreement is the usable part of this source."""
    assert abs(_error(20, 2, "Q4IM") - (-6.6e-3)) <= 5e-5 + 1e-6


def test_q4im_converges_without_locking():
    errs = {el: [_error(20 * k, 2 * k, el) for k in (1, 2, 4, 8)] for el in ("Q4", "Q4IM")}
    print(f"\n  FS2019 column, square meshes 20x2 .. 160x16: Q4 {np.round(errs['Q4'], 5)}; Q4IM "
          f"{np.round(errs['Q4IM'], 5)}")
    assert max(abs(e) for e in errs["Q4IM"]) <= 1e-2
