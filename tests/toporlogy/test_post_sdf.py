"""toporlogy.post.sdf / topology / primitives: sub-voxel distance, morphology, Euler characteristic."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.post import sdf as S
from toporlogy.post.primitives import Box, Cylinder, Plane, Sphere
from toporlogy.post.topology import grid_topology


def _grid(n: int, h: float = 1.0) -> S.Grid:
    return S.Grid(np.full(3, -n / 2 * h + 0.5 * h), h, (n, n, n))


@pytest.mark.parametrize("R", [10.3, 4.2])
def test_redistance_sphere_subvoxel(R: float) -> None:
    g = _grid(40)
    X = g.points()
    r = np.linalg.norm(X, axis=1)
    exact = (r - R).reshape(g.shape)
    phi = S.redistance((r ** 2 - R ** 2).reshape(g.shape), g.h)       # not a distance, same zero level
    band = np.abs(exact) < 3
    err = np.abs(phi - exact)[band]
    # measured 13-14 Nov 2026: 0.033 h (R = 10.3 h), 0.067 h (R = 4.2 h); binary EDT gives ~0.5 h
    assert err.max() < (0.05 if R > 8 else 0.1)


def test_opening_closing_radii() -> None:
    """Closing fillets a 90° concave corner: the corner point moves out by r(√2 − 1) along the bisector;
    opening rounds the convex corner of a box by r and leaves flat faces in place."""
    h = 0.5
    g = _grid(48, h)
    X = g.points()
    L_shape = np.minimum(Box((-8, -8, -8), (0, 8, 8)).sdf(X), Box((-8, -8, -8), (8, 0, 8)).sdf(X)).reshape(g.shape)
    r = 3.0
    cl = S.closing(L_shape, r, h)
    corner = g.sample(cl, np.array([[r * (1 - 1 / np.sqrt(2)) + 0.0, r * (1 - 1 / np.sqrt(2)), 0.0]]))
    # exact closed shape: the fillet circle centre (r, r), radius r → the point (r(1−1/√2), …) is on the surface
    assert abs(float(corner[0])) < 0.15 * h
    op = S.opening(Box((-6, -6, -6), (6, 6, 6)).sdf(X).reshape(g.shape), r, h)
    face = g.sample(op, np.array([[6.0, 0.0, 0.0]]))
    edge = g.sample(op, np.array([[6 - r + r / np.sqrt(2), 6 - r + r / np.sqrt(2), 0.0]]))
    assert abs(float(face[0])) < 0.1 * h and abs(float(edge[0])) < 0.15 * h


def test_euler_characteristic() -> None:
    x, y, z = np.mgrid[-16:17, -16:17, -16:17].astype(float)
    r = np.sqrt(x * x + y * y + z * z)
    assert grid_topology(r < 7) == {"bodies": 1, "cavities": 0, "euler": 1, "tunnels": 0}
    assert grid_topology((r < 12) & (r > 5)) == {"bodies": 1, "cavities": 1, "euler": 2, "tunnels": 0}
    torus = (np.sqrt(x * x + y * y) - 9) ** 2 + z * z < 9
    assert grid_topology(torus) == {"bodies": 1, "cavities": 0, "euler": 0, "tunnels": 1}
    two = (np.sqrt((x - 8) ** 2 + y * y + z * z) < 4) | (np.sqrt((x + 8) ** 2 + y * y + z * z) < 4)
    assert grid_topology(two)["bodies"] == 2


def test_primitive_sdfs_exact() -> None:
    rng = np.random.default_rng(1)
    P = rng.uniform(-5, 5, (500, 3))
    s = Sphere((1, 0, 0), 2.0)
    assert np.allclose(s.sdf(P), np.linalg.norm(P - [1, 0, 0], axis=1) - 2)
    c = Cylinder((0, 0, 0), (0, 0, 2), 1.5)
    assert np.allclose(c.sdf(P), np.hypot(P[:, 0], P[:, 1]) - 1.5)
    pl = Plane((0, 0, 1), (0, 0, 2))
    assert np.allclose(pl.sdf(P), P[:, 2] - 1)
    # projection lands on the surface; gradients are unit
    for sh in (s, c, pl, Box((-1, -2, -3), (2, 1, 1)), Cylinder((0, 0, 0), (1, 1, 0), 1.0, length=3.0)):
        Q = sh.project(P)
        assert np.abs(sh.sdf(Q)).max() < 1e-9
        assert np.allclose(np.linalg.norm(sh.grad(P), axis=1), 1.0)


def test_volume_threshold_matches_target() -> None:
    g = _grid(40, 0.5)
    X = g.points()
    rf = 1.0 / (1.0 + np.exp((np.linalg.norm(X, axis=1) - 5.0) / 0.6)).reshape(g.shape)
    target = 4 / 3 * np.pi * 4.5 ** 3
    t, phi, v = S.volume_threshold(rf, g, target, redistanced=True)
    assert abs(v / target - 1) < 1e-4
    assert abs(S.volume(phi, g.h) / target - 1) < 1e-4
