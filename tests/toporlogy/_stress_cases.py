"""Problems of docs/olcumler/gerilme_onkayit.md, built once for the tests and for ``docs/olcumler/gerilme.py``.

Closed forms with consistent nodal loads (uniaxial tension, pure bending, two parallel materials) and the peak-stress
problems (point load, re-entrant corner, clamped edge, and the controls).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.optimization.stress_recovery import StressRecovery
from toporlogy.optimization.stress_singularity import RegionStress, region_stress

NU = 0.3
EXACT_ELEMENTS = {2: ("Q4", "Q4IM"), 3: ("H8", "H8IM")}
TYPES = ("integration_point", "centroid", "integration_point_max", "extrapolated", "nodal_averaged",
         "patch_recovered")


def face_loads(d: Domain, axis: int, upper: bool, traction: Any, where: Any = None) -> np.ndarray:
    """Consistent nodal loads (ndof, 1) of ``traction(X) -> (n, dim)`` on the box face normal to ``axis``
    (2 Gauss points per direction: exact for tractions linear on a face element). ``where(centre)`` selects face
    elements."""
    F = d.load_vector()
    side = d.shape[axis] - 1 if upper else 0
    others = [a for a in range(d.dim) if a != axis]
    gp = np.array([(1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2])
    area = d.h ** (d.dim - 1) * (d.thickness if d.dim == 2 else 1.0)
    for cell in itertools.product(*[range(d.shape[a]) for a in others]):
        ijk = np.zeros(d.dim, int)
        ijk[axis], ijk[others] = side, cell
        if not d.active[tuple(ijk)]:
            continue
        base = d.origin + d.h * ijk
        centre = base + 0.5 * d.h
        centre[axis] = base[axis] + (d.h if upper else 0.0)
        if where is not None and not where(centre):
            continue
        for q in itertools.product(range(2), repeat=len(others)):
            xi = np.zeros(d.dim)
            xi[axis] = 1.0 if upper else 0.0
            xi[others] = gp[list(q)]
            t = np.asarray(traction((base + d.h * xi)[None, :]), float)[0]
            for corner in itertools.product((0, 1), repeat=len(others)):
                node = ijk.copy()
                node[axis] += int(upper)
                node[others] += corner
                N = np.prod([xi[a] if c else 1 - xi[a] for a, c in zip(others, corner, strict=True)])
                n = int(d.node_number(node))
                F[d.dim * n:d.dim * n + d.dim, 0] += area * 0.5 ** len(others) * N * t
    return F


def node_at(d: Domain, *x: float) -> int:
    hit = d.nodes_where(lambda X: np.isclose(X, np.array(x)).all(1))
    assert hit.size == 1, f"no node at {x}"
    return int(hit[0])


@dataclass
class Case:
    domain: Domain
    physics: Elasticity
    U: np.ndarray
    E: float = 1.0

    def recovery(self, **kw: Any) -> StressRecovery:
        return StressRecovery(self.physics, self.U, kw.pop("E", self.E), **kw)


def _solve(d: Domain, fixed: np.ndarray, F: np.ndarray, element: str | None = None, nu: float = NU,
           E: Any = 1.0) -> Case:
    ph = Elasticity(d, fixed, nu, element)
    moduli = np.broadcast_to(np.asarray(E, float), (d.n_active,))
    return Case(d, ph, ph.solve_numpy(moduli, F), E)


# ------------------------------------------------------------------------------------------------ closed forms
SIGMA0, E_TENSION = 3.0, 70.0


def tension(element: str) -> Case:
    """Uniaxial tension σx = SIGMA0 of a block (rigid modes removed, free lateral contraction)."""
    dim = 3 if element.startswith("H8") else 2
    d = Domain((4, 3, 2), 0.5) if dim == 3 else Domain((6, 4), 0.5, thickness=0.7)
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    if dim == 2:
        fixed = np.concatenate([d.dofs(left, 0), d.dofs(left[:1], 1)])
    else:
        fixed = np.concatenate([d.dofs(left, 0), d.dofs(left[:1], [1, 2]),
                                d.dofs(np.array([node_at(d, 0.0, 1.5, 0.0)]), 2)])
    F = face_loads(d, 0, True, lambda X: np.tile(SIGMA0 * np.eye(dim)[0], (len(X), 1)))
    return _solve(d, fixed, F, element, E=E_TENSION)


def tension_exact(X: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((len(X), 6 if dim == 3 else 3))
    out[:, 0] = SIGMA0
    return out


H, L, W, SMAX = 1.0, 2.0, 0.5, 1.0


def bending(element: str, n: int) -> Case:
    """Pure bending σx = SMAX (y − H/2) / (H/2) of an L × H (× W) strip with n elements over H: symmetry plane at
    x = 0, linear traction at x = L."""
    dim = 3 if element.startswith("H8") else 2
    h = H / n
    shape = (2 * n, n) if dim == 2 else (2 * n, n, n // 2)
    d = Domain(shape, h)
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    if dim == 2:
        fixed = np.concatenate([d.dofs(left, 0), d.dofs(np.array([node_at(d, 0.0, H / 2)]), 1)])
    else:
        fixed = np.concatenate([d.dofs(left, 0), d.dofs(np.array([node_at(d, 0.0, H / 2, W / 2)]), [1, 2]),
                                d.dofs(np.array([node_at(d, 0.0, H, W / 2)]), 2)])
    F = face_loads(d, 0, True, lambda X: np.outer(SMAX * (X[:, 1] - H / 2) / (H / 2), np.eye(dim)[0]))
    return _solve(d, fixed, F, element)


def bending_exact(X: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((len(X), 6 if dim == 3 else 3))
    out[:, 0] = SMAX * (X[:, 1] - H / 2) / (H / 2)
    return out


def errors(case: Case, exact: Any, scale: float, region: Any = None, separate: Any = "material") -> dict[str, float]:
    """Largest |σ_h − σ| / scale of every output type over the points where ``region(X)`` holds. The Gauss-point
    maximum is compared with the exact von Mises stress maximised over the same points."""
    rec = case.recovery()
    dim = case.domain.dim
    sel = (lambda X: np.ones(len(X), bool)) if region is None else region

    def field_error(stress: np.ndarray, points: np.ndarray) -> float:
        s = stress[..., 0].reshape(-1, stress.shape[-2])                     # first load case
        X = points.reshape(-1, dim)
        m = sel(X)
        return float(np.abs(s[m] - exact(X[m], dim)).max() / scale)

    pf, cen, corner = rec.integration_points(), rec.centroids(), rec.extrapolated()
    avg, spr = rec.nodal_averaged(separate), rec.patch_recovered(separate)
    top = rec.ip_maximum()
    X = pf.points.reshape(-1, dim)
    inside = sel(X).reshape(pf.points.shape[:2])
    vm_exact = np.abs(exact(X, dim)[:, 0]).reshape(pf.points.shape[:2])      # uniaxial stress states
    vm_h = pf.von_mises()[..., 0]
    whole = inside.all(1)                                                    # elements fully inside the region
    assert whole.any()
    assert np.allclose(top.value[:, 0], vm_h.max(1))
    return {"integration_point": field_error(pf.stress, pf.points),
            "centroid": field_error(cen.stress, cen.points),
            "integration_point_max": float(np.abs(top.value[whole, 0] - vm_exact[whole].max(1)).max() / scale),
            "extrapolated": field_error(corner.stress, corner.points),
            "nodal_averaged": field_error(avg.stress, avg.points),
            "patch_recovered": field_error(spr.stress, spr.points)}


EPS = 1e-9
REGION_R = lambda X: X[:, 0] <= L / 2 + EPS                                  # noqa: E731
REGION_R_STRICT = lambda X: ((X[:, 0] >= H / 4 - EPS) & (X[:, 0] <= L / 2 + EPS)       # noqa: E731
                             & (X[:, 1] >= H / 4 - EPS) & (X[:, 1] <= 3 * H / 4 + EPS))
MESHES = {2: (8, 16, 32, 64), 3: (4, 8, 16)}


def orders(values: list[float]) -> list[float]:
    return [float(np.log2(values[i] / values[i + 1])) for i in range(len(values) - 1)]


def bimaterial(E1: float = 1.0, E2: float = 4.0) -> tuple[Case, np.ndarray]:
    """Two parallel layers (E1 below, E2 above, ν = 0) stretched with the same εx = 1: σx = E of the layer. Returns
    the case and E per active element."""
    d = Domain((6, 4), 0.5)
    E = np.where(d.element_centers()[:, 1] < 1.0, E1, E2)
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(np.array([node_at(d, 0.0, 0.0)]), 1)])
    F = face_loads(d, 0, True, lambda X: np.column_stack([np.where(X[:, 1] < 1.0, E1, E2), np.zeros(len(X))]))
    case = _solve(d, fixed, F, "Q4", nu=0.0, E=E)
    return case, E


# ------------------------------------------------------------------------------------------------ peak problems
LEVELS = (16, 32, 64, 128)


def point_load(n: int) -> Case:
    d = Domain((n, n), 1.0 / n)
    F = d.load_vector()
    F[2 * node_at(d, 0.5, 1.0) + 1, 0] = -1.0
    return _solve(d, d.dofs(d.nodes_where(lambda X: X[:, 1] == 0)), F)


def footprint_load(n: int) -> Case:
    """The point-load plate with the same total load spread over the footprint x ∈ [0.375, 0.625]."""
    d = Domain((n, n), 1.0 / n)
    F = face_loads(d, 1, True, lambda X: np.tile([0.0, -4.0], (len(X), 1)),
                   where=lambda c: 0.375 < c[0] < 0.625)
    assert np.isclose(F[1::2].sum(), -1.0)
    return _solve(d, d.dofs(d.nodes_where(lambda X: X[:, 1] == 0)), F)


def l_corner(n: int, fillet: float = 0.0) -> Case:
    """L-shaped part of the unit square (upper right quadrant removed), clamped on top of the vertical leg, uniform
    shear on the end of the horizontal leg. ``fillet``: radius of a voxelised (staircase) fillet at the inner
    corner (elements whose centre lies in the fillet material are added)."""
    h = 1.0 / n
    c = (np.indices((n, n)).reshape(2, -1).T + 0.5) * h
    solid = ~((c[:, 0] > 0.5) & (c[:, 1] > 0.5))
    if fillet:
        in_square = (c > 0.5).all(1) & (c < 0.5 + fillet).all(1)
        solid |= in_square & (((c - (0.5 + fillet)) ** 2).sum(1) >= fillet ** 2)
    d = Domain((n, n), h, frac=solid.reshape(n, n).astype(float))
    fixed = d.dofs(d.nodes_where(lambda X: np.isclose(X[:, 1], 1.0) & (X[:, 0] <= 0.5 + EPS)))
    # the loaded face x = 1 exists for y <= 0.5 only
    F = face_loads(d, 0, True, lambda X: np.tile([0.0, -2.0], (len(X), 1)))
    assert np.isclose(F[1::2].sum(), -1.0)
    return _solve(d, fixed, F)


def clamped_edge(n: int) -> Case:
    d = Domain((2 * n, n), 1.0 / n)
    F = face_loads(d, 0, True, lambda X: np.tile([0.0, -1.0], (len(X), 1)))
    return _solve(d, d.dofs(d.nodes_where(lambda X: X[:, 0] == 0)), F)


#: name -> (builder, box lo, box hi, singular point)
PEAK_PROBLEMS: dict[str, tuple[Any, tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "point_load": (point_load, (0.25, 0.75), (0.75, 1.0), (0.5, 1.0)),
    "l_corner": (l_corner, (0.25, 0.25), (0.75, 0.75), (0.5, 0.5)),
    "clamped_edge": (clamped_edge, (-0.25, 0.75), (0.25, 1.25), (0.0, 1.0)),
    "footprint_load": (footprint_load, (0.25, 0.75), (0.75, 1.0), (0.5, 1.0)),
}


def centroid_region(case: Case, lo: Any, hi: Any) -> RegionStress:
    """Region quantities of the centroid von Mises stress (first load case)."""
    cen = case.recovery().centroids()
    return region_stress(cen.von_mises()[:, 0], cen.points, lo, hi)


def bending_peak(n: int) -> RegionStress:
    """Negative control: largest centroid von Mises of the pure bending strip in x <= L/2 (it grows as 1 − h/H)."""
    return centroid_region(bending("Q4", n), (0.0, 0.0), (L / 2, H))
