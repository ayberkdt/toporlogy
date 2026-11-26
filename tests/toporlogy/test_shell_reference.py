"""Drill-independent 3-D solid reference for the curved shell segment of DR (docs/olcumler/kabuk_onkayit.md, DB0-DB5).

R = 1, 90°, L = 1 segment clamped at z = 0, t = 0.01 and 0.05, a unit force spread over the same patch at the shell's
mid-surface load point in both models (corner patch, direction +z; free-edge middle patch, direction −e_r(π/4)).
Solid: T10 (quadratic tetrahedra; ``fem.mesh`` has no H20 / H27 and no incompatible-mode hexahedron off the grid),
box mesh in (θ, z, r) mapped onto the cylinder node by node, so the edges are isoparametric arcs.
"""
from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from toporlogy.core.linear_elastic import isotropic_stiffness
from toporlogy.fem import shell as shell_module
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, box
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.assembly import assemble_vector
from toporlogy.fem.shell import FlatShell, cylinder_surface

R, L, NU = 1.0, 1.0, 0.3
ANGLE = math.pi / 2
DRILLS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CASES = ((0.01, "corner"), (0.01, "radial"), (0.05, "corner"), (0.05, "radial"))
KINDS = ("corner", "radial")
LEVELS = ((16, 12), (32, 24), (64, 48))                     # solid reference sequence (ratio 2)
SHELL_LEVELS = ((16, 12), (32, 24), (64, 48), (128, 96))
_C = math.cos(math.pi / 4)
# patch: (θ0, θ1, z0, z1, direction); aligned with meshes of n_theta a multiple of 16 and n_z a multiple of 6
PATCH = {
    "corner": (7 * math.pi / 16, math.pi / 2, 5 / 6, 1.0, np.array([0.0, 0.0, 1.0])),
    "radial": (math.pi / 4 - math.pi / 32, math.pi / 4 + math.pi / 32, 5 / 6, 1.0, np.array([-_C, -_C, 0.0])),
}
PATCH_VOLUME = (math.pi / 16) * (1 / 6) * R            # × t: Δθ Δz ∫ r dr = Δθ Δz R t


def _in_patch(theta: np.ndarray, z: np.ndarray, kind: str) -> np.ndarray:
    t0, t1, z0, z1, _ = PATCH[kind]
    return (theta > t0) & (theta < t1) & (z > z0) & (z < z1)


# ---------------------------------------------------------------------------------------------- solid
def _solid(nt: int, nz: int, nr: int, t: float, element: str = "T10") -> tuple[ElasticProblem, np.ndarray]:
    """ElasticProblem of the segment and the (θ, z, r) centroids of its cells (box coordinates, before mapping)."""
    m = box(nt, nz, nr, ANGLE, L, t, element=element, origin=(0.0, 0.0, R - t / 2))
    cent = m.cell_centroids()
    P = m.points.copy()
    m.points = np.column_stack([P[:, 2] * np.cos(P[:, 0]), P[:, 2] * np.sin(P[:, 0]), P[:, 1]])
    return ElasticProblem(m, isotropic_stiffness(1.0, NU)), cent


def _solid_load(pr: ElasticProblem, cent: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    """Uniform body force on the patch cells with unit resultant along the patch direction; (F, patch volume)."""
    mask = _in_patch(cent[:, 0], cent[:, 1], kind)
    dV = pr.geom.dV.sum(axis=1)
    vol = float(dV[mask].sum())
    fe = kn.body_load(pr.geom, PATCH[kind][4]) * mask[:, None] / vol
    return assemble_vector(pr.ndof, pr.cell_dofs, fe), vol


@cache
def solid_compliance(nt: int, nz: int, nr: int, t: float, element: str = "T10") -> tuple[float, float]:
    """(c corner, c radial) of the solid; one factorisation for both loads."""
    pr, cent = _solid(nt, nz, nr, t, element)
    F = np.column_stack([_solid_load(pr, cent, k)[0] for k in KINDS])
    bc = Dirichlet.fixed(pr.mesh.dofs(pr.mesh.node_sets["bottom"]), n_cases=2)
    U = pr.solve(F, bc).u
    c = np.einsum("ij,ij->j", F, U)
    return float(c[0]), float(c[1])


def richardson(c1: float, c2: float, c3: float) -> dict[str, float]:
    """Ratio-2 Richardson: p = log2(Δ1/Δ2), c∞ = c3 + Δ2² / (Δ1 − Δ2), ε = |c∞ − c3| / |c∞| (record derivation)."""
    d1, d2 = c2 - c1, c3 - c2
    ok = d1 * d2 > 0 and abs(d2) < abs(d1)
    p = math.log2(d1 / d2) if d1 * d2 > 0 else math.nan
    cinf = c3 + d2 * d2 / (d1 - d2) if d1 != d2 else math.nan
    return {"c1": c1, "c2": c2, "c3": c3, "p": p, "cinf": cinf, "eps": abs(cinf - c3) / abs(cinf),
            "monotone": float(ok)}


@cache
def reference(t: float, kind: str) -> dict[str, float]:
    i = KINDS.index(kind)
    return richardson(*(solid_compliance(nt, nz, 1, t)[i] for nt, nz in LEVELS))


# ---------------------------------------------------------------------------------------------- shell
def _shell(nt: int, nz: int, t: float, drill: float) -> FlatShell:
    m = cylinder_surface(R, L, nt, nz, angle=ANGLE)
    fixed = (6 * np.arange(nt + 1)[:, None] + np.arange(6)).ravel()
    return FlatShell(m, t, NU, fixed, drill=drill)


def shell_load(sh: FlatShell, kind: str) -> np.ndarray:
    """Patch facets (centroid inside) get A_e / A_P · d, a quarter per corner; resultant exactly d."""
    X = sh.mesh.nodes[sh.mesh.facets].mean(axis=1)
    mask = _in_patch(np.arctan2(X[:, 1], X[:, 0]), X[:, 2], kind)
    A = sh.sizes[:, 0] * sh.sizes[:, 1]
    w = np.where(mask, A, 0.0) / A[mask].sum()
    F = np.zeros(sh.ndof)
    for c in range(4):
        for d in range(3):
            np.add.at(F, 6 * sh.mesh.facets[:, c] + d, w / 4 * PATCH[kind][4][d])
    return F


@cache
def shell_compliance(nt: int, nz: int, t: float, drill: float) -> tuple[float, float]:
    sh = _shell(nt, nz, t, drill)
    E = np.ones(sh.mesh.ne)
    out = []
    for k in KINDS:
        F = shell_load(sh, k)
        out.append(float(F @ sh.solve(E, F).numpy()))
    return out[0], out[1]


def _shell_fine(t: float, kind: str, drill: float) -> float:
    return shell_compliance(*SHELL_LEVELS[-1], t, drill)[KINDS.index(kind)]


def _shell_mesh_change(t: float, kind: str, drill: float) -> float:
    i = KINDS.index(kind)
    c64, c128 = shell_compliance(64, 48, t, drill)[i], shell_compliance(128, 96, t, drill)[i]
    return abs(c128 - c64) / c128


def decision() -> dict[str, object]:
    """The pre-registered rule: E(α) = max over the four cases of |c_shell(α, 128 × 96) / c∞ − 1|."""
    E = {a: max(abs(_shell_fine(t, k, a) / reference(t, k)["cinf"] - 1) for t, k in CASES) for a in DRILLS}
    best = min(DRILLS, key=lambda a: (E[a], abs(math.log10(a) + 3)))
    keep = E[1e-3] <= 0.03 and E[1e-3] - E[best] <= 0.01
    return {"E": E, "best": best, "alpha": 1e-3 if keep else best}


# ---------------------------------------------------------------------------------------------- DB0 (fast)
@pytest.mark.parametrize("t", [0.01, 0.05])
def test_db0_setup(t: float) -> None:
    pr, cent = _solid(16, 12, 1, t)                          # the Jacobian check runs in the geometry
    vol = float(pr.geom.dV.sum())
    rel_v = vol / (ANGLE * R * t * L) - 1
    worst = 0.0
    rel_p = {}
    for k in KINDS:
        F, vp = _solid_load(pr, cent, k)
        rel_p[k] = vp / (PATCH_VOLUME * t) - 1
        worst = max(worst, float(np.abs(F.reshape(-1, 3).sum(axis=0) - PATCH[k][4]).max()))
        sh = _shell(16, 12, t, 1e-3)
        Fs = shell_load(sh, k).reshape(-1, 6)
        worst = max(worst, float(np.abs(Fs[:, :3].sum(axis=0) - PATCH[k][4]).max()), float(np.abs(Fs[:, 3:]).max()))
    print(f"DB0 t = {t}: volume {rel_v:+.3e}, patch volume corner {rel_p['corner']:+.3e}, radial "
          f"{rel_p['radial']:+.3e}, resultant error {worst:.1e}")
    assert abs(rel_v) <= 1e-4 and all(abs(v) <= 1e-4 for v in rel_p.values()) and worst <= 1e-12


# ---------------------------------------------------------------------------------------------- DB1-DB5 (slow)
@pytest.mark.slow
@pytest.mark.parametrize("t, kind", CASES)
def test_db1_reference_converges(t: float, kind: str) -> None:
    r = reference(t, kind)
    print(f"DB1 t = {t}, {kind}: c 16x12 {r['c1']:.6e}, 32x24 {r['c2']:.6e}, 64x48 {r['c3']:.6e}; p {r['p']:.3f}, "
          f"c_inf {r['cinf']:.6e}, eps {r['eps']:.3e}, monotone {bool(r['monotone'])}")
    assert r["monotone"] and r["eps"] <= 0.01


@pytest.mark.slow
@pytest.mark.parametrize("t, kind", CASES)
def test_db2_through_thickness(t: float, kind: str) -> None:
    i = KINDS.index(kind)
    c1, c2 = solid_compliance(32, 24, 1, t)[i], solid_compliance(32, 24, 2, t)[i]
    print(f"DB2 t = {t}, {kind}: nr = 1 {c1:.6e}, nr = 2 {c2:.6e}, rel {abs(c2 - c1) / c1:.3e}")
    assert abs(c2 - c1) / c1 <= 0.005


@pytest.mark.slow
@pytest.mark.parametrize("t, kind", CASES)
def test_db3_shell_mesh_convergence(t: float, kind: str) -> None:
    rel = {a: _shell_mesh_change(t, kind, a) for a in DRILLS}
    print(f"DB3 t = {t}, {kind}: " + ", ".join(f"{a:g}: {v:.3e}" for a, v in rel.items()))
    assert max(rel.values()) <= 0.01


@pytest.mark.slow
def test_db4_default_drill_follows_the_rule() -> None:
    refs_ok = all(reference(t, k)["monotone"] and reference(t, k)["eps"] <= 0.01 for t, k in CASES)
    thick_ok = all(abs(solid_compliance(32, 24, 2, t)[KINDS.index(k)] / solid_compliance(32, 24, 1, t)[KINDS.index(k)]
                       - 1) <= 0.005 for t, k in CASES)
    shell_ok = all(_shell_mesh_change(t, k, a) <= 0.01 for t, k in CASES for a in DRILLS)
    d = decision()
    alpha = d["alpha"] if (refs_ok and thick_ok and shell_ok) else 1e-3
    E = d["E"]
    assert isinstance(E, dict)
    print(f"DB4: preconditions DB1 {refs_ok}, DB2 {thick_ok}, DB3 {shell_ok}; E: "
          + ", ".join(f"{a:g}: {v:.4%}" for a, v in E.items()) + f"; argmin {d['best']:g}; rule -> {alpha:g}; "
          f"DRILL = {shell_module.DRILL:g}")
    assert alpha == shell_module.DRILL


@pytest.mark.slow
def test_db5_record() -> None:
    """Record (no threshold): the shell-vs-reference table, the shell's own Richardson value and H8 locking (H8 on
    64 × 48 × 2 only: the pre-registered 128 × 96 × 2 ran out of the 900 s test timeout in SuperLU on the first run)."""
    for t, k in CASES:
        i = KINDS.index(k)
        ref = reference(t, k)["cinf"]
        for a in DRILLS:
            cs = [shell_compliance(nt, nz, t, a)[i] for nt, nz in SHELL_LEVELS]
            rs = richardson(*cs[1:])
            print(f"DB5 t = {t}, {k}, drill {a:g}: " + ", ".join(f"{nt}x{nz} {c:.6e}" for (nt, nz), c in
                                                                 zip(SHELL_LEVELS, cs, strict=True))
                  + f"; d(128x96) {cs[-1] / ref - 1:+.4%}; shell Richardson c_inf {rs['cinf']:.6e} "
                  f"(p {rs['p']:.2f}), d {rs['cinf'] / ref - 1:+.4%}")
    for t in (0.01, 0.05):
        for nt, nz in ((64, 48),):
            h8 = solid_compliance(nt, nz, 2, t, "H8")
            print(f"DB5 H8 {nt}x{nz}x2 t = {t}: " + ", ".join(
                f"{k} {h8[i]:.6e} (c / c_inf {h8[i] / reference(t, k)['cinf']:.4f})" for i, k in enumerate(KINDS)))
            assert all(np.isfinite(h8))
