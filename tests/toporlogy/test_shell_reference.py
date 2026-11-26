"""Drill-independent 3-D solid reference for the curved shell segment of DR (docs/olcumler/kabuk_onkayit.md, DB0-DB5,
DL0-DL6).

R = 1, 90°, L = 1 segment clamped at z = 0, t = 0.01 and 0.05, a unit force spread over the same patch at the shell's
mid-surface load point in both models (corner patch, direction +z; free-edge middle patch, direction −e_r(π/4)).
Solid: T10 (quadratic tetrahedra; ``fem.mesh`` has no H20 / H27 and no incompatible-mode hexahedron off the grid),
box mesh in (θ, z, r) mapped onto the cylinder node by node, so the edges are isoparametric arcs.

DL: the through-thickness (layer) error of the one-layer reference estimated by Richardson in the layer count and
corrected for; the layered meshes are solved by the supernodal Cholesky in a nested dissection whose separators are
even (element-boundary) node planes of the quadratic node grid (an odd plane does not separate T10 cells).
"""
from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
import scipy.sparse as sp

from toporlogy.core.linear_elastic import isotropic_stiffness
from toporlogy.fem import shell as shell_module
from toporlogy.fem.elastic import checked_residual
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, box
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.assembly import assemble_vector
from toporlogy.fem.shell import FlatShell, cylinder_surface
from toporlogy.solvers import DirectSolver

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


# ---------------------------------------------------------------------------------------------- DL: layer correction
DL_DRILLS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
LAYERS = {(16, 12): (1, 2, 4, 8), (32, 24): (1, 2, 4)}
DL_SHELL_LEVELS = ((32, 24), (64, 48), (128, 96))
NEGLIGIBLE = 1e-5                  # a layer sequence with |Δ1|, |Δ2| ≤ 1e-5 c(N, 1) has no Richardson to speak of
GRID = 41                          # perturbation points per case in the separability check (ends included)
LEAF = 64                          # nodes per leaf box of the nested dissection (as toporlogy.solvers.ordering)
FREE_GB_64X48X2 = 3.8              # pre-registered: run 64 × 48 × 2 only with at least this much free memory


def even_dissection(node_shape: tuple[int, ...], leaf: int = LEAF) -> list[np.ndarray]:
    """Nested dissection of a quadratic-element node grid (C order): leaf boxes and separator planes in elimination
    order; a separator is always an even node plane (an element boundary), the one nearest the box centre along the
    longest axis that has one."""
    dims = tuple(int(n) for n in node_shape)
    strides = [int(np.prod(dims[i + 1:], dtype=np.int64)) for i in range(len(dims))]
    parts: list[np.ndarray] = []

    def nodes(lo: list[int], hi: list[int]) -> np.ndarray:
        g = np.meshgrid(*[np.arange(a, b, dtype=np.int64) for a, b in zip(lo, hi, strict=True)], indexing="ij")
        return np.asarray(sum(x.ravel() * s for x, s in zip(g, strides, strict=True)), dtype=np.int64)

    stack: list[tuple[list[int], list[int], bool]] = [([0] * len(dims), list(dims), False)]
    while stack:
        lo, hi, emit = stack.pop()
        ext = [b - a for a, b in zip(lo, hi, strict=True)]
        if min(ext) <= 0:
            continue
        if emit or int(np.prod(ext)) <= leaf:
            parts.append(nodes(lo, hi))
            continue
        for a in np.argsort(ext, kind="stable")[::-1]:
            ax = int(a)
            cand = [g for g in range(lo[ax] + 1, hi[ax] - 1) if g % 2 == 0]
            if cand:
                centre = (lo[ax] + hi[ax] - 1) / 2
                mid = min(cand, key=lambda g: abs(g - centre))
                hi1, lo2, los, his = list(hi), list(lo), list(lo), list(hi)
                hi1[ax], lo2[ax], los[ax], his[ax] = mid, mid + 1, mid, mid + 1
                stack += [(los, his, True), (lo2, hi, False), (lo, hi1, False)]   # halves first, separator last
                break
        else:
            parts.append(nodes(lo, hi))
    return parts


def even_blocks(node_shape: tuple[int, ...], free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(order, block sizes) of the reduced system for ``DirectSolver("cholesky")``; DOF 3 · node + d."""
    parts = even_dissection(node_shape)
    allnodes = np.concatenate(parts)
    n_total = 3 * int(np.prod(node_shape))
    assert np.array_equal(np.sort(allnodes), np.arange(n_total // 3))
    dofs = (3 * allnodes[:, None] + np.arange(3)).ravel()
    pos = np.full(n_total, -1, np.int64)
    pos[free] = np.arange(len(free))
    p = pos[dofs]
    keep = p >= 0
    part_of = np.repeat(np.arange(len(parts)), [3 * len(q) for q in parts])
    return p[keep], np.bincount(part_of[keep], minlength=len(parts)).astype(np.int64)


@cache
def layered_compliance(nt: int, nz: int, nr: int, t: float) -> tuple[float, float]:
    """(c corner, c radial) of the T10 solid with nr layers, supernodal Cholesky in the even nested dissection."""
    pr, cent = _solid(nt, nz, nr, t)
    F = np.column_stack([_solid_load(pr, cent, k)[0] for k in KINDS])
    free = np.setdiff1d(np.arange(pr.ndof), pr.mesh.dofs(pr.mesh.node_sets["bottom"]))
    Kff = sp.csr_matrix(pr.K)[free][:, free]
    order, blocks = even_blocks((2 * nt + 1, 2 * nz + 1, 2 * nr + 1), free)
    ds = DirectSolver("cholesky")
    upper = sp.triu(Kff, format="csr")
    ds.factor(upper, order=order, blocks=blocks)
    X = ds.solve(F[free])
    ds.release()
    checked_residual(upper, X, F[free])          # the library's check (relative residual or backward error)
    c = np.einsum("ij,ij->j", F[free], X)
    return float(c[0]), float(c[1])


def layer_extrapolation(c1: float, c2: float, c4: float) -> dict[str, float]:
    """Richardson in the layer count (1, 2, 4): c_L, p_L, r = c_L / c1 − 1, e_L = |c_L − c4| / c1 (pre-registered)."""
    d1, d2 = c2 - c1, c4 - c2
    negligible = max(abs(d1), abs(d2)) <= NEGLIGIBLE * c1
    if negligible:
        cl, el, p, ok = c4, abs(d2) / c1 + NEGLIGIBLE, math.nan, True
    else:
        ok = d1 * d2 > 0 and abs(d2) < abs(d1)
        cl = c4 + d2 * d2 / (d1 - d2) if d1 != d2 else math.nan
        el = abs(cl - c4) / c1
        p = math.log2(d1 / d2) if d1 * d2 > 0 else math.nan
    return {"c1": c1, "c2": c2, "c4": c4, "cL": cl, "p": p, "r": cl / c1 - 1, "eL": el, "ok": float(ok),
            "negligible": float(negligible)}


@cache
def layers(nt: int, nz: int, t: float, kind: str) -> dict[str, float]:
    i = KINDS.index(kind)
    return layer_extrapolation(*(layered_compliance(nt, nz, n, t)[i] for n in (1, 2, 4)))


@cache
def corrected(t: float, kind: str) -> dict[str, float]:
    """c_D = c_∞ (1 + r(32 × 24)) and its budget B = ε_in + e_L(32 × 24) + |r(32 × 24) − r(16 × 12)|."""
    ref, l32, l16 = reference(t, kind), layers(32, 24, t, kind), layers(16, 12, t, kind)
    m = abs(l32["r"] - l16["r"])
    return {"cinf": ref["cinf"], "cD": ref["cinf"] * (1 + l32["r"]), "eps": ref["eps"], "eL": l32["eL"], "m": m,
            "B": ref["eps"] + l32["eL"] + m}


def _dl_shell(t: float, kind: str, drill: float, variant: str) -> float:
    i = KINDS.index(kind)
    cs = [shell_compliance(nt, nz, t, drill)[i] for nt, nz in DL_SHELL_LEVELS]
    if variant == "128":
        return cs[-1]
    r = richardson(*cs)
    return r["cinf"] if r["monotone"] else cs[-1]


def _dl2(t: float, kind: str) -> tuple[float, float]:
    lay = layers(16, 12, t, kind)
    c8 = layered_compliance(16, 12, 8, t)[KINDS.index(kind)]
    return abs(c8 - lay["cL"]) / lay["c1"], max(lay["eL"], 1e-6)


def _dl3(t: float, kind: str) -> tuple[float, float]:
    r32, r16 = layers(32, 24, t, kind)["r"], layers(16, 12, t, kind)["r"]
    return abs(r32 - r16), max(0.25 * abs(r32), 1e-4)


def dl_preconditions() -> dict[str, bool]:
    return {
        "DB1": all(reference(t, k)["monotone"] and reference(t, k)["eps"] <= 0.01 for t, k in CASES),
        "DL1": all(layers(nt, nz, t, k)["ok"] for nt, nz in LAYERS for t, k in CASES),
        "DL2": all(_dl2(t, k)[0] <= _dl2(t, k)[1] for t, k in CASES),
        "DL3": all(_dl3(t, k)[0] <= _dl3(t, k)[1] for t, k in CASES),
        "DL4": all(_shell_mesh_change(t, k, a) <= 0.01 for t, k in CASES for a in DL_DRILLS),
    }


def dl_errors(variant: str = "128") -> dict[float, float]:
    """E(α) = max_k |c_shell(α, k) / c_D(k) − 1|."""
    return {a: max(abs(_dl_shell(t, k, a, variant) / corrected(t, k)["cD"] - 1) for t, k in CASES)
            for a in DL_DRILLS}


def dl_difference_range(a: float, b: float, variant: str) -> tuple[float, float]:
    """min and max of E(a) − E(b) over the grid c_D(k)(1 + s_k B(k)), s_k ∈ linspace(−1, 1, GRID), all cases."""
    s = np.linspace(-1.0, 1.0, GRID)
    Ea: np.ndarray = np.zeros((1,) * len(CASES))
    Eb: np.ndarray = np.zeros((1,) * len(CASES))
    for j, (t, k) in enumerate(CASES):
        cor = corrected(t, k)
        ref = cor["cD"] * (1 + s * cor["B"])
        shape = [1] * len(CASES)
        shape[j] = GRID
        Ea = np.maximum(Ea, np.abs(_dl_shell(t, k, a, variant) / ref - 1).reshape(shape))
        Eb = np.maximum(Eb, np.abs(_dl_shell(t, k, b, variant) / ref - 1).reshape(shape))
    diff = Ea - Eb
    return float(diff.min()), float(diff.max())


def dl_separable(a: float, b: float) -> bool:
    """Pre-registered: the sign of E(a) − E(b) is the same, non-zero, on the whole grid for both shell variants."""
    ranges = [dl_difference_range(a, b, v) for v in ("128", "rich")]
    return all(lo > 0 for lo, _ in ranges) or all(hi < 0 for _, hi in ranges)


def dl_decision() -> dict[str, object]:
    pre = dl_preconditions()
    E = dl_errors("128")
    best = min(DL_DRILLS, key=lambda a: (E[a], abs(math.log10(a) + 1)))
    if not all(pre.values()) or best == 1e-1:
        alpha = 1e-1
    else:
        alpha = best if dl_separable(best, 1e-1) else 1e-1
    return {"pre": pre, "E": E, "best": best, "alpha": alpha}


def _failed(check: str, measured: str, limit: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"pre-registered {check} failed: measured {measured} > {limit} "
                                                 "(kabuk_onkayit.md, DL0-DL6)")


# ---------------------------------------------------------------------------------------------- DL0 (fast)
@pytest.mark.parametrize("t", [pytest.param(0.01, marks=_failed("DL0", "c difference 1.2e-9", "1e-10")), 0.05])
def test_dl0_setup(t: float) -> None:
    nt, nz, nr = 16, 12, 2
    m = box(nt, nz, nr, ANGLE, L, t, element="T10", origin=(0.0, 0.0, R - t / 2))
    shape = (2 * nt + 1, 2 * nz + 1, 2 * nr + 1)
    grid_err = math.inf
    if len(m.points) == int(np.prod(shape)):
        idx = np.stack(np.unravel_index(np.arange(len(m.points)), shape), axis=1)
        grid = idx * np.array([ANGLE / (2 * nt), L / (2 * nz), t / (2 * nr)]) + np.array([0.0, 0.0, R - t / 2])
        grid_err = float(np.abs(m.points - grid).max())
    pr, cent = _solid(nt, nz, nr, t)
    F = np.column_stack([_solid_load(pr, cent, k)[0] for k in KINDS])
    U = pr.solve(F, Dirichlet.fixed(pr.mesh.dofs(pr.mesh.node_sets["bottom"]), n_cases=2)).u
    c_default = np.einsum("ij,ij->j", F, U)
    c_chol = np.array(layered_compliance(nt, nz, nr, t))
    rel = float(np.abs(c_chol / c_default - 1).max())
    print(f"DL0 t = {t}: grid {grid_err:.1e}; c default {c_default[0]:.10e} / {c_default[1]:.10e}, Cholesky "
          f"{c_chol[0]:.10e} / {c_chol[1]:.10e}, rel {rel:.1e}")
    assert grid_err <= 1e-12 and rel <= 1e-10


# ---------------------------------------------------------------------------------------------- DL1-DL6 (slow)
@pytest.mark.slow
@pytest.mark.parametrize("mesh", list(LAYERS), ids=lambda m: f"{m[0]}x{m[1]}")
@pytest.mark.parametrize("t, kind", CASES)
def test_dl1_layer_convergence(t: float, kind: str, mesh: tuple[int, int]) -> None:
    lay = layers(*mesh, t, kind)
    print(f"DL1 {mesh[0]}x{mesh[1]} t = {t}, {kind}: c n=1 {lay['c1']:.8e}, n=2 {lay['c2']:.8e}, n=4 {lay['c4']:.8e}; "
          f"p_L {lay['p']:.3f}, c_L {lay['cL']:.8e}, r {lay['r']:+.4e}, e_L {lay['eL']:.3e}, "
          f"negligible {bool(lay['negligible'])}, ok {bool(lay['ok'])}")
    assert lay["ok"]


@pytest.mark.slow
@pytest.mark.parametrize("t, kind", CASES)
def test_dl2_layer_extrapolation_predicts_eight_layers(t: float, kind: str) -> None:
    v, lim = _dl2(t, kind)
    c8 = layered_compliance(16, 12, 8, t)[KINDS.index(kind)]
    print(f"DL2 t = {t}, {kind}: c(16x12, 8) {c8:.8e}, c_L {layers(16, 12, t, kind)['cL']:.8e}, |diff| / c1 {v:.3e}, "
          f"limit {lim:.3e}")
    assert v <= lim


@pytest.mark.slow
@pytest.mark.parametrize("t, kind", [
    (0.01, "corner"),
    pytest.param(0.01, "radial", marks=_failed("DL3", "|r32 - r16| 2.234e-3", "4.248e-4")),
    pytest.param(0.05, "corner", marks=_failed("DL3", "|r32 - r16| 1.115e-4", "1.000e-4")),
    pytest.param(0.05, "radial", marks=_failed("DL3", "|r32 - r16| 1.859e-3", "9.188e-4")),
])
def test_dl3_layer_correction_independent_of_mesh(t: float, kind: str) -> None:
    v, lim = _dl3(t, kind)
    print(f"DL3 t = {t}, {kind}: r(16x12) {layers(16, 12, t, kind)['r']:+.4e}, r(32x24) "
          f"{layers(32, 24, t, kind)['r']:+.4e}, |diff| {v:.3e}, limit {lim:.3e}")
    assert v <= lim


@pytest.mark.slow
@pytest.mark.parametrize("t, kind", CASES)
def test_dl4_shell_mesh_convergence(t: float, kind: str) -> None:
    rel = {a: _shell_mesh_change(t, kind, a) for a in DL_DRILLS}
    print(f"DL4 t = {t}, {kind}: " + ", ".join(f"{a:g}: {v:.3e}" for a, v in rel.items()))
    assert max(rel.values()) <= 0.01


@pytest.mark.slow
def test_dl5_default_drill_follows_the_rule() -> None:
    d = dl_decision()
    E = d["E"]
    assert isinstance(E, dict)
    sep = {a: dl_separable(a, 1e-1) for a in DL_DRILLS if a != 1e-1}
    print(f"DL5: preconditions {d['pre']}; E: " + ", ".join(f"{a:g}: {v:.4%}" for a, v in E.items())
          + f"; argmin {d['best']:g}; separable from 1e-1: " + ", ".join(f"{a:g} {s}" for a, s in sep.items())
          + f"; rule -> {d['alpha']:g}; DRILL = {shell_module.DRILL:g}")
    assert d["alpha"] == shell_module.DRILL


@pytest.mark.slow
def test_dl6_record() -> None:
    """Record (no threshold): layer table, corrected reference with its budget, d(α, k) and E(α) for both shell
    variants, the separability range of E(1e-2) − E(1e-1) and the simple comparison with max_k B(k)."""
    for (nt, nz), ns in LAYERS.items():
        for t in (0.01, 0.05):
            cs = [layered_compliance(nt, nz, n, t) for n in ns]
            for i, k in enumerate(KINDS):
                print(f"DL6 layers {nt}x{nz} t = {t}, {k}: "
                      + ", ".join(f"n={n} {c[i]:.8e}" for n, c in zip(ns, cs, strict=True))
                      + "; rel to n=1: " + ", ".join(f"{c[i] / cs[0][i] - 1:+.4e}" for c in cs[1:]))
    for t, k in CASES:
        cor = corrected(t, k)
        print(f"DL6 reference t = {t}, {k}: c_inf {cor['cinf']:.8e}, c_D {cor['cD']:.8e} "
              f"(r {cor['cD'] / cor['cinf'] - 1:+.4e}); budget B {cor['B']:.3e} = eps_in {cor['eps']:.3e} + "
              f"e_L {cor['eL']:.3e} + m {cor['m']:.3e}")
    for v in ("128", "rich"):
        for t, k in CASES:
            print(f"DL6 d [{v}] t = {t}, {k}: " + ", ".join(
                f"{a:g} {_dl_shell(t, k, a, v) / corrected(t, k)['cD'] - 1:+.4%}" for a in DL_DRILLS))
        E = dl_errors(v)
        print(f"DL6 E [{v}]: " + ", ".join(f"{a:g} {e:.4%}" for a, e in E.items()))
        lo, hi = dl_difference_range(1e-2, 1e-1, v)
        print(f"DL6 E(1e-2) - E(1e-1) [{v}]: nominal {E[1e-2] - E[1e-1]:+.4%}, over the budget grid "
              f"[{lo:+.4%}, {hi:+.4%}]")
    for t, k in CASES:
        i = KINDS.index(k)
        print(f"DL6 shell t = {t}, {k}: " + "; ".join(
            f"{a:g}: " + ", ".join(f"{nt}x{nz} {shell_compliance(nt, nz, t, a)[i]:.8e}" for nt, nz in DL_SHELL_LEVELS)
            for a in DL_DRILLS))
    E = dl_errors("128")
    b_max = max(corrected(t, k)["B"] for t, k in CASES)
    print(f"DL6 simple comparison: |E(1e-2) - E(1e-1)| {abs(E[1e-2] - E[1e-1]):.4%} vs max_k B {b_max:.4%}; "
          f"separable(1e-2, 1e-1) {dl_separable(1e-2, 1e-1)}")


def _free_memory_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.virtual_memory().available) / 2**30


@pytest.mark.slow
def test_dl6_record_two_layers_64x48() -> None:
    """Conditional record (pre-registered): 64 × 48 × 2 only with ≥ 3.8 GB free memory at the start; run alone."""
    free = _free_memory_gb()
    if free is None or free < FREE_GB_64X48X2:
        print(f"DL6 64x48x2: not run, free memory {free} GB < {FREE_GB_64X48X2} GB")
        return
    for t in (0.01, 0.05):
        r2 = {}
        for nt, nz in ((16, 12), (32, 24), (64, 48)):
            c1, c2 = layered_compliance(nt, nz, 1, t), layered_compliance(nt, nz, 2, t)
            r2[(nt, nz)] = [c2[i] / c1[i] - 1 for i in range(2)]
        print(f"DL6 64x48x2 t = {t} (free {free:.2f} GB): " + "; ".join(
            f"{k} r2 " + ", ".join(f"{nt}x{nz} {r[i]:+.4e}" for (nt, nz), r in r2.items())
            for i, k in enumerate(KINDS)))
