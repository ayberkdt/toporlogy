"""Triangle surfaces: iso-surface extraction, topology and manifold checks, quality, decimation, comparison,
mirror-symmetry distance, subdivision.

A surface is a pair ``(V (n, 3) float, F (m, 3) int)`` with outward (counter-clockwise seen from outside) triangles.
Extraction uses VTK flying edges (``pyvista.ImageData.contour(method="flying_edges")``; Schroeder, Maynard & Geveci
2015, the marching-cubes family of Lorensen & Cline 1987 — sources not read in this session).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from toporlogy.references import cite

__all__ = ["extract", "to_polydata", "from_polydata", "edge_report", "components", "mesh_topology", "enclosed_volume",
           "area", "face_normals", "vertex_normals", "quality", "clean", "decimate", "flip_edges", "compare",
           "mirror_distance", "is_watertight", "subdivide"]


def to_polydata(V: np.ndarray, F: np.ndarray) -> Any:
    import pyvista as pv
    F = np.asarray(F, np.int64)
    faces = np.hstack([np.full((len(F), 1), 3, np.int64), F]).ravel()
    return pv.PolyData(np.asarray(V, float), faces)


def from_polydata(pd: Any) -> tuple[np.ndarray, np.ndarray]:
    pd = pd.triangulate()
    F = np.asarray(pd.faces).reshape(-1, 4)[:, 1:].astype(np.int64) if pd.n_cells else np.zeros((0, 3), np.int64)
    return np.asarray(pd.points, float).copy(), F


def enclosed_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Signed enclosed volume (divergence theorem); positive for outward triangles."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


enclosed_volume = cite("mirtich_1996", note="volume by the divergence theorem over the triangles")(enclosed_volume)


def face_normals(V: np.ndarray, F: np.ndarray, unit: bool = True) -> np.ndarray:
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if unit:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    return n


def area(V: np.ndarray, F: np.ndarray, per_face: bool = False) -> Any:
    a = 0.5 * np.linalg.norm(face_normals(V, F, unit=False), axis=1)
    return a if per_face else float(a.sum())


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    n = face_normals(V, F, unit=False)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def clean(V: np.ndarray, F: np.ndarray, tol: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate points (absolute ``tol``), drop degenerate (repeated-vertex or zero-area) triangles and unused
    points."""
    if tol > 0:
        pd = to_polydata(V, F).clean(tolerance=tol, absolute=True)
        V, F = from_polydata(pd)
    F = np.asarray(F, np.int64)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    a = area(V, F, per_face=True)
    F = F[a > 1e-14 * max(float(np.max(a, initial=0.0)), 1e-300)]
    used = np.unique(F)
    new = -np.ones(len(V), np.int64)
    new[used] = np.arange(len(used))
    return V[used], new[F]


def extract(phi: np.ndarray, grid: Any, level: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Iso-surface φ = level (flying edges), outward oriented, merged vertices."""
    im = grid.to_image(phi)
    srf = im.contour([level], scalars="phi", method="flying_edges")
    if srf.n_cells == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    V, F = from_polydata(srf.clean())
    V, F = clean(V, F)
    if enclosed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


@cite("botsch_kobbelt_2010", note="manifold and orientation tests from the edge incidences")
def edge_report(F: np.ndarray, n_vertices: int | None = None) -> dict[str, int]:
    """Open edges (1 triangle), non-manifold edges (> 2), edges whose two triangles disagree in orientation and
    non-manifold vertices (vertex star not one fan). A closed manifold surface has zero of each."""
    F = np.asarray(F, np.int64)
    if len(F) == 0:
        return {"edges": 0, "open": 0, "non_manifold": 0, "misoriented": 0, "non_manifold_vertices": 0}
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(d, axis=1)
    key, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    # orientation: a manifold edge must be used once in each direction
    fwd = (d[:, 0] < d[:, 1]).astype(np.int64)
    nf = np.bincount(inv, weights=fwd, minlength=len(key))
    two = cnt == 2
    mis = int((two & (nf != 1)).sum())
    # non-manifold vertices: number of faces around v vs. size of the fan reachable through manifold edges
    nv = int(n_vertices if n_vertices is not None else F.max() + 1)
    nmv = _non_manifold_vertices(F, nv)
    return {"edges": int(len(key)), "open": int((cnt == 1).sum()), "non_manifold": int((cnt > 2).sum()),
            "misoriented": mis, "non_manifold_vertices": nmv}


def _non_manifold_vertices(F: np.ndarray, nv: int) -> int:
    """Vertices whose incident faces form more than one edge-connected fan."""
    m = len(F)
    e = np.concatenate([np.sort(F[:, [0, 1]], 1), np.sort(F[:, [1, 2]], 1), np.sort(F[:, [2, 0]], 1)])
    fid = np.tile(np.arange(m), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    es, fs = e[order], fid[order]
    same = np.all(es[1:] == es[:-1], axis=1)
    pa, pb, ev = fs[:-1][same], fs[1:][same], es[:-1][same]

    def corner(f: np.ndarray, v: np.ndarray) -> np.ndarray:          # corner id = 3·face + local index of v
        return f * 3 + np.argmax(F[f] == v[:, None], axis=1)

    # two faces sharing an edge link their corners at both edge end points
    ra = np.concatenate([corner(pa, ev[:, 0]), corner(pa, ev[:, 1])])
    rb = np.concatenate([corner(pb, ev[:, 0]), corner(pb, ev[:, 1])])
    G = sp.coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(3 * m, 3 * m))
    _, lab = csgraph.connected_components(G, directed=False)
    pairs = np.unique(np.stack([F.ravel(), lab], axis=1), axis=0)
    fans = np.bincount(pairs[:, 0], minlength=nv)
    return int((fans > 1).sum())


def components(F: np.ndarray, n_vertices: int) -> tuple[int, np.ndarray]:
    """Edge-connected triangle components: (count, label per face)."""
    F = np.asarray(F, np.int64)
    m = len(F)
    if m == 0:
        return 0, np.zeros(0, np.int64)
    e = np.concatenate([np.sort(F[:, [0, 1]], 1), np.sort(F[:, [1, 2]], 1), np.sort(F[:, [2, 0]], 1)])
    fid = np.tile(np.arange(m), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    es, fs = e[order], fid[order]
    same = np.all(es[1:] == es[:-1], axis=1)
    G = sp.coo_matrix((np.ones(int(same.sum())), (fs[:-1][same], fs[1:][same])), shape=(m, m))
    n, lab = csgraph.connected_components(G, directed=False)
    return int(n), lab


def mesh_topology(V: np.ndarray, F: np.ndarray) -> dict[str, Any]:
    """Shells, bodies (outer shells), cavities (inner shells), χ and genus per shell, solid χ = Σχ/2."""
    n, lab = components(F, len(V))
    shells = []
    for c in range(n):
        Fc = F[lab == c]
        vs = np.unique(Fc)
        e = np.unique(np.sort(np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]]), 1), axis=0)
        chi = len(vs) - len(e) + len(Fc)
        vol = enclosed_volume(V, Fc)
        shells.append({"faces": int(len(Fc)), "euler": int(chi), "genus": (2 - chi) / 2, "volume": vol})
    outer = [s for s in shells if s["volume"] > 0]
    inner = [s for s in shells if s["volume"] <= 0]
    chi_solid = sum(s["euler"] for s in shells) / 2
    return {"shells": len(shells), "bodies": len(outer), "cavities": len(inner), "euler": chi_solid,
            "tunnels": float(sum(s["genus"] for s in outer)), "shell_list": shells}


def is_watertight(F: np.ndarray, n_vertices: int | None = None) -> bool:
    r = edge_report(F, n_vertices)
    return r["open"] == 0 and r["non_manifold"] == 0 and r["misoriented"] == 0 and r["non_manifold_vertices"] == 0


def quality(V: np.ndarray, F: np.ndarray) -> dict[str, float]:
    """Triangle quality: minimum angle (deg) and aspect ratio R/(2r) (1 = equilateral) statistics."""
    if len(F) == 0:
        return {}
    P = V[F]
    L = np.stack([np.linalg.norm(P[:, 1] - P[:, 2], axis=1), np.linalg.norm(P[:, 2] - P[:, 0], axis=1),
                  np.linalg.norm(P[:, 0] - P[:, 1], axis=1)], axis=1)
    A = area(V, F, per_face=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = []
        for i in range(3):
            a, b, c = L[:, i], L[:, (i + 1) % 3], L[:, (i + 2) % 3]
            cosang.append(np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1))
        ang = np.degrees(np.arccos(np.stack(cosang, axis=1)))
        s = L.sum(axis=1) / 2
        r_in = A / s
        R_c = L.prod(axis=1) / (4 * A)
        ar = R_c / (2 * r_in)
    amin = np.nan_to_num(ang.min(axis=1), nan=0.0)
    ar = np.where(np.isfinite(ar), ar, 1e9)
    return {"triangles": int(len(F)), "min_angle_deg": float(amin.min()),
            "min_angle_p1_deg": float(np.percentile(amin, 1)),
            "min_angle_mean_deg": float(amin.mean()), "share_min_angle_below_10deg": float(np.mean(amin < 10)),
            "aspect_max": float(ar.max()), "aspect_p99": float(np.percentile(ar, 99)), "aspect_mean": float(ar.mean()),
            "edge_mean": float(L.mean()), "edge_min": float(L.min()), "edge_max": float(L.max()),
            "degenerate": int((1e-12 * max(float(A.max()), 1e-300) >= A).sum())}


def decimate(V: np.ndarray, F: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reduce to about ``target_faces`` triangles, then Delaunay edge flips (:func:`flip_edges`).

    VTK quadric decimation with volume preservation (Garland-Heckbert error quadrics, as implemented by
    ``vtkQuadricDecimation`` — paper not read in this session); if that result is not a closed manifold of the same
    topology, VTK decimate-pro (topology preserved) is tried; if both fail the input is returned. Measured on a
    61 k-triangle cantilever surface (14 Nov 2026): decimate-pro at 88 % created 4 non-manifold edges, quadric kept
    a closed manifold of the same χ with −0.0 % volume change.
    """
    info: dict[str, Any] = {"requested": int(target_faces), "before": int(len(F)), "applied": False}
    if target_faces <= 0 or len(F) <= target_faces:
        return V, F, info
    red = 1.0 - target_faces / len(F)
    t0 = mesh_topology(V, F)
    pd = to_polydata(V, F)
    tries = [("vtkQuadricDecimation(volume_preservation)", lambda: pd.decimate(red, volume_preservation=True)),
             ("vtkDecimatePro(preserve_topology)",
              lambda: pd.decimate_pro(red, feature_angle=60.0, preserve_topology=True, splitting=False,
                                      boundary_vertex_deletion=False))]
    for name, fn in tries:
        V2, F2 = clean(*from_polydata(fn()))
        if not len(F2):
            continue
        if enclosed_volume(V2, F2) < 0:
            F2 = F2[:, ::-1].copy()
        t1 = mesh_topology(V2, F2)
        if is_watertight(F2, len(V2)) and t0["bodies"] == t1["bodies"] and t0["euler"] == t1["euler"]:
            F2, nflip = flip_edges(V2, F2)
            info.update({"applied": True, "after": int(len(F2)), "method": name, "edge_flips": nflip})
            return V2, F2, info
        info.setdefault("rejected", []).append(name)
    info["reason"] = "no decimation kept a closed manifold of the same topology; input kept"
    return V, F, info


@cite("botsch_kobbelt_2010", note="edge flips towards a Delaunay triangulation on near-planar edge stars")
def flip_edges(V: np.ndarray, F: np.ndarray, passes: int = 5, max_dihedral_deg: float = 20.0
               ) -> tuple[np.ndarray, int]:
    """Flip interior edges whose two opposite angles sum to more than 180° (Delaunay criterion), only where the two
    triangles are nearly coplanar (dihedral < ``max_dihedral_deg``), the new edge does not exist yet and both end
    points keep valence ≥ 4. Keeps orientation and the closed manifold. Returns (F, number of flips)."""
    faces = [list(map(int, f)) for f in np.asarray(F, np.int64)]
    emap: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(faces):
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            emap.setdefault((min(a, b), max(a, b)), []).append(i)
    val = np.bincount(np.asarray(faces).ravel(), minlength=len(V)).astype(np.int64)   # faces per vertex = valence
    cmax = np.cos(np.radians(max_dihedral_deg))
    nflip = 0

    def normal(f: list[int]) -> np.ndarray:
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        return n / max(float(np.linalg.norm(n)), 1e-300)

    def angle(p: int, q: int, r: int) -> float:              # angle at p in triangle p, q, r
        u, w = V[q] - V[p], V[r] - V[p]
        return float(np.arctan2(np.linalg.norm(np.cross(u, w)), np.dot(u, w)))

    for _ in range(passes):
        changed = 0
        for key in list(emap.keys()):
            fs = emap.get(key)
            if fs is None or len(fs) != 2:
                continue
            a, b = key
            f1, f2 = faces[fs[0]], faces[fs[1]]
            # orient so that f1 contains the directed edge a→b
            if any(f1[k] == b and f1[(k + 1) % 3] == a for k in range(3)):
                fs = [fs[1], fs[0]]
                f1, f2 = f2, f1
            c = next(v for v in f1 if v not in key)
            d = next(v for v in f2 if v not in key)
            if c == d or (min(c, d), max(c, d)) in emap or val[a] <= 3 or val[b] <= 3:
                continue
            if angle(c, a, b) + angle(d, a, b) <= np.pi + 1e-9:
                continue
            if float(normal(f1) @ normal(f2)) < cmax:
                continue
            g1, g2 = [a, d, c], [d, b, c]
            n0 = normal(f1) + normal(f2)
            if float(normal(g1) @ n0) <= 0 or float(normal(g2) @ n0) <= 0:
                continue
            i1, i2 = fs
            faces[i1], faces[i2] = g1, g2
            del emap[key]
            emap[(min(c, d), max(c, d))] = [i1, i2]
            for e, old, new in (((a, d), i2, i1), ((b, c), i1, i2)):
                lst = emap[(min(e), max(e))]
                lst[lst.index(old)] = new
            val[a] -= 1
            val[b] -= 1
            val[c] += 1
            val[d] += 1
            changed += 1
        nflip += changed
        if not changed:
            break
    return np.asarray(faces, np.int64), nflip


def mirror_distance(V: np.ndarray, F: np.ndarray, axis: Any, coordinate: float) -> dict[str, float]:
    """Geometric mirror-symmetry error of a surface S = (V, F) about the plane ``axis = coordinate``.

    With R the reflection, d(q) = dist(R(q), S) for q on S: how far the mirror image of each surface point is from
    the surface. R is an isometry and its own inverse, so dist(R(q), S) = dist(q, R(S)): the distances S → R(S) and
    R(S) → S have the same distribution and the symmetric Hausdorff distance is d_H(S, R(S)) = sup_q d(q). It is 0
    for a mirror-symmetric surface. Samples: the vertices and the face centroids (closest point on the triangles,
    exact; VTK cell locator); ``hausdorff`` is the largest sample (a lower bound of the supremum), ``mean`` and
    ``p95`` are weighted by face area over the centroids. ``bbox_diagonal`` scales the lengths (same unit as V).
    """
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    k = "xyz".index(axis.lower()) if isinstance(axis, str) else int(axis)
    if not 0 <= k < 3:
        raise ValueError(f"axis must be x, y, z (or 0, 1, 2), got {axis!r}")
    C = V[F].mean(axis=1)
    S = np.vstack([V, C])
    S[:, k] = 2.0 * float(coordinate) - S[:, k]
    _, cp = to_polydata(V, F).find_closest_cell(S, return_closest_point=True)
    d = np.linalg.norm(S - cp, axis=1)
    dc = d[len(V):]
    w = area(V, F, per_face=True)
    order = np.argsort(dc)
    cw = np.cumsum(w[order])
    p95 = float(dc[order][min(int(np.searchsorted(cw, 0.95 * cw[-1])), len(dc) - 1)])
    return {"hausdorff": float(d.max()), "mean": float(np.sum(w * dc) / np.sum(w)), "p95": p95,
            "bbox_diagonal": float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))), "samples": int(len(d))}


def compare(V: np.ndarray, F: np.ndarray, V0: np.ndarray, F0: np.ndarray) -> dict[str, float]:
    """Final (V, F) against a reference surface (V0, F0): volume change, one-sided Hausdorff distances (sampled at
    vertices and face centres) and normal deviation (final face normal vs the reference face closest to its centre).
    """
    ref = to_polydata(V0, F0)
    fin = to_polydata(V, F)
    C = V[F].mean(axis=1)
    S = np.vstack([V, C])
    cid, cp = ref.find_closest_cell(S, return_closest_point=True)
    d_fr = np.linalg.norm(S - cp, axis=1)
    C0 = V0[F0].mean(axis=1)
    S0 = np.vstack([V0, C0])
    _, cp0 = fin.find_closest_cell(S0, return_closest_point=True)
    d_rf = np.linalg.norm(S0 - cp0, axis=1)
    n_f = face_normals(V, F)
    n_r = face_normals(V0, F0)[cid[len(V):]]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n_f, n_r), -1, 1)))
    v0, v1 = enclosed_volume(V0, F0), enclosed_volume(V, F)
    return {"volume_ref": v0, "volume": v1, "volume_change_rel": (v1 - v0) / v0 if v0 else float("nan"),
            "hausdorff_final_to_ref": float(d_fr.max()), "hausdorff_ref_to_final": float(d_rf.max()),
            "mean_distance_final_to_ref": float(d_fr.mean()),
            "normal_dev_max_deg": float(ang.max()), "normal_dev_p99_deg": float(np.percentile(ang, 99)),
            "normal_dev_mean_deg": float(ang.mean())}


def subdivide(V: np.ndarray, F: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle split into k² congruent triangles (each edge into k parts). The new vertices lie on the old
    triangles, so the surface (the piecewise-linear shape) is exactly the same; vertices on shared edges are shared
    (a watertight surface stays watertight). k = 1 returns the input."""
    k = int(k)
    if k < 1:
        raise ValueError("k >= 1 is required")
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    if k == 1:
        return V.copy(), F.copy()
    # barycentric lattice (i, j), i + j <= k, of every triangle; a point is identified by its two or three
    # integer weights on the original vertex ids, so points on a shared edge get the same key in both triangles
    ij = [(i, j) for i in range(k + 1) for j in range(k + 1 - i)]
    local = {p: n for n, p in enumerate(ij)}
    keys: dict[tuple[tuple[int, int], ...], int] = {}
    pts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []
    for a, b, c in F.tolist():
        ids = []
        for i, j in ij:
            w = {a: k - i - j, b: i, c: j}
            key = tuple(sorted((v, wt) for v, wt in w.items() if wt))
            if key not in keys:
                keys[key] = len(pts)
                pts.append(((k - i - j) * V[a] + i * V[b] + j * V[c]) / k)
            ids.append(keys[key])
        for i in range(k):
            for j in range(k - i):
                tris.append((ids[local[(i, j)]], ids[local[(i + 1, j)]], ids[local[(i, j + 1)]]))
                if i + j < k - 1:
                    tris.append((ids[local[(i + 1, j)]], ids[local[(i + 1, j + 1)]], ids[local[(i, j + 1)]]))
    return np.asarray(pts), np.asarray(tris, np.int64)
