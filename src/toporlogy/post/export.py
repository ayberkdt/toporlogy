"""Writers: binary STL, binary PLY, OBJ, JSON quality report; one call for a finished part (+ print variant).

STEP is not written. gmsh can only write STEP from OpenCASCADE geometry; turning a triangle mesh into one planar
B-rep face per triangle gives a huge, unusable STEP. A proper STEP needs a surface fit (NURBS patches over a
quad-dominant layout, e.g. a reverse-engineering tool) — outside this package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toporlogy.geometry import write_stl
from toporlogy.references import cite

__all__ = ["write_ply", "write_obj", "write_json", "export_result", "voxel_surface", "export_phase_surfaces"]


def _stl(path: str | Path, V: np.ndarray, F: np.ndarray) -> str:
    """Binary STL of a finished part (:func:`toporlogy.geometry.write_stl`, header "toporlogy.post")."""
    return str(write_stl(path, V, F, header="toporlogy.post"))


@cite("turk_1994", note="binary little-endian PLY header")
def write_ply(path: str | Path, V: np.ndarray, F: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("ply\nformat binary_little_endian 1.0\ncomment toporlogy.post\n"
            f"element vertex {len(V)}\nproperty double x\nproperty double y\nproperty double z\n"
            f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    faces = np.zeros(len(F), dtype=np.dtype([("c", "u1"), ("i", "<i4", 3)]))
    faces["c"] = 3
    faces["i"] = F
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(np.asarray(V, "<f8").tobytes())
        fh.write(faces.tobytes())
    return str(path)


def write_obj(path: str | Path, V: np.ndarray, F: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("# toporlogy.post\n")
        np.savetxt(fh, np.asarray(V, float), fmt="v %.9g %.9g %.9g")
        np.savetxt(fh, np.asarray(F, np.int64) + 1, fmt="f %d %d %d")
    return str(path)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path: str | Path, data: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=1), encoding="utf-8")
    return str(path)


def export_result(res: Any, folder: str | Path, name: str = "part", formats: tuple[str, ...] = ("stl",),
                  checks: Any = None, print_stock: bool = True, extra: dict[str, Any] | None = None
                  ) -> dict[str, str]:
    """Write the finished surface (``formats`` from stl/ply/obj), the print variant when any role has stock, and
    ``<name>_report.json`` (pipeline report + checks + print-variant report + ``extra``)."""
    from toporlogy.post.pipeline import print_variant
    folder = Path(folder)
    out: dict[str, str] = {}
    writers = {"stl": _stl, "ply": write_ply, "obj": write_obj}
    for f in formats:
        out[f] = writers[f](folder / f"{name}.{f}", res.V, res.F)
    rep: dict[str, Any] = {"pipeline": res.report}
    if print_stock and any(r.stock > 0 for r in res.roles):
        pv = print_variant(res)
        out["stl_print"] = _stl(folder / f"{name}_print_stock.stl", pv["V"], pv["F"])
        rep["print_variant"] = pv["report"]
    if checks is not None:
        rep["checks"] = checks.as_dict() if hasattr(checks, "as_dict") else checks
    if extra:
        rep.update(extra)
    out["report"] = write_json(folder / f"{name}_report.json", rep)
    return out


def voxel_surface(mask: np.ndarray, h: float = 1.0, origin: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Boundary of the voxel set ``mask`` (3-D bool array) as triangles (V, F), outward normals, two per exposed
    cell face; no smoothing. Every directed edge has its reverse (closed), so the divergence-theorem volume is
    exactly ``mask.sum() · h³``. Vertices are not merged across faces."""
    m = np.asarray(mask, bool)
    if m.ndim != 3:
        raise ValueError("voxel_surface needs a 3-D mask")
    o = np.zeros(3) if origin is None else np.asarray(origin, float).reshape(3)
    tris: list[np.ndarray] = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        pad = [(0, 0)] * 3
        pad[a] = (1, 1)
        mp = np.pad(m, pad)
        lo = np.take(mp, np.arange(0, mp.shape[a] - 1), axis=a)
        hi = np.take(mp, np.arange(1, mp.shape[a]), axis=a)
        for sign, faces in ((1, lo & ~hi), (-1, ~lo & hi)):
            idx = np.argwhere(faces).astype(float)          # face at coordinate idx[a] (cell index k → plane k)
            if not len(idx):
                continue
            eb, ec = np.eye(3)[b], np.eye(3)[c]
            p0 = idx
            p1, p2, p3 = p0 + eb, p0 + eb + ec, p0 + ec
            t1 = np.stack([p0, p1, p2], 1) if sign > 0 else np.stack([p0, p2, p1], 1)
            t2 = np.stack([p0, p2, p3], 1) if sign > 0 else np.stack([p0, p3, p2], 1)
            tris += [t1, t2]
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    T = np.concatenate(tris)
    V = o + h * T.reshape(-1, 3)
    return V, np.arange(len(V), dtype=np.int64).reshape(-1, 3)


def export_phase_surfaces(labels: np.ndarray, names: list[str], folder: str | Path, h: float = 1.0,
                          origin: Any = None, prefix: str = "phase") -> dict[str, str]:
    """One binary STL per solid phase of a 3-D label field (0 void, i ≥ 1 phase ``names[i - 1]``, −1 outside):
    ``<prefix>_<name>.stl``, the voxel boundary of that phase (:func:`voxel_surface`). Phases with no element are
    skipped."""
    lab = np.asarray(labels)
    out: dict[str, str] = {}
    for i, nm in enumerate(names, start=1):
        mask = lab == i
        if not mask.any():
            continue
        V, F = voxel_surface(mask, h, origin)
        out[nm] = _stl(Path(folder) / f"{prefix}_{nm}.stl", V, F)
    return out
