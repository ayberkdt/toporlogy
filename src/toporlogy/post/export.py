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

from toporlogy.references import cite
from toporlogy.post.mesh import face_normals

__all__ = ["write_stl", "read_stl", "write_ply", "write_obj", "write_json", "export_result"]


@cite("3dsystems_1989", note="binary STL: 80-byte header, uint32 count, 50 bytes per facet")
def write_stl(path: str | Path, V: np.ndarray, F: np.ndarray, name: str = "toporlogy.post") -> str:
    """Binary STL (little-endian), units of V."""
    tri = np.asarray(V, float)[np.asarray(F, np.int64)]
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = face_normals(V, F)
    rec["v"] = tri
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(np.uint32(len(F)).tobytes())
        fh.write(rec.tobytes())
    return str(path)


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL → merged (V, F) (exact duplicate vertices merged)."""
    with open(path, "rb") as fh:
        fh.seek(80)
        n = int(np.frombuffer(fh.read(4), "<u4")[0])
        rec = np.frombuffer(fh.read(50 * n), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    P = rec["v"].reshape(-1, 3).astype(float)
    V, inv = np.unique(P, axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3).astype(np.int64)


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
    writers = {"stl": write_stl, "ply": write_ply, "obj": write_obj}
    for f in formats:
        out[f] = writers[f](folder / f"{name}.{f}", res.V, res.F)
    rep: dict[str, Any] = {"pipeline": res.report}
    if print_stock and any(r.stock > 0 for r in res.roles):
        pv = print_variant(res)
        out["stl_print"] = write_stl(folder / f"{name}_print_stock.stl", pv["V"], pv["F"])
        rep["print_variant"] = pv["report"]
    if checks is not None:
        rep["checks"] = checks.as_dict() if hasattr(checks, "as_dict") else checks
    if extra:
        rep.update(extra)
    out["report"] = write_json(folder / f"{name}_report.json", rep)
    return out
