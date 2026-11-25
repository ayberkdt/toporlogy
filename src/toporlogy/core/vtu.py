"""VTK XML UnstructuredGrid (``.vtu``) export of fields on the regular grid, and a reader for the roundtrip.

Only the **active** elements are written, in the order of ``domain.active_elements``, as VTK_QUAD (2-D) or
VTK_HEXAHEDRON (3-D); the element node order of :class:`~toporlogy.core.domain.Domain` (``edof[:, ::dim] // dim``)
is the VTK order of both cell types. Every node of the grid is written (points of void elements stay unconnected),
so point arrays have ``domain.nn`` rows; 2-D points get z = 0.

Values are written as ASCII with Python's shortest round-tripping ``repr``, so a float64 array reads back bit for bit
(NaN and ±Inf included, as ``nan`` / ``inf``; other readers may not parse those). The metadata (tensor and Voigt
convention, units, source) is a JSON text stored in the field data as a UInt8 array ``toporlogy_meta_json``, which
any VTK reader keeps and this module decodes.

Pre-registered checks: ``docs/olcumler/vtu_onkayit.md`` (W1-W6).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["write_vtu", "read_vtu", "VtuData", "VOIGT_CONVENTION", "VTK_QUAD", "VTK_HEXAHEDRON"]

VTK_QUAD, VTK_HEXAHEDRON = 9, 12
VOIGT_CONVENTION = ("Voigt [xx, yy, zz, yz, xz, xy] (3-D), [xx, yy, xy] (2-D); stress with physical shear, strain "
                    "with engineering shear (gamma = 2 eps_ij); global coordinates")
META_ARRAY = "toporlogy_meta_json"
_TYPES = {np.dtype(np.float64): "Float64", np.dtype(np.int64): "Int64", np.dtype(np.uint8): "UInt8"}
_NP = {"Float64": np.float64, "Float32": np.float32, "Int64": np.int64, "Int32": np.int32, "UInt8": np.uint8}


@dataclass
class VtuData:
    """What :func:`read_vtu` returns."""

    points: np.ndarray                                   # (n_points, 3)
    connectivity: np.ndarray                             # (n_cells, nodes per cell)
    cell_types: np.ndarray                               # (n_cells,)
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _text(a: np.ndarray) -> str:
    if a.dtype.kind == "f":
        return " ".join(repr(float(v)) for v in a.ravel())
    return " ".join(str(int(v)) for v in a.ravel())


def _array(parent: ET.Element, name: str | None, a: np.ndarray) -> None:
    a = np.asarray(a)
    if a.dtype.kind == "f":
        a = a.astype(np.float64, copy=False)
    elif a.dtype.kind in "iu" and a.dtype != np.uint8:
        a = a.astype(np.int64, copy=False)
    elif a.dtype.kind == "b":
        a = a.astype(np.uint8)
    if a.dtype not in _TYPES:
        raise ValueError(f"array {name!r}: dtype {a.dtype} is not supported (float, int, bool)")
    attrs = {"type": _TYPES[a.dtype], "format": "ascii",
             "NumberOfComponents": str(1 if a.ndim == 1 else int(np.prod(a.shape[1:])))}
    if name is not None:
        attrs["Name"] = name
    el = ET.SubElement(parent, "DataArray", attrs)
    el.text = _text(a)


def _rows(name: str, a: Any, n: int, what: str, grid_shape: tuple[int, ...] | None = None,
          active: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(a)
    if grid_shape is not None and active is not None and v.shape[:len(grid_shape)] == grid_shape:
        v = v.reshape((-1,) + v.shape[len(grid_shape):])[active]
    if v.ndim == 0 or v.shape[0] != n:
        raise ValueError(f"{what} array {name!r} has shape {np.shape(a)}; expected {n} rows ({what}s)"
                         + (f" or the grid shape {grid_shape}" if grid_shape is not None else ""))
    return v


def write_vtu(path: str | Path, domain: Any, point_data: Mapping[str, Any] | None = None,
              cell_data: Mapping[str, Any] | None = None, meta: Mapping[str, Any] | None = None) -> str:
    """Write the active grid of ``domain`` with ``point_data`` (``domain.nn`` rows each) and ``cell_data``
    (``domain.n_active`` rows each, or an array of ``domain.shape`` [+ components], of which the active elements are
    taken). ``meta`` is stored as JSON together with :data:`VOIGT_CONVENTION`. Returns the path."""
    if meta and "convention" in meta and meta["convention"] != VOIGT_CONVENTION:
        raise ValueError(f"meta['convention'] must be VOIGT_CONVENTION (the arrays are written in that convention), "
                         f"got {meta['convention']!r}")
    dim, nn = int(domain.dim), int(domain.nn)
    active = np.asarray(domain.active_elements, np.int64)
    conn = (np.asarray(domain.edof)[active][:, ::dim] // dim).astype(np.int64)
    X = np.asarray(domain.node_coords(), float)
    pts = np.zeros((nn, 3))
    pts[:, :dim] = X
    info = {"convention": VOIGT_CONVENTION, "dim": dim, "h": float(domain.h), "shape": list(domain.shape),
            "cells": "active elements in the order of domain.active_elements", **dict(meta or {})}

    root = ET.Element("VTKFile", {"type": "UnstructuredGrid", "version": "1.0", "byte_order": "LittleEndian",
                                  "header_type": "UInt64"})
    grid = ET.SubElement(root, "UnstructuredGrid")
    fd = ET.SubElement(grid, "FieldData")
    _array(fd, META_ARRAY, np.frombuffer(json.dumps(info, sort_keys=True).encode("utf-8"), np.uint8))
    piece = ET.SubElement(grid, "Piece", {"NumberOfPoints": str(nn), "NumberOfCells": str(len(active))})
    pd = ET.SubElement(piece, "PointData")
    for k, v in (point_data or {}).items():
        _array(pd, k, _rows(k, v, nn, "point"))
    cd = ET.SubElement(piece, "CellData")
    for k, v in (cell_data or {}).items():
        _array(cd, k, _rows(k, v, len(active), "cell", tuple(domain.shape), active))
    _array(ET.SubElement(piece, "Points"), None, pts)
    cells = ET.SubElement(piece, "Cells")
    _array(cells, "connectivity", conn.ravel())                   # one component: VTK splits it by offsets
    _array(cells, "offsets", np.arange(1, len(active) + 1, dtype=np.int64) * conn.shape[1])
    _array(cells, "types", np.full(len(active), VTK_QUAD if dim == 2 else VTK_HEXAHEDRON, np.uint8))
    p = Path(path)
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)
    return str(p)


def _values(el: ET.Element) -> np.ndarray:
    dt = _NP[el.attrib["type"]]
    words = (el.text or "").split()
    a = np.array([float(w) for w in words], dtype=dt) if np.dtype(dt).kind == "f" else \
        np.array([int(w) for w in words], dtype=dt)
    k = int(el.attrib.get("NumberOfComponents", "1"))
    return a if k == 1 else a.reshape(-1, k)


def read_vtu(path: str | Path) -> VtuData:
    """Read a file written by :func:`write_vtu` (ASCII UnstructuredGrid, one piece)."""
    root = ET.parse(Path(path)).getroot()
    grid = root.find("UnstructuredGrid")
    piece = grid.find("Piece") if grid is not None else None
    if grid is None or piece is None:
        raise ValueError(f"{path}: not a VTK UnstructuredGrid with a piece")
    meta: dict[str, Any] = {}
    for el in grid.iterfind("FieldData/DataArray"):
        if el.attrib.get("Name") == META_ARRAY:
            meta = json.loads(bytes(_values(el).astype(np.uint8)).decode("utf-8"))
    pts_el = piece.find("Points/DataArray")
    if pts_el is None:
        raise ValueError(f"{path}: no points")
    pts = _values(pts_el)
    cells = {el.attrib["Name"]: _values(el) for el in piece.iterfind("Cells/DataArray")}
    off = cells["offsets"]
    per = int(off[0]) if len(off) else 0
    return VtuData(points=pts, connectivity=cells["connectivity"].reshape(-1, per) if per else
                   cells["connectivity"].reshape(0, 0), cell_types=cells["types"],
                   point_data={el.attrib["Name"]: _values(el) for el in piece.iterfind("PointData/DataArray")},
                   cell_data={el.attrib["Name"]: _values(el) for el in piece.iterfind("CellData/DataArray")},
                   meta=meta)
