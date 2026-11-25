"""Subprocess worker: closed STL → 10-node tetrahedra with gmsh; writes X, T, faces to an .npz.

Run as ``python _gmsh_worker.py in.stl out.npz size angle_deg [modes]``. Kept in its own process so that a gmsh hang
(seen once after an OpenCASCADE STEP read in metres in the same process) cannot stall the caller, and so that gmsh's
runtime never meets torch / MKL in one process. Imports only gmsh and numpy.

Modes (comma-separated ``modes``, default ``reparametrize,discrete``, tried in order): "reparametrize" =
``classifySurfaces`` + ``createGeometry`` (the surface is remeshed at the target size); if that fails, "discrete" =
the STL triangles are kept as the boundary.
gmsh 10-node order: vertices 0-3, edges (0,1) (1,2) (2,0) (3,0) (3,2) (3,1).
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np


def _collect(gmsh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN001
    tags, coord, _ = gmsh.model.mesh.getNodes()
    X: np.ndarray = np.asarray(coord, float).reshape(-1, 3)
    lut = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    lut[tags.astype(np.int64)] = np.arange(len(tags))
    ty, _, en = gmsh.model.mesh.getElements(3)
    T: np.ndarray = lut[np.asarray(en[list(ty).index(11)], np.int64)].reshape(-1, 10)
    ty2, _, en2 = gmsh.model.mesh.getElements(2)
    Fb: np.ndarray = lut[np.asarray(en2[list(ty2).index(9)], np.int64)].reshape(-1, 6)
    used = np.unique(T)
    if len(used) != len(X):
        new = -np.ones(len(X), np.int64)
        new[used] = np.arange(len(used))
        X, T, Fb = X[used], new[T], new[Fb]
    return X, T, Fb


def main(src: str, dst: str, size: float, angle_deg: float, modes: str = "reparametrize,discrete") -> int:
    import gmsh
    t0 = time.perf_counter()
    last = ""
    for mode in modes.split(","):
        gmsh.initialize()
        try:
            gmsh.option.restoreDefaults()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.merge(src)
            if mode == "reparametrize":
                gmsh.model.mesh.classifySurfaces(angle_deg * math.pi / 180, True, True, math.pi)
                gmsh.model.mesh.createGeometry()
            surf = gmsh.model.getEntities(2)
            loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in surf])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            # straight-sided T10 (midside nodes at edge midpoints): curved midside nodes snapped to the reparametrised
            # STL patches inverted 18 Gauss-point Jacobians on a cantilever (14 Nov 2026); the surface is already a
            # fine faceted approximation, so curving buys nothing
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            if mode == "discrete":
                gmsh.model.mesh.generate(3)
                gmsh.model.mesh.setOrder(2)
            else:
                gmsh.option.setNumber("Mesh.ElementOrder", 2)
                gmsh.model.mesh.generate(3)
            X, T, Fb = _collect(gmsh)
            np.savez(dst, X=X, T=T, faces=Fb, mode=np.array(mode), seconds=np.array(time.perf_counter() - t0))
            return 0
        except Exception as exc:  # noqa: BLE001 - try the next mode
            last = f"{mode}: {exc}"
        finally:
            gmsh.finalize()
    sys.stderr.write(f"gmsh failed: {last}\n")
    return 2


if __name__ == "__main__":
    a = sys.argv
    sys.exit(main(a[1], a[2], float(a[3]), float(a[4]), a[5] if len(a) > 5 else "reparametrize,discrete"))
