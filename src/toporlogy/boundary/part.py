"""Part geometry seen by selectors and loads: ``PartInfo``, ``part_info`` (cached per domain).
"""
from __future__ import annotations

import weakref
from toporlogy.domain import Domain
from typing import Any

import numpy as np


class EmptySelectionError(ValueError):
    """A selector picked no node of the part."""


# =============================================================================================== part topology
class PartInfo:
    """Surface, faces, normals and areas of a domain's part (computed once per domain, see :func:`part_info`)."""

    def __init__(self, domain: Domain) -> None:
        d = domain
        self.domain, self.dim, self.h = d, d.dim, d.h
        self.nper = 2 ** d.dim
        act = d.active_elements
        self.elem_nodes = d.edof[act][:, ::d.dim] // d.dim                 # (na, 2^dim)
        cnt = np.bincount(self.elem_nodes.ravel(), minlength=d.nn)
        self.node_count = cnt
        self.node_active = cnt > 0
        self.surface = (cnt > 0) & (cnt < self.nper)
        self.X = d.node_coords()
        xa = self.X[self.node_active]
        self.L = float(np.linalg.norm(xa.max(axis=0) - xa.min(axis=0))) or d.h
        self.face_area = d.h * d.h if d.dim == 3 else d.h * d.thickness
        self.elem_volume = d.h ** 3 if d.dim == 3 else d.h * d.h * d.thickness
        self._normals: np.ndarray | None = None
        self._faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ smooth outward normals
    @property
    def normals(self) -> np.ndarray:
        """(nn, dim) unit outward normal per node (zero where undefined, e.g. interior nodes)."""
        if self._normals is None:
            d = self.domain
            frac = np.where(d.active, d.frac, 0.0)
            P = np.pad(frac, 2)
            W = np.lib.stride_tricks.sliding_window_view(P, (4,) * self.dim)   # node i ↔ elements i−2 … i+1
            off = np.array([-1.5, -0.5, 0.5, 1.5])
            G = np.stack(np.meshgrid(*([off] * self.dim), indexing="ij"), axis=0)   # (dim, 4, …)
            w = np.maximum(0.0, 2.5 - np.sqrt((G ** 2).sum(axis=0)))
            letters = "abc"[: self.dim]
            m = np.stack([np.einsum(f"...{letters},{letters}->...", W, G[k] * w) for k in range(self.dim)], axis=-1)
            m = m.reshape(-1, self.dim)
            nrm = np.linalg.norm(m, axis=1)
            n = np.zeros_like(m)
            ok = nrm > 1e-9
            n[ok] = -m[ok] / nrm[ok, None]
            self._normals = n
        return self._normals

    # ------------------------------------------------------------------ exposed faces (staircase surface)
    @property
    def faces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(face nodes (nf, 2^(dim−1)), staircase normal n_f (nf, dim), geometric area factor (nf,), smooth face
        normal (nf, dim)).

        The smooth normal of a face is the mean of its corner normals when the element that owns it is partial
        (frac < 1: the true surface cuts it, the face is a staircase step) and n_f otherwise (a full element with an
        empty neighbour: the face lies on the surface). The area factor is |n_f · smooth normal| — exactly 1 on
        grid-aligned faces, also next to edges where the corner normals tilt."""
        if self._faces is None:
            d = self.domain
            act = d.active
            corners = np.indices((2,) * self.dim).reshape(self.dim, -1).T
            fn, fnorm, own = [], [], []
            for k in range(self.dim):
                for s in (-1, 1):
                    nb = np.zeros_like(act)
                    src = [slice(None)] * self.dim
                    dst = [slice(None)] * self.dim
                    if s > 0:
                        src[k], dst[k] = slice(1, None), slice(None, -1)
                    else:
                        src[k], dst[k] = slice(None, -1), slice(1, None)
                    nb[tuple(dst)] = act[tuple(src)]
                    idx = np.argwhere(act & ~nb)
                    if not len(idx):
                        continue
                    cs = corners[corners[:, k] == (1 if s > 0 else 0)]
                    fn.append(np.stack([d.node_number(idx + c) for c in cs], axis=1))
                    e = np.zeros((len(idx), self.dim))
                    e[:, k] = s
                    fnorm.append(e)
                    own.append(d.frac[tuple(idx.T)])
            F = np.concatenate(fn)
            N = np.concatenate(fnorm)
            partial = np.concatenate(own) < 1.0 - 1e-9
            nh = self.normals[F].mean(axis=1)
            ln = np.linalg.norm(nh, axis=1)
            use = partial & (ln > 1e-9)
            SN = N.copy()
            SN[use] = nh[use] / ln[use, None]
            fac = np.abs(np.einsum("ij,ij->i", N, SN))
            self._faces = (F, N, fac, SN)
        return self._faces

    def patch(self, nodes: np.ndarray, geometric: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Faces whose corners all lie in ``nodes`` and the consistent nodal area of ``nodes`` on them.

        Returns (face indices, nodal area aligned with ``nodes``). ``geometric`` applies the |n_f·n̂| factor.
        """
        F, _, fac, _ = self.faces
        inset = np.zeros(self.domain.nn, bool)
        inset[nodes] = True
        sel = np.flatnonzero(inset[F].all(axis=1))
        a = self.face_area * (fac[sel] if geometric else np.ones(len(sel)))
        area = np.zeros(self.domain.nn)
        np.add.at(area, F[sel].ravel(), np.repeat(a / F.shape[1], F.shape[1]))
        return sel, area[nodes]

    def patch_normals(self, nodes: np.ndarray) -> np.ndarray:
        """(k, dim) outward normal of ``nodes`` from the smooth normals of the complete faces of the selection
        (area-weighted); the node normal where a node is on no such face. Flat patches give the exact normal up
        to their edges."""
        F, _, fac, SN = self.faces
        sel, _ = self.patch(nodes)
        acc = np.zeros((self.domain.nn, self.dim))
        np.add.at(acc, F[sel].ravel(), np.repeat(fac[sel, None] * SN[sel], F.shape[1], axis=0))
        out = acc[nodes]
        ln = np.linalg.norm(out, axis=1)
        out = np.where(ln[:, None] > 1e-12, out / np.maximum(ln, 1e-300)[:, None], self.normals[nodes])
        return out


_INFO: weakref.WeakKeyDictionary[Domain, PartInfo] = weakref.WeakKeyDictionary()


def part_info(domain: Domain) -> PartInfo:
    """Cached :class:`PartInfo` of ``domain``."""
    info = _INFO.get(domain)
    if info is None:
        info = PartInfo(domain)
        _INFO[domain] = info
    return info


def _cross(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """r × f row-wise; scalar (n,) in 2-D."""
    if r.shape[-1] == 2:
        return r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]
    return np.cross(r, f)


def _vec(v: Any, dim: int, what: str) -> np.ndarray:
    a = np.asarray(v, float).reshape(-1)
    if a.size != dim:
        raise ValueError(f"{what} must have {dim} components, got {a.size}")
    return a


def _mom(v: Any, dim: int) -> np.ndarray:
    """Moment as an array: (3,) in 3-D, (1,) in 2-D (about the out-of-plane axis)."""
    a = np.asarray(v, float).reshape(-1)
    want = 3 if dim == 3 else 1
    if a.size != want:
        raise ValueError(f"moment must have {want} component(s) in {dim}-D, got {a.size}")
    return a
