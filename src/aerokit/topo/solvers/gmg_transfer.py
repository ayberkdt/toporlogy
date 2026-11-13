"""Operator-dependent prolongation for the structured GMG (collapsed-stencil, "black-box" type; phase A2).

The fine nodes of a 2:1 grid fall in seven parity classes besides the coarse nodes: x-, y-, z-edge midpoints, three
face-centre classes and cell centres. The prolongation is a sequence of local solves on the operator's nodal 3×3
block stencil ``S[n, o]`` (o ∈ {−1, 0, 1}³):

1. coarse nodes: injection;
2. edge midpoint m on an a-axis edge: the stencil is summed over the planes ⟂ a ("collapsed"),
   ``L = Σ_{o_a=−1} S``, ``C = Σ_{o_a=0} S``, ``R = Σ_{o_a=+1} S``, and ``u_m = −C⁻¹ (L u_{m−e_a} + R u_{m+e_a})``;
3. face centre f (normal a): the stencil is summed along a, ``N_{(o_b, o_c)} = Σ_{o_a} S``, and
   ``u_f = −N_0⁻¹ Σ_{8 in-plane neighbours} N_k u_k`` (vertices and edge midpoints from steps 1–2);
4. cell centre c: ``u_c = −S_0⁻¹ Σ_{26 neighbours} S_k u_k``.

Each fine node inside or on a coarse element depends only on that element's 8 coarse nodes, so the prolongation is
element-local and the element-level Galerkin product ``K_c[E] = Σ_children P_{E,c}ᵀ K_child P_{E,c}`` is exact.
When a local block ``C`` is singular or not finite, that node falls back to trilinear weights. Fixed DOFs (zero rows
and columns of the masked operator) get zero rows.

This is the construction of Dendy's black-box multigrid extended blockwise to elasticity (general knowledge; the
original paper was not read in this session and is not in ``aerokit.references``). The restriction is the exact
transpose, applied in reverse order.
"""
from __future__ import annotations

import itertools

import torch

from aerokit.topo.solvers import gmg_kernels as gk

__all__ = ["BlackBoxTransfer"]

_OFFS = [o for o in itertools.product((-1, 0, 1), repeat=3)]          # 27, index 13 = (0, 0, 0)
_OI = {o: i for i, o in enumerate(_OFFS)}
_EDGE_P = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
_FACE_P = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]                              # normal x, y, z
_CELL_P = (1, 1, 1)


def _edge_offs(a: int) -> list[tuple[int, int, int]]:
    m, p = [0, 0, 0], [0, 0, 0]
    m[a], p[a] = -1, 1
    return [tuple(m), tuple(p)]  # type: ignore[list-item]


def _face_offs(a: int) -> list[tuple[int, int, int]]:
    return [o for o in _OFFS if o[a] == 0 and o != (0, 0, 0)]


_CELL_OFFS = [o for o in _OFFS if o != (0, 0, 0)]


def _class_offs(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    k = sum(p)
    if k == 1:
        return _edge_offs(p.index(1))
    if k == 2:
        return _face_offs(p.index(0))
    return _CELL_OFFS


def _count(p: tuple[int, int, int], nc: tuple[int, int, int]) -> tuple[int, int, int]:
    """Number of class-p fine nodes per axis on a coarse grid with ``nc`` nodes per axis."""
    return tuple(n - 1 if q else n for q, n in zip(p, nc, strict=True))  # type: ignore[return-value]


def _sl(start: tuple[int, ...], count: tuple[int, ...], step: int = 2) -> tuple[slice, ...]:
    return tuple(slice(s, s + step * (c - 1) + 1, step) for s, c in zip(start, count, strict=True))


class BlackBoxTransfer:
    """Prolongation / restriction between a fine level and its 2:1 coarse grid, built from the fine operator.

    Parameters
    ----------
    fine : gmg_kernels.Level
        Fine level (``s``/``KE`` or ``Ke`` form, float64) with its free mask.
    """

    def __init__(self, fine: gk.Level) -> None:
        self.fshape = fine.shape
        nx, ny, nz = fine.shape
        self.nf = (nx + 1, ny + 1, nz + 1)
        self.nc = (nx // 2 + 1, ny // 2 + 1, nz // 2 + 1)
        self.cshape = (nx // 2, ny // 2, nz // 2)
        dev, dt = fine.free.device, torch.float64
        # zero-padded element data: element e at padded index e + 1
        if fine.Ke is None:
            sp = torch.zeros((nx + 2, ny + 2, nz + 2), dtype=dt, device=dev)
            sp[1:-1, 1:-1, 1:-1] = fine.s.view(nx, ny, nz)
            self._src = ("s", sp, fine.KE)
        else:
            kp = torch.zeros((nx + 2, ny + 2, nz + 2, 24, 24), dtype=dt, device=dev)
            kp[1:-1, 1:-1, 1:-1] = fine.Ke.view(nx, ny, nz, 24, 24)
            self._src = ("Ke", kp, None)
        fp = torch.zeros((nx + 3, ny + 3, nz + 3, 3), dtype=dt, device=dev)
        fp[1:-1, 1:-1, 1:-1] = fine.free
        self._fpad = fp
        self.free = fine.free
        self.masked_rows = fine.Ke is None          # fine level: the mask is not in the element data
        self.classes = [p for p in _EDGE_P] + [p for p in _FACE_P] + [_CELL_P]
        self.W = {p: self._weights(p) for p in self.classes}
        self.n_trilinear_fallback = int(sum(int(v) for v in self._fallbacks.values()))
        del self._src, self._fpad

    # ------------------------------------------------------------------ stencil and weights
    def _stencil(self, p: tuple[int, int, int]) -> torch.Tensor:
        """S (cx, cy, cz, 27, 3, 3) of the class-p fine nodes, masked by the free DOFs."""
        cnt = _count(p, self.nc)
        kind, data, KE = self._src
        S = torch.zeros(cnt + (27, 3, 3), dtype=data.dtype, device=data.device)
        for i, ci in enumerate(gk.CORNERS):
            esl = _sl(tuple(p[a] - ci[a] + 1 for a in range(3)), cnt)       # element n − c_i (padded index)
            blk = data[esl]
            for j, cj in enumerate(gk.CORNERS):
                o = tuple(cj[a] - ci[a] for a in range(3))
                if kind == "s":
                    S[..., _OI[o], :, :] += blk[..., None, None] * KE[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                else:
                    S[..., _OI[o], :, :] += blk[..., 3 * i:3 * i + 3, 3 * j:3 * j + 3]
        fr = self._fpad[_sl(tuple(q + 1 for q in p), cnt)]                   # (…, 3) row mask
        fcol = torch.stack([self._fpad[_sl(tuple(p[a] + o[a] + 1 for a in range(3)), cnt)] for o in _OFFS], -2)
        return S * fr[..., None, :, None] * fcol[..., :, None, :]

    def _weights(self, p: tuple[int, int, int]) -> torch.Tensor:
        S = self._stencil(p)
        k = sum(p)
        if k == 1:
            a = p.index(1)
            oa = torch.tensor([o[a] for o in _OFFS], device=S.device)
            C = S[..., oa == 0, :, :].sum(-3)
            N = torch.stack([S[..., oa == -1, :, :].sum(-3), S[..., oa == 1, :, :].sum(-3)], -3)
            tri = 0.5
        elif k == 2:
            a = p.index(0)
            offs = _face_offs(a)
            C = sum(S[..., _OI[tuple(v if b != a else s for b, v in enumerate((0, 0, 0)))], :, :]
                    for s in (-1, 0, 1))
            N = torch.stack([sum(S[..., _OI[tuple(o[b] if b != a else s for b in range(3))], :, :]
                                 for s in (-1, 0, 1)) for o in offs], -3)
            tri_w = torch.tensor([0.25 if sum(abs(v) for v in o) == 1 else 0.0 for o in offs],
                                 dtype=S.dtype, device=S.device)
        else:
            C = S[..., 13, :, :]
            N = S[..., [i for i in range(27) if i != 13], :, :]
            tri_w = torch.tensor([1.0 / 6.0 if sum(abs(v) for v in o) == 1 else 0.0 for o in _CELL_OFFS],
                                 dtype=S.dtype, device=S.device)
        # fixed components: identity rows in C (their N rows are zero → zero weights)
        fr = self._fpad[_sl(tuple(q + 1 for q in p), _count(p, self.nc))]
        C = C + torch.diag_embed(1.0 - fr)
        nb = N.shape[-3]
        W = -torch.linalg.solve(C[..., None, :, :].expand(N.shape), N)
        bad = ~torch.isfinite(W).flatten(-3).all(-1)
        cond_ok = torch.linalg.cond(C) < 1e12
        bad |= ~cond_ok
        if k == 1:
            tw = torch.full((nb,), tri, dtype=S.dtype, device=S.device)
        else:
            tw = tri_w
        if bad.any():
            eye = torch.eye(3, dtype=S.dtype, device=S.device)
            Wt = (tw[:, None, None] * eye) * fr[bad][:, None, :, None]
            W[bad] = Wt
        self._fallbacks = getattr(self, "_fallbacks", {})
        self._fallbacks[p] = int(bad.sum())
        return W

    # ------------------------------------------------------------------ apply
    def prolong(self, uc: torch.Tensor) -> torch.Tensor:
        """Coarse nodal field (ncx, ncy, ncz, 3) → fine nodal field (unmasked; the caller multiplies by its mask)."""
        u = uc.new_zeros(self.nf + (3,))
        u[_sl((0, 0, 0), self.nc)] = uc
        for p in self.classes:
            cnt = _count(p, self.nc)
            offs = _class_offs(p)
            nb = torch.stack([u[_sl(tuple(p[a] + o[a] for a in range(3)), cnt)] for o in offs], -2)
            W = self.W[p].to(uc.dtype) if self.W[p].dtype != uc.dtype else self.W[p]
            u[_sl(p, cnt)] = torch.einsum("...kij,...kj->...i", W, nb)
        return u

    def restrict(self, r: torch.Tensor) -> torch.Tensor:
        """Transpose of :meth:`prolong` (input not modified)."""
        r = r.clone()
        for p in reversed(self.classes):
            cnt = _count(p, self.nc)
            offs = _class_offs(p)
            W = self.W[p].to(r.dtype) if self.W[p].dtype != r.dtype else self.W[p]
            cs = _sl(p, cnt)
            contrib = torch.einsum("...kji,...j->...ki", W, r[cs])
            for k, o in enumerate(offs):
                r[_sl(tuple(p[a] + o[a] for a in range(3)), cnt)] += contrib[..., k, :]
            r[cs] = 0.0
        return r[_sl((0, 0, 0), self.nc)].contiguous()

    def to(self, dtype: torch.dtype) -> BlackBoxTransfer:
        """Shallow copy with the weights cast (for the float32 V-cycle)."""
        t = object.__new__(BlackBoxTransfer)
        t.__dict__.update(self.__dict__)
        t.W = {p: w.to(dtype) for p, w in self.W.items()}
        return t

    # ------------------------------------------------------------------ Galerkin
    def element_P(self, i0: int, i1: int) -> torch.Tensor:
        """Local prolongation (m, 27, 3, 24) of the coarse elements with x-index in [i0, i1): fine node
        q = 9 qx + 3 qy + qz of the element (q ∈ {0, 1, 2}³) from its 8 coarse nodes (CORNERS order)."""
        NX, NY, NZ = self.cshape
        n = (i1 - i0, NY, NZ)
        m = n[0] * NY * NZ
        dev, dt = self.free.device, self.W[_CELL_P].dtype
        P = torch.zeros((m, 27, 3, 24), dtype=dt, device=dev)
        qi = (lambda q: 9 * q[0] + 3 * q[1] + q[2])  # noqa: E731
        eye = torch.eye(3, dtype=dt, device=dev)
        for k, c in enumerate(gk.CORNERS):
            P[:, qi(tuple(2 * v for v in c)), :, 3 * k:3 * k + 3] = eye
        for p in self.classes:
            offs = _class_offs(p)
            free_axes = [a for a in range(3) if p[a] == 0]
            for corner in itertools.product((0, 2), repeat=len(free_axes)):
                q = list(p)
                for a, v in zip(free_axes, corner, strict=True):
                    q[a] = v
                start = [(q[a] - p[a]) // 2 for a in range(3)]
                start[0] += i0
                W = self.W[p][tuple(slice(start[a], start[a] + n[a]) for a in range(3))].reshape(m, len(offs), 3, 3)
                acc = torch.zeros((m, 3, 24), dtype=dt, device=dev)
                for kk, o in enumerate(offs):
                    acc += W[:, kk] @ P[:, qi(tuple(q[a] + o[a] for a in range(3)))]
                P[:, qi(tuple(q))] = acc
        return P

    def galerkin(self, fine: gk.Level, slab: int | None = None) -> torch.Tensor:
        """Coarse element matrices (nc, 24, 24) = Σ_children P̃_cᵀ K_child P̃_c (fine rows masked)."""
        NX, NY, NZ = self.cshape
        if slab is None:
            slab = max(1, 4096 // max(NY * NZ, 1))
        out = []
        qi = (lambda q: 9 * q[0] + 3 * q[1] + q[2])  # noqa: E731
        child_rows = []
        for ci in range(8):
            off = (ci // 4, (ci // 2) % 2, ci % 2)
            child_rows.append([qi(tuple(off[a] + c[a] for a in range(3))) for c in gk.CORNERS])
        child_rows_t = torch.tensor(child_rows, device=self.free.device)          # (8, 8)
        if fine.Ke is None:
            s_ch = gk.children(fine.s, self.fshape)                               # (nc, 8)
            f_ch = gk.children(gk.gather(fine.free, self.fshape), self.fshape)    # (nc, 8, 24)
        else:
            K_ch = gk.children(fine.Ke, self.fshape)                              # (nc, 8, 24, 24)
        for i0 in range(0, NX, slab):
            i1 = min(NX, i0 + slab)
            P = self.element_P(i0, i1)                                            # (m, 27, 3, 24)
            m = P.shape[0]
            Pc = P[:, child_rows_t].reshape(m, 8, 24, 24)                         # (m, 8, 24 fine, 24 coarse)
            e0, e1 = i0 * NY * NZ, i1 * NY * NZ
            if fine.Ke is None:
                Pc = Pc * f_ch[e0:e1][..., None]
                T = torch.einsum("jk,mckl->mcjl", fine.KE, Pc)
                Kc = torch.einsum("mc,mcji,mcjl->mil", s_ch[e0:e1], Pc, T)
            else:
                T = torch.matmul(K_ch[e0:e1], Pc)
                Kc = torch.einsum("mcji,mcjl->mil", Pc, T)
            out.append(Kc)
        Kc = torch.cat(out, 0)
        return 0.5 * (Kc + Kc.transpose(1, 2))
