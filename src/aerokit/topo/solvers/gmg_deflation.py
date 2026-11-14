"""Rigid-body deflation of floating stiff islands for the structured GMG (phase A2).

A stiff component (elements with ``E ≥ tau · max E``, face-connected) that does not touch a fixed DOF is held only
by soft material, so its six rigid-body motions are near-null vectors of ``K`` that a geometric coarse grid cannot
represent (measured in docs/olcumler/A0.md §5 and A2.md). They are added as an exact two-level correction around
the multigrid cycle ``B``:

``z0 = Q r;  z1 = z0 + B(r − A z0);  z = z1 + Q(r − A z1)``,  ``Q = Z (Zᵀ A Z)⁺ Zᵀ``,

which is symmetric when ``B`` is (balancing form). ``Z`` (rigid modes of each island, centred at its node centroid,
zero on fixed DOFs) is never formed: ``Zᵀ r`` and ``Z c`` are index-add / gather operations over the island nodes,
and ``Zᵀ A Z`` is assembled element by element. Deflation leaves the solution unchanged; only the preconditioner
improves. Islands are rare in converged designs (none in the measured histories), common in random test fields.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from aerokit.topo.solvers import gmg_kernels as gk

__all__ = ["IslandDeflation"]


class IslandDeflation:
    """Deflation space of the floating stiff components of a fine level.

    Parameters
    ----------
    L : gmg_kernels.Level
        Fine level (``s``/``KE`` form, float64) with its free mask.
    fixed : bool tensor (nx+1, ny+1, nz+1, 3)
        User-fixed DOFs (a component touching one is anchored, not deflated).
    tau : float
        Stiffness threshold relative to max(E).
    max_islands : int
        Cap; the largest islands are kept.
    """

    def __init__(self, L: gk.Level, fixed: torch.Tensor, tau: float = 1e-3, max_islands: int = 200,
                 chunk: int = 16384) -> None:
        from scipy.ndimage import label
        nx, ny, nz = L.shape
        dev = L.free.device
        s = L.s
        assert s is not None and L.KE is not None
        stiff = (s >= tau * float(s.max())).view(nx, ny, nz).cpu().numpy()
        lab_np, n = label(stiff)
        self.n_components = int(n)
        self.n = 0
        if n <= 1:
            return
        lab = torch.as_tensor(lab_np.reshape(-1), dtype=torch.float64, device=dev)
        node_lab = gk.node_max(lab, L.shape).long()                         # (nx+1, ny+1, nz+1)
        anchored = torch.unique(node_lab[fixed.any(-1)])
        sizes = torch.bincount(node_lab.reshape(-1), minlength=n + 1)
        cand = [c for c in range(1, n + 1) if c not in set(anchored.tolist())]
        if not cand:
            return
        cand.sort(key=lambda c: -int(sizes[c]))
        cand = cand[:max_islands]
        remap = torch.full((n + 1,), -1, dtype=torch.long, device=dev)
        remap[torch.as_tensor(cand, device=dev)] = torch.arange(len(cand), device=dev)
        zid = remap[node_lab.reshape(-1)]                                     # (nn,)
        self.sel = torch.nonzero(zid >= 0).flatten()
        self.zid = zid[self.sel]
        self.n = len(cand)
        X = torch.stack(torch.meshgrid(*[torch.arange(v + 1, device=dev, dtype=torch.float64) for v in L.shape],
                                       indexing="ij"), -1).reshape(-1, 3)[self.sel]
        cnt = torch.bincount(self.zid, minlength=self.n).to(torch.float64)
        cen = torch.zeros((self.n, 3), dtype=torch.float64, device=dev).index_add_(0, self.zid, X) / cnt[:, None]
        self.R = X - cen[self.zid]                                            # (nsel, 3)
        self.free_sel = L.free.reshape(-1, 3)[self.sel]                       # (nsel, 3)
        self.node_shape = L.free.shape
        self.island_nodes = cnt.long().tolist()
        # Gram matrix Zᵀ A Z, element by element (elements with a node in an island)
        zid_full = torch.full((int(np.prod(self.node_shape[:3])),), -1, dtype=torch.long, device=dev)
        zid_full[self.sel] = self.zid
        Rf = torch.zeros((zid_full.numel(), 3), dtype=torch.float64, device=dev)
        Rf[self.sel] = self.R
        zid_g = zid_full.view(self.node_shape[:3])
        ze = gk.gather(zid_g[..., None].expand(*self.node_shape[:3], 3).double(), L.shape)[:, 0::3].long()  # (ne, 8)
        els = torch.nonzero((ze >= 0).any(1)).flatten()
        Rg = Rf.view(*self.node_shape[:3], 3)
        G = torch.zeros((self.n * 6) ** 2, dtype=torch.float64, device=dev)
        fe_all = gk.gather(L.free, L.shape)
        Re_all = gk.gather(Rg, L.shape)                                       # (ne, 24): per corner (Rx, Ry, Rz)
        for i in range(0, els.numel(), chunk):
            e = els[i:i + chunk]
            Phi = self._phi(Re_all[e].view(-1, 8, 3), ze[e]) * fe_all[e].view(-1, 8, 3)[..., None]  # (m, 8, 3, 6)
            K = s[e][:, None, None] * L.KE[None] * fe_all[e][:, :, None] * fe_all[e][:, None, :]
            K = K.view(-1, 8, 3, 8, 3)
            loc = torch.einsum("maik,maibj,mbjl->mabkl", Phi, K, Phi)       # (m, 8, 8, 6, 6)
            ia, ib = ze[e][:, :, None].expand(-1, 8, 8), ze[e][:, None, :].expand(-1, 8, 8)
            ok = (ia >= 0) & (ib >= 0)
            rows = (ia[ok] * 6)[:, None, None] + torch.arange(6, device=dev)[None, :, None]
            cols = (ib[ok] * 6)[:, None, None] + torch.arange(6, device=dev)[None, None, :]
            G.index_add_(0, (rows * (self.n * 6) + cols).reshape(-1), loc[ok].reshape(-1))
        G = G.view(self.n * 6, self.n * 6)
        G = 0.5 * (G + G.T)
        d = torch.sqrt(torch.clamp(torch.diagonal(G), min=1e-300))
        Gs = G / d[:, None] / d[None, :]
        w, V = torch.linalg.eigh(Gs)
        keep = w > 1e-10 * float(w.max())
        self.Ginv = (V[:, keep] / w[keep]) @ V[:, keep].T / d[:, None] / d[None, :]
        self.rank = int(keep.sum())

    @staticmethod
    def _phi(R: torch.Tensor, zid: torch.Tensor) -> torch.Tensor:
        """Rigid modes (…, 3, 6) at nodes with relative coordinates R (…, 3); zero where zid < 0."""
        z = torch.zeros(R.shape[:-1] + (3, 6), dtype=R.dtype, device=R.device)
        z[..., 0, 0] = z[..., 1, 1] = z[..., 2, 2] = 1.0
        z[..., 0, 3], z[..., 1, 3] = -R[..., 1], R[..., 0]
        z[..., 1, 4], z[..., 2, 4] = -R[..., 2], R[..., 1]
        z[..., 0, 5], z[..., 2, 5] = -R[..., 2], R[..., 0]
        return z * (zid >= 0)[..., None, None].to(R.dtype)

    def Zt(self, r: torch.Tensor) -> torch.Tensor:
        """Zᵀ r → (6n,) float64."""
        rs = r.reshape(-1, 3)[self.sel].to(torch.float64) * self.free_sel
        R = self.R
        t = torch.stack([rs[:, 0], rs[:, 1], rs[:, 2], -R[:, 1] * rs[:, 0] + R[:, 0] * rs[:, 1],
                         -R[:, 2] * rs[:, 1] + R[:, 1] * rs[:, 2], -R[:, 2] * rs[:, 0] + R[:, 0] * rs[:, 2]], 1)
        return torch.zeros((self.n, 6), dtype=torch.float64, device=r.device).index_add_(0, self.zid, t).reshape(-1)

    def Z(self, c: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Z c as a nodal field shaped/typed like ``like``."""
        cc = c.view(self.n, 6)[self.zid]
        R = self.R
        v = torch.stack([cc[:, 0] - R[:, 1] * cc[:, 3] - R[:, 2] * cc[:, 5],
                         cc[:, 1] + R[:, 0] * cc[:, 3] - R[:, 2] * cc[:, 4],
                         cc[:, 2] + R[:, 1] * cc[:, 4] + R[:, 0] * cc[:, 5]], 1) * self.free_sel
        out = torch.zeros((like.numel() // 3, 3), dtype=like.dtype, device=like.device)
        out[self.sel] = v.to(like.dtype)
        return out.view_as(like)

    def Q(self, r: torch.Tensor) -> torch.Tensor:
        return self.Z(self.Ginv @ self.Zt(r), r)

    def wrap(self, A: Any, B: Any, r: torch.Tensor) -> torch.Tensor:
        """Balancing correction around the cycle ``B`` with operator ``A`` (same dtype as r)."""
        z0 = self.Q(r)
        z1 = z0 + B(r - A(z0))
        return z1 + self.Q(r - A(z1))

    def info(self) -> dict[str, Any]:
        return {"components": self.n_components, "islands": self.n, "rank": getattr(self, "rank", 0),
                "island_nodes": getattr(self, "island_nodes", [])[:20]}
