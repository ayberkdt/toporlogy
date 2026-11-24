"""NumPy transcriptions of two published educational codes, used only as independent references by
tests/toporlogy/test_independent_vv.py. Nothing here imports toporlogy's FE, filter or optimizer code.

Sources read on 2026-11-24 (the MATLAB text itself, line by line):

* ``top88.m`` — "AN 88 LINE TOPOLOGY OPTIMIZATION CODE Nov, 2010", downloaded from the DTU TopOpt group page
  (topopt.mek.dtu.dk, file top88.m, 4836 bytes);
* ``top3d.m`` — "AN 169 LINE 3D TOPOLOGY OPITMIZATION CODE BY LIU AND TOVAR (JUL 2013)", GitHub kliu99/Top3d release
  1.0 (8013 bytes). The erratum on top3d.app (lk_H8 scale 1/144 instead of 1/72) is already in that file.

The transcription keeps the MATLAB numbering (column-major reshapes, 1-based node / DOF numbers converted to 0-based
only at indexing time), the element matrices, the filter construction, the loads and supports and the OC loop as
printed. Only plotting and printing are left out. Deliberate deviations, each without numerical effect on the
quantities compared:

* SciPy's ``spsolve`` replaces MATLAB's backslash (CHOLMOD); both solve the same SPD system to rounding;
* ``max_loop`` caps top88's ``while change > 0.01`` (top88 has no cap; top3d's ``maxloop = 200`` is kept as is);
* top3d preallocates ``iH`` with ``(2(ceil(rmin)−1)+1)^2`` entries per element — a square, not a cube — and MATLAB grows
  the arrays when the 3-D loop writes past the end; the triplet lists here are built at their final length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.references import cite

__all__ = ["Top88", "Top3D", "lk_H8"]


def _filter_matrix(shape_f: tuple[int, ...], rmin: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """``H`` and ``Hs = sum(H, 2)`` of top88 lines 27-43 / top3d lines 39-61 (cone weights on index distance).

    ``shape_f`` is the MATLAB array shape of the element field, (nely, nelx) or (nely, nelx, nelz); element numbers
    are the column-major positions, as ``e1 = (k1−1)·nelx·nely + (i1−1)·nely + j1``.
    """
    r = math.ceil(rmin) - 1
    idx = np.indices(shape_f).reshape(len(shape_f), -1, order="F").T       # (nele, dim) 0-based, e1 order
    rows, cols, vals = [], [], []
    for off in np.ndindex(*([2 * r + 1] * len(shape_f))):
        d = np.asarray(off) - r
        nb = idx + d
        ok = np.all((nb >= 0) & (nb < np.asarray(shape_f)), axis=1)
        e1 = np.ravel_multi_index(tuple(idx[ok].T), shape_f, order="F")
        e2 = np.ravel_multi_index(tuple(nb[ok].T), shape_f, order="F")
        rows.append(e1)
        cols.append(e2)
        vals.append(np.full(len(e1), max(0.0, rmin - math.sqrt(float(np.sum(d * d))))))
    n = int(np.prod(shape_f))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    H.eliminate_zeros()                                   # MATLAB's sparse() drops the zero weights as well
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


@dataclass
class _History:
    compliance: list[float] = field(default_factory=list)
    change: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)


class _OCCode:
    """FE analysis, ft = 2 sensitivities and the OC loop, shared verbatim by top88 (lines 54-83) and top3d."""

    shape_f: tuple[int, ...]
    KE: np.ndarray
    edofMat: np.ndarray          # 0-based DOF numbers, (nele, nloc)
    F: np.ndarray
    freedofs: np.ndarray
    H: sp.csr_matrix
    Hs: np.ndarray
    penal: float
    volfrac: float
    E0 = 1.0
    Emin = 1e-9

    def __init__(self) -> None:
        self.nele = int(np.prod(self.shape_f))
        nloc = self.KE.shape[0]
        self.iK = np.kron(self.edofMat, np.ones((nloc, 1), np.int64)).ravel()       # reshape(kron(edofMat,ones(n,1))')
        self.jK = np.kron(self.edofMat, np.ones((1, nloc), np.int64)).ravel()

    def _col(self, a: np.ndarray) -> np.ndarray:
        """MATLAB ``a(:)``."""
        return np.asarray(a).reshape(-1, order="F")

    def _field(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v).reshape(self.shape_f, order="F")

    def analyse(self, xPhys: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """``c``, ``ce`` and ``U`` of the physical density (FE-ANALYSIS and OBJECTIVE blocks)."""
        xp = self._col(xPhys)
        sK = (self.KE.reshape(-1, 1, order="F") * (self.Emin + xp ** self.penal * (self.E0 - self.Emin))[None, :])
        sK = sK.reshape(-1, order="F")
        ndof = self.F.shape[0]
        K = sp.csr_matrix((sK, (self.iK, self.jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2
        U = np.zeros(ndof)
        fd = self.freedofs
        U[fd] = spla.spsolve(K[fd][:, fd].tocsc(), self.F[fd])
        Ue = U[self.edofMat]
        ce = self._field(np.sum((Ue @ self.KE) * Ue, axis=1))
        c = float(np.sum((self.Emin + self._field(xp) ** self.penal * (self.E0 - self.Emin)) * ce))
        return c, ce, U

    def sensitivities(self, xPhys: np.ndarray, ce: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``dc``, ``dv`` after the ft = 2 modification (density filter chain rule)."""
        dc = -self.penal * (self.E0 - self.Emin) * self._field(self._col(xPhys)) ** (self.penal - 1) * ce
        dv = np.ones(self.shape_f)
        dc = self._field(self.H @ (self._col(dc) / self.Hs))
        dv = self._field(self.H @ (self._col(dv) / self.Hs))
        return dc, dv

    def physical(self, x: np.ndarray) -> np.ndarray:
        """``xPhys(:) = (H*x(:))./Hs`` (the ft = 2 physical density, as computed inside the OC loop)."""
        return self._field((self.H @ self._col(x)) / self.Hs)

    def evaluate(self, x: np.ndarray) -> dict[str, np.ndarray | float]:
        """Objective and sensitivities of a fixed design x (MATLAB field layout) with xPhys = filter(x)."""
        xPhys = self.physical(x)
        c, ce, U = self.analyse(xPhys)
        dc, dv = self.sensitivities(xPhys, ce)
        return {"c": c, "dc": dc, "dv": dv, "xPhys": xPhys, "U": U}

    def optimise(self, max_loop: int, tolx: float = 0.01) -> dict[str, object]:
        """The INITIALIZE / START ITERATION blocks: x = volfrac, xPhys = x, OC with move 0.2, bisection 1e-3."""
        x = np.full(self.shape_f, self.volfrac)
        xPhys = x.copy()
        loop, change = 0, 1.0
        hist = _History()
        while change > tolx and loop < max_loop:
            loop += 1
            c, ce, _ = self.analyse(xPhys)
            dc, dv = self.sensitivities(xPhys, ce)
            hist.compliance.append(c)
            l1, l2, move = 0.0, 1e9, 0.2
            xnew = x
            while (l2 - l1) / (l1 + l2) > 1e-3:
                lmid = 0.5 * (l2 + l1)
                xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move,
                                                                                    x * np.sqrt(-dc / dv / lmid)))))
                xPhys = self.physical(xnew)
                if np.sum(xPhys) > self.volfrac * self.nele:
                    l1 = lmid
                else:
                    l2 = lmid
            change = float(np.max(np.abs(xnew - x)))
            x = xnew
            hist.change.append(change)
            hist.volume.append(float(np.mean(xPhys)))
        return {"compliance": hist.compliance, "change": hist.change, "volume": hist.volume, "x": x,
                "xPhys": xPhys, "iterations": loop}


@cite("andreassen_2011", note="top88.m, Nov 2010 version from the DTU TopOpt page, lines 1-83 (plotting left out)")
class Top88(_OCCode):
    """``top88(nelx, nely, volfrac, penal, rmin, ft=2)``: half MBB beam, E0 = 1, Emin = 1e-9, ν = 0.3.

    Element fields have MATLAB shape (nely, nelx); row 0 is the top row of elements (node 1 is the top-left node).
    """

    def __init__(self, nelx: int, nely: int, volfrac: float, penal: float, rmin: float, nu: float = 0.3) -> None:
        self.nelx, self.nely, self.volfrac, self.penal, self.rmin = nelx, nely, float(volfrac), float(penal), rmin
        self.shape_f = (nely, nelx)
        A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float)
        A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float)
        B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float)
        B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float)
        self.KE = 1 / (1 - nu ** 2) / 24 * (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12],
                                                                                                [B12.T, B11]]))
        nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((1 + nely, 1 + nelx), order="F")
        edofVec = (2 * nodenrs[:-1, :-1] + 1).reshape(-1, order="F")
        edof1 = edofVec[:, None] + np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1, -2, -1])
        self.edofMat = edof1 - 1
        ndof = 2 * (nely + 1) * (nelx + 1)
        self.F = np.zeros(ndof)
        self.F[2 - 1] = -1.0                                       # F = sparse(2,1,-1,...)
        fixed1 = np.union1d(np.arange(1, 2 * (nely + 1) + 1, 2), [2 * (nelx + 1) * (nely + 1)])
        self.fixeddofs = fixed1 - 1
        self.freedofs = np.setdiff1d(np.arange(ndof), self.fixeddofs)
        self.H, self.Hs = _filter_matrix(self.shape_f, rmin)
        super().__init__()

    def node_xy(self, node0: np.ndarray) -> np.ndarray:
        """Coordinates (x right, y up, element size 1) of 0-based node numbers (node 1 = top-left)."""
        n = np.asarray(node0)
        return np.column_stack([n // (self.nely + 1), self.nely - n % (self.nely + 1)])


@cite("liu_tovar_2014", note="lk_H8 (with the 1/144 erratum) and the 3-D loop of top3d.m, JUL 2013 release 1.0")
def lk_H8(nu: float) -> np.ndarray:
    """top3d's closed-form 24×24 trilinear hexahedron stiffness (unit cube, E = 1)."""
    A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                  [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]], float)
    k = 1 / 144 * A.T @ np.array([1.0, nu])
    k = np.concatenate([[np.nan], k])                              # k[1..14] as in MATLAB
    K1 = np.array([[k[1], k[2], k[2], k[3], k[5], k[5]], [k[2], k[1], k[2], k[4], k[6], k[7]],
                   [k[2], k[2], k[1], k[4], k[7], k[6]], [k[3], k[4], k[4], k[1], k[8], k[8]],
                   [k[5], k[6], k[7], k[8], k[1], k[2]], [k[5], k[7], k[6], k[8], k[2], k[1]]])
    K2 = np.array([[k[9], k[8], k[12], k[6], k[4], k[7]], [k[8], k[9], k[12], k[5], k[3], k[5]],
                   [k[10], k[10], k[13], k[7], k[4], k[6]], [k[6], k[5], k[11], k[9], k[2], k[10]],
                   [k[4], k[3], k[5], k[2], k[9], k[12]], [k[11], k[4], k[6], k[12], k[10], k[13]]])
    K3 = np.array([[k[6], k[7], k[4], k[9], k[12], k[8]], [k[7], k[6], k[4], k[10], k[13], k[10]],
                   [k[5], k[5], k[3], k[8], k[12], k[9]], [k[9], k[10], k[2], k[6], k[11], k[5]],
                   [k[12], k[13], k[10], k[11], k[6], k[4]], [k[2], k[12], k[9], k[4], k[5], k[3]]])
    K4 = np.array([[k[14], k[11], k[11], k[13], k[10], k[10]], [k[11], k[14], k[11], k[12], k[9], k[8]],
                   [k[11], k[11], k[14], k[12], k[8], k[9]], [k[13], k[12], k[12], k[14], k[7], k[7]],
                   [k[10], k[9], k[8], k[7], k[14], k[11]], [k[10], k[8], k[9], k[7], k[11], k[14]]])
    K5 = np.array([[k[1], k[2], k[8], k[3], k[5], k[4]], [k[2], k[1], k[8], k[4], k[6], k[11]],
                   [k[8], k[8], k[1], k[5], k[11], k[6]], [k[3], k[4], k[5], k[1], k[8], k[2]],
                   [k[5], k[6], k[11], k[8], k[1], k[8]], [k[4], k[11], k[6], k[2], k[8], k[1]]])
    K6 = np.array([[k[14], k[11], k[7], k[13], k[10], k[12]], [k[11], k[14], k[7], k[12], k[9], k[2]],
                   [k[7], k[7], k[14], k[10], k[2], k[9]], [k[13], k[12], k[10], k[14], k[7], k[11]],
                   [k[10], k[9], k[2], k[7], k[14], k[7]], [k[12], k[2], k[9], k[11], k[7], k[14]]])
    return 1 / ((nu + 1) * (1 - 2 * nu)) * np.block([[K1, K2, K3, K4], [K2.T, K5, K6, K3.T],
                                                     [K3.T, K6, K5.T, K2.T], [K4, K3, K2, K1.T]])


@cite("liu_tovar_2014", note="top3d.m, JUL 2013 release 1.0: loads, supports, edofMat, filter, OC loop")
class Top3D(_OCCode):
    """``top3d(nelx, nely, nelz, volfrac, penal, rmin)``: cantilever, face x = 0 clamped, −1 in y on every node of
    the edge x = nelx, y = 0 (z = 0..nelz). Element fields have MATLAB shape (nely, nelx, nelz), row 0 at the top."""

    def __init__(self, nelx: int, nely: int, nelz: int, volfrac: float, penal: float, rmin: float,
                 nu: float = 0.3) -> None:
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.volfrac, self.penal, self.rmin = float(volfrac), float(penal), rmin
        self.shape_f = (nely, nelx, nelz)
        il, jl, kl = np.meshgrid(nelx, 0, np.arange(nelz + 1))
        loadnid = kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl)
        loaddof = 3 * loadnid.reshape(-1, order="F") - 1
        iif, jf, kf = np.meshgrid(0, np.arange(nely + 1), np.arange(nelz + 1))
        fixednid = (kf * (nelx + 1) * (nely + 1) + iif * (nely + 1) + (nely + 1 - jf)).reshape(-1, order="F")
        fixeddof = np.concatenate([3 * fixednid, 3 * fixednid - 1, 3 * fixednid - 2])
        nele = nelx * nely * nelz
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        self.F = np.zeros(ndof)
        np.add.at(self.F, loaddof - 1, -1.0)                       # sparse() sums repeated entries
        self.loaddof = loaddof - 1
        self.fixeddofs = np.unique(fixeddof - 1)
        self.freedofs = np.setdiff1d(np.arange(ndof), self.fixeddofs)
        self.KE = lk_H8(nu)
        nodegrd = np.arange(1, (nely + 1) * (nelx + 1) + 1).reshape((nely + 1, nelx + 1), order="F")
        nodeids = nodegrd[:-1, :-1].reshape(-1, order="F")
        nodeidz = np.arange(0, (nelz - 1) * (nely + 1) * (nelx + 1) + 1, (nely + 1) * (nelx + 1))
        nodeids = nodeids[:, None] + nodeidz[None, :]
        edofVec = 3 * nodeids.reshape(-1, order="F") + 1
        layer = 3 * (nely + 1) * (nelx + 1)
        base = [0, 1, 2, 3 * nely + 3, 3 * nely + 4, 3 * nely + 5, 3 * nely + 0, 3 * nely + 1, 3 * nely + 2, -3, -2, -1]
        offs = np.array(base + [layer + b for b in base])
        assert len(offs) == 24 and nele == len(edofVec)
        self.edofMat = edofVec[:, None] + offs[None, :] - 1
        self.H, self.Hs = _filter_matrix(self.shape_f, rmin)
        super().__init__()

    def node_xyz(self, node0: np.ndarray) -> np.ndarray:
        """Coordinates (element size 1) of 0-based node numbers: n = k(nelx+1)(nely+1) + i(nely+1) + (nely − y)."""
        n = np.asarray(node0)
        layer = (self.nelx + 1) * (self.nely + 1)
        k, rest = n // layer, n % layer
        i, r = rest // (self.nely + 1), rest % (self.nely + 1)
        return np.column_stack([i, self.nely - r, k])
