"""NumPy transcriptions of published educational codes, used only as independent references by
tests/toporlogy/test_independent_vv.py and tests/toporlogy/test_published_values.py. Nothing here imports toporlogy's
FE, filter or optimizer code.

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

Read on 2026-11-25 (full text of the arXiv PDFs; the codes are printed in their appendices):

* topBuck250 — Ferrari, Sigmund & Guest, "Topology Optimization with linearized buckling criteria in 250 lines of
  Matlab", arXiv:2101.02973v1 (published as SMO 63:3045-3066, 2021), Appendix C, lines 3-154, and the replacement of
  lines 64-70 for the wall example of its §5.2. See :class:`TopBuck250` for what is transcribed.
* Stress_3D_Sensitivity — Deng, Vulimiri & To, "An efficient 146-line 3D sensitivity analysis code of stress-based
  topology optimization written in MATLAB", arXiv:2104.01210v1, Appendix A, lines 2-46 (the forward part: FE, stress,
  von Mises, p-norm). See :func:`deng_pnorm`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.references import cite

__all__ = ["Top88", "Top3D", "lk_H8", "TopBuck250", "DENG_B", "deng_pnorm"]


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


# ================================================================================ topBuck250 (Ferrari et al. 2021)
def _imfilter(a: np.ndarray, kernel: np.ndarray, bcF: str | int) -> np.ndarray:
    """MATLAB ``imfilter(a, kernel, bcF)``: correlation, same size; 'symmetric' mirrors the border including the edge
    element (SciPy's ``reflect``), 0 pads with zeros."""
    if bcF == "symmetric":
        return np.asarray(ndi.correlate(a, kernel, mode="reflect"))
    return np.asarray(ndi.correlate(a, kernel, mode="constant", cval=0.0))


@cite("ferrari_sigmund_guest_2021", note="topBuck250, arXiv:2101.02973v1 Appendix C lines 3-154; wall: §5.2")
class TopBuck250:
    """The analysis part of ``topBuck250`` at one design: physical density, stiffness, displacements, element stress,
    stress stiffness G and the buckling eigenvalues (Appendix C, lines 3-154).

    ``setup="column"`` uses the code's own loads, supports and passive set (lines 64-69: left edge clamped, x-load
    −1e-3 spread over the 17 right-edge nodes around mid-height, a 20 × 10 passive-solid block under the load);
    ``setup="wall"`` uses the replacement printed in §5.2 for the wall-reinforcement example (only for 320 × 320,
    since the supported node columns are printed as numbers).

    Transcribed literally: material (line 3), numbering and connectivity (lines 11-16), the element matrix from the
    lists c1, c2 (lines 18-25), the stress and Gauss-point operators (lines 34-39), loads / supports / passive sets,
    filter and projection (lines 73-83, ft = 2 path), the initial design (lines 93-96), the physical density
    (lines 101-111), the stiffness interpolation and solve (lines 114-120), centroid stresses (line 124), the
    integration of G (lines 126-148) and the eigenproblem of ``K⁻¹G`` for the most negative eigenvalues
    (lines 150-152). Element fields have MATLAB shape (nely, nelx), row 0 at the top.

    Deviations, without effect on the numbers compared: G is assembled from the full 4 × 4 node-pair matrix of every
    element instead of the 10 unique coefficients and the index arrays IkG / a1 / a2 (same matrix); K is assembled
    from the full 8 × 8 matrix instead of its lower triangle plus symmetrisation (same matrix); SuperLU replaces
    MATLAB's Cholesky ``decomposition``; ARPACK's ``eigs(..., which="SR")`` replaces MATLAB's ``eigs(..., 'sa')``
    on the same operator v ↦ K⁻¹ G v. Not transcribed: sensitivities, KS aggregation, the OC/MMA update, plotting.

    ``stress_at="gauss"`` is a labelled deviation from the published code (one line): the stress entering G is taken
    at each Gauss point instead of the element centroid. It exists to separate the two formulations in the tests.
    """

    E0, Emin, nu = 1.0, 1e-6, 0.3                                                  # line 3

    def __init__(self, nelx: int, nely: int, Lx: float, setup: str = "column", rmin: float = 4.0, ft: int = 2,
                 ftBC: str = "N", eta: float = 0.5, beta: float = 2.0, stress_at: str = "centroid") -> None:
        if stress_at not in ("centroid", "gauss"):
            raise ValueError("stress_at: 'centroid' (published) or 'gauss'")
        nu = self.nu
        self.nelx, self.nely, self.Lx, self.stress_at = nelx, nely, float(Lx), stress_at
        self.shape_f = (nely, nelx)
        self.Ly = nely / nelx * Lx                                                  # line 11
        self.nEl = nEl = nelx * nely
        elNrs = np.arange(1, nEl + 1).reshape((nely, nelx), order="F")
        nodeNrs = np.arange(1, (1 + nely) * (1 + nelx) + 1).reshape((1 + nely, 1 + nelx), order="F")
        cVec = (2 * nodeNrs[:-1, :-1] + 1).reshape(-1, order="F")
        self.cMat = cVec[:, None] + np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1]) - 1
        self.nDof = nDof = (1 + nely) * (1 + nelx) * 2
        # ------------------------------------------------------------ element stiffness, lines 18-25
        c1 = np.array([12, 3, -6, -3, -6, -3, 0, 3, 12, 3, 0, -3, -6, -3, -6, 12, -3, 0, -3, -6, 3, 12, 3,
                       -6, 3, -6, 12, 3, -6, -3, 12, 3, 0, 12, -3, 12], float)
        c2 = np.array([-4, 3, -2, 9, 2, -3, 4, -9, -4, -9, 4, -3, 2, 9, -2, -4, -3, 4, 9, 2, 3, -4, -9, -2,
                       3, 2, -4, 3, -2, 9, -4, -9, 4, -4, -3, -4], float)
        assert len(c1) == len(c2) == 36
        Ke = 1 / (1 - nu ** 2) / 24 * (c1 + nu * c2)
        flat = np.zeros(64)
        flat[np.flatnonzero(np.tril(np.ones((8, 8))).ravel(order="F") == 1)] = Ke    # Ke0(tril(ones(8))==1) = Ke'
        Ke0 = flat.reshape((8, 8), order="F")
        self.Ke0 = Ke0 + Ke0.T - np.diag(np.diag(Ke0))
        # ------------------------------------------------------------ stress and Gauss-point operators, lines 34-39
        self.Cmat0 = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)
        self.xiG = math.sqrt(1 / 3) * np.array([-1.0, 1.0])
        self.xe = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * Lx / nelx / 2
        lMat = np.zeros((3, 4))
        lMat[0, 0] = 1
        lMat[1, 3] = 1
        lMat[2, 1:3] = 1
        self.lMat = lMat
        # ------------------------------------------------------------ loads, supports, passive domains
        F = np.zeros(nDof)
        if setup == "column":                                                       # lines 64-69
            fixed1 = np.arange(1, 2 * (nely + 1) + 1)
            lcDof = 2 * nodeNrs[nely // 2 + np.arange(-8, 9), -1] - 1               # nodeNrs(nely/2+1+[-8:8], end)
            modF = 1e-3 / self.Ly / (len(lcDof) - 1)
            F[lcDof - 1] = -modF
            pasS = elNrs[nely // 2 - 1 + np.arange(-9, 11), nelx - 10:].reshape(-1, order="F")
            pasV = np.zeros(0, np.int64)
        elif setup == "wall":                                                       # §5.2, replaces lines 64-70
            if nelx != 320 or nely != 320:
                raise ValueError("the wall's supported node columns are printed for 320 x 320 only")
            cols = np.concatenate([np.arange(1, 10), np.arange(120, 129), np.arange(257, 266), np.arange(313, 322)])
            fixedNodes = nodeNrs[-1, cols - 1]
            fixed1 = np.concatenate([2 * fixedNodes, 2 * fixedNodes - 1])
            lcDof = 2 * nodeNrs[:, 0] - 1
            modF = 1e-2 / self.Ly / (len(lcDof) - 1)
            F[lcDof - 1] = modF
            tx, ty, f5 = nelx // 40, nely // 40, nely // 5
            # MATLAB ranges a:b (1-based, inclusive) become Python a-1:b
            a1 = elNrs[:ty, :]                                                      # (1:nely/40, :)
            a2 = elNrs[:, np.r_[0:tx, nelx - tx:nelx]]                              # (:, [1:nelx/40, end-nelx/40+1:end])
            a3 = elNrs[2 * f5 - 1:, 2 * nelx // 5 - 1:nelx - nelx // 5]           # (2nely/5:end, 2nelx/5:end-nelx/5)
            a4 = elNrs[2 * f5 - ty - 1:, 2 * nelx // 5 - tx - 1:2 * nelx // 5 - 1]
            a5 = elNrs[2 * f5 - ty - 1:, nelx - nelx // 5:nelx - nelx // 5 + tx]
            a6 = elNrs[2 * f5 - ty - 1:2 * f5 - 1, 2 * nelx // 5 - 1:nelx - nelx // 5]
            pasS = np.unique(np.concatenate([a.reshape(-1, order="F") for a in (a1, a2, a4, a5, a6)]))
            pasV = a3.reshape(-1, order="F")
        else:
            raise ValueError("setup: 'column' or 'wall'")
        F[lcDof[0] - 1] /= 2                                                        # consistent load on end nodes
        F[lcDof[-1] - 1] /= 2
        self.F = F
        self.lcDof = lcDof - 1
        self.fixed = np.unique(fixed1 - 1)
        self.free = np.setdiff1d(np.arange(nDof), self.fixed)
        self.pasS, self.pasV = pasS - 1, pasV - 1
        self.act = np.setdiff1d(np.arange(nEl), np.union1d(self.pasS, self.pasV))
        # ------------------------------------------------------------ filter and projection, lines 73-83
        self.bcF: str | int = "symmetric" if ftBC == "N" else 0
        r = math.ceil(rmin)
        dy, dx = np.meshgrid(np.arange(-r + 1, r), np.arange(-r + 1, r))
        self.h = np.maximum(0.0, rmin - np.sqrt(dx ** 2 + dy ** 2))
        self.Hs = _imfilter(np.ones(self.shape_f), self.h, self.bcF)
        self.ft, self.eta, self.beta = ft, float(eta), float(beta)

    # ------------------------------------------------------------------ helpers
    def dN(self, xi: float, zi: float) -> np.ndarray:
        """Line 38: logical derivatives of the four shape functions."""
        return 0.25 * np.array([[zi - 1, 1 - zi, 1 + zi, -1 - zi], [xi - 1, -1 - xi, 1 + xi, 1 - xi]])

    def B0(self, gradN: np.ndarray) -> np.ndarray:
        """Line 39: strain-displacement matrix."""
        return self.lMat @ np.kron(gradN, np.eye(2))

    def grad_n(self, xi: float, zi: float) -> np.ndarray:
        return np.linalg.solve(self.dN(xi, zi) @ self.xe, self.dN(xi, zi))         # (dN*xe)\dN

    def prj(self, v: np.ndarray) -> np.ndarray:
        e, b = self.eta, self.beta
        return (np.tanh(b * e) + np.tanh(b * (v - e))) / (np.tanh(b * e) + np.tanh(b * (1 - e)))

    def node_xy(self, node0: np.ndarray) -> np.ndarray:
        """Coordinates (x right, y up, element size Lx/nelx) of 0-based node numbers (node 1 = top-left)."""
        n = np.asarray(node0)
        hx = self.Lx / self.nelx
        return np.column_stack([n // (self.nely + 1) * hx, (self.nely - n % (self.nely + 1)) * hx])

    # ------------------------------------------------------------------ the published steps
    def initial_x(self, volfrac: float) -> np.ndarray:
        """Lines 85-95 (no x0): x on the active set, 1 on pasS, 0 on pasV. Column vector in element order."""
        x = np.zeros(self.nEl)
        x[self.act] = (volfrac * (self.nEl - len(self.pasV)) - len(self.pasS)) / len(self.act)
        x[self.pasS] = 1.0
        return x

    def physical(self, x: np.ndarray) -> np.ndarray:
        """Lines 96 and 101-111 at loop 1 for ft ≤ 2 (no η Newton loop): xPhys = x, filtered values on the active
        set, then the projection of every element."""
        xPhys = np.array(x, float)
        xTilde = _imfilter(np.asarray(x).reshape(self.shape_f, order="F"), self.h, self.bcF) / self.Hs
        xPhys[self.act] = xTilde.reshape(-1, order="F")[self.act]
        if self.ft > 1:
            if self.ft == 3:
                raise NotImplementedError("ft = 3 (η Newton loop) is not transcribed")
            xPhys = self.prj(xPhys)
        return xPhys

    def analyse(self, xPhys: np.ndarray, penalK: float, penalG: float, n_eig: int,
                eigen: bool = True) -> dict[str, Any]:
        """Lines 114-154: K, U, c = FᵀU, centroid stresses, G and the n_eig + 4 eigenvalues μ = −D (descending)."""
        E0, Emin = self.E0, self.Emin
        xp = np.asarray(xPhys, float)
        cMat, nDof, free = self.cMat, self.nDof, self.free
        sK = Emin + xp ** penalK * (E0 - Emin)                                      # line 114
        iK = np.repeat(cMat, 8, axis=1).ravel()
        jK = np.tile(cMat, (1, 8)).ravel()
        K = sp.csr_matrix(((sK[:, None] * self.Ke0.ravel()[None, :]).ravel(), (iK, jK)), shape=(nDof, nDof))
        Kff = K[free][:, free].tocsc()
        lu = spla.splu(Kff)
        U = np.zeros(nDof)
        U[free] = lu.solve(self.F[free])                                            # line 120
        Ue = U[cMat]
        sGP = Ue @ (self.Cmat0 @ self.B0(self.grad_n(0.0, 0.0))).T                  # line 124: (nEl, 3)
        kn = np.zeros((self.nEl, 4, 4))
        for xi in self.xiG:                                                          # lines 126-142
            for zi in self.xiG:
                w = 1.0 * 1.0 * np.linalg.det(self.dN(xi, zi) @ self.xe)           # wxi(j)*weta(k)*det(...)
                gN = self.grad_n(xi, zi)
                a, b = gN[0], gN[1]
                s = sGP if self.stress_at == "centroid" else Ue @ (self.Cmat0 @ self.B0(gN)).T
                kn += w * (s[:, 0, None, None] * np.outer(a, a)[None] + s[:, 1, None, None] * np.outer(b, b)[None]
                           + s[:, 2, None, None] * (np.outer(a, b) + np.outer(b, a))[None])
        kn *= (E0 * xp ** penalG)[:, None, None]                                     # line 143: sG0
        Ge = np.zeros((self.nEl, 8, 8))
        Ge[:, 0::2, 0::2] = kn                                                       # x-x and y-y node pairs
        Ge[:, 1::2, 1::2] = kn
        G = sp.csr_matrix((Ge.ravel(), (iK, jK)), shape=(nDof, nDof))
        out: dict[str, Any] = {"K": K, "U": U, "c": float(self.F @ U), "sGP": sGP, "G": G, "xPhys": xp}
        if eigen:
            Gff = G[free][:, free].tocsr()
            n = len(free)
            op = spla.LinearOperator((n, n), matvec=lambda v: lu.solve(Gff @ v), dtype=float)   # matFun
            D = spla.eigs(op, k=n_eig + 4, which="SR", v0=np.random.default_rng(0).standard_normal(n),
                          return_eigenvectors=False)
            out["mu"] = np.sort(-np.real(D))[::-1]                                  # lines 151-152
        return out


# ================================================================================ Deng, Vulimiri & To (2021)
#: The strain matrix printed in Stress_3D_Sensitivity, lines 126-144 (B = [B_1, B_2, B_3], 5 significant digits),
#: rows εx, εy, εz, γxy, γyz, γzx; columns u, v, w of the eight nodes in top3d's element order.
DENG_B = np.hstack([
    np.array([[-0.044658, 0, 0, 0.044658, 0, 0, 0.16667, 0],
              [0, -0.044658, 0, 0, -0.16667, 0, 0, 0.16667],
              [0, 0, -0.044658, 0, 0, -0.16667, 0, 0],
              [-0.044658, -0.044658, 0, -0.16667, 0.044658, 0, 0.16667, 0.16667],
              [0, -0.044658, -0.044658, 0, -0.16667, -0.16667, 0, -0.62201],
              [-0.044658, 0, -0.044658, -0.16667, 0, 0.044658, -0.62201, 0]]),
    np.array([[0, -0.16667, 0, 0, -0.16667, 0, 0, 0.16667],
              [0, 0, 0.044658, 0, 0, -0.16667, 0, 0],
              [-0.62201, 0, 0, -0.16667, 0, 0, 0.044658, 0],
              [0, 0.044658, -0.16667, 0, -0.16667, -0.16667, 0, -0.62201],
              [0.16667, 0, -0.16667, 0.044658, 0, 0.044658, -0.16667, 0],
              [0.16667, -0.16667, 0, -0.16667, 0.044658, 0, -0.16667, 0.16667]]),
    np.array([[0, 0, 0.62201, 0, 0, -0.62201, 0, 0],
              [-0.62201, 0, 0, 0.62201, 0, 0, 0.16667, 0],
              [0, 0.16667, 0, 0, 0.62201, 0, 0, 0.16667],
              [0.16667, 0, 0.62201, 0.62201, 0, 0.16667, -0.62201, 0],
              [0.16667, -0.62201, 0, 0.62201, 0.62201, 0, 0.16667, 0.16667],
              [0, 0.16667, 0.62201, 0, 0.62201, 0.16667, 0, -0.62201]])])


def deng_D(nu: float = 0.3) -> np.ndarray:
    """Stress_3D_Sensitivity lines 77-79: isotropic elasticity matrix, E = 1, engineering shear strains."""
    return 1.0 / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1 - nu, nu, nu, 0, 0, 0], [nu, 1 - nu, nu, 0, 0, 0], [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0], [0, 0, 0, 0, (1 - 2 * nu) / 2, 0], [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]])


def deng_pnorm(ref: Top3D, xPhys: np.ndarray, q: float, p: float, B: np.ndarray = DENG_B,
               nu: float = 0.3) -> dict[str, Any]:
    """Forward part of Stress_3D_Sensitivity (lines 2-46): FE, relaxed element stress ``x^q D B u_e``, von Mises
    and the p-norm ``(Σ vm^p)^(1/p)``.

    The FE lines 3-32 are top3d's in everything that enters the numbers: the same lk_H8 text (A, k and K1-K6 of
    lines 81-124 read equal to top3d's), the same load (−1 in y on the nodes of the edge x = nelx, y = 0), support
    (face x = 0), edofMat, E0 = 1, Emin = 1e-9, ``(K + K')/2`` and backslash. ``ref`` therefore supplies them (a
    :class:`Top3D` with ``penal = pl``). The element order of ``x(i)`` is top3d's column-major order.
    """
    c, _, U = ref.analyse(xPhys)
    x = ref._col(xPhys)
    S = x[:, None] ** q * (U[ref.edofMat] @ (deng_D(nu) @ B).T)                     # line 36
    mises = np.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 0] - S[:, 2]) ** 2 + (S[:, 1] - S[:, 2]) ** 2
                           + 6 * np.sum(S[:, 3:6] ** 2, axis=1)))                    # lines 38-39
    return {"pnorm": float(np.sum(mises ** p) ** (1 / p)), "mises": mises, "S": S, "U": U, "c": c}
