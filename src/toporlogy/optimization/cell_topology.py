"""Inverse homogenisation on a general periodic mesh: the material layout of one periodic cell that maximises an
effective property (``docs/foundations/prereg/IH_inverse_homogenization.md``).

x̃ = F x with the periodic density filter below, E_e = emin + x̃_e^p (1 − emin) times the unit-modulus ``D``, the
effective stiffness from :func:`toporlogy.fem.mesh.homogenization.homogenize_mesh_elasticity` (self-adjoint dC/dE_e),
the volume fraction Σ|Ω_e| x̃_e / Σ|Ω_e| ≤ ``volume``, MMA. Objective ``"bulk"``: the 2-D bulk modulus
κ = (C11 + C12 + C21 + C22)/4 (Voigt [xx, yy, xy]), maximised. For the SIMP material to stay inside the
Hashin–Shtrikman bounds the exponent must exceed
:func:`toporlogy.core.micromechanics.required_interpolation_parameters` (3.08 for ν = 0.3, plane stress).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import ElasticProblem

__all__ = ["PeriodicDensityFilter", "CellOptimizeResult", "cell_objective", "optimize_cell", "CELL_OBJECTIVES"]

CELL_OBJECTIVES = ("bulk", "poisson", "poisson_stress")


class PeriodicDensityFilter:
    """x̃ = F x: cone weights max(0, rmin − d_ij) |Ω_j| with d_ij the minimum-image distance of the cell centroids
    over the lattice translates n·a (n ∈ {−1, 0, 1}^dim), each row normalised by its sum. ``matrix`` is F."""

    def __init__(self, mesh: Mesh, lattice: Any, rmin: float, cell_volumes: np.ndarray | None = None) -> None:
        if not rmin > 0:
            raise ValueError(f"rmin must be > 0 (a length), got {rmin}")
        L = np.atleast_2d(np.asarray(lattice, float))
        if L.shape != (mesh.dim, mesh.dim):
            raise ValueError(f"lattice must be ({mesh.dim}, {mesh.dim}), got {L.shape}")
        c = mesh.cell_centroids()
        n = len(c)
        vol = np.ones(n) if cell_volumes is None else np.asarray(cell_volumes, float)
        if vol.shape != (n,) or not (vol > 0).all():
            raise ValueError("cell_volumes must be (n_cells,) and > 0")
        tree = cKDTree(c)
        rows, cols, vals = [], [], []
        for shift in itertools.product((-1, 0, 1), repeat=mesh.dim):
            t = np.asarray(shift, float) @ L
            for i, nb in enumerate(tree.query_ball_point(c - t, r=float(rmin))):
                nb = np.asarray(nb, np.int64)
                if not len(nb):
                    continue
                w = float(rmin) - np.linalg.norm(c[nb] + t - c[i], axis=1)
                keep = w > 0
                rows.append(np.full(int(keep.sum()), i))
                cols.append(nb[keep])
                vals.append(w[keep] * vol[nb[keep]])
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        H.sum_duplicates()
        self.matrix = (sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H).tocsr()
        self._T = self.matrix.T.tocsr()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ np.asarray(x, float))

    def backward(self, g: np.ndarray) -> np.ndarray:
        """Fᵀ g (the chain rule through the filter)."""
        return np.asarray(self._T @ np.asarray(g, float))


@dataclass
class CellOptimizeResult:
    """Design ``x``, filtered ``x_phys``, the objective value (κ for "bulk"), the effective ``C``, the volume
    fraction, iterations, the stop reason ("max_iter" or "kkt"), the last KKT residual and the history."""

    x: np.ndarray
    x_phys: np.ndarray
    objective: float
    C: np.ndarray
    volume: float
    iterations: int
    stop: str
    kkt: float | None
    history: dict[str, list[float]] = field(default_factory=dict)


def _bulk(C: np.ndarray) -> tuple[float, np.ndarray]:
    """κ and dκ/dC for 2-D Voigt C."""
    w = np.zeros((3, 3))
    w[:2, :2] = 0.25
    return float(np.sum(w * C)), w


def _evaluate(mesh: Mesh, lattice: Any, D: Any, x_phys: np.ndarray, penal: float, emin: float,
              objective: str) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Objective value and gradient with respect to x̃, C, and dC11/dx̃ (for the stiffness constraint)."""
    if objective not in CELL_OBJECTIVES:
        raise ValueError(f"objective must be one of {CELL_OBJECTIVES}, got {objective!r}")
    if mesh.dim != 2:
        raise ValueError("the cell objectives are written for 2-D cells")
    E = emin + x_phys ** penal * (1.0 - emin)
    h = homogenize_mesh_elasticity(mesh, D, lattice, E, sensitivities=True)
    assert h.dC is not None
    C = h.C
    if objective == "bulk":
        val, w = _bulk(C)
        dE = np.einsum("ij,eij->e", w, h.dC)
    elif objective == "poisson":                              # maximise −C12/C11 (stiffness ratio)
        val = -C[0, 1] / C[0, 0]
        dE = -(h.dC[:, 0, 1] / C[0, 0] - C[0, 1] * h.dC[:, 0, 0] / C[0, 0] ** 2)
    else:                                                     # "poisson_stress": maximise −ν12 = S21/S11
        S = np.linalg.inv(C)
        dS = -np.einsum("ik,ekl,lj->eij", S, h.dC, S, optimize=True)
        val = S[1, 0] / S[0, 0]
        dE = dS[:, 1, 0] / S[0, 0] - S[1, 0] * dS[:, 0, 0] / S[0, 0] ** 2
    # same operation order as before the poisson objective (bitwise bulk path, IP2 / IA4)
    return (val, dE * penal * x_phys ** (penal - 1.0) * (1.0 - emin), C,
            h.dC[:, 0, 0] * penal * x_phys ** (penal - 1.0) * (1.0 - emin))


def _e1_gradient(mesh: Mesh, lattice: Any, D: Any, x_phys: np.ndarray, penal: float, emin: float
                 ) -> tuple[float, np.ndarray]:
    """E1 = 1/S11 and dE1/dx̃ (the stiffness under uniaxial stress along x, shear free)."""
    E = emin + x_phys ** penal * (1.0 - emin)
    h = homogenize_mesh_elasticity(mesh, D, lattice, E, sensitivities=True)
    assert h.dC is not None
    S = np.linalg.inv(h.C)
    dS00 = -np.einsum("k,ekl,l->e", S[0], h.dC, S[:, 0], optimize=True)
    return 1.0 / S[0, 0], -dS00 / S[0, 0] ** 2 * penal * x_phys ** (penal - 1.0) * (1.0 - emin)


def cell_objective(mesh: Mesh, lattice: Any, D: Any, x_phys: np.ndarray, penal: float, emin: float,
                   objective: str = "bulk") -> tuple[float, np.ndarray, np.ndarray]:
    """Objective value (κ for "bulk", −C12/C11 for "poisson"; both maximised), its gradient with respect to x̃ and
    C, for the physical densities ``x_phys``."""
    val, grad, C, _ = _evaluate(mesh, lattice, D, x_phys, penal, emin, objective)
    return val, grad, C


def _moduli_gradients(mesh: Mesh, lattice: Any, D: Any, x_phys: np.ndarray, penal: float, emin: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(E1, E2, G12) = 1/S_ii from the full compliance and their gradients (3, n_cells) with respect to x̃."""
    E = emin + x_phys ** penal * (1.0 - emin)
    h = homogenize_mesh_elasticity(mesh, D, lattice, E, sensitivities=True)
    assert h.dC is not None
    S = np.linalg.inv(h.C)
    fac = penal * x_phys ** (penal - 1.0) * (1.0 - emin)
    vals = 1.0 / np.diag(S)
    grads = np.stack([np.einsum("k,ekl,l->e", S[i], h.dC, S[:, i], optimize=True) / S[i, i] ** 2 * fac
                      for i in range(3)])
    return vals, grads


def _project(xb: np.ndarray, beta: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """tanh projection (the formula of :class:`toporlogy.design.chain.HeavisideProjection`) and its derivative."""
    a, d = np.tanh(beta * eta), np.tanh(beta * (1.0 - eta))
    th = np.tanh(beta * (xb - eta))
    return (a + th) / (a + d), beta * (1.0 - th ** 2) / (a + d)


def optimize_cell(mesh: Mesh, lattice: Any, D: Any, volume: float, rmin: float, objective: str = "bulk",
                  penal: float = 3.5, emin: float = 1e-9, max_iter: int = 300, move: float = 0.2,
                  x0: np.ndarray | None = None, kkt_tol: float | None = None,
                  projection: tuple[float, Sequence[float]] | None = None,
                  stage_iter: int = 100, min_C11: float | None = None,
                  min_E1: float | None = None,
                  min_moduli: tuple[float, float, float] | None = None) -> CellOptimizeResult:
    """Maximise the effective ``objective`` of the periodic cell under a volume fraction (module notes).

    ``projection=(eta, betas)``: the filtered field is projected by the tanh step with threshold ``eta``; β runs
    through ``betas``, each stage until ``kkt_tol`` or ``stage_iter`` iterations, MMA restarted per stage, and the
    volume is taken on the projected field (``prereg/IP_projection.md``). Without it one stage of ``max_iter``.
    ``min_C11``: the extra constraint C11 ≥ min_C11 (with ``objective="poisson"``, ``prereg/IA_auxetic_cell.md``);
    ``min_E1``: E1 = 1/S11 ≥ min_E1 (with ``objective="poisson_stress"``, ``prereg/IAb_auxetic_compliance.md``);
    ``min_moduli=(E1, E2, G12)``: all three 1/S_ii bounded below (``prereg/IAc_auxetic_bounded.md``)."""
    from toporlogy.optimization.diagnostics import kkt_estimate
    from toporlogy.optimizers.mma import MMA
    if not 0 < volume < 1:
        raise ValueError(f"volume must be in (0, 1), got {volume}")
    vol = ElasticProblem(mesh, np.asarray(D, float)).geom.dV.sum(axis=1)
    vw = vol / vol.sum()
    filt = PeriodicDensityFilter(mesh, lattice, rmin, vol)
    x = np.full(mesh.n_cells, float(volume)) if x0 is None else np.asarray(x0, float).copy()
    if x.shape != (mesh.n_cells,):
        raise ValueError(f"x0 must have {mesh.n_cells} entries")
    if projection is None:
        stages: list[float | None] = [None]
        caps = [int(max_iter)]
        eta = 0.5
    else:
        eta = float(projection[0])
        stages = [float(b) for b in projection[1]]
        if not stages or not all(b > 0 for b in stages if b is not None) or not 0 < eta < 1:
            raise ValueError("projection must be (eta in (0, 1), non-empty positive betas)")
        caps = [int(stage_iter)] * len(stages)
    hist: dict[str, list[float]] = {"objective": [], "volume": [], "change": [], "beta": []}
    f_ref: float | None = None
    stop, kkt, it = "max_iter", None, 0

    def physical(x: np.ndarray, beta: float | None) -> tuple[np.ndarray, np.ndarray | None]:
        xb = filt(x)
        if beta is None:
            return xb, None
        return _project(xb, beta, eta)

    val, C = 0.0, np.zeros((3, 3))
    xp = filt(x)
    for beta, cap in zip(stages, caps, strict=True):
        opt = MMA(move=move)
        k = 0
        stop = "max_iter"
        while k < cap:
            xp, dp = physical(x, beta)
            val, dval, C, dc11 = _evaluate(mesh, lattice, D, xp, penal, emin, objective)
            f_ref = abs(val) if f_ref is None else f_ref
            if dp is None:
                f0, df0 = -val / f_ref, -filt.backward(dval) / f_ref              # MMA minimises
                dg = (filt.backward(vw) / volume)[None, :]
            else:
                f0, df0 = -val / f_ref, -filt.backward(dval * dp) / f_ref
                dg = (filt.backward(vw * dp) / volume)[None, :]
            v = float(vw @ xp)
            g = np.array([v / volume - 1.0])
            if min_C11 is not None:
                g = np.append(g, 1.0 - C[0, 0] / min_C11)
                dc = -dc11 / min_C11
                dg = np.vstack([dg, filt.backward(dc if dp is None else dc * dp)[None, :]])
            if min_E1 is not None:
                e1, de1 = _e1_gradient(mesh, lattice, D, xp, penal, emin)
                g = np.append(g, 1.0 - e1 / min_E1)
                de = -de1 / min_E1
                dg = np.vstack([dg, filt.backward(de if dp is None else de * dp)[None, :]])
            if min_moduli is not None:
                mv, mg = _moduli_gradients(mesh, lattice, D, xp, penal, emin)
                lim = np.asarray(min_moduli, float)
                g = np.concatenate([g, 1.0 - mv / lim])
                rows = -mg / lim[:, None]
                dg = np.vstack([dg] + [filt.backward(r if dp is None else r * dp)[None, :] for r in rows])
            hist["objective"].append(val)
            hist["volume"].append(v)
            hist["beta"].append(float("nan") if beta is None else beta)
            if kkt_tol is not None:
                kkt = float(kkt_estimate(x, df0, g, dg)["residual"])
                if kkt <= kkt_tol:
                    stop = "kkt"
                    break
            it += 1
            k += 1
            xn = opt.step(x, f0, df0, g, dg)
            hist["change"].append(float(np.abs(xn - x).max()))
            x = xn
        else:
            xp, _ = physical(x, beta)
            val, _, C, _ = _evaluate(mesh, lattice, D, xp, penal, emin, objective)
    return CellOptimizeResult(x, xp, val, C, float(vw @ xp), it, stop, kkt, hist)
