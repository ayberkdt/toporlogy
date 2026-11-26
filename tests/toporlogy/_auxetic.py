"""A negative Poisson ratio cell by inverse homogenisation (docs/olcumler/auxetic_hucre_onkayit.md, A1-A4); used by
``test_auxetic_cell.py`` and ``docs/olcumler/auxetic_hucre.py``."""
from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from scipy import ndimage

from toporlogy.core.domain import Domain
from toporlogy.fem.homogenization import homogenize_elasticity
from toporlogy.optimizers.mma import MMA

N, NU, P, EMIN, RMIN = 30, 0.3, 3.0, 1e-9, 1.5
VOL, C11_MIN, MAX_ITER = 0.40, 0.02, 200


def _kernel() -> np.ndarray:
    r = int(np.ceil(RMIN))
    i, j = np.mgrid[-r:r + 1, -r:r + 1]
    w = np.maximum(0.0, RMIN - np.sqrt(i * i + j * j))
    return w / w.sum()


def filt(x: np.ndarray) -> np.ndarray:
    """Periodic cone filter (normalised once: the adjoint is the same convolution)."""
    return ndimage.convolve(x, _kernel(), mode="wrap")


def start() -> np.ndarray:
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return 0.4 + 0.2 * np.cos(2 * np.pi * i / N) * np.cos(2 * np.pi * j / N)


def evaluate(x: np.ndarray) -> dict[str, Any]:
    """Objective ν = C12/C11 and constraints g = [vol/VOL − 1, 1 − C11/C11_MIN] with gradients in x (N, N)."""
    d = Domain((N, N))
    xt = filt(x)
    E = EMIN + (1 - EMIN) * xt.ravel() ** P
    h = homogenize_elasticity(d, E, NU, sensitivities=True)
    assert h.dC is not None
    C, dC = h.C, h.dC
    dE = (P * (1 - EMIN) * xt.ravel() ** (P - 1))
    nu = C[0, 1] / C[0, 0]
    dnu_dE = (dC[:, 0, 1] * C[0, 0] - C[0, 1] * dC[:, 0, 0]) / C[0, 0] ** 2
    dnu = filt((dnu_dE * dE).reshape(N, N))
    vol = xt.mean()
    dvol = filt(np.full((N, N), 1.0 / N ** 2))
    dc11 = filt((dC[:, 0, 0] * dE).reshape(N, N))
    g = np.array([vol / VOL - 1.0, 1.0 - C[0, 0] / C11_MIN])
    dg = np.stack([dvol.ravel() / VOL, -dc11.ravel() / C11_MIN])
    return {"f0": nu, "df0": dnu.ravel(), "g": g, "dg": dg, "C": C, "vol": vol}


@cache
def run() -> dict[str, Any]:
    opt = MMA(move=0.1)
    x = start().ravel()
    hist = []
    for _ in range(MAX_ITER):
        ev = evaluate(x.reshape(N, N))
        hist.append((float(ev["f0"]), float(ev["vol"]), float(ev["C"][0, 0])))
        xn = opt.step(x, ev["f0"], ev["df0"], ev["g"], ev["dg"])
        change = float(np.abs(xn - x).max())
        x = xn
        if change < 1e-3:
            break
    ev = evaluate(x.reshape(N, N))
    xt = filt(x.reshape(N, N))
    binary = (xt > 0.5).astype(float)
    Eb = EMIN + (1 - EMIN) * binary.ravel()
    Cb = homogenize_elasticity(Domain((N, N)), Eb, NU).C
    return {"iterations": len(hist), "history": hist, "x": x.reshape(N, N), "xphys": xt, "C": ev["C"],
            "nu": float(ev["f0"]), "vol": float(ev["vol"]), "binary": binary, "C_binary": Cb,
            "nu_binary": float(Cb[0, 1] / Cb[0, 0]), "vol_binary": float(binary.mean())}


def periodic_one_body(b: np.ndarray) -> bool:
    """One body that percolates through the periodic tiling in both directions (face contact)."""
    t = np.tile(b > 0.5, (3, 3))
    lab, _ = ndimage.label(t)
    centre = lab[N:2 * N, N:2 * N][b > 0.5]
    ids = set(np.unique(centre)) - {0}
    if len(ids) != 1:
        return False
    k = ids.pop()
    rows, cols = np.nonzero(lab == k)
    return bool(rows.min() < N and rows.max() >= 2 * N and cols.min() < N and cols.max() >= 2 * N)
