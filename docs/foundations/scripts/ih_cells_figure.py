"""Figure of the post hoc IH cells (docs/foundations/prereg/IH_inverse_homogenization.md, post hoc start study):
the filtered density of the square and the rhombic cell, each tiled 3 x 3 so that the periodic structure reads.

    python docs/foundations/scripts/ih_cells_figure.py

Data: docs/foundations/data/ih_cells_x_phys.json (written from the post hoc runs); the meshes are rebuilt here
exactly as in scripts/ih_starts_posthoc.py. Output: docs/foundations/fig_ih_cells.{pdf,png} + datawiz audit.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np

import datawiz as dw
from toporlogy.fem.mesh.mesh import rectangle

HERE = Path(__file__).resolve().parents[1]


def meshes() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    m0 = rectangle(40, 40, 1.0, 1.0, "T3")
    A = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2]])
    m1 = replace(m0, points=m0.points @ A.T)
    return {"square_void_hole": (m0.points, m0.cells, np.eye(2)), "rhombus_void_hole": (m1.points, m1.cells, A.T)}


def main() -> None:
    raw = json.loads((HERE / "data" / "ih_cells_x_phys.json").read_text(encoding="utf-8"))
    x = {k: np.asarray(v) for k, v in raw.items()}
    res = json.loads((HERE / "IH_starts_posthoc.json").read_text(encoding="utf-8"))
    fig, axs = dw.figur("makale", hedef="genel-cift", oran=0.42, ncols=2)
    titles = {"square_void_hole": "Square lattice", "rhombus_void_hole": "Triangular lattice (60° cell)"}
    accent = dw.paletler.tema("akademik")["kategorik"][0] if hasattr(dw, "paletler") else "#0072B2"
    mappable = None
    for ax, key in zip(axs, ("square_void_hole", "rhombus_void_hole"), strict=True):
        P, cells, L = meshes()[key]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                t = i * L[0] + j * L[1]
                tri = mtri.Triangulation(P[:, 0] + t[0], P[:, 1] + t[1], cells)
                mappable = ax.tripcolor(tri, facecolors=x[key], cmap="Greys", vmin=0.0, vmax=1.0,
                                        edgecolors="none", rasterized=True)
        corner = np.array([[0, 0], L[0], L[0] + L[1], L[1], [0, 0]])
        ax.plot(corner[:, 0], corner[:, 1], color=accent, lw=1.0)
        lo = np.min([(P + i * L[0] + j * L[1]).min(axis=0) for i in (-1, 1) for j in (-1, 1)], axis=0)
        hi = np.max([(P + i * L[0] + j * L[1]).max(axis=0) for i in (-1, 1) for j in (-1, 1)], axis=0)
        pad = 0.02 * float((hi - lo).max())
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="datalim")         # equal boxes: the titles share one line
        ax.set_axis_off()
        r = res[key]
        ax.set_title(titles[key] + r": $\kappa/\kappa_{\mathrm{HS}}$" + f" = {r['ratio_hs']:.3f}", loc="left")
    assert mappable is not None
    dw.lejant.renk_cubugu(axs[-1], mappable, "Filtered density (–)")
    dw.etiket.panel_harfleri(axs, hedef="genel-cift")
    rep = dw.kaydet(fig, str(HERE / "fig_ih_cells"), hedef="genel-cift", cvd=True)
    print(rep if isinstance(rep, str) else {k: v for k, v in rep.items() if k != "figur"})


if __name__ == "__main__":
    main()
