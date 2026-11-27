"""Figure of the post hoc IH cells (docs/foundations/prereg/IH_inverse_homogenization.md, post hoc start study):
the filtered density of the square and the rhombic cell, each tiled 3 x 3 so that the periodic structure reads.

    python docs/foundations/scripts/ih_cells_figure.py            # IH post hoc cells -> fig_ih_cells
    python docs/foundations/scripts/ih_cells_figure.py ip         # IP projected cells -> fig_ip_cells
    python docs/foundations/scripts/ih_cells_figure.py ia         # IA auxetic cell -> fig_ia_cell

Data: docs/foundations/data/ih_cells_x_phys.json (written from the post hoc runs); the meshes are rebuilt here
exactly as in scripts/ih_starts_posthoc.py. Output: docs/foundations/fig_ih_cells.{pdf,png} + datawiz audit.
"""
from __future__ import annotations

import json
import sys
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
    return {"square_void_hole": (m0.points, m0.cells, np.eye(2)), "rhombus_void_hole": (m1.points, m1.cells, A.T),
            "square_auxetic": (m0.points, m0.cells, np.eye(2))}


def main(which: str = "ih") -> None:
    data, values, out = {"ih": ("ih_cells_x_phys.json", "IH_starts_posthoc.json", "fig_ih_cells"),
                         "ip": ("ip_cells_x_phys.json", "IP_designs.json", "fig_ip_cells"),
                         "ia": ("ia_cell_x_phys.json", "IA_design.json", "fig_ia_cell")}[which]
    raw = json.loads((HERE / "data" / data).read_text(encoding="utf-8"))
    x = {k: np.asarray(v) for k, v in raw.items()}
    res = json.loads((HERE / values).read_text(encoding="utf-8"))
    keys = list(x)
    hedef = "genel-cift" if len(keys) == 2 else "genel-tek"
    fig, axs = dw.figur("makale", hedef=hedef, oran=0.42 if len(keys) == 2 else 0.8, ncols=len(keys))
    axs = np.atleast_1d(axs)
    titles = {"square_void_hole": "Square lattice", "rhombus_void_hole": "Triangular lattice (60° cell)",
              "square_auxetic": "Auxetic square cell"}
    accent = dw.paletler.tema("akademik")["kategorik"][0] if hasattr(dw, "paletler") else "#0072B2"
    mappable = None
    for ax, key in zip(axs, keys, strict=True):
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
        if which == "ia":
            d = res["design"]
            ax.set_title(f"{titles[key]}: " + r"$\nu_{21}$" + f" = {d['nu21_stress_y']:.2f}, " + r"$\nu_{12}$"
                         + f" = {d['nu12_stress_x']:.2f}", loc="left")
        else:
            r = res[key]
            ax.set_title(titles[key] + r": $\kappa/\kappa_{\mathrm{HS}}$" + f" = {r['ratio_hs']:.3f}", loc="left")
    assert mappable is not None
    dw.lejant.renk_cubugu(axs[-1], mappable, "Filtered density (–)" if which == "ih" else "Projected density (–)")
    if len(keys) > 1:
        dw.etiket.panel_harfleri(axs, hedef=hedef)
    rep = dw.kaydet(fig, str(HERE / out), hedef=hedef, cvd=True)
    print(rep if isinstance(rep, str) else {k: v for k, v in rep.items() if k != "figur"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ih")
