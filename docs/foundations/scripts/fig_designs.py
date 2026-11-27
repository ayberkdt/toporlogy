"""Figure of the time-aware topology optimisation designs (docs/foundations/designs.npz from save_designs.py).

    python docs/foundations/scripts/fig_designs.py      -> docs/foundations/fig_designs.{pdf,png} + .denetim.txt

(a)-(c): two-phase viscoelastic history (VM): min-max design and the two single-phase designs; (d)-(e): Norton
creep MBB half-beam (CO6): elastic-compliance design and service-time (1000 s) design. Density x̄ on one shared
sequential grey scale (solid dark).
"""
from __future__ import annotations

from pathlib import Path

import datawiz as dw
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
D = np.load(ROOT / "docs" / "foundations" / "designs.npz")


def grid(x: np.ndarray, cent: np.ndarray) -> np.ndarray:
    xs, ys = np.unique(np.round(cent[:, 0], 9)), np.unique(np.round(cent[:, 1], 9))
    G = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, np.round(cent[:, 0], 9))
    iy = np.searchsorted(ys, np.round(cent[:, 1], 9))
    G[iy, ix] = x
    return G


def sci(v: float) -> str:
    m, e = f"{v:.2e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


fig, axs = dw.figur("makale", hedef="genel-cift", oran=0.40, nrows=2, ncols=3, layout="compressed")
panels = [
    (axs[0, 0], "vm_minmax", "vm_centroids", "(a) min–max", (r"$J_A$", r"$J_B$")),
    (axs[0, 1], "vm_A", "vm_centroids", "(b) phase A only", (r"$J_A$", r"$J_B$")),
    (axs[0, 2], "vm_B", "vm_centroids", "(c) phase B only", (r"$J_A$", r"$J_B$")),
    (axs[1, 0], "co_elastic", "co_centroids", "(d) elastic J(0)", ("J(0)", "J(1000 s)")),
    (axs[1, 1], "co_creep", "co_centroids", "(e) creep J(1000 s)", ("J(0)", "J(1000 s)")),
]
im = None
for ax, key, ck, title, names in panels:
    G = grid(D[key], D[ck])
    im = ax.imshow(G, origin="lower", cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
    J = D[key + "_J"]
    ax.set_title(f"{title}\n{names[0]} = {sci(J[0])}, {names[1]} = {sci(J[1])}", fontsize=8, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():                     # thin domain outline (void cells are white)
        s.set_visible(True)
        s.set_color("0.6")
        s.set_linewidth(0.5)
axs[1, 2].axis("off")
dw.lejant.renk_cubugu(axs[1, 2], im, "Density x̄ (–)", yon="sag")
rep = dw.kaydet(fig, str(ROOT / "docs" / "foundations" / "fig_designs"), hedef="genel-cift", cvd=True)
print(rep)
