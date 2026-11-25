"""README banner: a real toporlogy result drawn as an SVG, light and dark.

The design is docs/assets/banner.yaml (a cantilever). The hero picture is the final design coloured by its strain
energy density (log scale); the strip shows the same run at earlier iterations; the logo tile is the final shape.

usage: python docs/assets/make_banner.py [--cache FILE.npz]     (from the repository root; about three minutes)
writes docs/assets/banner-light.svg and banner-dark.svg; the README picks one with <picture>. With --cache the
optimisation result is stored in / read from FILE.npz, so a design change does not re-run it.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imsave
from scipy import ndimage

from toporlogy.configuration.build import build_setup

HERE = Path(__file__).resolve().parent
SNAPSHOTS = (6, 18, 45)
W, H = 1280, 440
FONT = "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"bg0": "#FFF9F3", "bg1": "#FBF3EB", "ink": "#1E2230", "muted": "#596070", "line": "#E6D9CC",
              "pill": "#FFFFFF", "tile": "#1E2230", "thumb": "#3A3F4E",
              "ramp": ["#DB6E52", "#CC4F53", "#B13A5C", "#842A62", "#4E1C60"]},
    "dark": {"bg0": "#1B1F2B", "bg1": "#11141C", "ink": "#F4EFE8", "muted": "#AEB4C0", "line": "#2A3040",
             "pill": "#232836", "tile": "#2A3042", "thumb": "#D9DCE3",
             "ramp": ["#8A55C4", "#B44D8E", "#E0605A", "#F9A26C", "#FFE7A8"]},
}


# ------------------------------------------------------------------ data
def compute() -> dict[str, np.ndarray]:
    raw = yaml.safe_load((HERE / "banner.yaml").read_text(encoding="utf-8"))
    s = build_setup(raw)
    st, opts = s.study, s.run_options
    snaps: dict[int, np.ndarray] = {}

    def keep(it: int, ev, change: float) -> None:
        if it + 1 in SNAPSHOTS:
            snaps[it + 1] = ev.xphys.reshape(st.domain.shape).copy()
    res = st.run(max_iter=int(opts["max_iter"]), tol=float(opts["tol"]), callback=keep)
    xp = res.xphys.reshape(st.domain.shape)
    U = st.evaluate_density(xp).U
    with torch.no_grad():
        E = st.interp(torch.as_tensor(xp, dtype=torch.float64)).reshape(-1).numpy()
    act = np.flatnonzero(st.domain.active.reshape(-1))
    dens = np.zeros(xp.size)
    dens[act] = E[act] * st.physics.element_energy(U)[:, 0]
    out = {"xphys": xp, "energy": dens.reshape(xp.shape), "objective": np.array(res.objective),
           "iterations": np.array(res.iterations)}
    out.update({f"it{k}": v for k, v in snaps.items()})
    return out


# ------------------------------------------------------------------ pictures
def _png(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    imsave(buf, np.clip(rgba, 0, 1), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _alpha(x: np.ndarray, up: int) -> np.ndarray:
    """Anti-aliased {x̄ > 0.5} on an ``up`` times finer grid, image rows top → bottom."""
    g = ndimage.gaussian_filter(np.pad(x, 2, mode="edge"), 0.8)     # rounds the element staircase off
    f = ndimage.zoom(g.T[::-1], up, order=3)[2 * up:-2 * up, 2 * up:-2 * up]
    return np.clip((f - 0.5) * 1.5 * up + 0.5, 0.0, 1.0)          # edge about one fine pixel wide


def hero(x: np.ndarray, energy: np.ndarray, ramp: list[str], up: int = 5) -> str:
    """Final design coloured by strain energy density, transparent outside the solid. The colour is the rank of the
    (smoothed) density among the solid elements, so the ramp is used evenly; the order is that of the energy."""
    solid = x > 0.5
    w = ndimage.gaussian_filter(solid.astype(float), 1.6)          # smoothing inside the solid only, so the void
    e = ndimage.gaussian_filter(energy * solid, 1.6) / np.maximum(w, 1e-9)   # (energy 0) does not pale the edges
    ranks = np.sort(e[solid])
    t = np.searchsorted(ranks, e) / len(ranks)
    t = ndimage.grey_dilation(np.where(solid, t, 0), size=3) * ~solid + t * solid      # colour reaches the edge
    tt = ndimage.zoom(np.pad(t, 1, mode="edge").T[::-1], up, order=1)[up:-up, up:-up]
    rgba = LinearSegmentedColormap.from_list("ramp", ramp)(tt)
    rgba[..., 3] = _alpha(x, up)
    return _png(rgba)


def mono(x: np.ndarray, colour: str, up: int = 2, grey: bool = True) -> str:
    """Density as opacity of one colour (grey designs stay grey), or the crisp shape."""
    c = np.array([int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    f = ndimage.zoom(np.pad(x, 1, mode="edge").T[::-1], up, order=1)[up:-up, up:-up]
    a = np.clip(f, 0, 1) ** 1.4 if grey else _alpha(x, up)
    return _png(np.dstack([np.broadcast_to(c, f.shape + (3,)), a]))


# ------------------------------------------------------------------ layout
def pill(x: float, y: float, text: str, t: dict) -> tuple[str, float]:
    w = 9.4 * len(text) + 32
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="32" rx="16" fill="{t["pill"]}" '
            f'stroke="{t["line"]}" stroke-width="1.5"/><text x="{x + w / 2:.0f}" y="{y + 22:.0f}" font-size="16" '
            f'fill="{t["ink"]}" text-anchor="middle">{text}</text>'), w


def svg(d: dict[str, np.ndarray], t: dict) -> str:
    x = d["xphys"]
    nx, ny = x.shape
    # hero: right half
    hw = 600
    s = hw / nx
    hh = s * ny
    hx, hy = W - 64 - hw, 64
    wall = "".join(f'<line x1="{hx - 4:.1f}" y1="{hy + k:.1f}" x2="{hx - 16:.1f}" y2="{hy + k + 12:.1f}" '
                   f'stroke="{t["muted"]}" stroke-width="1.6" stroke-linecap="round"/>'
                   for k in range(0, int(hh) - 8, 14))
    wall += (f'<line x1="{hx - 3:.1f}" y1="{hy - 6:.1f}" x2="{hx - 3:.1f}" y2="{hy + hh + 6:.1f}" '
             f'stroke="{t["muted"]}" stroke-width="2.5" stroke-linecap="round"/>')
    ax, ay = hx + hw + 2, hy + hh / 2
    arrow = (f'<line x1="{ax:.1f}" y1="{ay - 58:.1f}" x2="{ax:.1f}" y2="{ay - 8:.1f}" stroke="{t["ink"]}" '
             f'stroke-width="2.5" stroke-linecap="round"/><path d="M{ax - 7:.1f} {ay - 15:.1f} L{ax:.1f} {ay - 3:.1f} '
             f'L{ax + 7:.1f} {ay - 15:.1f}" fill="none" stroke="{t["ink"]}" stroke-width="2.5" '
             f'stroke-linecap="round" stroke-linejoin="round"/>')
    grid = "".join(f'<line x1="{hx + s * i:.1f}" y1="{hy:.1f}" x2="{hx + s * i:.1f}" y2="{hy + hh:.1f}"/>'
                   for i in range(0, nx + 1, 20))
    grid += "".join(f'<line x1="{hx:.1f}" y1="{hy + s * j:.1f}" x2="{hx + hw:.1f}" y2="{hy + s * j:.1f}"/>'
                    for j in range(0, ny + 1, 20))
    # legend under the hero
    ly = hy + hh + 34
    stops = "".join(f'<stop offset="{k / (len(t["ramp"]) - 1):.2f}" stop-color="{c}"/>'
                    for k, c in enumerate(t["ramp"]))
    legend = (f'<text x="{hx:.0f}" y="{ly + 9:.0f}" font-size="16" fill="{t["muted"]}">low</text>'
              f'<rect x="{hx + 34:.0f}" y="{ly - 1:.0f}" width="96" height="10" rx="5" fill="url(#ramp)"/>'
              f'<text x="{hx + 140:.0f}" y="{ly + 9:.0f}" font-size="16" fill="{t["muted"]}">high strain energy</text>'
              f'<text x="{hx + hw:.0f}" y="{ly + 9:.0f}" font-size="16" fill="{t["muted"]}" text-anchor="end">'
              f'{nx} × {ny} elements · 35 % material</text>')
    # left column
    L = 64
    logo = (f'<rect x="{L}" y="68" width="68" height="68" rx="18" fill="{t["tile"]}"/>'
            f'<image x="{L + 8}" y="{102 - 26 * ny / nx:.1f}" width="52" height="{52 * ny / nx:.1f}" '
            f'href="{hero(x, d["energy"], THEMES["dark"]["ramp"], up=1)}"/>')
    pills, px, py = [], float(L), 214.0
    for text in ("Differentiable FEM", "OC · MMA · GCMMA · SiMPL", "2-D and 3-D grids", "Watertight STL"):
        p, w = pill(px, py, text, t)
        if px + w > L + 500:
            px, py = float(L), py + 44
            p, w = pill(px, py, text, t)
        pills.append(p)
        px += w + 10
    tw, th, ty = 120, 60, int(hy + hh + 43) - 84                   # labels on the legend's baseline
    thumbs = []
    for k, it in enumerate(SNAPSHOTS):
        tx = L + k * (tw + 36)
        thumbs.append(f'<image x="{tx}" y="{ty}" width="{tw}" height="{th}" href="{mono(d[f"it{it}"], t["thumb"])}"/>'
                      f'<text x="{tx}" y="{ty + th + 24}" font-size="16" fill="{t["muted"]}">iteration {it}</text>')
        if k < len(SNAPSHOTS) - 1:
            cx = tx + tw + 18
            thumbs.append(f'<path d="M{cx - 6} {ty + th / 2 - 6} l6 6 l-6 6" fill="none" stroke="{t["muted"]}" '
                          f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img"
 aria-labelledby="title desc">
<title id="title">toporlogy</title>
<desc id="desc">Topology optimisation you can take apart. The picture is a cantilever designed by toporlogy, coloured
by strain energy density, with the same run at iterations {", ".join(map(str, SNAPSHOTS))}.</desc>
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t["bg0"]}"/><stop offset="1"
 stop-color="{t["bg1"]}"/></linearGradient>
<linearGradient id="ramp" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient>
</defs>
<rect width="{W}" height="{H}" rx="24" fill="url(#bg)"/>
<g stroke="{t["line"]}" stroke-width="1">{grid}</g>
<image x="{hx:.1f}" y="{hy:.1f}" width="{hw:.1f}" height="{hh:.1f}" href="{hero(x, d["energy"], t["ramp"])}"/>
{wall}{arrow}{logo}
<g font-family="{FONT}">
<text x="{L + 82}" y="122" font-size="60" font-weight="700" fill="{t["ink"]}" letter-spacing="-1.5">toporlogy</text>
<text x="{L}" y="182" font-size="24" fill="{t["ink"]}">Topology optimisation you can take apart.</text>
{"".join(pills)}
{"".join(thumbs)}
{legend}
</g>
</svg>
"""


if __name__ == "__main__":
    cache = Path(sys.argv[sys.argv.index("--cache") + 1]) if "--cache" in sys.argv else None
    if cache is not None and cache.is_file():
        data = dict(np.load(cache))
    else:
        data = compute()
        if cache is not None:
            np.savez_compressed(cache, **data)
    for name, theme in THEMES.items():
        (HERE / f"banner-{name}.svg").write_text(svg(data, theme), encoding="utf-8")
    print("objective", float(data["objective"]), "iterations", int(data["iterations"]), file=sys.stderr)
