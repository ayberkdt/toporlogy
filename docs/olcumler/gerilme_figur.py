"""Figures of docs/olcumler/gerilme.md from gerilme.json and gerilme_alanlar.json (datawiz, a local plotting
package that is not a dependency of toporlogy). Run through ``python docs/olcumler/gerilme.py --figures``.

FİGÜR BRİFİ (hepsi için): okur uzman; ortam web (depo belgesi), ondalık virgül; set gerilme.seti.json.
1 gerilme_mertebe  — mesaj: merkez ve yama kurtarma O(h²), Gauss noktası ve köşe/düğüm değerleri O(h).
2 gerilme_tekillik — mesaj: üç tekil noktada tepe inceltmeyle büyümeye devam ediyor, yayılı yükte yerleşiyor.
3 gerilme_tasarim  — mesaj: iki 0/1 tasarımda da yük düğümleri boşlukta kalıyor.
4 gerilme_lbracket — mesaj: sabit geometri inceltildikçe tepe oranı büyüyor, sınırın çok üstüne çıkıyor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    import datawiz as dw
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    d = json.loads((HERE / "gerilme.json").read_text(encoding="utf-8"))
    fields = json.loads((HERE / "gerilme_alanlar.json").read_text(encoding="utf-8"))
    S = dw.Seti(str(HERE / "gerilme.seti.json"), hedef="web-640", ortam="web", ondalik_ayirici=",",
                palet="okabe-ito")
    number = mticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    def comma(v: float, digits: int = 2) -> str:
        return f"{v:.{digits}f}".replace(".", ",")

    # one colour, one entity across the set: the two O(h) readings of figure 1 and the two controls are grey context
    for name in ("merkez", "yama kurtarma", "noktasal yük", "içbükey köşe", "ankastre kenar",
                 "Toplulaştırılmış 0/1", "Augmented Lagrangian 0/1"):
        S.varlik(name)
    S.varlik("Gauss noktası", baglam=True, cizgi=":")
    S.varlik("köşe = düğüm ortalaması", baglam=True, cizgi="--")
    for name in ("yayılı yük, kontrol", "dolu parça"):
        S.varlik(name, baglam=True)

    # ---------------------------------------------------------------- 1: orders of pure bending (Q4, region R)
    b = d["pure_bending"]["Q4_R"]
    n = np.array(b["n"], float)
    series = [("integration_point", "Gauss noktası"), ("extrapolated", "köşe = düğüm ortalaması"),
              ("centroid", "merkez"), ("patch_recovered", "yama kurtarma")]
    fig, ax = S.figur(oran=0.62)
    for key, name in series:
        ax.plot(n, b["error"][key],
                **{**S.stil(name, isaretli=True), "label": f"{name} (p = {comma(b['order'][key][-1])})"})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n, [f"{int(k)}" for k in n])
    ax.set_xlim(7, 75)
    ax.set_xlabel("Yükseklik boyunca eleman sayısı (adet)")
    ax.set_ylabel("En büyük hata / en büyük tam gerilme (–)")
    dw.etiket.cizgi_sonu(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "gerilme_mertebe"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 2: peak growth at singular points
    names = [("point_load", "noktasal yük"), ("l_corner", "içbükey köşe"), ("clamped_edge", "ankastre kenar"),
             ("footprint_load", "yayılı yük, kontrol")]
    fig, ax = S.figur(oran=0.62)
    for key, name in names:
        p = d["peaks"][key]
        peak = np.array(p["peak"])
        ax.plot(p["n"], peak / peak[0],
                **{**S.stil(name, isaretli=True), "label": f"{name} (eğim {comma(p['slopes'][-1])})"})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(d["peaks"]["point_load"]["n"], [str(k) for k in d["peaks"]["point_load"]["n"]])
    ax.set_yticks([1, 2, 4, 8])
    ax.yaxis.set_major_formatter(number)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(14, 150)
    ax.set_xlabel("Birim uzunluk başına eleman sayısı (adet)")
    ax.set_ylabel("Tepe gerilme / en kaba ağdaki (–)")
    dw.etiket.cizgi_sonu(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "gerilme_tekillik"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 3: the two 0/1 designs
    lb = d["lbracket"]
    methods = [("aggregated", "Toplulaştırılmış"), ("augmented_lagrangian", "Augmented Lagrangian")]
    cmap = ListedColormap(mpl.colormaps["Blues"](np.linspace(0.18, 1.0, 256)))
    load = "#d55e00"
    fig, axs = S.figur(oran=0.57, mozaik=[["a", "b"]])
    image = None
    for key, title in methods:
        ax = axs["a" if key == "aggregated" else "b"]
        x = np.array(fields[key]["x_binary"])
        ratio = np.ma.masked_where(x <= 0.5, np.array(fields[key]["ratio_binary"]))
        image = ax.imshow(ratio.T, origin="lower", extent=(0, 40, 0, 40), cmap=cmap, vmin=0.0, vmax=1.75,
                          interpolation="nearest")
        ax.plot([0, 40, 40, 16, 16, 0, 0], [0, 0, 16, 16, 40, 40, 0], color="0.55", lw=0.8)
        ax.add_patch(Rectangle((34, 12), 6, 4, fill=False, ls="--", lw=0.9, ec="0.35"))
        ax.plot([37, 38, 39, 40], [16] * 4, "v", color=load, ms=5, clip_on=False)
        top = lb["posthoc"][key]["binary"]["raw_solid_max"]
        ax.plot(*top["at"], "o", mfc="none", mec="black", ms=11, mew=1.2)
        ax.set_title(f"{title}: en büyük {comma(top['ratio'])}")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 42)
        ax.set_ylim(-0.5, 41)
        ax.set_xticks([0, 16, 40])
        ax.set_yticks([0, 16, 40])
        ax.set_xlabel("x (birim)")
        ax.set_ylabel("y (birim)")
    bar = fig.colorbar(image, ax=list(axs.values()), shrink=0.72, label="Gerilme / sınır (–)")
    bar.ax.yaxis.set_major_formatter(number)
    keys = [Line2D([], [], ls="none", marker="v", ms=6, color=load, label="yük düğümü"),
            Line2D([], [], ls="--", lw=0.9, color="0.35", label="dışlanan bölge"),
            Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec="black", label="en büyük oran")]
    fig.legend(handles=keys, loc="outside lower center", ncols=3, frameon=False)
    S.kaydet(fig, str(HERE / "gerilme_tasarim"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 4: refinement of the fixed geometry
    fig, ax = S.figur(oran=0.62)
    for key, name in (("aggregated", "Toplulaştırılmış 0/1"), ("augmented_lagrangian", "Augmented Lagrangian 0/1"),
                      ("solid", "dolu parça")):
        lv = lb["refinement"][key]["levels"]
        ax.plot([r["k"] for r in lv], [r["peak_centroid"]["ratio"] for r in lv], **S.stil(name, isaretli=True))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.yaxis.set_major_formatter(number)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(0.92, 8.8)
    ax.set_xlabel("Eleman başına bölme k (adet)")
    ax.set_ylabel("Tepe gerilme / sınır gerilme (–)")
    dw.etiket.cizgi_sonu(ax)
    dw.izgara.uygula(ax, "silik", "y")
    dw.notlar.referans(ax, y=1.0, metin="sınır")
    S.kaydet(fig, str(HERE / "gerilme_lbracket"), bicimler=("png", "pdf"))
