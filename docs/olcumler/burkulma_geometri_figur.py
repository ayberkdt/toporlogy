"""Figures of docs/olcumler/burkulma_geometri.md from burkulma_geometri.json (datawiz, a local plotting package that
is not a dependency of toporlogy).

    python docs/olcumler/burkulma_geometri_figur.py

FİGÜR BRİFİ (ikisi için): okur uzman; ortam web (depo belgesi), ondalık virgül; set burkulma_geometri.seti.json.
1 burkulma_yakinsama — mesaj: Euler kolonunda ve gömülü basınç elemanında λ1 ağla yerleşiyor; tekil duvarda inceltme
  sürdükçe düşmeye devam ediyor ve sıkıştırma ucunu dışlamak bunu durdurmuyor. Veri: λ1 / λ1(en kaba ağ) (nicel)
  × inceltme h0/h (1, 2, 4, 8; log2), dört seri (nominal). Tür: işaretli çizgi, lejant (çizgi sonu etiketleri çakışıyor).
2 hacim_esit — mesaj: aynı hacimdeki dört tasarımda uyum %26'ya, en büyük von Mises %21'e, λ1 %10'a kadar değişiyor.
  Veri: üç nicelik / V1 değeri (nicel) × değişken (nominal, 4). Tür: nokta grafiği (ortak ölçek), referans 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    import datawiz as dw
    import matplotlib.ticker as mticker

    d = json.loads((HERE / "burkulma_geometri.json").read_text(encoding="utf-8"))
    S = dw.Seti(str(HERE / "burkulma_geometri.seti.json"), hedef="web-640", ortam="web", ondalik_ayirici=",",
                palet="okabe-ito")
    number = mticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    for name in ("Euler kolonu", "gömülü basınç elemanı", "tekil duvar", "duvar, uç dışlanmış"):
        S.varlik(name)
    for name in ("uyum", "en büyük von Mises", "λ1"):
        S.varlik(name)
    if S.kenar_oner():                                    # one plot-area margin for both figures (set alignment)
        S.kenar(*S.kenar_oner())
    mark = {"markersize": 6.0, "markeredgecolor": "#333333", "markeredgewidth": 0.6}

    # ---------------------------------------------------------------- 1: refinement series
    A = d["A"]
    series = [("Euler kolonu", A["A1"]["lambda_q4"]), ("gömülü basınç elemanı", [e["lambda1"] for e in
                                                                                A["A3"]["embedded"]]),
              ("tekil duvar", A["A4"]["lambda1"]), ("duvar, uç dışlanmış", A["A5"]["lambda1"])]
    fig, ax = S.figur(oran=0.62)
    for name, lam in series:
        lam = np.asarray(lam, float)
        r = 2.0 ** np.arange(len(lam))
        ax.plot(r, lam / lam[0], **{**S.stil(name, isaretli=True), **mark, "label": name})
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlim(0.85, 14)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(number)
    ax.set_xlabel("İnceltme h0/h (–)")
    ax.set_ylabel("λ1 / en kaba ağdaki λ1 (–)")
    dw.lejant.yerlestir(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "burkulma_yakinsama"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 2: volume-equal variants
    V = d["VC"]["variants"]
    order = [("V0_grey", "gri"), ("V1_threshold", "eşik"), ("V2_blur1", "bulanık σ 1"), ("V3_blur2", "bulanık σ 2")]
    qty = [("compliance", "uyum"), ("vm_max_solid", "en büyük von Mises"), ("lambda1", "λ1")]
    fig, ax = S.figur(oran=0.62)
    y = np.arange(1, len(order) + 1)[::-1]                # categories at 1..4: no zero line through a row
    for k, (key, name) in enumerate(qty):
        ref = V["V1_threshold"][key]
        vals = [V[v][key] / ref for v, _ in order]
        ax.plot(vals, y + (k - 1) * 0.18, **{**S.stil(name, isaretli=True), **mark, "linestyle": "none", "label": name})
    ax.set_yticks(y, [lab for _, lab in order])
    ax.xaxis.set_major_formatter(number)
    ax.set_xlabel("Değer / eşik tasarımındaki değer (–)")
    ax.set_ylabel("Tasarım (aynı hacim)")
    dw.notlar.referans(ax, x=1.0, metin="eşik")
    dw.lejant.yerlestir(ax)
    dw.izgara.uygula(ax, "silik", "x")
    S.kaydet(fig, str(HERE / "hacim_esit"), bicimler=("png", "pdf"))


if __name__ == "__main__":
    main()
