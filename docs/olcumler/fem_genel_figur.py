"""Figures of docs/olcumler/fem_genel.md from fem_genel.json (datawiz, a local plotting package that is not a
dependency of toporlogy).

    python docs/olcumler/fem_genel_figur.py

FİGÜR BRİFİ (hepsi için): okur uzman; ortam web (depo belgesi), ondalık virgül; set fem_genel.seti.json; yöntem
renkleri setin varlıkları (aynı yöntem her figürde aynı renk ve işaret).
1 fem_genel_nu         — mesaj: ν → 0,5'te Q4 ve T3 kilitlenir (hata ≈ 1), Taylor–Hood'un hatası değişmez.
                         Veri: ν (sıralı, 8 düzey) × yöntem → göreli L2 hata (log). Tür: işaretli çizgi, ν eşit aralıklı
                         sıralı eksen (doğrusal ν eksen 0,49 … 0,4999'u ezer), doğrudan etiket.
2 fem_genel_konsol     — mesaj: eleman en-boy oranı büyüdükçe (L/h ↑, n_x sabit) Q4, T3, H8 kayma kilitlenmesine girer,
                         ikinci dereceden elemanlar R ≈ 1'de kalır. x: L/h (log), y: R (0–1,05).
3 fem_genel_yakinsama  — mesaj: ν = 0,4999'da Taylor–Hood yer değiştirmede h³, basınçta h² ile yakınsar; Q4 ve T3'ün
                         yer değiştirme hatası azalmaz. log–log, x: 1/h.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    import datawiz as dw
    import matplotlib.ticker as mticker

    d = json.loads((HERE / "fem_genel.json").read_text(encoding="utf-8"))
    S = dw.Seti(str(HERE / "fem_genel.seti.json"), hedef="web-640", ortam="web", ondalik_ayirici=",",
                palet="okabe-ito")
    names = {"Q4": "Q4", "T3": "T3", "Q9": "Q9", "T6": "T6", "Q4IM": "Q4IM (ızgara)", "TH:T6-T3": "TH T6-T3",
             "TH:Q9-Q4": "TH Q9-Q4", "H8": "H8", "T10": "T10"}
    for k in ("Q4", "T3", "Q9", "T6", "Q4IM", "TH:T6-T3", "TH:Q9-Q4"):
        S.varlik(names[k])
    S.varlik("H8", baglam=True, cizgi="--")
    S.varlik("T10", baglam=True, cizgi=":")

    def comma(v: float) -> str:
        return f"{v:g}".replace(".", ",")

    # ---------------------------------------------------------------- 1: nu sweep, plane strain, n = 16, E fixed
    nus = [-0.9, -0.5, 0.0, 0.3, 0.45, 0.49, 0.499, 0.4999]
    rows = [r for r in d["nu_sweep"] if r.get("fixed") == "E" and r.get("plane") == "strain" and "l2" in r]
    fig, ax = S.figur(oran=0.62)
    xs = np.arange(len(nus))
    for k in ("Q4", "T3", "Q4IM", "T6", "Q9", "TH:T6-T3", "TH:Q9-Q4"):
        by = {r["nu"]: r["l2"] for r in rows if r["method"] == k}
        x = [i for i, nu in enumerate(nus) if nu in by]
        ax.plot(x, [by[nus[i]] for i in x], **{**S.stil(names[k], isaretli=True), "label": names[k]})
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{v:.4f}".replace(".", ",").replace("-", "−") for v in nus], rotation=30, ha="right")
    ax.set_xlim(-0.3, len(nus) + 1.6)
    ax.set_xlabel("Poisson oranı ν (–)")
    ax.set_ylabel("Göreli L2 yer değiştirme hatası (–)")
    dw.etiket.cizgi_sonu(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "fem_genel_nu"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 2: cantilever, family F2, ny = 2
    cant = [r for r in d["cantilever"] if r["family"] == "F2" and r["ny"] == 2]
    fig, ax = S.figur(oran=0.62)
    for k in ("Q4", "T3", "H8", "T6", "Q9", "T10"):
        rr = sorted((r for r in cant if r["method"] == k), key=lambda r: r["slenderness"])
        if rr:
            ax.plot([r["slenderness"] for r in rr], [r["R"] for r in rr],
                    **{**S.stil(names[k], isaretli=True), "label": names[k]})
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 50, 100], ["5", "10", "50", "100"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(4, 260)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_xlabel("Narinlik L/h (–), n_x = 40, n_y = 2")
    ax.set_ylabel("Şekil değiştirme enerjisi oranı U_h / U (–)")
    dw.etiket.cizgi_sonu(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "fem_genel_konsol"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 3: convergence at nu = 0.4999, distorted mesh
    th = [r for r in d["th_convergence"] if r["nu"] == 0.4999]
    fig, ax = S.figur(oran=0.62)
    for k in ("Q4", "T3", "TH:T6-T3", "TH:Q9-Q4"):
        rr = sorted((r for r in th if r["method"] == k), key=lambda r: r["n"])
        ax.plot([r["n"] for r in rr], [r["l2"] for r in rr],
                **{**S.stil(names[k], isaretli=True), "label": names[k]})
    for k in ("TH:T6-T3", "TH:Q9-Q4"):
        rr = sorted((r for r in th if r["method"] == k), key=lambda r: r["n"])
        ax.plot([r["n"] for r in rr], [r["p_error"] for r in rr],
                **{**S.stil(names[k], isaretli=True), "linestyle": "--", "label": f"{names[k]}, p"})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 16, 32], ["8", "16", "32"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(7, 90)
    ax.set_xlabel("Kenar başına eleman sayısı 1/h (adet)")
    ax.set_ylabel("Göreli L2 hata (–)")
    dw.etiket.cizgi_sonu(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "fem_genel_yakinsama"), bicimler=("png", "pdf"))


if __name__ == "__main__":
    main()
