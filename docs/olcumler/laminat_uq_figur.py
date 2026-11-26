"""Figures of the record ``laminat_uq.md`` from ``laminat_uq.json`` (datawiz set ``laminat_uq.seti.json``).

    python docs/olcumler/laminat_uq_figur.py
"""
from __future__ import annotations

import json
from pathlib import Path

import datawiz as dw
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "laminat_uq.json").read_text(encoding="utf-8"))
S = dw.Seti(str(HERE / "laminat_uq.seti.json"), hedef="web-640", ondalik_ayirici=",", palet="okabe-ito")
S.varlik("Ağırlıklı toplam", renk="#a86b00")    # darker Okabe-Ito orange: 3:1 against white


def convergence() -> None:
    c = D["uq"]["convergence"]
    n = np.array(c["n"])
    fig, ax = S.figur(oran=0.58, ondalik=",")
    names = {"mc": "Monte Carlo", "lhs": "Latin hiperküpü", "sobol": "Sobol"}
    for key, label in names.items():
        slope = f"{c['slope'][key]:.2f}".replace(".", ",").replace("-", "−")
        ax.plot(n, c["rmse"][key], **{**S.stil(label, isaretli=True), "label": f"{label} · eğim {slope}"})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n[::1], [str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlabel("Örnek sayısı n")
    ax.set_ylabel("Ortalama tahmininin RMSE'si (40 tohum)")
    ax.set_title("Karıştırılmış Sobol dizisi Ishigami ortalamasında en hızlı yakınsıyor", loc="left")
    dw.izgara.uygula(ax, "yok")
    dw.etiket.cizgi_sonu(ax)
    S.kaydet(fig, str(HERE / "laminat_uq_yakinsama"), cvd=True)


def pareto() -> None:
    p = D["pareto"]
    fig, axs = S.figur(oran=0.5, mozaik=[["a", "b"]], ondalik=",")
    ax = axs["a"]
    f1 = np.linspace(0, 1, 200)
    ax.plot(f1, 1 - f1**2, color="#8a8a8a", lw=1.6, label="Gerçek cephe", zorder=1)
    eps = np.array([q["f"] for q in p["toy_epsilon"]])
    ws = np.array([q["f"] for q in p["toy_weighted"]])
    ax.plot(eps[:, 0], eps[:, 1], ls="none", ms=5, mec="#3a3a3a", mew=0.6, label="ε-kısıt", zorder=2,
            marker=S.varlik("ε-kısıt")["isaret"], color=S.varlik("ε-kısıt")["renk"])
    ax.plot(ws[:, 0], ws[:, 1], ls="none", ms=9, mfc="none", mew=1.6, label="Ağırlıklı toplam",
            zorder=3, marker=S.varlik("Ağırlıklı toplam")["isaret"], color=S.varlik("Ağırlıklı toplam")["renk"])
    ax.set_xlabel("Birinci amaç f₁")
    ax.set_ylabel("İkinci amaç f₂")
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.06, 1.3)
    off = ws[ws[:, 1] > 1.3]
    if len(off):
        # a zero weight leaves f2 free: that solution is only weakly efficient; it is named, not hidden
        value = f"({off[0, 0]:.0f}; {off[0, 1]:.2f})".replace(".", ",")
        ax.annotate(f"w₁ = 1 çözümü {value}\neksen dışında", (0.0, 1.27), fontsize=9, va="top", ha="left",
                    xytext=(8, 0), textcoords="offset points")
    ax.set_title("Konveks olmayan cephe", loc="left")

    bx = axs["b"]
    w = [p["structural_weighted"][k] for k in sorted(p["structural_weighted"], key=float)]
    v = np.array([[r["c1"], r["c2"]] for r in w])
    bx.plot(v[:, 0], v[:, 1], lw=1.6, ms=5, marker=S.varlik("Ağırlıklı toplam")["isaret"],
            color=S.varlik("Ağırlıklı toplam")["renk"], label="Ağırlıklı toplam")
    cold = p["structural_epsilon"]
    bx.plot([cold["c1"]], [cold["c2"]], ls="none", ms=5, mec="#3a3a3a", mew=0.6, marker=S.varlik("ε-kısıt")["isaret"],
            color=S.varlik("ε-kısıt")["renk"], label="ε-kısıt")
    bx.annotate("ε-kısıt, düz başlangıç\n(yerel minimum)", (cold["c1"], cold["c2"]), xytext=(4, 12),
                textcoords="offset points", ha="center", va="bottom", fontsize=9)
    bx.set_xlabel("Birinci yük durumu uyumu c₁")
    bx.set_ylabel("İkinci yük durumu uyumu c₂")
    bx.set_title("Konsol, iki yük durumu", loc="left")
    bx.margins(0.12, 0.14)
    S.panel_harfleri([ax, bx])
    for a in (ax, bx):
        dw.izgara.uygula(a, "yok")
    dw.lejant.ortak(fig)
    S.kaydet(fig, str(HERE / "laminat_uq_pareto"), cvd=True)


def reduced() -> None:
    rows = D["rom"]["by_rank"]
    rows = [r for r in rows if r["max_rel_error"] > 0]
    r = np.array([q["rank"] for q in rows])
    fig, ax = S.figur(oran=0.55, ondalik=",")
    ax.plot(r, [q["max_rel_bound"] for q in rows], **S.stil("Hata sınırı", isaretli=True))
    ax.plot(r, [q["max_rel_error"] for q in rows], **S.stil("Gerçek hata", isaretli=True))
    ax.set_yscale("log")
    ax.set_xlabel("İndirgenmiş taban boyutu r")
    ax.set_ylabel("En büyük bağıl enerji hatası (50 test)")
    ax.set_title("Hata sınırı gerçek hatanın hep üstünde kalıyor", loc="left")
    dw.izgara.uygula(ax, "yok")
    dw.etiket.cizgi_sonu(ax)
    S.kaydet(fig, str(HERE / "laminat_uq_rom"), cvd=True)


if __name__ == "__main__":
    convergence()
    pareto()
    reduced()
