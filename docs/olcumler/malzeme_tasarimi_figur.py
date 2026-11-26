"""Figures of docs/olcumler/malzeme_tasarimi.md from malzeme_tasarimi.json (datawiz, a local plotting package that
is not a dependency of toporlogy).

    python docs/olcumler/malzeme_tasarimi_figur.py

FİGÜR BRİFİ (ikisi için): okur uzman; ortam web (depo belgesi), ondalık virgül; set malzeme_tasarimi.seti.json.
1 malzeme_tasarimi_yonlu_modul — mesaj: sonlu elemanın eğik çekme modülü kapalı formla 1e-12 içinde örtüşüyor
  (A5). Veri: açı (nicel, 0–90°), E_x / E1 (nicel); kapalı form sürekli çizgi, FE 7 nokta. Biçim: çizgi + nokta.
2 malzeme_tasarimi_tarama — mesaj: tek global açıda uyum 0°'de en küçük; MMA her başlangıçta 1° içine iniyor ama
  ±1,6° salınıyor (A6). Veri: açı (−90…89°, 1° adım), uyum (nicel); üç MMA sonucu. Biçim: çizgi + nokta.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    import datawiz as dw

    d = json.loads((HERE / "malzeme_tasarimi.json").read_text(encoding="utf-8"))
    S = dw.Seti(str(HERE / "malzeme_tasarimi.seti.json"), hedef="web-640", ortam="web", ondalik_ayirici=",",
                palet="okabe-ito")
    S.varlik("kapalı form")
    S.varlik("sonlu eleman")
    S.varlik("1° tarama", baglam=True)
    S.varlik("MMA sonucu")

    # ---------------------------------------------------------------- 1: directional modulus (A5)
    rows = d["A5"]
    th = np.linspace(0.0, 90.0, 181)
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    ex = 1.0 / (c ** 4 / 1.0 + (1.0 / 0.1 - 2 * 0.3 / 1.0) * s ** 2 * c ** 2 + s ** 4 / 0.2)
    fig, ax = S.figur(oran=0.62)
    ax.plot(th, ex, **{**S.stil("kapalı form"), "label": "kapalı form"})
    ax.plot([r["angle_deg"] for r in rows], [r["Ex_fe"] for r in rows],
            **{**S.stil("sonlu eleman", isaretli=True), "linestyle": "none", "label": "sonlu eleman"})
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fiber açısı θ (°)")
    ax.set_ylabel("Eğik çekme modülü E_x / E1 (–)")
    dw.lejant.yerlestir(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "malzeme_tasarimi_yonlu_modul"), bicimler=("png", "pdf"))

    # ---------------------------------------------------------------- 2: 1-D sweep and MMA (A6)
    a = d["A6_A7"]
    ang, cs = np.array(a["sweep_angles_deg"]), np.array(a["sweep_compliance"])
    fig, ax = S.figur(oran=0.62)
    ax.plot(ang, cs, **{**S.stil("1° tarama"), "label": "1° tarama"})
    runs = a["global_runs"]
    ax.plot([r["theta_opt_deg"] for r in runs], [r["c_opt"] for r in runs],
            **{**S.stil("MMA sonucu", isaretli=True), "linestyle": "none", "label": "MMA sonucu (3 başlangıç)"})
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("Global fiber açısı θ (°)")
    ax.set_ylabel("Uyum c (kuvvet × uzunluk)")
    dw.lejant.yerlestir(ax)
    dw.izgara.uygula(ax, "silik", "y")
    S.kaydet(fig, str(HERE / "malzeme_tasarimi_tarama"), bicimler=("png", "pdf"))
    print("min", float(cs.min()), "at", float(ang[int(np.argmin(cs))]), math.isfinite(float(cs.max())))


if __name__ == "__main__":
    main()
