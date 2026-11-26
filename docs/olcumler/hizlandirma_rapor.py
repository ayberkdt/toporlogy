"""Tables and the figure of docs/olcumler/hizlandirma.md from the raw before/after measurement.

    python docs/olcumler/hizlandirma_rapor.py docs/olcumler/hizlandirma_sonrasi.json --figure <out>

Per tag and code ("base" / "new") the per-iteration median of every component is taken from each run of
``hizlandirma_olcum.py``; with several passes (``--new-first`` second pass) the passes are listed separately and the
ratio is given per pass. The figure (datawiz, dumbbell on a log axis) shows the per-iteration time base → new.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

COMPS = ("factor", "opt_evals", "assembly", "solve", "residual", "sensitivity", "filter", "optimizer", "other")


def load(paths: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = []
    for k, p in enumerate(paths, 1):
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            r["pass"] = k
            rows.append(r)
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" in r:
            continue
        out[r["tag"]][r["code"]].append(r)
    return out


def fmt(v: float) -> str:
    return f"{v:.3g}" if v < 10 else f"{v:.1f}"


def tables(data: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    L = ["| Satır | Serbest DOF | Taban (s/it) | Yeni (s/it) | Oran | Taban: çarpanlama / OC değerl. | "
         "Yeni: çarpanlama / OC değerl. | CPU ortanca % (taban / yeni) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, by in data.items():
        if "base" not in by:
            continue
        for k, (b, n) in enumerate(zip(by["base"], by["new"], strict=False)):
            mb, mn = b["median_per_iteration_s"], n["median_per_iteration_s"]
            lb, ln = (b.get("load_cpu_percent") or {}).get("median"), (n.get("load_cpu_percent") or {}).get("median")
            L.append(f"| {tag} (geçiş {k + 1}) | {b['n_free']} | {fmt(mb['iteration'])} | "
                     f"{fmt(mn['iteration'])} | {mb['iteration'] / mn['iteration']:.2f}× | {fmt(mb['factor'])} / "
                     f"{fmt(mb['opt_evals'])} | {fmt(mn['factor'])} / {fmt(mn['opt_evals'])} | "
                     f"{lb if lb is None else round(lb)} / {ln if ln is None else round(ln)} |")
    return "\n".join(L)


def components(data: dict[str, dict[str, list[dict[str, Any]]]], tags: list[str]) -> str:
    L = ["| Satır | Kod | " + " | ".join(COMPS) + " | iterasyon |", "|---|---|" + "---:|" * (len(COMPS) + 1)]
    for tag in tags:
        for code in ("base", "new"):
            for r in data.get(tag, {}).get(code, [])[:1]:
                m = r["median_per_iteration_s"]
                L.append(f"| {tag} | {code} | " + " | ".join(f"{1e3 * m.get(c, 0.0):.1f}" for c in COMPS)
                         + f" | {1e3 * m['iteration']:.1f} |")
    return "\n".join(L) + "\n\n(ms / iterasyon, ortanca)"


def cholesky(data: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    """Cholesky rows of every pass against the SuperLU row (new code) of the first pass; pass 1 ran the pre-registered
    H3 code, pass 2 the post hoc P1 code."""
    L = ["| Satır | Geçiş (kod) | Serbest DOF | SuperLU çarpanlama (s) | Cholesky çarpanlama (s) | Oran | "
         "SuperLU çözüm (s) | Cholesky çözüm (s) | İterasyon SuperLU → Cholesky (s) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, by in data.items():
        if not tag.endswith("_chol") or "new" not in by:
            continue
        ref = data.get(tag[:-5], {}).get("new")
        if not ref:
            continue
        mr = ref[0]["median_per_iteration_s"]
        for r in by["new"]:
            mc = r["median_per_iteration_s"]
            label = {"1": "1 (H3)", "2": "2 (P1)"}.get(str(r.get("pass", "")), str(r.get("git", ""))[:7])
            L.append(f"| {tag[:-5]} | {label} | {ref[0]['n_free']} | {fmt(mr['factor'])} | {fmt(mc['factor'])} | "
                     f"{mr['factor'] / mc['factor']:.2f}× | {fmt(mr['solve'])} | {fmt(mc['solve'])} | "
                     f"{fmt(mr['iteration'])} → {fmt(mc['iteration'])} |")
    return "\n".join(L)


def figure(data: dict[str, dict[str, list[dict[str, Any]]]], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import datawiz as dw
    names, before, after = [], [], []
    for tag, by in data.items():
        if "base" in by and "new" in by:
            names.append(f"{tag} ({by['base'][0]['n_free']:,} DOF)".replace(",", " "))
            before.append(statistics.median(r["median_per_iteration_s"]["iteration"] for r in by["base"]))
            after.append(statistics.median(r["median_per_iteration_s"]["iteration"] for r in by["new"]))
    fig, ax = dw.figur("makale", hedef="genel-cift", oran=0.62)
    dw.ciz.halter(ax, names, before, after, etiketler=("taban 10daa03", "yeni"), sirala="fark")
    ax.set_xscale("log")
    ax.set_xlabel("İterasyon süresi (s, ortanca, log ölçek)")
    ax.set_ylabel("Satır")
    dw.kaydet(fig, out, hedef="genel-cift", cvd=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+", help="one JSON per pass, in pass order")
    ap.add_argument("--figure")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    data = load([Path(p) for p in a.raw])
    print(tables(data))
    print()
    print(components(data, ["mbb2d", "cantilever3d", "mbb2d_300x100", "cantilever3d_40x20x10",
                            "multiload3d_40x16x12", "mbb2d_600x200"]))
    print()
    print(cholesky(data))
    if a.figure:
        figure(data, a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
