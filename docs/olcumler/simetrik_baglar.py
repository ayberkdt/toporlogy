"""PR-11 experiment: the bracket (a) without ties, (b) with mirrored ties, (c) with SymmetryTying, against a one-sided
tie (roadmap §2.3). Same YAML otherwise (150 iterations, MMA, beta continuation).

usage: python docs/olcumler/simetrik_baglar.py --configs <folder>   (writes <arm>.yaml for the four arms)
       python -m toporlogy run <folder>/<arm>.yaml --out <folder>/<arm>   (each arm)
       python docs/olcumler/simetrik_baglar.py <folder> [out.json]   (the record)
The study files are git-ignored and read from the main checkout (ROOT below). The stored run of 18 November
(``ref_stored``) is listed as well: it was optimised before commit 29e2ac5 changed the tie paths, so today's ties
are not the ties it was optimised with (``keep_void_cells`` counts today's keep cells that are void in its design).
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

import toporlogy.optimization.connectivity as C
from toporlogy.configuration.build import build_setup
from toporlogy.execution.runs import read_run
from toporlogy.optimization.diagnostics import field_symmetry

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "final_2026" / "ansys" / "toporlogy_2026-11-18"
PLANE = ("x", 500.0)
ARMS = ("r_one_sided_tie", "a_no_ties", "b_sym_ties", "c_sym_tying")


def config_of(arm: str) -> dict:
    raw = yaml.safe_load((STUDY / "official_bracket.yaml").read_text(encoding="utf-8"))
    if arm == "a_no_ties":
        raw["connectivity"] = {**raw["connectivity"], "ties": None}
    elif arm == "b_sym_ties":
        raw["connectivity"] = {**raw["connectivity"], "symmetry": {"x": 500}}
    elif arm == "c_sym_tying":
        raw["chain"] = [{"type": "symmetry", "planes": {"x": 500}}] + list(raw["chain"])
    return raw


def gate_masks_and_domain(arm: str, xphys_file: Path):
    """Domain of the arm and the interface masks its connectivity gate uses (read from the gate's call of
    analyse_connectivity: the masks are built inside the post step, not kept in the Setup)."""
    raw = config_of(arm)
    raw["post"] = [{"type": "connectivity_gate"}]
    seen: dict = {}
    orig = C.analyse_connectivity

    def spy(dom, x, masks, *a, **k):
        seen.setdefault("masks", masks)
        return orig(dom, x, masks, *a, **k)
    C.analyse_connectivity = spy                       # the step imports it when it is built
    try:
        s = build_setup(raw)
        d = s.study.domain
        xp = np.load(xphys_file)["xphys"].reshape(d.shape)
        with tempfile.TemporaryDirectory() as td:
            s.post[0][1](types.SimpleNamespace(xphys=xp), s.study, Path(td))
    finally:
        C.analyse_connectivity = orig
    return d, xp, seen["masks"]


def necks(d, xp, masks) -> dict:
    """Minimum cut (designable cells) between every pair of interfaces, as the gate computes it; ``None`` when the
    pair touches or is joined through frozen cells only (keep, ties, interface pads), so no design change can cut it."""
    solid = (xp >= 0.5) & d.active
    frozen = d.keep | np.logical_or.reduce(list(masks.values()))
    names = list(masks)
    cuts = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c, _ = C.min_cut_cells(solid, masks[names[i]], masks[names[j]], "face", frozen)
            cuts[f"{names[i]}-{names[j]}"] = c
    lab, n = ndimage.label(frozen & solid)
    cut = [v for v in cuts.values() if v is not None]
    return {"pairs": len(cuts), "pairs_without_cut": sum(v is None for v in cuts.values()),
            "min_cut_cells": min(cut) if cut else None, "frozen_solid_components": int(n), "cuts": cuts}


def row(arm: str, folder: Path) -> dict:
    s = read_run(folder)
    d, xp, masks = gate_masks_and_domain("r_one_sided_tie" if arm == "ref_stored" else arm, folder / "result.npz")
    sym = field_symmetry(d, {"keep": d.keep.astype(float), "xphys": xp}, *PLANE)["fields"]
    gate = (s.get("post") or {}).get("connectivity_gate", {})
    neck = gate.get("min_neck") or {}
    return {"arm": arm, "objective": s["objective"], "verdict": s.get("verdict"), "status": s.get("status"),
            "iterations": s["iterations"], "kkt_residual": s.get("kkt_residual"),
            "max_violation": s.get("max_violation"),
            "constraints": {c["name"]: c["value"] for c in s["constraints"]},
            "n_bodies": gate.get("n_bodies"), "gate": gate.get("verdict"),
            "robust_checks": [(c["label"], c["n_bodies"]) for c in gate.get("checks", [])],
            "min_neck_cells": neck.get("cells"), "min_neck_area": neck.get("area"),
            "keep_cells": int(d.keep.sum()), "keep_void_cells": int((xp[d.keep] < 0.5).sum()),
            "e_sym_keep": sym["keep"]["e_sym"], "e_sym_xphys": sym["xphys"]["e_sym"],
            "necks": necks(d, xp, masks), "seconds": s.get("seconds")}


if __name__ == "__main__":
    if sys.argv[1] == "--configs":
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        for a in ARMS:
            (out / f"{a}.yaml").write_text(yaml.safe_dump(config_of(a), sort_keys=False), encoding="utf-8")
        sys.exit(0)
    runs = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("simetrik_baglar.json")
    rows = [row("ref_stored", STUDY / "run")] if (STUDY / "run" / "result.npz").is_file() else []
    rows += [row(a, runs / a) for a in ARMS if (runs / a / "result.npz").is_file()]
    dst.write_text(json.dumps({"plane": PLANE, "rows": rows}, indent=1, default=float) + "\n", encoding="utf-8")
    for r in rows:
        n = r["necks"]
        print(f"{r['arm']:16s} c {r['objective']:.1f} {r['verdict']} kkt {r['kkt_residual']:.3f} "
              f"bodies {r['n_bodies']} neck {r['min_neck_cells']} uncuttable {n['pairs_without_cut']}/{n['pairs']} "
              f"keep void {r['keep_void_cells']} e_sym keep {r['e_sym_keep']:.3f} x {r['e_sym_xphys']:.3f}")
