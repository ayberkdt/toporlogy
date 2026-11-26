"""Measured values of docs/olcumler/faz_aci_onkayit.md (MA1, MA2, MA2b, MA9) → faz_aci.json.

    PYTHONPATH=src;tests/toporlogy python docs/olcumler/faz_aci.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tests" / "toporlogy"))

import test_phase_angle as TA  # noqa: E402

from toporlogy.configuration.build import build_setup  # noqa: E402
from toporlogy.core.domain import Domain  # noqa: E402
from toporlogy.design.orientation import AngleBlock, OrientedEmbed  # noqa: E402
from toporlogy.design.phases import phase_report  # noqa: E402
from toporlogy.optimization.responses import PhaseVolume  # noqa: E402
from toporlogy.optimization.study import Constraint  # noqa: E402


def ma1() -> dict[str, dict[str, float]]:
    out = {}
    for rule in ("recursive", "simplex"):
        d = Domain((12, 6))
        blk = AngleBlock("element", rmin=1.5)
        first = TA.matrix() if rule == "recursive" else TA.fibre("fibre0")
        one, _ = TA.angle_study(d, [TA.fibre()], rule, blk, [Constraint(PhaseVolume("fibre"), 0.3)])
        two, _ = TA.angle_study(d, [first, TA.fibre()], rule, blk, [Constraint(PhaseVolume("fibre"), 0.3)])
        n = d.n_design
        rng = np.random.default_rng(21)
        t, s = rng.uniform(0.2, 0.9, n), rng.uniform(0.0, 1.0, n)
        a = one.evaluate(np.concatenate([t, s]))
        b = two.evaluate(np.concatenate([np.zeros(n), t, s]))
        out[rule] = {"f0": abs(b.f0 / a.f0 - 1), "g": TA.rel(b.g, a.g), "df0": TA.rel(b.df0[n:], a.df0),
                     "dg": TA.rel(b.dg[:, n:], a.dg)}
    return out


def ma2() -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for case in TA.MA2_CASES:
        for dr in TA.MA2_DIRECTIONS:
            st, x, dv, ev = TA._ma2_fd(case, dr)
            ad = np.concatenate([[float(ev.df0 @ dv)], ev.dg @ dv])
            fd6 = TA._central(st, x, dv, 1e-6)
            rich = (4 * TA._central(st, x, dv, 5e-4) - TA._central(st, x, dv, 1e-3)) / 3

            def err(fd: np.ndarray, ad: np.ndarray = ad) -> list[float]:
                return [float(abs(f - a) / abs(f)) if f != 0 else float(abs(a)) for f, a in zip(fd, ad, strict=True)]
            out[f"{case}/{dr}"] = {"h1e-6": err(fd6), "richardson": err(rich)}
    return out


def ma2b() -> float:
    st, _ = TA._ma2_study("simplex-element")
    rng = np.random.default_rng(13)
    x = rng.uniform(0.2, 0.9, st.n_var)
    dv = rng.normal(size=st.n_var)
    ev = st.evaluate(x)
    hs = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])
    r = [abs(st.evaluate(x + h * dv, gradients=False).f0 - ev.f0 - h * float(ev.df0 @ dv)) for h in hs]
    return float(np.polyfit(np.log(hs), np.log(r), 1)[0])


def ma9() -> dict[str, object]:
    raw = yaml.safe_load(TA.EXAMPLE.read_text(encoding="utf-8"))
    setup = build_setup(raw)
    st = setup.study
    t0 = time.perf_counter()
    res = st.run(max_iter=60, tol=1e-3)
    sec = time.perf_counter() - t0
    emb = st.chain.links[0]
    assert isinstance(emb, OrientedEmbed)
    with torch.no_grad():
        st.chain(torch.as_tensor(res.x))
        emb.take()
    coh = emb.angle_filter.coherence.numpy()[st.domain.design] if emb.angle_filter is not None else None
    rep = phase_report(st.interp, res.xphys, st.domain)
    return {"iterations": res.iterations, "termination": res.termination, "seconds": sec,
            "f0_first": float(res.history["f0"][0]), "objective": float(res.objective),
            "constraints": {k: float(v) for k, v in res.constraint_values.items()}, "feasible": res.feasible,
            "phase_volume_fraction": {k: v["volume_fraction"] for k, v in rep["phases"].items()},
            "void_fraction": rep["void"]["volume_fraction"], "grey_fraction": rep["grey_fraction"],
            "grey_rule": rep["grey_rule"], "min_coherence": None if coh is None else float(coh.min()),
            "n_var": st.n_var}


def main() -> None:
    torch.set_num_threads(1)
    git = ["git", "-C", str(REPO)]
    commit = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()  # noqa: S603, S607
    dirty = bool(subprocess.run([*git, "status", "--porcelain", "src", "tests"], capture_output=True,  # noqa: S603, S607
                                text=True).stdout)
    res = {"commit": commit, "dirty": dirty, "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "MA1": ma1(),
           "MA2": ma2(), "MA2b_slope": ma2b(), "MA9": ma9()}
    (HERE / "faz_aci.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("commit", "dirty", "MA1", "MA2b_slope", "MA9")}, indent=1))


if __name__ == "__main__":
    main()
