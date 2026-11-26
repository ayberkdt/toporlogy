"""Post hoc (docs/olcumler/hizlandirma.md §4): the H1 thread policy alone, in one process, alternating torch 1 and 6
intra-op threads around the same study (this tree's code). Separate processes for base and new (hizlandirma_olcum.py)
mix the policy with the load drift between processes; here the two settings alternate every few iterations.

Per round and setting: ``Study.run`` of ``--iters`` iterations, the threads already set (no switch inside the timed
loop; ``Study.torch_threads = None``), per-iteration time from the history. Output: raw times and medians.

    python docs/olcumler/hizlandirma_iplik_ab.py docs/olcumler/hizlandirma_iplik_ab.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
ROWS = [("mbb2d", []), ("cantilever3d", []), ("eigenfrequency2d", []), ("inverter2d", []),
        ("buckling_column2d", []), ("stress_lbracket2d", []), ("multiload3d", []),
        ("mbb2d_300x100", ["domain.shape=[300,100]"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.resolve import read_config
    try:
        import psutil
    except ImportError:
        psutil = None
    rows = []
    for tag, sets in ROWS:
        ex = tag.split("_")[0] if sets else tag
        path = next(p for p in [REPO / "examples" / f"{ex}.yaml", REPO / "examples" / "research" / f"{ex}.yaml"]
                    if p.is_file())
        st = build_setup(read_config(str(path)), sets).study
        st.torch_threads = None                              # the setting is made here, outside the timed loop
        t_it: dict[int, list[float]] = {1: [], 6: []}
        load = []
        if psutil is not None:
            psutil.cpu_percent(None)
        for _ in range(a.rounds):
            for th in (1, 6):
                torch.set_num_threads(th)
                st.run(max_iter=1, tol=0.0)                   # warm the pool after the switch
                r = st.run(max_iter=a.iters, tol=0.0)
                sec = np.diff(np.r_[0.0, r.history["seconds"]])
                t_it[th] += [float(v) for v in sec[1:]]       # the first iteration holds the run's start-up
                if psutil is not None:
                    load.append(psutil.cpu_percent(None))
        m1, m6 = statistics.median(t_it[1]), statistics.median(t_it[6])
        rows.append({"tag": tag, "overrides": sets, "per_iteration_s": {"1": t_it[1], "6": t_it[6]},
                     "median_1": m1, "median_6": m6, "ratio_6_over_1": m6 / m1, "cpu_percent": load})
        print(f"{tag:22s} 1 thread {m1:8.4f} s/it  6 threads {m6:8.4f} s/it  6/1 = {m6 / m1:5.2f}  "
              f"cpu {statistics.median(load) if load else float('nan'):.0f} %", flush=True)
    torch.set_num_threads(6)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "iters": a.iters, "rounds": a.rounds,
            "torch": torch.__version__}
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
