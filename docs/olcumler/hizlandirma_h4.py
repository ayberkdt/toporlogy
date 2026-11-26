"""H4a (docs/olcumler/hizlandirma_onkayit.md §2): ``Study.constraint_values`` call time with the earlier and the new
``Passive.forward`` / ``Volume.__call__``, alternating in one process (blocks of ``--calls`` calls, ``--rounds``
rounds, torch 1 thread as inside ``Study.run``).

    python docs/olcumler/hizlandirma_h4.py docs/olcumler/hizlandirma_h4.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch  # noqa: F401  (before toporlogy, as the other measurement scripts)

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.resolve import read_config
    from toporlogy.design.chain import Passive
    from toporlogy.optimization.responses import Volume

    def old_passive(self, x):  # the implementation before H4
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        return torch.where(self.keep, one, torch.where(self.off, zero, x))

    def old_volume(self, ctx):  # the implementation before H4
        d = ctx.domain
        if ctx.xphys.dim() != d.dim:
            raise ValueError("one design field")
        if self.weighted:
            fr = torch.as_tensor(d.frac, dtype=ctx.xphys.dtype, device=ctx.xphys.device)
            return (fr * ctx.xphys).sum() / float(d.frac.sum())
        act = torch.as_tensor(d.active, device=ctx.xphys.device)
        return ctx.xphys[act].sum() / float(d.n_active)

    impl = {"new": (Passive.forward, Volume.__call__), "old": (old_passive, old_volume)}
    rows = []
    for ex in ("cantilever3d", "mbb2d", "symmetric_am3d"):
        st = build_setup(read_config(str(REPO / "examples" / f"{ex}.yaml"))).study
        x = st.default_start()
        times: dict[str, list[float]] = {"old": [], "new": []}
        vals = {}
        for _ in range(a.rounds):
            for which in ("old", "new"):
                Passive.forward, Volume.__call__ = impl[which]
                st.constraint_values(x)
                t = time.perf_counter()
                for _ in range(a.calls):
                    v = st.constraint_values(x)
                times[which].append((time.perf_counter() - t) / a.calls)
                vals[which] = v.tobytes()
        Passive.forward, Volume.__call__ = impl["new"]
        mo, mn = statistics.median(times["old"]), statistics.median(times["new"])
        rows.append({"example": ex, "old_us": [1e6 * t for t in times["old"]], "new_us": [1e6 * t for t in times["new"]],
                     "median_old_us": 1e6 * mo, "median_new_us": 1e6 * mn, "reduction": 1 - mn / mo,
                     "same_value_bits": vals["old"] == vals["new"]})
        print(f"{ex:16s} old {1e6 * mo:7.0f} us  new {1e6 * mn:7.0f} us  reduction {100 * (1 - mn / mo):5.1f} %  "
              f"same bits {vals['old'] == vals['new']}", flush=True)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "calls": a.calls, "rounds": a.rounds, "torch_threads": 1}
    try:
        import psutil
        meta["cpu_percent_after"] = psutil.cpu_percent(interval=1.0)
    except ImportError:
        pass
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
