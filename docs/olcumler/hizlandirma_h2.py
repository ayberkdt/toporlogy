"""H2 alone (docs/olcumler/hizlandirma.md §2): time of building SuperLU's input, the earlier construction
``(A + triu(A, 1).T).tocsr()[order][:, order].tocsc()`` against the cached index map (``_PermutedPattern.matrix``),
alternating in one process on the same matrix, and the SuperLU factorisation time of that input for scale. The
SuperLU call itself did not change (same arrays, test_speedups H2a).

    python docs/olcumler/hizlandirma_h2.py docs/olcumler/hizlandirma_h2.json 60x20 24x8x4 300x100 40x20x10 600x200
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import toporlogy as T
from toporlogy.solvers.direct import _PermutedPattern


def main() -> int:
    out = []
    for arg in sys.argv[2:]:
        shape = tuple(int(v) for v in arg.split("x"))
        d = T.Domain(shape)
        X = d.node_coords()
        m = T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 0] == 0)), 0.3)
        A = m.matrix(1e-9 + np.random.default_rng(0).random(m.n_elements) ** 3)
        order = m.elimination_order()
        t = time.perf_counter()
        pm = _PermutedPattern(A, order)
        t_build = time.perf_counter() - t
        old, new = [], []
        for _ in range(5):
            t = time.perf_counter()
            Ko = (A + sp.triu(A, k=1).T).tocsr()[order][:, order].tocsc()
            old.append(time.perf_counter() - t)
            t = time.perf_counter()
            Kn = pm.matrix(A.data)
            new.append(time.perf_counter() - t)
        same = (np.array_equal(Ko.indptr, Kn.indptr) and np.array_equal(Ko.indices, Kn.indices)
                and Ko.data.tobytes() == Kn.data.tobytes())
        t = time.perf_counter()
        spla.splu(Kn, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        t_lu = time.perf_counter() - t
        row = {"shape": shape, "n": len(order), "old_s": old, "new_s": new, "map_build_s": t_build,
               "median_old_s": statistics.median(old), "median_new_s": statistics.median(new), "splu_s": t_lu,
               "same_arrays": same}
        print(f"{arg:10s} n={len(order):7d} old {row['median_old_s'] * 1e3:8.2f} ms  "
              f"new {row['median_new_s'] * 1e3:7.2f} ms  (map once {t_build * 1e3:.1f} ms)  splu {t_lu:.3f} s  "
              f"same {same}", flush=True)
        out.append(row)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
