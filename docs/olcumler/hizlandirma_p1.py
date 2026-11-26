"""Post hoc P1 (docs/olcumler/hizlandirma.md): factorisation and solve time of SuperLU (nested dissection) and of the
supernodal Cholesky with amalgamation 0 / 256 / 1024, the same matrix (cantilever, random field E = 1e-9 + r³),
torch 1 thread; minimum of two factorisations after a warm-up one.

    python docs/olcumler/hizlandirma_p1.py docs/olcumler/hizlandirma_p1.json 300x100 40x20x10
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import toporlogy as T
from toporlogy.solvers.direct import DirectSolver
from toporlogy.solvers.supernodal import SupernodalCholesky


def best_of(f, n=2):  # noqa: ANN001, ANN201 - small timing helper
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> int:
    torch.set_num_threads(1)
    out = []
    for shape in [tuple(int(v) for v in a.split("x")) for a in sys.argv[2:]]:
        d = T.Domain(shape)
        X = d.node_coords()
        m = T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 0] == 0)), 0.3)
        A = m.matrix(1e-9 + np.random.default_rng(0).random(m.n_elements) ** 3)
        o, b = m.elimination_order(), m.elimination_blocks()
        B = np.random.default_rng(1).random((len(o), 1))
        res: dict = {"shape": shape, "n": len(o)}
        s = DirectSolver("splu")
        s.factor(A, order=o)
        tf = best_of(lambda s=s, A=A, o=o: s.factor(A, order=o))
        t = time.perf_counter()
        X0 = s.solve(B)
        res["splu"] = (tf, time.perf_counter() - t)
        for am in (0, 256, 1024):
            ch = SupernodalCholesky(o, b, amalgamate=am)
            ch.factor(A)
            tf = best_of(lambda ch=ch, A=A: ch.factor(A))
            t = time.perf_counter()
            X1 = ch.solve(B)
            ts = time.perf_counter() - t
            res[f"chol{am}"] = (tf, ts, len(ch._blocks), ch.factor_nnz,
                                float(np.linalg.norm(X1 - X0) / np.linalg.norm(X0)))
        print(res, flush=True)
        out.append(res)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
