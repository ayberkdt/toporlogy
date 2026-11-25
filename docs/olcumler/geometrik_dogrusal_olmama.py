"""Measurements for docs/olcumler/geometrik_dogrusal_olmama.md (geometrically non-linear elasticity).

    PYTHONPATH=src python docs/olcumler/geometrik_dogrusal_olmama.py [locking] [cantilever] [benchmark]

No argument = all three. Writes ``geometrik_dogrusal_olmama.json`` next to this file (sections not run keep their
previous value).

* ``locking``: linear Q4 cantilever (``Elasticity``) against Timoshenko — the measurement the cantilever tolerances
  were set from (run before the pre-registration with the same code).
* ``cantilever``: large-deflection cantilever (V5 of the pre-registration) against the elastica.
* ``benchmark``: linear vs non-linear end-compliance design of a clamped-clamped beam (V7).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from toporlogy.core.domain import Domain
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearElasticity
from toporlogy.optimization.nonlinear import EndCompliance
from toporlogy.optimization.responses import Compliance, Volume
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "toporlogy"))
from _elastica import elastica_quadrature  # noqa: E402

OUT = Path(__file__).resolve().with_name("geometrik_dogrusal_olmama.json")


def linear_locking(slender: int = 50) -> list[dict]:
    rows = []
    for n in (2, 4, 8, 16):
        H, h = 1.0, 1.0 / n
        d = Domain((slender * n, n), h=h)
        X = d.node_coords()
        el = Elasticity(d, d.dofs(np.flatnonzero(np.abs(X[:, 0]) < 1e-9)), nu=0.0)
        tip = np.flatnonzero(np.abs(X[:, 0] - slender * H) < 1e-9)
        w = np.ones(len(tip))
        w[[0, -1]] = 0.5
        w /= w.sum()
        F = np.zeros(d.ndof)
        F[2 * tip + 1] = -w
        U = el.solve_numpy(np.ones(el.n_elements), F)[:, 0]
        L, I_ = slender * H, H ** 3 / 12
        v = -float(np.sum(w * U[2 * tip + 1]))
        tim = L ** 3 / (3 * I_) + L / (5 / 6 * 0.5 * H)
        rows.append({"n": n, "elements": slender * n * n, "v_over_timoshenko": v / tim})
    return rows


def cantilever(slender: int = 50) -> list[dict]:
    rows = []
    for material in ("svk", "neo_hooke"):
        for alpha in (1.0, 2.0):
            u_ref, v_ref, th_ref = elastica_quadrature(alpha)
            for n in (4, 8, 16):
                t = time.perf_counter()
                H, h = 1.0, 1.0 / n
                d = Domain((slender * n, n), h=h)
                X = d.node_coords()
                L = slender * H
                tip = np.flatnonzero(np.abs(X[:, 0] - L) < 1e-9)
                w = np.ones(len(tip))
                w[[0, -1]] = 0.5
                w /= w.sum()
                F = np.zeros(d.ndof)
                F[2 * tip + 1] = -alpha * (H ** 3 / 12) / L ** 2 * w
                nl = NonlinearElasticity(d, d.dofs(np.flatnonzero(X[:, 0] < 1e-9)), nu=0.0, material=material,
                                         rtol=1e-10)
                U = nl.solve_numpy(np.ones(nl.n_elements), F)[:, 0]
                u, v = -float(np.sum(w * U[2 * tip])) / L, -float(np.sum(w * U[2 * tip + 1])) / L
                rep = nl.last_reports[0]
                rows.append({"material": material, "alpha": alpha, "n": n, "u": u, "v": v, "u_ref": u_ref,
                             "v_ref": v_ref, "err_u": (u - u_ref) / u_ref, "err_v": (v - v_ref) / v_ref,
                             "steps": len(rep.steps), "iterations": rep.iterations, "cuts": rep.cuts,
                             "seconds": time.perf_counter() - t})
                print(rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------------------------- V7 benchmark
BENCH = {"shape": (96, 16), "volfrac": 0.4, "rmin": 3.0, "penal": 3.0, "emin_ratio": 1e-6, "max_iter": 150,
         "tol": 0.01, "move": 0.2, "deflection_over_span": 0.10, "material": "neo_hooke", "nu": 0.3}


def _bench_setup(nonlinear: bool, P: float) -> Study:
    b = BENCH
    d = Domain(b["shape"], h=1.0)
    X = d.node_coords()
    span = X[:, 0].max()
    fixed = d.dofs(np.flatnonzero((X[:, 0] < 1e-9) | (np.abs(X[:, 0] - span) < 1e-9)))
    top = np.abs(X[:, 1] - X[:, 1].max()) < 1e-9
    mid = np.flatnonzero(top & (np.abs(X[:, 0] - span / 2) <= 1.0 + 1e-9))      # 3 nodes at mid-span, top edge
    F = np.zeros(d.ndof)
    F[2 * mid + 1] = -P / len(mid)
    if nonlinear:
        phys = NonlinearElasticity(d, fixed, nu=b["nu"], material=b["material"],
                                   energy_interpolation=EnergyInterpolation(E0=1.0))
        obj = EndCompliance()
    else:
        phys = Elasticity(d, fixed, nu=b["nu"])
        obj = Compliance()
    chain = Chain([Embed(d), DensityFilter(d, b["rmin"]), Passive(d)])
    st = Study(d, phys, F, chain, SIMP(d, E0=1.0, penal=b["penal"], emin_ratio=b["emin_ratio"]), obj,
               [Constraint(Volume(), b["volfrac"])], MMA(move=b["move"]))
    st.mid_dofs = 2 * mid + 1                                                     # type: ignore[attr-defined]
    return st


def benchmark(save: bool = True) -> dict:
    b = BENCH
    t0 = time.perf_counter()
    lin1 = _bench_setup(False, 1.0)
    rl = lin1.run(max_iter=b["max_iter"], tol=b["tol"])
    span = float(b["shape"][0])
    delta1 = -float(np.mean(rl.U[lin1.mid_dofs, 0]))                            # linear mid deflection at P = 1
    P = b["deflection_over_span"] * span / delta1
    t1 = time.perf_counter()
    nls = _bench_setup(True, P)
    def log(it: int, ev: object, change: float) -> None:
        r = nls.physics.last_reports[0]
        print(f"NL it {it:3d}  C = {ev.f0:.6g}  change = {change:.3f}  t = {time.perf_counter() - t1:.0f} s  "
              f"steps {len(r.steps)} cuts {r.cuts} restarts {r.restarts} cutoff {r.cutoff}", flush=True)

    rn = nls.run(max_iter=b["max_iter"], tol=b["tol"], callback=log)
    t2 = time.perf_counter()
    linP = _bench_setup(False, P)
    out = {"P": P, "delta_linear_at_P1": delta1, "seconds_linear_opt": t1 - t0, "seconds_nonlinear_opt": t2 - t1,
           "linear_run": {"iterations": rl.iterations, "termination": rl.termination, "feasible": rl.feasible},
           "nonlinear_run": {"iterations": rn.iterations, "termination": rn.termination, "feasible": rn.feasible}}
    for tag, x in (("x_L", rl.x), ("x_NL", rn.x)):
        eN = nls.evaluate(x, gradients=False)
        eL = linP.evaluate(x, gradients=False)
        out[tag] = {"C_NL": eN.f0, "C_L": eL.f0, "volume": eN.values["volume"],
                    "mid_deflection_NL": -float(np.mean(eN.U[nls.mid_dofs, 0])),
                    "mid_deflection_L": -float(np.mean(eL.U[linP.mid_dofs, 0])),
                    "grey": float(np.mean(4 * eN.xphys * (1 - eN.xphys))),
                    "cutoffs": nls.physics.cutoffs()}
        if save:
            np.save(OUT.with_name(f"geometrik_dogrusal_olmama_{tag}.npy"), eN.xphys)
    out["V7a_ratio_C_NL"] = out["x_NL"]["C_NL"] / out["x_L"]["C_NL"]
    out["V7b_ratio_C_L"] = out["x_L"]["C_L"] / out["x_NL"]["C_L"]
    print(json.dumps(out, indent=1), flush=True)
    return out


def main(argv: list[str]) -> None:
    which = argv or ["locking", "cantilever", "benchmark"]
    data = json.loads(OUT.read_text(encoding="utf8")) if OUT.exists() else {}
    torch.manual_seed(0)
    np.random.seed(0)
    for w in which:
        t = time.perf_counter()
        data[w] = {"locking": linear_locking, "cantilever": cantilever, "benchmark": benchmark}[w]()
        data.setdefault("seconds", {})[w] = time.perf_counter() - t
    data["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "torch": torch.__version__, "platform": platform.platform(),
                           "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    OUT.write_text(json.dumps(data, indent=1), encoding="utf8")


if __name__ == "__main__":
    main(sys.argv[1:])
