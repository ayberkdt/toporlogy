"""Measured values of docs/olcumler/malzeme_tasarimi_onkayit.md (pass / fail is decided by
tests/toporlogy/test_material_design.py with the same definitions; this script records the numbers) and the
threshold-free measurements: A4b (linear θ filter), K4 (order dependence of the recursive rule), A6 sweep, A7,
the two-phase rule comparison and timings.

    python docs/olcumler/malzeme_tasarimi.py            # writes docs/olcumler/malzeme_tasarimi.json
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "toporlogy"))

import test_material_design as T  # noqa: E402

from toporlogy.core.domain import Domain  # noqa: E402
from toporlogy.core.linear_elastic import ElasticTensor  # noqa: E402
from toporlogy.core.material_field import PhaseMaterial  # noqa: E402
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive  # noqa: E402
from toporlogy.design.interp import SIMP  # noqa: E402
from toporlogy.design.orientation import PeriodicAngleFilter, wrap_angle  # noqa: E402
from toporlogy.design.phases import TensorPhases, phase_fractions  # noqa: E402
from toporlogy.fem.elastic import Elasticity  # noqa: E402
from toporlogy.fem.tensor_elastic import TensorElasticity  # noqa: E402
from toporlogy.optimization.responses import Compliance, PhaseVolume  # noqa: E402
from toporlogy.optimization.study import Constraint, Study  # noqa: E402
from toporlogy.optimizers.mma import MMA  # noqa: E402


def a4() -> dict[str, float]:
    d = Domain((10, 10))
    ii, jj = np.indices(d.shape)
    th = np.where((ii + jj) % 2 == 0, math.radians(85.0), math.radians(-85.0))
    out = PeriodicAngleFilter(d, 2.5, math.pi)(torch.as_tensor(th)).numpy()
    lin = DensityFilter(d, 2.5)(torch.as_tensor(th)).numpy()
    dev = np.degrees(np.abs(wrap_angle(out - math.pi / 2, math.pi)))
    dev_lin = np.degrees(np.abs(wrap_angle(lin - math.pi / 2, math.pi)))
    inner = (slice(3, 7), slice(3, 7))
    return {"A4a_max_deviation_deg": float(dev.max()), "A4b_linear_filter_max_deviation_deg": float(dev_lin.max()),
            "A4b_linear_filter_interior_min_deviation_deg": float(dev_lin[inner].min())}


def k4_recursive_order() -> float:
    d = Domain((2, 2))
    ph = T.phases(2)
    perm = [2, 0, 1]
    a = TensorPhases(d, ph, rule="recursive")
    b = TensorPhases(d, [ph[i] for i in perm], rule="recursive")
    eta = T._dirichlet(3, 500, 5)

    def box(e: np.ndarray) -> np.ndarray:                     # η → t (inverse stick-breaking)
        m = e.shape[0] - 1
        t, rem = np.zeros((m, e.shape[1])), np.ones(e.shape[1])
        for j in range(m - 1, -1, -1):
            t[j] = np.divide(e[j + 1], rem, out=np.zeros_like(rem), where=rem > 0)
            rem = rem - e[j + 1]
        return t
    eb = eta[[0] + [1 + i for i in perm]]
    Ca = a.stiffness_of(torch.as_tensor(box(eta))).numpy()
    Cb = b.stiffness_of(torch.as_tensor(box(eb))).numpy()
    return T.rel(Cb, Ca)


def a5() -> list[dict[str, float]]:
    rows = []
    for a in range(0, 91, 15):
        th = torch.tensor(math.radians(a), dtype=torch.float64, requires_grad=True)
        c, s2v = T._plate_compliance(th)
        (g,) = torch.autograd.grad(c, th)
        Ex = T.directional_modulus(1.0, 0.2, 0.1, 0.3, math.radians(a))
        ref = s2v * T.directional_compliance_derivative(1.0, 0.2, 0.1, 0.3, math.radians(a))
        rows.append({"angle_deg": a, "Ex_fe": s2v / float(c), "Ex_closed": float(Ex),
                     "Ex_rel_err": abs(s2v / float(c) / Ex - 1), "dc_dtheta": float(g), "dc_dtheta_closed": ref,
                     "grad_rel_err": abs(float(g) / ref - 1) if a not in (0, 90) else None,
                     "grad_abs_over_c": abs(float(g)) / float(c) if a in (0, 90) else None})
    return rows


def a6_a7() -> dict[str, object]:
    angles, cs = T.sweep_cantilever()
    k = int(np.argmin(cs))
    runs = []
    for start in (0.0, 45.0, -45.0):
        st, emb = T._cantilever_angle_study("global", None, start, 30)
        t0 = time.perf_counter()
        res = st.run(x0=emb.reduce(np.zeros(0)), max_iter=30, tol=1e-9)
        lo, hi = emb.angle_block.bounds
        th = math.degrees(lo + (hi - lo) * float(res.x[0]))
        runs.append({"start_deg": start, "theta_opt_deg": th, "c_opt": res.objective, "iterations": res.iterations,
                     "seconds": time.perf_counter() - t0,
                     "angle_error_deg": abs(float(wrap_angle(th - angles[k], 180.0))),
                     "c_ratio_to_sweep_min": res.objective / cs[k]})
    st, emb = T._cantilever_angle_study("element", 2.0, float(angles[k]), 40)
    res = st.run(x0=emb.reduce(np.zeros(0)), max_iter=40, tol=1e-9)
    st.evaluate(res.x, gradients=False)
    coh = st.chain[0].angle_filter.coherence
    lo, hi = emb.angle_block.bounds
    theta_field = np.degrees(st.chain[0]._angle.detach().numpy()) if st.chain[0]._angle is not None else None
    return {"sweep_angles_deg": angles.tolist(), "sweep_compliance": cs.tolist(), "sweep_argmin_deg": float(angles[k]),
            "sweep_min": float(cs[k]), "sweep_local_minima_per_period": T._local_minima(cs), "global_runs": runs,
            "A7": {"c_uniform_opt": float(cs[k]), "c_element": res.objective, "iterations": res.iterations,
                   "improvement": 1 - res.objective / cs[k],
                   "min_coherence": float(coh[torch.as_tensor(Domain((40, 20)).design)].min()),
                   "theta_field_deg": None if theta_field is None else theta_field.tolist()}}


def rule_comparison() -> list[dict[str, object]]:
    d = Domain((48, 24))
    F = T.tip_load(d)
    out = []
    ph = [PhaseMaterial("P1", ElasticTensor.isotropic(0.3, 0.2), density=0.4),
          PhaseMaterial("P3", ElasticTensor.orthotropic(**T.ORTHO, R=T.rotation_about_z(30.0)), density=1.0)]
    for rule in T.RULES:
        tp = TensorPhases(d, ph, rule=rule)
        te = TensorElasticity(d, T.left_fixed(d))
        te.bind(tp)
        chain = Chain([Embed(d, 2), DensityFilter(d, 2.0), Passive(d)])
        cons = [Constraint(PhaseVolume("P1"), 0.15), Constraint(PhaseVolume("P3"), 0.2)]
        st = Study(d, te, F, chain, tp, Compliance(), cons, MMA(move=0.1))
        t0 = time.perf_counter()
        res = st.run(max_iter=80, tol=1e-3)
        eta = phase_fractions(torch.as_tensor(res.xphys)).numpy()
        out.append({"rule": rule, "compliance": res.objective, "iterations": res.iterations,
                    "seconds": time.perf_counter() - t0, "feasible": res.feasible,
                    "volume_P1": float(eta[1].mean()), "volume_P3": float(eta[2].mean()),
                    "grey_fraction": float(np.mean(eta.max(axis=0) < 0.9))})
    return out


def timing() -> dict[str, float]:
    d = Domain((120, 60))
    F = T.tip_load(d)
    x = torch.as_tensor(np.random.default_rng(0).uniform(0.2, 1.0, d.shape))
    el = Elasticity(d, T.left_fixed(d))
    simp = SIMP(d)
    tp = TensorPhases(d, [PhaseMaterial("s", ElasticTensor.isotropic(1.0, 0.3))])
    te = TensorElasticity(d, T.left_fixed(d))
    te.bind(tp)

    def run(phys: object, interp: object, n: int = 5) -> float:
        ts = []
        for _ in range(n):
            xt = x.clone().requires_grad_(True)
            t0 = time.perf_counter()
            E = interp(xt)
            U = phys.solve(E.reshape(-1)[torch.as_tensor(phys.elements)], F)
            torch.autograd.grad((torch.as_tensor(F)[:, None] * U).sum(), xt)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    nt = torch.get_num_threads()
    a, b = run(el, simp), run(te, tp)
    torch.set_num_threads(1)
    try:
        a1, b1 = run(el, simp), run(te, tp)
    finally:
        torch.set_num_threads(nt)
    f = tp.material_field(x)
    return {"grid": "120 x 60", "torch_threads": nt, "simp_elasticity_s": a, "tensor_s": b, "ratio": b / a,
            "simp_elasticity_s_1thread": a1, "tensor_s_1thread": b1, "ratio_1thread": b1 / a1,
            "material_field_bytes": f.nbytes(), "element_matrix_bytes": int(te.KE.nbytes)}


def main() -> None:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S607
                         cwd=REPO).stdout.strip()
    out = {"commit": sha, "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "platform": platform.platform()},
           "A4": a4(), "K4_recursive_order_dependence": k4_recursive_order(), "A5": a5(), "A6_A7": a6_a7(),
           "rule_comparison": rule_comparison(), "timing": timing()}
    (HERE / "malzeme_tasarimi.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("A6_A7",)}, indent=1)[:4000])
    print("A6 runs", out["A6_A7"]["global_runs"], "A7", {k: v for k, v in out["A6_A7"]["A7"].items()
                                                         if k != "theta_field_deg"})


if __name__ == "__main__":
    main()
