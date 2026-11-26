"""Raw measurements of the record ``laminat_uq.md`` (laminates, UQ / robust design, Pareto, identification, POD ROM).

    python docs/olcumler/laminat_uq.py            # writes docs/olcumler/laminat_uq.json (several minutes)

It reuses the problem set-ups of the tests (``tests/toporlogy/test_{laminate,uq,pareto,identification,rom}.py``), so
the numbers here are the ones the checks see. All materials, strengths, distributions and "measurements" are
synthetic. The script is also the working API example of each module.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "toporlogy")]

import toporlogy  # noqa: E402,F401  (PARDISO before torch)
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

import test_identification as TI  # noqa: E402
import test_laminate as TL  # noqa: E402
import test_pareto as TP  # noqa: E402
import test_rom as TR  # noqa: E402
import test_uq as TU  # noqa: E402
from toporlogy import Constraint  # noqa: E402
from toporlogy.core import laminate as L  # noqa: E402
from toporlogy.optimization import identification as ident  # noqa: E402
from toporlogy.optimization import pareto, rom, uq  # noqa: E402
from toporlogy.optimization.laminate_failure import first_ply_failure  # noqa: E402


def environment() -> dict:
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"date": datetime.now().astimezone().isoformat(timespec="seconds"), "git_sha": git("rev-parse", "HEAD"),
            "git_dirty_src": bool(git("status", "--porcelain", "--", "src", "tests")),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "platform": platform.platform(), "torch_threads": torch.get_num_threads()}


# ------------------------------------------------------------------------------------------------- laminate
def laminate_record() -> dict:
    angles = np.linspace(-90, 90, 181)
    Qb = L.transformed_reduced_stiffness(L.ply_reduced_stiffness(TL.MAT), angles)
    l2_classic = max(TL.rel(q, TL.classical_qbar(a)) for a, q in zip(angles, Qb, strict=True))
    l2_3d = max(TL.rel(q, TL.route_3d(a)) for a, q in zip(angles, Qb, strict=True))
    l4 = 0.0
    for lam in TL.random_stacks():
        K = np.zeros((6, 6))
        for k, z, w in TL.quad(lam):
            q = TL.route_3d(lam.angles[k])
            K += w * np.block([[q, z * q], [z * q, z * z * q]])
        h = lam.thickness
        scale = np.abs(lam.abd()[:3, :3]).max() * np.block([[np.ones((3, 3)), h * np.ones((3, 3))],
                                                            [h * np.ones((3, 3)), h * h * np.ones((3, 3))]])
        l4 = max(l4, float((np.abs(lam.abd() - K) / scale).max()))
    # API example: [0/45/-45/90]s under in-plane load and cool-down (synthetic data)
    lam = L.Laminate.from_angles(TL.MAT, [0, 45, -45, 90], TL.T_PLY, TL.ALPHA, symmetric=True)
    load = [100.0, 20.0, 0.0]
    example = {"stack": "[0/45/-45/90]s", "ply_thickness": TL.T_PLY, "A": lam.A.tolist(), "D": lam.D.tolist(),
               "classification": lam.classification(), "N": load, "dT0": -100.0}
    for crit in ("tsai_wu", "hashin"):
        r = first_ply_failure(lam, TL.STRENGTH, crit, N=load, dT0=-100.0)
        example[crit] = {"max_index": float(r.index.max()), "load_factor": r.load_factor,
                         "critical_ply": r.critical_ply, "critical_angle": float(lam.angles[r.critical_ply]),
                         "critical_mode": r.critical_mode, "sign_change_inside_ply": r.sign_change_inside_ply}
        example[crit]["load_factor_without_thermal"] = first_ply_failure(lam, TL.STRENGTH, crit, N=load).load_factor
    sens_a, sens_t = L.abd_design_derivatives(L.Laminate.from_angles(TL.MAT, TL.LAYUP, TL.THICK))
    return {"L2_max_rel_classical": l2_classic, "L2_max_rel_3d_route": l2_3d, "L4_max_scaled_error": l4,
            "L9_dA11_dtheta_per_deg": sens_a[:, 0, 0].tolist(), "L9_dA11_dt": sens_t[:, 0, 0].tolist(),
            "example": example}


# ------------------------------------------------------------------------------------------------- UQ
def uq_record() -> dict:
    out: dict = {"exact": {"mean": TU.A, "variance": TU.B ** 2, "q95": float(TU.Q95), "cvar95": float(TU.CVAR95)}}
    for method in uq.SAMPLING_METHODS:
        y = TU.values(2 ** 14, method, 1)
        out[f"normal_{method}"] = {"mean": float(y.mean()), "variance": float(y.var(ddof=1)),
                                   "q95": uq.quantile_estimate(y, 0.95).value, "cvar95": uq.cvar_estimate(y, 0.95).value,
                                   "mean_se_iid": uq.mean_estimate(y).standard_error}
    space = uq.ParameterSpace((uq.Parameter.uniform("x", 0.0, 1.0),))
    hits = 0
    for seed in range(1000):
        e = uq.mean_estimate(space.sample(200, "mc", seed).values[:, 0] ** 2)
        hits += e.ci_low <= 1 / 3 <= e.ci_high
    out["U3_coverage"] = hits / 1000
    sizes = [2 ** k for k in range(6, 14)]
    r = TU.rmse_table(sizes)
    out["convergence"] = {"n": sizes, "seeds": 40, "rmse": {m: v.tolist() for m, v in r.items()},
                          "slope": {m: float(np.polyfit(np.log(sizes), np.log(v), 1)[0]) for m, v in r.items()}}
    si = uq.sobol_indices(lambda p: float(TU.ishigami(np.array([[p["x1"], p["x2"], p["x3"]]]))[0]),
                          TU.ishigami_space(), 2 ** 13, seed=4)
    out["sobol_indices"] = {"first": si.first.tolist(), "total": si.total.tolist(), "variance": si.variance,
                            "evaluations": si.n_evaluations}
    tail = uq.exceedance_probability(uq.ParameterSpace((uq.Parameter.normal("x", 0.0, 1.0),))
                                     .sample(1000, "mc", 10).values[:, 0], 3.0902323061678132)
    out["rare_event"] = {"exact": 1e-3, "estimate": tail.value, "ci": [tail.ci_low, tail.ci_high],
                         "reliable": tail.reliable}
    t = time.perf_counter()
    out["U10c_cantilever"] = TU.robust_comparison()
    out["U10c_seconds"] = time.perf_counter() - t
    out["post_hoc_column"] = TU.column_comparison()
    return out


# ------------------------------------------------------------------------------------------------- Pareto
def pareto_record() -> dict:
    ws, eps = TP.nonconvex_front()
    sweep = TP.structural_sweep()
    warm = TP.structural_run(pareto.CaseCompliance(0), [Constraint(pareto.CaseCompliance(1), sweep["epsilon"])], 50,
                             x0=sweep["weighted"][0.5]["x"])
    row = lambda r: {"c1": float(r["c"][0]), "c2": float(r["c"][1]), "volume": r["volume"],  # noqa: E731
                     "feasible": bool(r["feasible"]), "iterations": r["iterations"]}
    return {"toy_weighted": [{"w1": float(p.parameter[0]), "f": p.objectives.tolist()} for p in ws],
            "toy_epsilon": [{"eps": float(p.parameter[0]), "f": p.objectives.tolist(), "feasible": p.feasible}
                            for p in eps],
            "structural_weighted": {str(w): row(r) for w, r in sweep["weighted"].items()},
            "structural_epsilon": {"bound": sweep["epsilon"], **row(sweep["constrained"])},
            "post_hoc_epsilon_warm": row(warm)}


# ------------------------------------------------------------------------------------------------- identification
def identification_record() -> dict:
    def pack(r: ident.Identification) -> dict:
        return {"theta": r.theta.tolist(), "std": None if r.std is None else r.std.tolist(), "rank": r.rank,
                "condition": r.condition if math.isfinite(r.condition) else None,
                "singular_values": r.singular_values.tolist(), "weighted_rms_train": r.weighted_rms}

    jac = TI.jacobian_of(TI.TRAIN)
    y = TI.model_of(TI.TRAIN)(TI.TRUE)
    clean = ident.identify(TI.model_of(TI.TRAIN), ident.Observations(y, 0.01 * np.abs(y), "synthetic"), TI.START,
                           (TI.LOW, TI.HIGH), TI.NAMES, jacobian=jac)
    noisy, held = TI.noisy_identification()
    only0 = (("0", (50.0, 0, 0)),)
    y0 = TI.model_of(only0)(TI.TRUE)
    degenerate = ident.identify(TI.model_of(only0), ident.Observations(y0, 0.01 * np.abs(y0), "synthetic"), TI.START,
                                (TI.LOW, TI.HIGH), TI.NAMES, jacobian=TI.jacobian_of(only0))
    runs, agree = ident.multi_start_identify(TI.model_of(TI.TRAIN),
                                             TI.noisy(y, 0.01, 20, "1 % noise"), (TI.LOW, TI.HIGH), TI.NAMES,
                                             n_starts=16, seed=5, jacobian=jac)
    mis, mis_rms = TI.mismatch_case()
    return {"data_kind": "synthetic", "names": list(TI.NAMES), "true": TI.TRUE.tolist(),
            "I1_jacobian_error": ident.check_jacobian(TI.model_of(TI.TRAIN), jac, TI.START, TI.START),
            "noise_free": pack(clean), "noise_free_max_rel_error": float(np.abs(clean.theta / TI.TRUE - 1).max()),
            "noisy": pack(noisy), "noisy_z": ((noisy.theta - TI.TRUE) / noisy.std).tolist(),
            "noisy_held_out_rms": ident.weighted_rms(TI.model_of(TI.HELD), held, noisy.theta),
            "zero_degree_only": {**pack(degenerate), "null_space": degenerate.null_space.tolist()},
            "multi_start": {"agree": agree, "of": len(runs), "costs": [r.cost for r in runs]},
            "mismatch": {**pack(mis), "held_out_rms": mis_rms}}


# ------------------------------------------------------------------------------------------------- ROM
def rom_record() -> dict:
    physics, regions, F, system = TR.problem()
    train, test = TR.thetas(40, 1), TR.thetas(50, 2)
    full = rom.PODModel(system, train)
    truth = [system.solve(t) for t in test]
    Ks = [system.stiffness(t) for t in test]
    ranks = sorted({r for r in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, full.rank) if r <= full.rank})
    rows = []
    for r in ranks:
        model = rom.PODModel(system, train, r)
        err, bound, eff = [], [], []
        for t, u, K in zip(test, truth, Ks, strict=True):
            sol = model.solve(t)
            d = u - sol.u
            e, nu = math.sqrt(max(float(d @ (K @ d)), 0.0)), math.sqrt(float(u @ (K @ u)))
            err.append(e / nu)
            bound.append(sol.relative_error_bound)
            eff.append(sol.error_bound / e if e > 0 else math.nan)
        rows.append({"rank": r, "max_rel_error": max(err), "median_rel_error": float(np.median(err)),
                     "max_rel_bound": max(bound), "effectivity_min": float(np.nanmin(eff)),
                     "effectivity_max": float(np.nanmax(eff))})
    gaps = TR.compliance_rows(system, rom.PODModel(system, train, TR.RANK))
    rel_other, sol_other, e_other = TR.other_load_error((physics, regions, F, system,
                                                         rom.PODModel(system, train, TR.RANK)))
    model = rom.PODModel(system, train, TR.RANK)

    def median_ms(fn) -> float:
        times = []
        for t in test:
            s = time.perf_counter()
            fn(t)
            times.append(time.perf_counter() - s)
        return 1e3 * float(np.median(times))

    return {"n_free_dofs": system.n, "train": 40, "test": 50, "numerical_rank": full.rank,
            "singular_values": full.singular_values.tolist(), "by_rank": rows,
            "R5_max_gap_error_over_e2": max(abs(g - e2) / e2 for _, g, e2 in gaps),
            "R5_max_gap_error_over_s": max(abs(g - e2) / s for s, g, e2 in gaps),
            "R6_other_load": {"rel_error": rel_other, "rel_bound": sol_other.relative_error_bound,
                              "bound_over_error": sol_other.error_bound / e_other},
            "R7_median_ms": {"full_solve": median_ms(system.solve), "reduced_only": median_ms(model.reduced_coefficients),
                             "reduced_with_bound": median_ms(model.solve), "rank": TR.RANK}}


def main() -> None:
    record = {"environment": environment()}
    for name, fn in (("laminate", laminate_record), ("uq", uq_record), ("pareto", pareto_record),
                     ("identification", identification_record), ("rom", rom_record)):
        t = time.perf_counter()
        record[name] = fn()
        print(f"{name}: {time.perf_counter() - t:.1f} s", flush=True)
    out = HERE / "laminat_uq.json"
    out.write_text(json.dumps(record, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
