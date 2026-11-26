"""Measurements of ``nonlineer_cozucu.md`` (pre-registration ``nonlineer_cozucu_onkayit.md``): E2, E3, E4, D2 and
the data of the figure (truss path, load-control negative control, J2 cyclic curve).

    python docs/olcumler/nonlineer_cozucu.py --run        # writes nonlineer_cozucu.json (~10 min)
    python docs/olcumler/nonlineer_cozucu.py --figures    # nonlineer_cozucu_sekil.{png,pdf} from the JSON (datawiz)

Pass/fail checks are the tests (``tests/toporlogy/test_constitutive.py``, ``test_nonlinear_solvers.py``); this script
only measures (E2–E4, D2 have no threshold) and records the environment.
"""
from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "toporlogy"))
OUT = HERE / "nonlineer_cozucu.json"


def _env() -> dict:
    import scipy
    import torch
    git = shutil.which("git") or "git"
    sha = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()  # noqa: S603
    return {"sha": sha, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "platform": platform.platform(), "date": time.strftime("%Y-%m-%d %H:%M %z")}


# ---------------------------------------------------------------------------------------------- E2: jumps
def _cantilever_24x8(P: float):
    from toporlogy.core.domain import Domain
    d = Domain((24, 8), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero((np.abs(X[:, 0] - 24) < 1e-9) & (np.abs(X[:, 1] - 4) < 1.1))
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -P / len(tip)
    return d, fixed, F


def e2_jumps() -> dict:
    from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearConvergenceError, NonlinearElasticity
    d, fixed, F = _cantilever_24x8(0.01)
    nl = NonlinearElasticity(d, fixed, nu=0.3, material="neo_hooke", energy_interpolation=EnergyInterpolation())
    rng = np.random.default_rng(1)
    xa = 0.05 + 0.95 * rng.random(nl.n_elements)
    xb = (rng.random(nl.n_elements) > 0.5).astype(float)
    ts = np.linspace(0.9, 1.0, 21)
    rows = []
    for t in ts:
        x = xa + t * (xb - xa)
        E = 1e-6 + (1 - 1e-6) * x ** 3
        U = nl.solve_numpy(E, F)
        c = nl.cutoffs()[0]
        g = nl.adjoint_gradient(E, U, F[:, None], [c])
        dJdt = float(g @ (3 * (1 - 1e-6) * x ** 2 * (xb - xa)))
        rows.append({"t": float(t), "cutoff": c, "J": float(F @ U[:, 0]), "dJdt": dJdt,
                     "restarts": nl.last_reports[0].restarts, "max_u": float(np.abs(U).max())})
    switches = []
    for k in range(len(rows) - 1):
        a, b = rows[k], rows[k + 1]
        if a["cutoff"] != b["cutoff"]:
            x = xa + a["t"] * (xb - xa)
            E = 1e-6 + (1 - 1e-6) * x ** 3
            fixed_hi = NonlinearElasticity(d, fixed, nu=0.3, material="neo_hooke",
                                           energy_interpolation=EnergyInterpolation(cutoff=b["cutoff"], cutoff_step=0))
            try:
                J_hi: float | None = float(F @ fixed_hi.solve_numpy(E, F)[:, 0])
                err_hi = ""
            except NonlinearConvergenceError as err:     # the raised model does not converge on the earlier design
                J_hi, err_hi = None, str(err)[:300]
            dt = b["t"] - a["t"]
            switches.append({"t_before": a["t"], "t_after": b["t"], "cutoff_before": a["cutoff"],
                             "cutoff_after": b["cutoff"], "J_before": a["J"], "J_after": b["J"],
                             "J_same_design_high_cutoff": J_hi, "high_cutoff_error": err_hi,
                             "model_jump_relative": None if J_hi is None else abs(J_hi - a["J"]) / abs(a["J"]),
                             "jump_over_slope_estimate":
                                 abs(b["J"] - a["J"]) / abs(0.5 * (a["dJdt"] + b["dJdt"]) * dt)})
    smooth = [abs(rows[k + 1]["J"] - rows[k]["J"]) / abs(0.5 * (rows[k]["dJdt"] + rows[k + 1]["dJdt"])
                                                         * (rows[k + 1]["t"] - rows[k]["t"]))
              for k in range(len(rows) - 1) if rows[k]["cutoff"] == rows[k + 1]["cutoff"]]
    return {"rows": rows, "switches": switches, "model_epochs": nl.model_epoch, "model_log": nl.model_log,
            "smooth_interval_ratio_range": [min(smooth), max(smooth)] if smooth else None}


# ---------------------------------------------------------------------------------------------- E3: options
def e3_options(iters: int = 30, P: float = 0.02) -> dict:
    import torch  # noqa: F401
    from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
    from toporlogy.design.interp import SIMP
    from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearConvergenceError, NonlinearElasticity
    from toporlogy.optimization.nonlinear import EndCompliance
    from toporlogy.optimization.responses import Volume
    from toporlogy.optimization.study import Constraint, Study
    from toporlogy.optimizers.mma import MMA

    d, fixed, F = _cantilever_24x8(P)

    def study(policy: str) -> tuple[Study, NonlinearElasticity, MMA]:
        nl = NonlinearElasticity(d, fixed, nu=0.3, material="neo_hooke",
                                 energy_interpolation=EnergyInterpolation(policy=policy))
        opt = MMA(move=0.2)
        chain = Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)])
        st = Study(d, nl, F, chain, SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6), EndCompliance(),
                   [Constraint(Volume(), 0.4)], opt)
        return st, nl, opt

    out = {}
    for variant in ("adaptive", "monotone", "reject"):
        policy = "adaptive" if variant == "adaptive" else "monotone"
        st, nl, opt = study(policy)
        x = 0.4 * np.ones(d.n_design)
        t0 = time.time()
        hist, cut_hist, restarts, rejects, failures = [], [], 0, 0, 0
        ev = st.evaluate(x)
        epoch = nl.model_epoch
        for _ in range(iters):
            xn = opt.step(x, ev.f0, ev.df0, ev.g, ev.dg)
            cur = max(c for c in nl.cutoffs() if c is not None)
            try:
                evn = st.evaluate(xn)
            except NonlinearConvergenceError:
                failures += 1
                break
            new_cut = max(c for c in nl.cutoffs() if c is not None)
            if variant == "reject" and new_cut > cur:
                for _k in range(5):                       # shrink the step toward x, the model of the epoch
                    trial = x + 0.5 * (xn - x)
                    nl._cutoff_floor = cur                # keep the epoch's model while trying (diagnostic driver)
                    nl.interpolation = EnergyInterpolation(policy="monotone", cutoff=cur, cutoff_step=0.0)
                    try:
                        evt = st.evaluate(trial)
                        ok = True
                    except NonlinearConvergenceError:
                        ok = False
                    nl.interpolation = EnergyInterpolation(policy="monotone")
                    rejects += 1
                    xn = trial
                    if ok:
                        evn = evt
                        new_cut = cur
                        break
                else:
                    evn = st.evaluate(xn)                 # accept with the raised model
                    new_cut = max(c for c in nl.cutoffs() if c is not None)
            if variant != "adaptive" and nl.model_epoch != epoch:
                opt.reset()                               # history of a different problem: drop it
                restarts += 1
            epoch = nl.model_epoch
            x, ev = xn, evn
            hist.append(float(ev.f0))
            cut_hist.append(new_cut)
        out[variant] = {"x": x.tolist(), "history": hist, "cutoffs": cut_hist, "restarts": restarts,
                        "rejected_steps": rejects, "failures": failures, "model_epochs": nl.model_epoch,
                        "seconds": time.time() - t0,
                        "monotonicity_violations": int(sum(1 for a, b in zip(hist, hist[1:], strict=False) if b > a))}
    # common re-evaluation at the highest cutoff any run reached (fixed model)
    c_common = max(max(v["cutoffs"]) for v in out.values() if v["cutoffs"])
    for _variant, v in out.items():
        nl = NonlinearElasticity(d, fixed, nu=0.3, material="neo_hooke",
                                 energy_interpolation=EnergyInterpolation(cutoff=c_common, cutoff_step=0.0))
        chain = Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)])
        st = Study(d, nl, F, chain, SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6), EndCompliance(),
                   [Constraint(Volume(), 0.4)], MMA())
        try:
            e = st.evaluate(np.asarray(v["x"]), gradients=False)
            v["J_common_cutoff"] = float(e.f0)
            v["volume"] = float(e.values.get("volume", np.nan))
        except NonlinearConvergenceError as err:
            v["J_common_cutoff"] = None
            v["common_error"] = str(err)[:200]
        del v["x"]
    return {"load": P, "iterations": iters, "common_cutoff": c_common, "variants": out}


# ---------------------------------------------------------------------------------------------- E4: density
def e4_density() -> dict:
    rows = []
    rho = np.linspace(0.0, 1.0, 1001)
    for emin in (1e-9, 1e-6):
        for p_simp in (1.0, 2.0, 3.0):
            E = emin + rho ** p_simp * (1 - emin)
            rt = E ** (1 / 3.0)                           # the reconstruction of EnergyInterpolation (p = 3)
            err = np.abs(rt - rho)
            rows.append({"emin": emin, "simp_penal": p_simp, "interp_exponent": 3.0,
                         "max_abs_error": float(err.max()), "rho_at_max": float(rho[int(err.argmax())]),
                         "error_at_cutoff_0.1": float(np.interp(0.1, rho, err)),
                         "rho_tilde_at_rho0": float(rt[0])})
    return {"rows": rows}


# ---------------------------------------------------------------------------------------------- D2: V4 measure
def d2_v4() -> dict:
    from test_nonlinear import _beam, _orders
    out = {}
    for material in ("svk", "neo_hooke"):
        _u, _v, rep = _beam(4, material, 2.0, slender=50, load_steps=10)
        steps = []
        for s in rep.steps:
            r = s.residuals
            best = None
            for k in range(1, len(r) - 1):
                if r[k] < r[k - 1] and r[k] > 1e-11 and r[k + 1] > 1e-11 and r[k - 1] > 1e-11:
                    p = math.log(r[k + 1] / r[k]) / math.log(r[k] / r[k - 1])
                    if best is None or p > best[0]:
                        best = (p, abs(math.log(r[k] / r[k - 1])))
            steps.append({"load_factor": s.load_factor, "residuals": r, "backward_errors": s.backward_errors,
                          "orders": _orders(r), "max_order": None if best is None else best[0],
                          "denominator_of_max": None if best is None else best[1]})
        out[material] = steps
    return out


# ---------------------------------------------------------------------------------------------- figure data
def fig_data() -> dict:
    from _truss import TrussAtLoad, TwoBarTruss

    from toporlogy.core.constitutive import J2Plasticity, drive_material_point
    from toporlogy.solvers.continuation import ArcLengthSettings, arc_length, load_control
    from toporlogy.solvers.newton import NewtonSettings, newton
    tr = TwoBarTruss()
    path = arc_length(tr, np.zeros(2), 0.0, ArcLengthSettings(ds=0.005, rtol=1e-12, eta_tol=0.0, max_steps=2000),
                      lambda x, lam: -x[1] >= 2.5 * tr.h)

    def attempt(lam, x):
        o = newton(TrussAtLoad(tr, lam), x, float(np.linalg.norm(lam * tr.load)), NewtonSettings(rtol=1e-12))
        return o.converged, (o.x if o.converged else x), o.record, o.relative_residual, o.backward_error

    lc = load_control(attempt, np.zeros(2), 0.0, 1.2, 1 / 20, 1e-4, keep_points=True)
    m = J2Plasticity(0.3, 1e-3, 0.05)
    legs = [np.linspace(0, 5e-3, 41)[1:], np.linspace(5e-3, -5e-3, 41)[1:], np.linspace(-5e-3, 8e-3, 41)[1:]]
    eps = np.concatenate(legs)
    strain = np.zeros((len(eps), 6))
    strain[:, 0] = eps
    h = drive_material_point(m, strain, np.zeros((len(eps), 6)), [True] + [False] * 5)
    from test_constitutive import _uniaxial_reference
    s_ref, _a_ref = _uniaxial_reference(eps, 0.05)
    return {"truss": {"h": tr.h, "w1": tr.w1, "w2": tr.w2, "w": (-path.component(1)).tolist(),
                      "lam": path.lam.tolist(), "rejected": path.rejected,
                      "det_sign": [p.det_sign for p in path.points],
                      "load_control": {"ok": lc.ok, "lam_reached": lc.lam, "cuts": lc.cuts,
                                       "last_code": lc.records[-1].code,
                                       "w": [0.0] + [float(-x[1]) for _l, x in lc.points],
                                       "lam": [0.0] + [float(l_) for l_, _x in lc.points]}},
            "j2_cyclic": {"eps": eps.tolist(), "sigma": h.stress[:, 0].tolist(), "sigma_closed_form": s_ref.tolist(),
                          "alpha": [float(s["alpha"][0]) for s in h.states],
                          "dissipation_total": float(h.dissipation.sum())}}


def run() -> None:
    t0 = time.time()
    res = {"environment": _env(), "figure": fig_data()}
    print("figure data", time.time() - t0)
    res["E4"] = e4_density()
    res["D2"] = d2_v4()
    print("D2", time.time() - t0)
    res["E2"] = e2_jumps()
    print("E2", time.time() - t0)
    res["E3"] = e3_options()
    print("E3", time.time() - t0)
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


def figures() -> None:
    """FİGÜR BRİFİ — nonlineer_cozucu_sekil
    Mesaj : yay boyu, yük kontrolünün limit noktasında kaybettiği kararsız dalı analitik eğri üzerinde izliyor; J2
            malzeme noktası çevrimsel yolu kapalı formla çakışıyor.
    Okur  : uzman · ortam web (depo belgesi), ondalık virgül · 2 panel (a) kafes λ–w/h (b) J2 σ/σ_y–ε/ε_y.
    Biçim : çizgi (analitik, gri bağlam) + işaret (sayısal); yük kontrolü ayrı işaret; doğrudan etiket.
    """
    import datawiz as dw

    d = json.loads(OUT.read_text(encoding="utf-8"))["figure"]
    tr = d["truss"]
    h = tr["h"]
    fig, axs = dw.figur("makale", hedef="web-640", ncols=2, oran=0.42, ondalik=",")
    ax = axs[0]
    w = np.linspace(0, 2.5 * h, 400)
    L3 = (1 + h * h) ** 1.5
    Pm = 2 * h ** 3 / (3 * math.sqrt(3) * L3)
    ax.plot(w / h, w * (2 * h - w) * (h - w) / L3 / Pm, color="0.6", lw=2.2, label="analitik", zorder=1)
    ws, ls = np.array(tr["w"]) / h, np.array(tr["lam"])
    ax.plot(ws[::3], ls[::3], "o", ms=3.2, color="#0072B2", label="yay boyu", zorder=3)
    lc = tr["load_control"]
    ax.plot(np.array(lc["w"]) / h, lc["lam"], "s", ms=3.6, mfc="none", color="#D55E00", label="yük kontrolü",
            zorder=4)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("Tepe çökmesi w/h (–)")
    ax.set_ylabel("Yük çarpanı λ = P/P_max (–)")
    dw.lejant.yerlestir(ax)
    ax2 = axs[1]
    j = d["j2_cyclic"]
    e = np.array(j["eps"]) / 1e-3
    s = np.array(j["sigma"]) / 1e-3
    ax2.plot(e, np.array(j["sigma_closed_form"]) / 1e-3, color="0.6", lw=2.2, label="kapalı form", zorder=1)
    ax2.plot(e[::2], s[::2], "o", ms=3.0, color="#009E73", label="radial return", zorder=3)
    ax2.set_xlabel("Şekil değiştirme ε/ε_y (–)")
    ax2.set_ylabel("Gerilme σ/σ_y (–)")
    dw.lejant.yerlestir(ax2)
    for a in axs:
        dw.izgara.uygula(a, "silik", "y")
    dw.kaydet(fig, str(HERE / "nonlineer_cozucu_sekil"), hedef="web-640", cvd=True, bicimler=("png", "pdf"))


def run_e3b(P: float) -> None:
    """Post hoc E3b: the comparison of E3 at a load where the model does change (E3 had no change at P = 0.02)."""
    res = json.loads(OUT.read_text(encoding="utf-8"))
    res["E3b_posthoc"] = e3_options(30, P)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"E3b at P = {P}: written")


if __name__ == "__main__":
    if "--e3b" in sys.argv:
        run_e3b(float(sys.argv[sys.argv.index("--e3b") + 1]))
    if "--run" in sys.argv:
        run()
    if "--figures" in sys.argv:
        figures()
