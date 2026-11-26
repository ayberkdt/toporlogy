"""Measurements of the general-mesh FEM record (docs/olcumler/fem_genel_onkayit.md §3-§4) -> fem_genel.json.

    python docs/olcumler/fem_genel.py                 # full run (about 15 min on a 12-thread laptop)
    python docs/olcumler/fem_genel.py --quick         # small subset, for a smoke test

Sections: environment, nu sweep (E fixed and G fixed, plane strain 2-D, 3-D, plane stress), MMS convergence on
distorted meshes (V3 tables), Taylor-Hood convergence near / at incompressibility (L1e), inf-sup constants (L2),
cantilever slenderness sweep (L3, families F1 square elements and F2 fixed n_x = 20 n_y), conditioning vs locking
(L4) and solver/resource figures (ndof, nnz, SuperLU fill, times, process peak memory).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from toporlogy.core.linear_elastic import isotropic_plane_strain  # noqa: E402
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, MixedElasticity, perturb, rectangle  # noqa: E402
from toporlogy.fem.mesh.benchmarks import cantilever_case, mms_case, moduli  # noqa: E402
from toporlogy.fem.mesh.linear import Factorization, condition_1norm, jacobi_scaled, jacobi_pcg  # noqa: E402
from toporlogy.fem.mesh.manufactured import Manufactured  # noqa: E402

NUS = (-0.9, -0.5, 0.0, 0.3, 0.45, 0.49, 0.499, 0.4999)
OUT = HERE / "fem_genel.json"


def _peak_mb() -> float | None:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", getattr(mi, "rss", 0))) / 2 ** 20
    except Exception:                                   # noqa: BLE001 - optional figure
        return None


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else ("inf" if math.isinf(v) else v)
    if isinstance(x, np.integer):
        return int(x)
    return x


def environment() -> dict:
    import scipy
    try:
        import skfem
        skv = skfem.__version__
    except ImportError:
        skv = None
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S603,S607
                         cwd=REPO).stdout.strip()
    from toporlogy.solvers import pardiso
    return {"sha": sha, "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "scikit-fem": skv, "platform": platform.platform(),
            "processor": platform.processor(), "pardiso": pardiso.state()}


def nu_sweep(quick: bool) -> list[dict]:
    rows = []
    two = ["Q4", "Q9", "T3", "T6", "Q4IM", "TH:T6-T3", "TH:Q9-Q4", "Q4-P0"]
    three = ["H8", "T4", "T10", "H8IM", "TH:T10-T4"]
    nus = (0.3, 0.4999) if quick else NUS
    for fixed in ("E", "G"):
        for meth in two + ([] if quick else three):
            n = 16 if meth in two else 6
            for nu in nus + ((0.5,) if meth.startswith(("TH", "Q4-P0")) else ()):
                try:
                    rows.append(mms_case(meth, n, nu, fixed=fixed))
                except ValueError as e:                 # mixed u-p at lambda = 0 is refused
                    rows.append({"method": meth, "n": n, "nu": nu, "fixed": fixed, "refused": str(e)})
        for meth in ("Q4", "T3"):
            for nu in nus:
                rows.append(mms_case(meth, 16, nu, fixed=fixed, plane="stress"))
    return rows


def mms_convergence(quick: bool) -> list[dict]:
    levels = {"Q4": (4, 8, 16, 32), "T3": (4, 8, 16, 32), "Q9": (4, 8, 16, 32), "T6": (4, 8, 16, 32),
              "H8": (4, 8, 16), "T4": (4, 8, 16, 24), "T10": (2, 4, 8, 12)}
    if quick:
        levels = {"Q4": (4, 8), "T6": (4, 8)}
    return [_mms_smooth(name, n) for name, ns in levels.items() for n in ns]


def _mms_smooth(name: str, n: int) -> dict:
    """V3 problem: c = 1, nu = 0.3, plane strain / 3-D, vertices ±0.2 h, seed 100 + n (as the test)."""
    from toporlogy.core.linear_elastic import isotropic_stiffness
    from toporlogy.fem.mesh import box, get
    dim = get(name).dim
    base = rectangle(n, n, element=name) if dim == 2 else box(n, n, n, element=name)
    m = perturb(base, 0.2 / n, seed=100 + n)
    E, mu, lam = moduli(0.3)
    ms = Manufactured(dim, 1.0, lam)
    t0 = time.perf_counter()
    P = ElasticProblem(m, isotropic_plane_strain(E, 0.3) if dim == 2 else isotropic_stiffness(E, 0.3))
    _ = P.K
    ta = time.perf_counter() - t0
    k = get(name).degree
    f = P.body_force(lambda X: ms.body_force(mu, X), degree=2 * k + 4)
    t0 = time.perf_counter()
    s = P.solve(f, Dirichlet.fixed(m.dofs(m.boundary_nodes())))
    ts = time.perf_counter() - t0
    err = P.errors(s.u[:, 0], ms.u, ms.grad)
    fac = P.factorization()
    return {"method": name, "n": n, "h": 1.0 / n, **err, "ndof": P.ndof, "nnz": int(P.K.nnz),
            "fill_nnz": fac.fill_nnz if fac else None, "solver": fac.method if fac else None, "t_assembly": ta,
            "t_solve": ts, "peak_mb": _peak_mb()}


def th_convergence(quick: bool) -> list[dict]:
    rows = []
    ns = (8, 16) if quick else (8, 16, 32)
    for meth in ("TH:T6-T3", "TH:Q9-Q4", "T6", "Q9", "Q4", "T3"):
        for nu in (0.4999, 0.5):
            if nu == 0.5 and not meth.startswith("TH"):
                continue
            for n in ns:
                rows.append(mms_case(meth, n, nu, variant="B", distort=0.2, seed=200 + n))
    return rows


def inf_sup(quick: bool) -> list[dict]:
    rows = []
    for el, ps in (("T6", "vertex"), ("Q9", "vertex"), ("Q4", "P0")):
        for n in ((4, 8) if quick else (4, 8, 16)):
            m = rectangle(n, n, element=el)
            mx = MixedElasticity(m, 1.0, math.inf, pressure=ps)
            w = mx.inf_sup(Dirichlet.fixed(m.dofs(m.boundary_nodes())))
            z = int((w <= 1e-10 * w.max()).sum())
            rows.append({"pair": f"{el}-{'P1' if ps == 'vertex' else 'P0'}", "n": n, "zero_modes": z,
                         "beta": math.sqrt(w[z]) if z < len(w) else 0.0, "beta2_smallest": [float(x) for x in w[:4]]})
    return rows


def cantilever(quick: bool) -> list[dict]:
    rows = []
    sl = (5, 100) if quick else (5, 10, 50, 100)
    # F1: square elements
    for meth in ("Q4", "Q4IM", "Q9", "T3", "T6"):
        for ny in ((2,) if quick else (1, 2, 4)):
            for s in sl:
                rows.append({**cantilever_case(meth, s, ny), "family": "F1"})
    for meth in ("H8", "H8IM", "T10") if not quick else ("H8IM",):
        for ny in ((1,) if quick else (1, 2)):
            for s in sl:
                rows.append({**cantilever_case(meth, s, ny, nu=0.0, dim=3), "family": "F1"})
    # F2: fixed n_x = 20 n_y, element aspect L/h / 20
    for meth in ("Q4", "Q9", "T3", "T6"):
        for ny in ((2,) if quick else (1, 2, 4)):
            for s in sl:
                rows.append({**cantilever_case(meth, s, ny, nx=20 * ny), "family": "F2"})
    for meth in ("H8", "T10") if not quick else ("H8",):
        for ny in ((1,) if quick else (1, 2)):
            for s in sl:
                rows.append({**cantilever_case(meth, s, ny, nx=20 * ny, nu=0.0, dim=3), "family": "F2"})
    return rows


def conditioning(quick: bool) -> list[dict]:
    rows = []
    nus = (0.3, 0.4999) if quick else (0.3, 0.49, 0.499, 0.4999)
    n = 16
    for nu in nus:
        E, mu, lam = moduli(nu)
        ms = Manufactured(2)
        m = rectangle(n, n, element="Q4")
        P = ElasticProblem(m, isotropic_plane_strain(E, nu))
        f = P.body_force(lambda X, ms=ms, mu=mu: ms.body_force(mu, X), degree=6)
        c = m.dofs(m.boundary_nodes())
        free = np.setdiff1d(P.active, c)
        s = P.solve(f, Dirichlet.fixed(c))
        K = P.K[free][:, free]
        fac = P.factorization()
        Ks, sc = jacobi_scaled(K)
        fs = Factorization(Ks, "spd")
        us = sc * fs.solve(sc * f[free])
        r = f[free] - K @ s.u[free, 0]
        _, it, conv = jacobi_pcg(K, f[free])
        rows.append({"method": "Q4", "nu": nu, "kappa1": condition_1norm(K, fac),
                     "kappa1_jacobi": condition_1norm(Ks, fs),
                     "scaled_solve_change": float(np.abs(us - s.u[free, 0]).max() / np.abs(s.u[free, 0]).max()),
                     "refinement_change": float(np.abs(fac.solve(r)).max() / np.abs(s.u[free, 0]).max()),
                     "pcg_iterations": it, "pcg_converged": conv,
                     "l2": P.errors(s.u[:, 0], ms.u, ms.grad)["l2"]})
        mt = rectangle(n, n, element="T6")
        mx = MixedElasticity(mt, mu, lam)
        bc = Dirichlet.fixed(mt.dofs(mt.boundary_nodes()))
        Km, fm, gauge = mx.system(bc)
        f2 = mx.disp.body_force(lambda X, ms=ms, mu=mu: ms.body_force(mu, X), degree=8)
        so = mx.solve(f2, bc)
        try:
            jacobi_pcg(Km, np.ones(Km.shape[0]), structure="symmetric_indefinite")
            refused = False
        except ValueError:
            refused = True
        rows.append({"method": "TH:T6-T3", "nu": nu, "kappa1": condition_1norm(Km, fm), "kappa1_jacobi": None,
                     "pcg_refused": refused, "gauge": gauge,
                     "l2": ElasticProblem.errors(mx.disp, so.u[:, 0], ms.u, ms.grad)["l2"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.perf_counter()
    data: dict = {"environment": environment(), "quick": a.quick}
    for name, fn in (("nu_sweep", nu_sweep), ("mms_convergence", mms_convergence), ("th_convergence", th_convergence),
                     ("inf_sup", inf_sup), ("cantilever", cantilever), ("conditioning", conditioning)):
        t = time.perf_counter()
        data[name] = fn(a.quick)
        print(f"{name}: {len(data[name])} rows, {time.perf_counter() - t:.0f} s", flush=True)
    data["wall_time_s"] = time.perf_counter() - t0
    data["peak_mb"] = _peak_mb()
    Path(a.out).write_text(json.dumps(_clean(data), indent=1), encoding="utf8")
    print(f"wrote {a.out} ({data['wall_time_s']:.0f} s)")


if __name__ == "__main__":
    main()
