"""Measurements and data export of docs/olcumler/tensor_elastisite.md (records P6b, P8, P10b and the reports that
go with P7, P9c, P11, L7c, K1d); the pass/fail checks are the tests (tests/toporlogy/test_tensor.py,
test_linear_elastic.py, test_anisotropy.py).

    python docs/olcumler/tensor_elastisite.py [--out runs/tensor_elastisite] [--no-figure]

Writes ``tensor_elastisite.json`` (raw results, environment, git SHA), ``tensor_elastisite_yonler.csv`` (directional
E, ν extrema over m, G extrema, β for six reference materials on a Fibonacci sphere), ``<out>/yonler.npz`` (same data;
*.npz is not tracked) and the figure ``tensor_elastisite_sekil.png`` with its datawiz audit
(``tensor_elastisite_sekil.denetim.txt``). All materials are synthetic, normalised values (no catalogue data).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from toporlogy.core import anisotropy as A  # noqa: E402
from toporlogy.core import linear_elastic as L  # noqa: E402
from toporlogy.core import material as M  # noqa: E402
from toporlogy.core import tensor as T  # noqa: E402

SWEEP = [-0.99, -0.9, -0.5, 0.0, 0.2, 0.3, 0.45, 0.49, 0.499, 0.4999]
ORTHO = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)


def _env() -> dict[str, Any]:
    import scipy
    import torch
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,  # noqa: S607
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,  # noqa: S607
                           text=True).stdout.strip()
    return {"git": sha, "dirty": bool(dirty), "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "machine": platform.platform()}


def _materials() -> dict[str, np.ndarray]:
    """Voigt compliances of the reference materials (synthetic)."""
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    Mm = (Q * np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])) @ Q.T
    out = {"isotropic_nu0.3": L.isotropic_compliance(1.0, 0.3),
           "orthotropic": L.orthotropic_compliance(**ORTHO),
           "cubic": np.linalg.inv(L.cubic_stiffness(1.7, 1.2, 0.75)),
           "transversely_isotropic": L.transversely_isotropic_compliance(1.0, 5.0, 0.3, 0.2, 0.4),
           "spd_family_k10": L.orthotropic_compliance(100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 0.0),
           "random_spd": np.linalg.inv(T.mandel_to_stiffness(0.5 * (Mm + Mm.T)))}
    return {k: 0.5 * (v + v.T) for k, v in out.items()}


def directional(n_dirs: int = 600) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, extrema = [], {}
    N = L.fibonacci_sphere(n_dirs)
    for name, S in _materials().items():
        E = A.young_modulus(S, N)
        nu = A.poisson_extrema(S, N)
        G = A.shear_extrema(S, N)
        b = A.linear_compressibility(S, N)
        for k in range(n_dirs):
            rows.append({"material": name, "nx": N[k, 0], "ny": N[k, 1], "nz": N[k, 2], "E": E[k],
                         "nu_min": nu["nu_min"][k], "nu_max": nu["nu_max"][k], "G_min": G["G_min"][k],
                         "G_max": G["G_max"][k], "beta": b[k]})
        ex = {}
        for q in A.QUANTITIES:
            r = A.directional_extrema(S, q, 2000)
            ex[q] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        ex["volumetric_compressibility"] = A.volumetric_compressibility(S)
        ex["strong_ellipticity"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in L.strong_ellipticity(np.linalg.inv(S), 2000).items()}
        extrema[name] = ex
    return rows, extrema


def cubic_exploration() -> list[dict[str, Any]]:
    """P8 (measurement only): cubic C11 = 1, approaching each positive-definiteness boundary by t."""
    out = []
    for boundary in ("C11-C12", "C11+2C12", "C44"):
        for t in (1e-1, 1e-2, 1e-3, 1e-4):
            C11, C12, C44 = 1.0, 0.5, 0.5
            if boundary == "C11-C12":
                C12 = C11 - t
            elif boundary == "C11+2C12":
                C12 = -(C11 - t) / 2
            else:
                C44 = t
            C = L.cubic_stiffness(C11, C12, C44)
            S = np.linalg.inv(C)
            S = 0.5 * (S + S.T)
            r = A.directional_extrema(S, "poisson", 2000)
            e = A.directional_extrema(S, "young", 2000)
            out.append({"boundary": boundary, "t": t, "C11": C11, "C12": C12, "C44": C44,
                        "nu_min": r["min"], "nu_max": r["max"], "n_nu_min": r["n_min"].tolist(),
                        "m_nu_min": r["m_min"].tolist(), "n_nu_max": r["n_max"].tolist(),
                        "m_nu_max": r["m_max"].tolist(), "E_min": e["min"], "E_max": e["max"],
                        "E_ratio": e["max"] / e["min"], "label": r["label"],
                        "mandel_condition": float(np.linalg.cond(T.stiffness_to_mandel(C)))})
    return out


def spd_family() -> list[dict[str, Any]]:
    out = []
    for k in (1.0, 10.0, 100.0, 1000.0):
        for a in (0.5, -0.5):
            S = L.orthotropic_compliance(k * k, 1.0, 1.0, 1.0, 1.0, 1.0, a * k, a * k, 0.0)
            d = L.check_positive_definite(S, "compliance", "S")
            out.append({"k": k, "nu12": a * k, "nu21": a / k, "cauchy_schwarz_bound": k, **d})
    return out


def parametrisation_comparison(n: int = 2000) -> list[dict[str, Any]]:
    """P10b (measurement only): fraction of random steps that leave the SPD set, from a near-boundary orthotropic
    material (ν12 = 0.6 k with ν12² < E1/2 needing |a| < 0.707; k = 10), in the engineering constants (log of E and
    G, additive ν) and in the Cholesky chart, for step sizes s (Gaussian, per parameter)."""
    rng = np.random.default_rng(81)
    base = dict(E1=100.0, E2=1.0, E3=1.0, G12=1.0, G13=1.0, G23=1.0, nu12=6.0, nu13=6.0, nu23=0.0)
    th0 = A.cholesky_parameters(np.linalg.inv(L.orthotropic_compliance(**base)))
    out = []
    for s in (0.01, 0.03, 0.1, 0.3, 1.0):
        bad_e = bad_c = 0
        for _ in range(n):
            z = rng.standard_normal(9)
            p = {k: (v * math.exp(s * zz) if k[0] in "EG" else v + s * zz)
                 for (k, v), zz in zip(base.items(), z, strict=True)}
            try:
                L.check_positive_definite(L.orthotropic_compliance(**p), "compliance")
            except ValueError:
                bad_e += 1
            C = A.stiffness_from_cholesky(th0 + s * rng.standard_normal(21))
            try:
                L.check_positive_definite(C)
            except ValueError:
                bad_c += 1
        out.append({"step": s, "n": n, "leave_spd_engineering": bad_e / n, "leave_spd_cholesky": bad_c / n})
    return out


def hs_audit() -> dict[str, Any]:
    import torch

    from toporlogy.core.domain import Domain
    from toporlogy.design.interp import HashinShtrikman, hs_admissible_exponents
    nus = np.linspace(-0.9, 0.49, 140)
    table = {f"{d}d_{p}": [list(hs_admissible_exponents(float(v), d, p)) for v in nus]
             for d, p in ((2, "stress"), (2, "strain"), (3, "stress"))}
    x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    nu_dev = {}
    for dim, shape in ((2, (2, 2)), (3, (2, 2, 2))):
        for v in (-0.3, 0.0, 0.3, 0.45):
            hs = HashinShtrikman(Domain(shape), nu=v)
            nu_dev[f"{dim}d_nu{v}"] = (hs.poisson(x) - v).tolist()
    return {"nu": nus.tolist(), "exponents_bulk_shear": table, "x": x.tolist(),
            "hs_poisson_minus_solid": nu_dev,
            "note": "HashinShtrikman keeps the solid nu in the element matrix; nu_HS(x) - nu is the error of that "
                    "constant-nu approximation (2-D: plane stress)"}


def reports() -> dict[str, Any]:
    out: dict[str, Any] = {}
    C = M.Orthotropic(**ORTHO).stiffness()
    P = [0, 1, 5]

    def schur(c: np.ndarray, o: list[int]) -> np.ndarray:
        return c[np.ix_(P, P)] - c[np.ix_(P, o)] @ np.linalg.solve(c[np.ix_(o, o)], c[np.ix_(o, P)])
    Cz = M.rotate_stiffness(C, M.rotation_about_z(35.0))
    Cg = M.rotate_stiffness(C, M.rotation_from_axes([1.0, 0.4, 0.7], [0.0, 1.0, -0.3]))
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    out["L7c_szz_only_vs_full"] = {"about_z_35deg": rel(schur(Cz, [2]), M.plane_stress(Cz)),
                                   "general_rotation": rel(schur(Cg, [2]), M.plane_stress(Cg))}
    rng = np.random.default_rng(24)
    errs = []
    for _ in range(10):
        H = 1e-10 * rng.standard_normal((3, 3))
        F = np.eye(3) + H
        E = T.green_lagrange(H[None])[0]
        errs.append(float(np.abs(0.5 * (F.T @ F - np.eye(3)) - E).max() / np.abs(E).max()))
    out["K1d_F_form_relative_error_max"] = max(errs)
    grad = []
    for nu in SWEEP:
        _, dn = A.isotropic_stiffness_derivatives(1.0, nu)
        grad.append({"nu": nu, "dC_dnu_max": float(np.abs(dn).max()),
                     "mandel_condition": float(np.linalg.cond(T.stiffness_to_mandel(L.isotropic_stiffness(1.0, nu))))})
    out["P9c_gradient_vs_conditioning"] = grad
    return out


def figure(res: dict[str, Any], path: Path) -> dict[str, Any]:
    import datawiz as dw

    fig, axs = dw.figur("makale", hedef="genel-cift", oran=0.75, ncols=2, nrows=2)
    a, b, c, d = axs.ravel()
    import matplotlib.pyplot as plt
    col = plt.rcParams["axes.prop_cycle"].by_key()["color"]    # one colour = one entity across panels
    # (a) conditioning of the isotropic sweep (E fixed)
    rows = res["P6_isotropic_sweep"]["E"]
    nu = np.array([r["nu"] for r in rows])
    a.semilogy(nu, [r["cond_C"] for r in rows], marker="o", color="0.25")  # datawiz: izin log, 4 mertebe
    a.set_xlabel("Poisson oranı ν")
    a.set_ylabel("Mandel κ(C), log ölçek")
    a.set_xlim(-1.0, 0.5)
    # (b) orthotropic in-plane Poisson extrema
    S = L.orthotropic_compliance(**ORTHO)
    th = np.linspace(0.0, math.pi, 181)
    n = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    ex = A.poisson_extrema(S, n)
    deg = np.degrees(th)
    b.plot(deg, ex["nu_max"], color=col[0], label="maks")
    b.plot(deg, ex["nu_min"], color=col[1], linestyle="--", label="min")
    b.set_xlabel("Yük yönü θ (°)")
    b.set_ylabel("Poisson oranı")
    b.set_xticks([0, 45, 90, 135, 180])
    dw.lejant.yerlestir(b)
    # (c) unbounded nu in an SPD family
    fam = [r for r in res["P7_spd_family"] if r["nu12"] > 0]
    k = np.array([r["k"] for r in fam])
    c.loglog(k, [r["nu12"] for r in fam], marker="o", color=col[2], label="ν12")  # datawiz: izin log, 4 mertebe
    c.loglog(k, k, linestyle="--", color="0.45", label="sınır k")  # datawiz: izin log, 4 mertebe
    c.axhline(0.5, color="0.55", linewidth=0.8, linestyle=":")
    c.set_xlabel("k = √(E1/E2), log ölçek")
    c.set_ylabel("Poisson oranı, log ölçek")
    dw.lejant.yerlestir(c)
    # (d) SIMP / HS admissible exponent
    hs = res["P11_hs"]
    labels = {"2d_stress": "2B gerilme", "2d_strain": "2B şekil d.", "3d_stress": "3B"}
    styles = {"2d_stress": "-", "2d_strain": "--", "3d_stress": "-."}
    for key, lab in labels.items():
        p = [max(v) for v in hs["exponents_bulk_shear"][key]]
        d.plot(hs["nu"], p, linestyle=styles[key], color=col[3 + list(labels).index(key)], label=lab)
    d.axhline(3.0, color="0.55", linewidth=0.8, linestyle=":")
    d.set_xlabel("Poisson oranı ν")
    d.set_ylabel("SIMP üssü p*")
    d.set_ylim(0, 8)
    d.set_xlim(-1.0, 0.5)
    dw.lejant.yerlestir(d)
    dw.etiket.panel_harfleri(axs.ravel(), bicim="({})")
    rep = dw.kaydet(fig, str(path), bicimler=("png",), hedef="genel-cift", cvd=True)
    return {"report": str(rep.get("denetim", "")) if isinstance(rep, dict) else str(rep)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs" / "tensor_elastisite"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from tensor_elastisite.json")
    args = ap.parse_args()
    if args.figure_only:
        res0 = json.loads((HERE / "tensor_elastisite.json").read_text(encoding="utf-8"))
        res0["figure"] = figure(res0, HERE / "tensor_elastisite_sekil")
        (HERE / "tensor_elastisite.json").write_text(json.dumps(res0, indent=1, default=float) + "\n",
                                                     encoding="utf-8")
        return 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"environment": _env()}
    res["P6_isotropic_sweep"] = {f: A.isotropic_sweep(SWEEP, f) for f in ("E", "G")}
    rows, res["directional_extrema"] = directional()
    res["P7_spd_family"] = spd_family()
    res["P8_cubic_exploration"] = cubic_exploration()
    res["P10b_parametrisation"] = parametrisation_comparison()
    res["P11_hs"] = hs_audit()
    res.update(reports())
    with open(HERE / "tensor_elastisite_yonler.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez(out_dir / "yonler.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    if not args.no_figure:
        res["figure"] = figure(res, HERE / "tensor_elastisite_sekil")
    (HERE / "tensor_elastisite.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'tensor_elastisite.json'}, tensor_elastisite_yonler.csv ({len(rows)} rows), {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
