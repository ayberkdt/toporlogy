"""Fingerprint of :class:`toporlogy.fem.nonlinear.NonlinearElasticity` before and after the constitutive-protocol /
general-Newton refactor (check P0 of ``nonlineer_cozucu_onkayit.md``).

    python docs/olcumler/nonlineer_cozucu_parmakizi.py --write     # once, on the base commit 10daa03 (before)
    python docs/olcumler/nonlineer_cozucu_parmakizi.py             # after: compares, exit 1 unless bit-for-bit equal

Fixed inputs (seeded RNG), SuperLU (``DirectSolver("splu")``, deterministic). Every array of every configuration is
stored as its SHA-256 and a few summary numbers; the comparison is on the bytes. The entries: converged U, the
relative-residual and backward-error history of every Newton iteration, cuts / restarts / cutoff, the element
force and tangent at a deformed state, the strain energy and the adjoint gradient.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
OUT = HERE / "nonlineer_cozucu_oncesi.json"


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def _cases():
    from toporlogy.core.domain import Domain
    from toporlogy.core.material import Orthotropic, unit_elasticity_field
    from toporlogy.fem.nonlinear import EnergyInterpolation

    for mat in ("svk", "neo_hooke"):
        for dim, plane in ((2, "stress"), (2, "strain"), (3, "stress")):
            for interp in (False, True):
                yield f"{mat}-{dim}D-{plane}-{'ei' if interp else 'full'}", mat, dim, plane, interp, None
    # per-element D field (SVK only): one rotation per element about z
    d = Domain((6, 3), h=0.5, thickness=0.8)
    th = np.random.default_rng(7).uniform(0, np.pi, d.n_active)
    c, s = np.cos(th), np.sin(th)
    R = np.zeros((d.n_active, 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    ortho = Orthotropic(1.0, 0.4, 0.4, 0.3, 0.3, 0.2, 0.25, 0.25, 0.3)
    yield "svk-2D-field", "svk", 2, "stress", False, unit_elasticity_field(ortho, 2, "stress", R)
    del EnergyInterpolation


def fingerprint() -> dict[str, dict[str, object]]:
    from toporlogy.core.domain import Domain
    from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearConvergenceError, NonlinearElasticity
    from toporlogy.solvers.direct import DirectSolver

    out: dict[str, dict[str, object]] = {}
    for name, mat, dim, plane, interp, field in _cases():
        d = Domain((6, 3) if dim == 2 else (4, 2, 2), h=0.5, thickness=0.8)
        X = d.node_coords()
        fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
        tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
        F = np.zeros((d.ndof, 2))
        F[dim * tip + 1, 0] = -0.04 / len(tip)
        F[dim * tip, 1] = 0.05 / len(tip)
        kw = dict(nu=0.3, material=mat, plane=plane, solver=DirectSolver("splu"), load_steps=4)
        if field is not None:
            kw["elasticity"] = field
        if interp:
            kw["energy_interpolation"] = EnergyInterpolation(cutoff=0.3)
        nl = NonlinearElasticity(d, fixed, **kw)
        rng = np.random.default_rng(11)
        E = 0.05 + 0.95 * rng.random(nl.n_elements) if interp else 0.5 + 0.5 * rng.random(nl.n_elements)
        try:
            U = nl.solve_numpy(E, F)
        except NonlinearConvergenceError as e:       # the failure itself is part of the fingerprint
            out[name] = {"error": str(e)}
            continue
        rec: dict[str, object] = {"U": _sha(U), "U_max": float(np.abs(U).max())}
        for c, rep in enumerate(nl.last_reports):
            res = np.concatenate([np.asarray(s.residuals, float) for s in rep.steps])
            eta = np.concatenate([np.asarray(s.backward_errors, float) for s in rep.steps])
            rec[f"case{c}_residuals"] = _sha(res)
            rec[f"case{c}_eta"] = _sha(eta)
            rec[f"case{c}_n_iter"] = int(rep.iterations)
            rec[f"case{c}_steps"] = [float(s.load_factor) for s in rep.steps]
            rec[f"case{c}_converged"] = [bool(s.converged) for s in rep.steps]
            rec[f"case{c}_meta"] = [rep.cutoff, rep.cuts, rep.restarts, float(rep.relative_residual),
                                    float(rep.backward_error)]
        u = U[:, 0]
        st = nl.element_state(u, E, need_K=True, need_dE=True)
        rec["f_e"] = _sha(st["f"])
        rec["K_e"] = _sha(st["K"])
        rec["dfdE"] = _sha(st["dfdE"])
        rec["energy"] = float(nl.strain_energy(u, E))
        rec["f_int"] = _sha(nl.internal_force(u, E))
        rec["K_T"] = _sha(nl.tangent(u, E).toarray())
        g = nl.adjoint_gradient(E, U, F, nl.cutoffs())
        rec["grad"] = _sha(g)
        out[name] = rec
    return out


def main() -> int:
    got = fingerprint()
    if "--write" in sys.argv:
        OUT.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT} ({len(got)} configurations)")
        return 0
    ref = json.loads(OUT.read_text(encoding="utf-8"))
    bad = 0
    for name in sorted(set(ref) | set(got)):
        if name not in got or name not in ref:
            print(f"{name}: {'missing now' if name not in got else 'new'}")
            bad += 1
            continue
        diff = [k for k in ref[name] if ref[name][k] != got[name].get(k)]
        print(f"{name}: {'bit-for-bit equal' if not diff else 'DIFFERS in ' + ', '.join(diff)}")
        bad += bool(diff)
    print(f"{len(got) - bad} of {len(got)} configurations equal")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
