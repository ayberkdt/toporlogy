"""Drilling-scale sensitivity of the flat-facet shell on a curved segment (docs/olcumler/kabuk_onkayit.md, DR0-DR3)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from toporlogy.design.chain import Chain, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.shell import FlatShell, cylinder_surface
from toporlogy.optimization.mesh_study import MeshFilterLink, MeshVolume
from toporlogy.optimization.responses import Compliance
from toporlogy.optimization.shell_study import ShellDomain, ShellPhysics
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

NU, EMIN = 0.3, 1e-9
DRILLS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def _segment(n_theta: int, n_z: int, t: float, drill: float) -> FlatShell:
    """R = 1, 90°, L = 1 open segment clamped (six DOFs) on its first ring (z = 0)."""
    m = cylinder_surface(1.0, 1.0, n_theta, n_z, angle=np.pi / 2)
    fixed = (6 * np.arange(n_theta + 1)[:, None] + np.arange(6)).ravel()
    return FlatShell(m, t, NU, fixed, drill=drill)


def _load(sh: FlatShell, n_theta: int, n_z: int, kind: str) -> np.ndarray:
    """"corner": unit z force on the last node (z = L, θ = 90°); "radial": unit inward force on the free-ring middle
    node (the KS3 load)."""
    F = np.zeros(sh.ndof)
    if kind == "corner":
        F[6 * (sh.mesh.nn - 1) + 2] = 1.0
        return F
    node = n_theta // 2 + (n_theta + 1) * n_z
    X = sh.mesh.nodes[node]
    F[6 * node:6 * node + 2] = -X[:2] / np.linalg.norm(X[:2])
    return F


def _compliance(n_theta: int, n_z: int, t: float, drill: float, kind: str) -> float:
    sh = _segment(n_theta, n_z, t, drill)
    F = _load(sh, n_theta, n_z, kind)
    return float(F @ sh.solve(np.ones(sh.mesh.ne), F).numpy())


def _spread(c: dict[float, float]) -> float:
    return abs(c[1e-4] - c[1e-2]) / c[1e-3]


def test_dr0_setup_reproduces_the_recorded_values() -> None:
    rec = {1e-4: 184.90, 1e-3: 177.27, 1e-2: 175.33}
    got = {a: _compliance(8, 6, 0.05, a, "corner") for a in rec}
    print("DR0: " + ", ".join(f"{a:g}: {c:.4f}" for a, c in got.items()))
    assert all(abs(got[a] - rec[a]) <= 0.005 for a in rec)


def _failed(check: str, measured: str, limit: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"pre-registered {check} failed: measured {measured} > {limit} "
                                                 "(kabuk_onkayit.md, DR0-DR3)")


@pytest.mark.parametrize("t, kind", [
    (0.01, "corner"),
    pytest.param(0.01, "radial", marks=_failed("DR1", "2.50e-2", "1e-2")),
    pytest.param(0.05, "corner", marks=_failed("DR1", "5.57e-2", "1e-2")),
    pytest.param(0.05, "radial", marks=_failed("DR1", "2.13e-1", "1e-2")),
])
def test_dr1_static_compliance_spread_on_16x12(t: float, kind: str) -> None:
    c = {a: _compliance(16, 12, t, a, kind) for a in DRILLS}
    print(f"DR1 t = {t}, {kind}: " + ", ".join(f"{a:g}: {v:.6e}" for a, v in c.items())
          + f"; spread 1e-4 vs 1e-2: {_spread(c):.3e}")
    assert _spread(c) <= 0.01


def _run(t: float, drill: float) -> tuple[Any, np.ndarray, Study]:
    sh = _segment(16, 12, t, drill)
    F = _load(sh, 16, 12, "radial")
    sd: Any = ShellDomain(sh.mesh)
    st = Study(sd, ShellPhysics(sh), F, Chain([Embed(sd), MeshFilterLink(sd, 0.15), Passive(sd)]),
               SIMP(sd, penal=3.0, emin_ratio=EMIN), Compliance(), [Constraint(MeshVolume(sd), 0.4)], MMA())
    res = st.run(max_iter=30)
    return res, np.asarray(sd.cell_volume, float), st


def _optimise(t: float) -> dict[float, dict[str, Any]]:
    runs = {a: _run(t, a) for a in DRILLS}
    A = runs[1e-3][1]
    ref = runs[1e-3][0].xphys.reshape(-1)
    st_ref = runs[1e-3][2]
    out: dict[float, dict[str, Any]] = {}
    for a, (res, _A, _st) in runs.items():
        xp = res.xphys.reshape(-1)
        cross = st_ref.evaluate(res.x, False).f0          # design of α, analysed with the α = 1e-3 physics
        out[a] = {"c": res.objective, "it": res.iterations, "vol": res.constraint_values["volume"],
                  "xphys": xp, "max": float(np.abs(xp - ref).max()), "l1": float(A @ np.abs(xp - ref) / A.sum()),
                  "cross": cross}
    for a, r in out.items():
        r["cross_ratio"] = r["cross"] / out[1e-3]["c"]
    return out


@pytest.mark.parametrize("t", [pytest.param(0.05, marks=_failed("DR2 (a)", "5.80e-2", "2e-2")), 0.01])
def test_dr2_optimisation_spread(t: float) -> None:
    out = _optimise(t)
    for a, r in out.items():
        print(f"DR2 t = {t}, drill {a:g}: c {r['c']:.6e}, iterations {r['it']}, volume {r['vol']:.6f}, "
              f"vs 1e-3: max |dx| {r['max']:.3e}, L1 {r['l1']:.3e}, c at drill 1e-3 / c(1e-3 design) "
              f"{r['cross_ratio']:.6f}")
    c = {a: r["c"] for a, r in out.items()}
    A = ShellDomain(_segment(16, 12, t, 1e-3).mesh).cell_volume
    l1 = float(A @ np.abs(out[1e-4]["xphys"] - out[1e-2]["xphys"]) / A.sum())
    print(f"DR2 t = {t}: (a) compliance spread 1e-4 vs 1e-2 {_spread(c):.3e}, (b) L1 1e-4 vs 1e-2 {l1:.3e}")
    assert _spread(c) <= 0.02 and l1 <= 0.05


def test_dr3_record_mesh_dependence() -> None:
    """Record (no threshold): static spread over the five drilling scales on three meshes."""
    for n_theta, n_z in ((8, 6), (16, 12), (32, 24)):
        for t in (0.01, 0.05):
            for kind in ("corner", "radial"):
                c = {a: _compliance(n_theta, n_z, t, a, kind) for a in DRILLS}
                rel = ", ".join(f"{a:g}: {c[a] / c[1e-3] - 1:+.3e}" for a in DRILLS)
                print(f"DR3 {n_theta} x {n_z}, t = {t}, {kind}: c(1e-3) {c[1e-3]:.6e}; c / c(1e-3) - 1: {rel}; "
                      f"spread 1e-4 vs 1e-2 {_spread(c):.3e}")
                assert all(np.isfinite(list(c.values())))
