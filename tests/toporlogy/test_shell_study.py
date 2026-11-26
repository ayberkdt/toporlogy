"""The flat-facet shell inside Study (docs/olcumler/kabuk_onkayit.md, KS1-KS5)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from toporlogy.design.chain import Chain, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.mesh import perturb, rectangle
from toporlogy.fem.shell import FlatShell, cylinder_surface
from toporlogy.optimization.mesh_study import MeshFilterLink, MeshVolume
from toporlogy.optimization.mesh_topology import MeshDensityFilter
from toporlogy.optimization.responses import Compliance
from toporlogy.optimization.shell_study import ShellDomain, ShellPhysics
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

NU, T, EMIN = 0.3, 0.05, 1e-9


def _segment(n_theta: int, n_z: int) -> tuple[FlatShell, int]:
    """R = 1, 90°, L = 1 open segment clamped (six DOFs) on its first ring; also the free-ring middle node."""
    m = cylinder_surface(1.0, 1.0, n_theta, n_z, angle=np.pi / 2)
    fixed = (6 * np.arange(n_theta + 1)[:, None] + np.arange(6)).ravel()
    return FlatShell(m, T, NU, fixed), n_theta // 2 + (n_theta + 1) * n_z


def _inward(sh: FlatShell, node: int) -> np.ndarray:
    F = np.zeros(sh.ndof)
    X = sh.mesh.nodes[node]
    F[6 * node:6 * node + 2] = -X[:2] / np.linalg.norm(X[:2])
    return F


def _study(sh: FlatShell, F: np.ndarray, rmin: float) -> Study:
    sd: Any = ShellDomain(sh.mesh)        # duck-typed: Study, Embed, Passive, SIMP are annotated with Domain
    return Study(sd, ShellPhysics(sh), F, Chain([Embed(sd), MeshFilterLink(sd, rmin), Passive(sd)]),
                 SIMP(sd, penal=3.0, emin_ratio=EMIN), Compliance(), [Constraint(MeshVolume(sd), 0.4)], MMA())


def _two_cases() -> tuple[FlatShell, np.ndarray]:
    sh, tip = _segment(8, 6)
    return sh, np.column_stack([sh.pressure_load(1.0), _inward(sh, tip)])


def test_ks1_study_equals_direct_shell_compliance() -> None:
    sh, F = _two_cases()
    st = _study(sh, F, 0.3)
    x = 0.2 + 0.7 * np.random.default_rng(41).random(st.n_var)
    ev = st.evaluate(x)
    centroids = sh.mesh.nodes[sh.mesh.facets].mean(axis=1)
    xt = MeshDensityFilter(centroids, 0.3).matrix @ x
    E = EMIN + xt ** 3 * (1.0 - EMIN)
    c = sum(float(F[:, k] @ sh.solve(E, F[:, k]).numpy()) for k in range(F.shape[1]))
    area = 8 * 2.0 * np.sin(np.pi / 32) * 1.0
    A = ShellDomain(sh.mesh).cell_volume
    vol = float(A @ xt / A.sum())
    rc, rv = abs(ev.f0 - c) / abs(c), abs(ev.values["volume"] - vol) / vol
    ra = abs(A.sum() - area) / area
    print(f"KS1: value {rc:.2e}, volume {rv:.2e}, area {ra:.2e} (c = {c:.6e})")
    assert rc <= 1e-12 and rv <= 1e-12 and ra <= 1e-12


def test_ks2_study_gradient_on_a_curved_segment() -> None:
    sh, F = _two_cases()
    st = _study(sh, F, 0.3)
    x = 0.2 + 0.7 * np.random.default_rng(42).random(st.n_var)
    v = np.random.default_rng(43).normal(size=st.n_var)
    g = st.evaluate(x).df0 @ v
    s = 1e-3
    f = {k: st.evaluate(x + k * s * v, False).f0 for k in (-2, -1, 1, 2)}
    fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * s)
    rel = abs(g - fd) / abs(fd)
    print(f"KS2: rel {rel:.2e}")
    assert rel <= 1e-6


def test_ks3_study_run_on_a_clamped_segment() -> None:
    sh, tip = _segment(16, 12)
    st = _study(sh, _inward(sh, tip), 0.15)
    res = st.run(max_iter=30)
    c0 = res.history["f0"][0]
    vol = res.constraint_values["volume"]
    print(f"KS3: compliance {c0:.4e} -> {res.objective:.4e} in {len(res.history['f0'])} iterations, volume {vol:.5f}")
    assert res.objective < c0 and vol <= 0.4 * (1 + 1e-3)


def test_ks4_wrong_load_rows_refused() -> None:
    sh, _tip = _segment(8, 6)
    bad = np.zeros(3 * sh.mesh.nn)
    with pytest.raises(ValueError, match=f"{6 * sh.mesh.nn} state DOFs"):
        _study(sh, bad, 0.3)
    with pytest.raises(ValueError, match="6 per node"):
        ShellPhysics(sh).solve(torch.ones(sh.mesh.ne, dtype=torch.float64), bad)


def test_ks5_filter_on_centroids_equals_filter_on_mesh() -> None:
    m = perturb(rectangle(16, 8, 2.0, 1.0, "Q4"), 0.03, seed=3)
    for rmin in (0.2, 0.35):
        A = MeshDensityFilter(m, rmin).matrix
        B = MeshDensityFilter(m.cell_centroids(), rmin).matrix
        same = (A.indptr == B.indptr).all() and (A.indices == B.indices).all() and (A.data == B.data).all()
        print(f"KS5: rmin {rmin}: bitwise equal {bool(same)}")
        assert same
