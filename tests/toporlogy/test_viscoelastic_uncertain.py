"""Viscoelastic topology optimisation under uncertain relaxation spectra
(docs/foundations/prereg/VU_viscoelastic_uncertain.md, VU1-VU5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.viscoelastic_topology import (RobustViscoelastic, ViscoelasticStructure,
                                                          optimize_viscoelastic)

from test_viscoelastic_topology import TIMES, _cantilever, _schedule, _weights

MPA = 1e6


def _sample(fk: float, ft: float) -> GeneralizedMaxwell:
    return GeneralizedMaxwell(PronySeries(1.0 * MPA, [], []), PronySeries(0.5 * MPA, [4.5 * MPA * fk], [10.0 * ft]))


SAMPLES = [(fk, ft) for fk in (0.7, 1.0, 1.3) for ft in (0.5, 1.0, 2.0)]


def _set(mesh, times, loads, fixed, samples=SAMPLES):
    return [ViscoelasticStructure(mesh, _sample(fk, ft), times, loads, fixed) for fk, ft in samples]


def _cant_set():
    m, fixed, tip = _cantilever()
    loads = _schedule(m.n_nodes * 2, tip, TIMES, 2.0)
    return m, _set(m, TIMES, loads, fixed)


def test_vu1_value_equals_separate_mean_std() -> None:
    m, sts = _cant_set()
    rv = RobustViscoelastic(sts, kappa=2.0)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    w = torch.as_tensor(_weights(x))
    with torch.no_grad():
        got = float(rv.response(w).compliance[-1])
        Js = np.array([float(s.response(w).compliance[-1]) for s in sts])
    ref = Js.mean() + 2.0 * Js.std()
    assert abs(got - ref) <= 1e-13 * abs(ref)


def test_vu2_gradient_fd() -> None:
    m, sts = _cant_set()
    rv = RobustViscoelastic(sts, kappa=2.0)
    rng = np.random.default_rng(4)
    x = 0.3 + 0.6 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    xt = torch.as_tensor(x).requires_grad_(True)
    rv.response(_weights(xt)).compliance[-1].backward()
    g = float(xt.grad.numpy() @ d)
    errs = []
    with torch.no_grad():
        for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            jp = float(rv.response(_weights(torch.as_tensor(x + h * d))).compliance[-1])
            jm = float(rv.response(_weights(torch.as_tensor(x - h * d))).compliance[-1])
            errs.append(abs((jp - jm) / (2 * h) - g) / abs(g))
    assert min(errs) <= 1e-6, errs


def test_vu3_single_sample_kappa_zero_is_the_single_structure() -> None:
    m, sts = _cant_set()
    rv = RobustViscoelastic(sts[:1], kappa=0.0)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    a = torch.as_tensor(_weights(x)).requires_grad_(True)
    b = torch.as_tensor(_weights(x)).requires_grad_(True)
    ja = rv.response(a).compliance[-1]
    jb = sts[0].response(b).compliance[-1]
    ja.backward()
    jb.backward()
    assert float(ja) == float(jb)
    assert np.array_equal(a.grad.numpy(), b.grad.numpy())


def test_vu4_robust_design_has_smaller_spread() -> None:
    nx, ny = 30, 10
    m = rectangle(nx, ny, float(nx), float(ny))
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    corner = int(np.flatnonzero(np.isclose(P[:, 0], nx) & np.isclose(P[:, 1], 0.0))[0])
    fixed = np.concatenate([2 * left, [2 * corner + 1]])
    tl = int(np.flatnonzero(np.isclose(P[:, 0], 0.0) & np.isclose(P[:, 1], ny))[0])
    times = np.concatenate([[0.0], np.logspace(-1, 2, 13)])
    loads = np.zeros((len(times), m.n_nodes * 2))
    loads[:, 2 * tl + 1] = -1.0
    sts = _set(m, times, loads, fixed)
    rob = optimize_viscoelastic(RobustViscoelastic(sts, kappa=2.0), 0.5, 1.5, max_iter=300, tol=1e-3)
    nom = optimize_viscoelastic(sts[4], 0.5, 1.5, max_iter=300, tol=1e-3)                  # (1.0, 1.0) sample
    with torch.no_grad():
        def stats(xp):
            J = np.array([float(s.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))).compliance[-1])
                          for s in sts])
            return J.mean(), J.std()
        mr, sr = stats(rob.xphys)
        mn, sn = stats(nom.xphys)
    print(f"VU4 robust mean {mr:.6e} std {sr:.6e} (it {rob.iterations}, conv {rob.converged}); nominal mean {mn:.6e} "
          f"std {sn:.6e} (it {nom.iterations}, conv {nom.converged})")
    assert sr <= sn


def test_vu5_invalid_inputs() -> None:
    m, sts = _cant_set()
    m2, fixed2, tip2 = _cantilever(nx=6)
    other_mesh = ViscoelasticStructure(m2, _sample(1, 1), TIMES, _schedule(m2.n_nodes * 2, tip2, TIMES, 2.0), fixed2)
    other_time = ViscoelasticStructure(m, _sample(1, 1), TIMES * 2, sts[0].loads, np.setdiff1d(
        np.arange(m.n_nodes * 2), sts[0].free))
    for bad in [dict(structures=[sts[0], other_mesh]), dict(structures=[sts[0], other_time]),
                dict(structures=sts[:2], weights=[1.5, -0.5]), dict(structures=sts[:2], weights=[0.3, 0.3]),
                dict(structures=sts[:2], kappa=-1.0), dict(structures=[])]:
        with pytest.raises(ValueError):
            RobustViscoelastic(**bad)
