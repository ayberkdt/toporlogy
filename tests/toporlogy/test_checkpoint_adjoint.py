"""Checkpointed history adjoint for the viscoelastic march (docs/foundations/prereg/D3_checkpoint.md, D3a-D3d)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure

from test_viscoelastic_topology import TIMES, _cantilever, _general, _schedule, _weights

FP = Path(__file__).resolve().parent / "baseline" / "d3_fingerprint_b7e7637.json"


def _setup():
    m, fixed, tip = _cantilever()
    st = ViscoelasticStructure(m, _general(), TIMES, _schedule(m.n_nodes * 2, tip, TIMES, 2.0), fixed)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    return st, x


def _grad(st, x, k):
    xt = torch.as_tensor(x).requires_grad_(True)
    J = st.response(_weights(xt), checkpoint_every=k).compliance[-1]
    J.backward()
    return float(J), xt.grad.numpy().copy()


@pytest.mark.parametrize("k", [1, 4, 12])
def test_d3a_checkpointed_gradient_equals_full(k) -> None:
    st, x = _setup()
    _, g0 = _grad(st, x, None)
    _, gk = _grad(st, x, k)
    assert np.linalg.norm(gk - g0) <= 1e-12 * np.linalg.norm(g0)


def _saved_bytes(st, x, k):
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t
    xt = torch.as_tensor(x).requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        J = st.response(_weights(xt), checkpoint_every=k).compliance[-1]
    return total[0], J


def test_d3b_saved_memory_is_reduced() -> None:
    st, x = _setup()
    full, _ = _saved_bytes(st, x, None)
    ck, _ = _saved_bytes(st, x, 4)
    print(f"D3b saved bytes: full {full}, checkpoint_every=4 {ck}, ratio {ck / full:.3f}")
    assert ck <= 0.5 * full


def test_d3c_recompute_cost() -> None:
    st, x = _setup()
    st.n_solves = 0
    _grad(st, x, None)
    n_full = st.n_solves
    st.n_solves = 0
    _grad(st, x, 4)
    n_ck = st.n_solves
    print(f"D3c solves: full {n_full}, checkpointed {n_ck}")
    assert n_ck <= 2 * n_full


def test_d3d_default_is_unchanged() -> None:
    st, x = _setup()
    J, g = _grad(st, x, None)
    ref = json.loads(FP.read_text(encoding="utf-8"))
    assert ref["J"] == J
    assert np.array_equal(g, np.array(ref["grad"]))
