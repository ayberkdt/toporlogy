"""Viscoelastic topology optimisation with a two-phase load history (Foundations II, prereg VM_minmax_switching.md).

    python examples/foundations/viscoelastic_minmax.py        # about 1.5 min

A cantilever carries a vertical load for 10 s and a horizontal load from 10 s to 100 s; the material (synthetic)
relaxes in bulk only. Each single-phase design is near-singular for the other phase; the min-max design balances
both. Gradients are the exact discrete history adjoint (reverse mode through the time march).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "toporlogy"))
from test_viscoelastic_minmax_switch import _problem  # noqa: E402

from toporlogy.optimization.viscoelastic_topology import optimize_viscoelastic  # noqa: E402

st, kA, kB = _problem()
kw = dict(volume=0.4, rmin=0.075, max_iter=1500, kkt_tol=1e-4)
runs = {"min-max": optimize_viscoelastic(st, steps=[kA, kB], **kw),
        "phase A only": optimize_viscoelastic(st, step_index=kA, **kw),
        "phase B only": optimize_viscoelastic(st, step_index=kB, **kw)}
for name, r in runs.items():
    with torch.no_grad():
        c = st.response(torch.as_tensor(1e-9 + r.xphys ** 3 * (1 - 1e-9))).compliance
    print(f"{name:14s} J_A(10 s) = {float(c[kA]):.4e}  J_B(100 s) = {float(c[kB]):.4e}  "
          f"({r.iterations} iterations, stop: {r.termination})")
