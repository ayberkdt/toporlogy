"""Min-max over a two-load history with a design-dependent critical time
(docs/foundations/prereg/VM_minmax_switching.md, VM1-VM3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure, optimize_viscoelastic

from test_viscoelastic_topology import _nonprop


def _problem():
    nx, ny = 40, 20
    m = rectangle(nx, ny, 2.0, 1.0)
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    fixed = np.concatenate([2 * left, 2 * left + 1])
    a = int(np.flatnonzero(np.isclose(P[:, 0], 2.0) & np.isclose(P[:, 1], 0.0))[0])
    b = int(np.flatnonzero(np.isclose(P[:, 0], 2.0) & np.isclose(P[:, 1], 0.5))[0])
    t1 = np.concatenate([[0.0], np.logspace(-1, 1, 8)])
    t2 = np.concatenate([[10.5], np.logspace(np.log10(15.0), 2, 8)])
    times = np.concatenate([t1, t2])
    loads = np.zeros((len(times), m.n_nodes * 2))
    for k, t in enumerate(times):
        sA = 1.0 if t <= 10.0 else max(0.0, 1.0 - (t - 10.0) / 0.5)
        sB = 0.0 if t <= 10.0 else min(1.0, (t - 10.0) / 0.5)
        loads[k, 2 * a + 1] = -1.0 * sA
        loads[k, 2 * b] = 1.0 * sB
    kA = int(np.flatnonzero(times == 10.0)[0])
    kB = len(times) - 1
    return ViscoelasticStructure(m, _nonprop(), times, loads, fixed), kA, kB


@pytest.mark.slow
def test_vm_minmax_balances_two_loads() -> None:
    st, kA, kB = _problem()
    kw = dict(volume=0.4, rmin=1.5 * 0.05, max_iter=1500, kkt_tol=1e-4)
    mm = optimize_viscoelastic(st, steps=[kA, kB], **kw)
    sa = optimize_viscoelastic(st, step_index=kA, **kw)
    sb = optimize_viscoelastic(st, step_index=kB, **kw)
    with torch.no_grad():
        def JAB(xp):
            c = st.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))).compliance
            return float(c[kA]), float(c[kB])
        jm, ja, jb = JAB(mm.xphys), JAB(sa.xphys), JAB(sb.xphys)
    print(f"VM min-max {jm} ({mm.iterations} it, {mm.termination}, kkt {mm.kkt_residual}); A-only {ja} "
          f"({sa.iterations}, {sa.termination}); B-only {jb} ({sb.iterations}, {sb.termination})")
    for r in (mm, sa, sb):                                                                   # VM3
        assert (r.converged and r.kkt_residual <= 1e-4) or (not r.converged and r.iterations == 1500)
    assert max(jm) <= 0.95 * min(max(ja), max(jb))                                         # VM1
    assert abs(jm[0] - jm[1]) / max(jm) <= 0.02                                            # VM2
