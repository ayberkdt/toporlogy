"""Negative Poisson ratio cell (docs/olcumler/auxetic_hucre_onkayit.md, A1-A4, pre-registered before the script)."""
from __future__ import annotations

import numpy as np
import pytest

import _auxetic as A


def test_a1_gradients_by_central_differences() -> None:
    x0 = A.start()
    ev = A.evaluate(x0)
    v = np.random.default_rng(0).normal(size=x0.shape)
    t = 1e-6
    hi, lo = A.evaluate(x0 + t * v), A.evaluate(x0 - t * v)
    fd0 = (hi["f0"] - lo["f0"]) / (2 * t)
    assert abs(ev["df0"] @ v.ravel() - fd0) <= 1e-6 * abs(fd0)
    fdg = (hi["g"] - lo["g"]) / (2 * t)
    assert np.abs(ev["dg"] @ v.ravel() - fdg).max() <= 1e-6 * np.abs(fdg).max()


@pytest.mark.slow
def test_a2_a4_auxetic_cell() -> None:
    r = A.run()
    print(f"\n  A: {r['iterations']} it, grey nu {r['nu']:.4f} vol {r['vol']:.4f} C11 {r['C'][0, 0]:.4f}; "
          f"0/1 nu {r['nu_binary']:.4f} vol {r['vol_binary']:.4f}")
    assert r["vol"] <= A.VOL + 1e-3 and r["C"][0, 0] >= A.C11_MIN * (1 - 1e-3)        # A2
    assert r["nu_binary"] < 0                                                           # A3
    assert A.periodic_one_body(r["binary"])                                             # A4
    assert np.linalg.eigvalsh(r["C_binary"]).min() > 0
