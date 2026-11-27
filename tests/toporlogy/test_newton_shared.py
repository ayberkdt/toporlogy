"""NS1: the shared Newton loop of cure_slab and nonlinear_heat is behaviour neutral, bit for bit
(docs/foundations/prereg/NS_newton_shared.md, preregistered before the change). The baseline was written by the code
at 1f898bf, before the change: ``python tests/toporlogy/test_newton_shared.py --write`` (only for an intended change
of these solvers; say why in the commit)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporlogy.core import cure as C
from toporlogy.core import properties as P
from toporlogy.fem.mesh.cure_slab import CureSlab
from toporlogy.fem.mesh.history import StepRejected
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.fem.mesh.nonlinear_heat import NonlinearHeat

BASELINE = Path(__file__).resolve().parent / "baseline" / "ns_newton_fingerprint_1f898bf.json"


def _feed(h: Any, v: Any) -> None:
    """Feed ``v`` into the hash without any rounding: arrays as raw bytes with dtype and shape, floats as hex."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        for f in dataclasses.fields(v):
            h.update(f.name.encode())
            _feed(h, getattr(v, f.name))
    elif isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    elif isinstance(v, (list, tuple)):
        h.update(f"seq{len(v)}".encode())
        for x in v:
            _feed(h, x)
    elif isinstance(v, dict):
        for k in sorted(v):
            h.update(str(k).encode())
            _feed(h, v[k])
    elif isinstance(v, (float, np.floating)):
        h.update(float(v).hex().encode())
    else:
        h.update(repr(v).encode())


def _digest(*vs: Any) -> str:
    h = hashlib.sha256()
    for v in vs:
        _feed(h, v)
    return h.hexdigest()


def _slab(bc: str) -> CureSlab:
    mesh = rectangle(1, 8, 0.004, 0.02)
    kin = C.NthOrder(C.Arrhenius.from_reference(1e-3, 350.0, 4e4), 1.5)
    slab = CureSlab(mesh, kin, C.ReactionHeat(7.5e4), 1200.0, 1500.0, 0.2, 350.0, 0.0)
    if bc == "robin":
        slab.add_robin(mesh.facet_sets["top"], 40.0, 350.0)
    else:
        slab.add_dirichlet(mesh.node_sets["top"], 350.0)
    return slab


def _heat(element: str) -> NonlinearHeat:
    k = P.Polynomial([1.0, 0.01], 300.0, (250.0, 450.0), "W/(m K)", source="synthetic", positive=True)
    c = P.Polynomial([2.0, 0.01], 300.0, (250.0, 450.0), "J/(m^3 K)", source="synthetic", positive=True)
    mesh = rectangle(4, 4, 1.0, 1.0, element)
    model = NonlinearHeat(mesh, k, c, 300.0)
    model.add_dirichlet(np.unique(mesh.node_sets["left"]), lambda t: 300.0 + 400.0 * t)
    model.add_robin(mesh.facet_sets["right"], 2.0, 300.0)
    model.set_source(lambda x, t: 50.0 * (1.0 + x[:, 1]))
    return model


def fingerprints() -> dict[str, str]:
    out: dict[str, str] = {}
    for bc, t_end, dt in (("robin", 1200.0, 60.0), ("dirichlet", 600.0, 30.0)):
        slab = _slab(bc)
        res = slab.advance_to(t_end, dt)
        out[f"cure_{bc}"] = _digest(res, slab.history, slab.T, slab.alpha, slab.t)
    slab = _slab("robin")
    slab.advance_to(120.0, 60.0)
    before = (slab.T.copy(), slab.alpha.copy(), slab.t, len(slab.history))
    try:
        slab.step(600.0, max_iter=1)
        msg = "accepted"
    except StepRejected as e:
        msg = str(e)
    out["cure_refused"] = _digest(msg, before, slab.T, slab.alpha, slab.t, len(slab.history))
    for el in ("Q4", "T6"):
        model = _heat(el)
        res = model.advance_to(0.1, 0.02)
        steady = model.steady()
        out[f"heat_{el}"] = _digest(res, steady, model.T, model.t)
    model = _heat("Q4")
    Tb, tb = model.T.copy(), model.t
    try:
        model.step(0.02, max_iter=1)
        msg = "accepted"
    except StepRejected as e:
        msg = str(e)
    out["heat_refused"] = _digest(msg, Tb, tb, model.T, model.t)
    return out


def test_ns1_shared_newton_is_bitwise_neutral() -> None:
    want = json.loads(BASELINE.read_text(encoding="utf-8"))
    got = fingerprints()
    assert sorted(got) == sorted(want)
    diff = [k for k in want if got[k] != want[k]]
    assert not diff, f"fingerprints changed: {diff}"


if __name__ == "__main__":
    import sys
    if "--write" not in sys.argv:
        sys.exit("usage: python tests/toporlogy/test_newton_shared.py --write")
    BASELINE.write_text(json.dumps(fingerprints(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE.name}")
