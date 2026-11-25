"""E2 of docs/olcumler/malzeme_tasarimi_onkayit.md: SHA-256 fingerprints of the outputs of the existing
interpolations on fixed inputs. Run on the base commit before the change (``--write malzeme_tasarimi_oncesi.json``)
and after it (``--check malzeme_tasarimi_oncesi.json``); the outputs must be bit-identical.

    python docs/olcumler/malzeme_tasarimi_parmakizi.py --write docs/olcumler/malzeme_tasarimi_oncesi.json
    python docs/olcumler/malzeme_tasarimi_parmakizi.py --check docs/olcumler/malzeme_tasarimi_oncesi.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.design.interp import RAMP, SIMP, HashinShtrikman
from toporlogy.design.phases import MultiPhaseSIMP, phase_fractions, phase_start


def _h(a: torch.Tensor | np.ndarray) -> str:
    arr = a.detach().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def fingerprints() -> dict[str, str]:
    rng = np.random.default_rng(20261126)
    frac = rng.uniform(0.2, 1.0, (9, 5))
    d = Domain((9, 5), frac=frac)
    x = torch.as_tensor(rng.uniform(0.0, 1.0, (9, 5)), dtype=torch.float64).requires_grad_(True)
    t3 = torch.as_tensor(rng.uniform(0.0, 1.0, (3, 9, 5)), dtype=torch.float64).requires_grad_(True)
    out: dict[str, str] = {}
    for name, mod in (("simp", SIMP(d, E0=2.0, penal=3.0)), ("ramp", RAMP(d, E0=2.0, q=8.0)),
                      ("hs2", HashinShtrikman(d, E0=2.0, nu=0.3))):
        E = mod(x)
        (g,) = torch.autograd.grad(E.sum(), x)
        out[name] = _h(E)
        out[name + "_grad"] = _h(g)
    phases = [{"name": "a", "E": 0.3, "density": 0.4, "cost": 1.0}, {"name": "b", "E": 0.6, "density": 0.7},
              {"name": "c", "E": 1.0, "density": 1.0, "cost": 3.0}]
    mp = MultiPhaseSIMP(d, phases, penal=3.0)
    E = mp(t3)
    (g,) = torch.autograd.grad(E.sum(), t3)
    out["multiphase"] = _h(E)
    out["multiphase_grad"] = _h(g)
    out["multiphase_fractions"] = _h(mp.fractions(t3))
    out["multiphase_one"] = _h(MultiPhaseSIMP(d, phases[-1:], penal=3.0)(x))
    out["phase_fractions"] = _h(phase_fractions(t3))
    out["phase_start"] = _h(phase_start(3, d.n_design, [0.1, None, 0.3]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write")
    g.add_argument("--check")
    a = ap.parse_args()
    fp = fingerprints()
    if a.write:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        Path(a.write).write_text(json.dumps({"commit": sha, "fingerprints": fp}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(fp)} fingerprints at {sha}")
        return 0
    ref = json.loads(Path(a.check).read_text(encoding="utf-8"))["fingerprints"]
    bad = [k for k in ref if fp.get(k) != ref[k]]
    print(f"{len(ref) - len(bad)}/{len(ref)} identical" + (f"; differ: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
