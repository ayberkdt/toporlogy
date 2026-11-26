"""MA0 fingerprint (docs/olcumler/faz_aci_onkayit.md): sha256 of the float64 bytes of the single-phase angle path
and of the plain multi-phase Embed path, before and after the multi-phase angle change.

    PYTHONPATH=src python docs/olcumler/faz_aci_parmakizi.py <out.json>
    PYTHONPATH=src python docs/olcumler/faz_aci_parmakizi.py <out.json> --compare docs/olcumler/faz_aci_oncesi.json
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material_field import PhaseMaterial
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.orientation import AngleBlock, OrientedEmbed
from toporlogy.design.phases import TensorPhases
from toporlogy.fem.tensor_elastic import TensorElasticity
from toporlogy.optimization.responses import Compliance, PhaseVolume, Volume
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

FIBRE = dict(E1=1.0, E2=0.1, E3=0.1, G12=0.05, G13=0.05, G23=0.04, nu12=0.3, nu13=0.3, nu23=0.3)


def h(a: Any) -> str:
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def left_fixed(d: Domain) -> np.ndarray:
    nodes = np.arange(d.shape[1] + 1)                    # nodes on x = 0 (2-D), both DOFs
    return (nodes[:, None] * 2 + np.arange(2)).ravel()


def tip_load(d: Domain) -> np.ndarray:
    F = np.zeros(d.ndof)
    nx, ny = d.shape
    F[2 * (nx * (ny + 1) + ny // 2) + 1] = -1.0           # node (nx, ny // 2), y
    return F


def fibre() -> PhaseMaterial:
    return PhaseMaterial("fibre", ElasticTensor.orthotropic(**FIBRE))


def matrix() -> PhaseMaterial:
    return PhaseMaterial("matrix", ElasticTensor.isotropic(0.3, 0.3))


def single_phase_angle() -> dict[str, str]:
    out: dict[str, str] = {}
    d = Domain((12, 6))
    rng = np.random.default_rng(3)
    for loc, rmin in (("element", 1.5), ("global", None)):
        for dens in (True, False):
            key = f"{loc}-{'dens' if dens else 'nodens'}"
            emb = OrientedEmbed(d, AngleBlock(loc, start=math.radians(10.0), rmin=rmin), density=dens)
            tp = TensorPhases(d, [fibre()])
            tp.attach_angle(emb)
            x = torch.as_tensor(rng.uniform(0.1, 0.9, emb.n_in))
            out[key + ":grid"] = h(emb(x))
            out[key + ":angle"] = h(emb.take())
            out[key + ":reduce"] = h(emb.reduce(np.full(d.n_design if dens else 0, 0.4)))
            out[key + ":layout"] = hashlib.sha256(json.dumps(emb.layout.describe()).encode()).hexdigest()
            out[key + ":n_in"] = str(emb.n_in)
            te = TensorElasticity(d, left_fixed(d))
            te.bind(tp)
            chain = Chain([emb, DensityFilter(d, 1.5), Passive(d)]) if dens else Chain([emb])
            st = Study(d, te, tip_load(d), chain, tp, Compliance(), [Constraint(Volume(), 0.4 if dens else 2.0)],
                       MMA(move=0.1))
            ev = st.evaluate(x.numpy())
            for k in ("df0", "g", "dg", "xphys"):
                out[f"{key}:study:{k}"] = h(getattr(ev, k))
            out[f"{key}:study:f0"] = h(ev.f0)
            out[f"{key}:study:start"] = h(st.default_start())
    return out


def multiphase_embed() -> dict[str, str]:
    out: dict[str, str] = {}
    d = Domain((12, 6))
    rng = np.random.default_rng(5)
    for rule in ("simplex", "recursive"):
        for cons in ("phase", "solid"):
            key = f"{rule}-{cons}"
            tp = TensorPhases(d, [matrix(), fibre()], rule=rule)
            te = TensorElasticity(d, left_fixed(d))
            te.bind(tp)
            cs = ([Constraint(PhaseVolume("matrix"), 0.2), Constraint(PhaseVolume("fibre"), 0.15)]
                  if cons == "phase" else [Constraint(PhaseVolume(), 0.4)])
            st = Study(d, te, tip_load(d), Chain([Embed(d, 2), DensityFilter(d, 1.5), Passive(d)]), tp,
                       Compliance(), cs, MMA(move=0.1))
            out[key + ":start"] = h(st.default_start())
            out[key + ":n_var"] = str(st.n_var)
            ev = st.evaluate(rng.uniform(0.1, 0.9, st.n_var))
            for k in ("df0", "g", "dg", "xphys"):
                out[f"{key}:{k}"] = h(getattr(ev, k))
            out[key + ":f0"] = h(ev.f0)
    return out


def main() -> None:
    torch.set_num_threads(1)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout)
    res = {"commit": commit, "src_dirty": dirty, "single_phase_angle": single_phase_angle(),
           "multiphase_embed": multiphase_embed()}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if "--compare" in sys.argv:
        ref = json.loads(Path(sys.argv[sys.argv.index("--compare") + 1]).read_text(encoding="utf-8"))
        diff = [f"{g}/{k}" for g in ("single_phase_angle", "multiphase_embed")
                for k in sorted(set(ref[g]) | set(res[g])) if ref[g].get(k) != res[g].get(k)]
        n = sum(len(ref[g]) for g in ("single_phase_angle", "multiphase_embed"))
        print(f"{n - len(diff)}/{n} identical; differing: {diff}")
        sys.exit(1 if diff else 0)
    print(f"wrote {sys.argv[1]} ({sum(len(v) for v in res.values() if isinstance(v, dict))} entries)")


if __name__ == "__main__":
    main()
