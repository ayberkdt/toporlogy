"""Several tensor phases with a fibre-angle design block (OrientedEmbed with n_fields > 1).

Checks and thresholds pre-registered in docs/olcumler/faz_aci_onkayit.md (26 Nov 2026 12:32:58 +0300, before the
first line of code); results in the same file. Test names carry the check ids. Relative errors are max-norm:
max|a − b| / max|b| over the whole array.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material_field import PhaseMaterial
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.orientation import AngleBlock, OrientedEmbed
from toporlogy.design.phases import TensorPhases, phase_start
from toporlogy.fem.tensor_elastic import TensorElasticity
from toporlogy.optimization.responses import Compliance, Mass, PhaseVolume, Volume
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA
from toporlogy.optimizers.oc import OC

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "research" / "fiber_matrix_angle_cantilever2d.yaml"
FIBRE = dict(E1=1.0, E2=0.1, E3=0.1, G12=0.05, G13=0.05, G23=0.04, nu12=0.3, nu13=0.3, nu23=0.3)
CROSS = dict(E1=0.5, E2=0.2, E3=0.2, G12=0.08, G13=0.08, G23=0.06, nu12=0.25, nu13=0.25, nu23=0.3)


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def left_fixed(d: Domain) -> np.ndarray:
    """Both DOFs of the nodes on x = 0 (2-D)."""
    return (np.arange(d.shape[1] + 1)[:, None] * 2 + np.arange(2)).ravel()


def tip_load(d: Domain) -> np.ndarray:
    F = np.zeros(d.ndof)
    F[int(np.ravel_multi_index((d.shape[0], d.shape[1] // 2), (d.shape[0] + 1, d.shape[1] + 1))) * 2 + 1] = -1.0
    return F


def fibre(name: str = "fibre") -> PhaseMaterial:
    return PhaseMaterial(name, ElasticTensor.orthotropic(**FIBRE), density=1.0, cost=4.0)


def matrix() -> PhaseMaterial:
    return PhaseMaterial("matrix", ElasticTensor.isotropic(0.3, 0.3), density=0.5, cost=1.0)


def angle_study(d: Domain, phs: list[PhaseMaterial], rule: str, block: AngleBlock,
                constraints: list[Constraint]) -> tuple[Study, OrientedEmbed]:
    emb = OrientedEmbed(d, block, n_fields=len(phs))
    tp = TensorPhases(d, phs, rule=rule)
    tp.attach_angle(emb)
    te = TensorElasticity(d, left_fixed(d))
    te.bind(tp)
    st = Study(d, te, tip_load(d), Chain([emb, DensityFilter(d, 1.5), Passive(d)]), tp, Compliance(), constraints,
               MMA(move=0.1))
    return st, emb


# ====================================================================== MA1: one phase in a two-phase angle study
@pytest.mark.parametrize("rule", ["recursive", "simplex"])
def test_ma1_two_phase_angle_study_reduces_to_one_phase(rule: str) -> None:
    """MA1a (recursive, phases [isotropic A, fibre]) and MA1b (simplex, phases [fibre copy, fibre]): with the first
    field ≡ 0 the two-phase study is the one-phase [fibre] study."""
    d = Domain((12, 6))
    blk = AngleBlock("element", rmin=1.5)
    first = matrix() if rule == "recursive" else fibre("fibre0")
    one, e1 = angle_study(d, [fibre()], rule, blk, [Constraint(PhaseVolume("fibre"), 0.3)])
    two, e2 = angle_study(d, [first, fibre()], rule, blk, [Constraint(PhaseVolume("fibre"), 0.3)])
    n = d.n_design
    rng = np.random.default_rng(21)
    t, s = rng.uniform(0.2, 0.9, n), rng.uniform(0.0, 1.0, n)
    a = one.evaluate(np.concatenate([t, s]))
    b = two.evaluate(np.concatenate([np.zeros(n), t, s]))
    assert abs(b.f0 / a.f0 - 1) <= 1e-12
    assert rel(b.g, a.g) <= 1e-12
    assert rel(b.df0[n:], a.df0) <= 1e-10
    assert rel(b.dg[:, n:], a.dg) <= 1e-10


# ====================================================================== MA2: gradients, central differences
def _ma2_study(case: str) -> tuple[Study, OrientedEmbed]:
    keep = np.zeros((12, 6), bool)
    keep[-1, :] = True                              # one keep column (Passive acts; keep_angle there)
    d = Domain((12, 6), keep=keep)
    rule = "recursive" if case.startswith("recursive") else "simplex"
    blk = AngleBlock("global") if case == "simplex-global" else AngleBlock("element", rmin=1.5)
    phs = [matrix(), fibre()]
    if case == "simplex-3phase":
        phs = [matrix(), PhaseMaterial("cross", ElasticTensor.orthotropic(**CROSS), density=0.8, cost=2.0,
                                       rotate=False), fibre()]
    cons = [Constraint(PhaseVolume(p.name), 0.2) for p in phs] + [Constraint(Mass(), 20.0)]
    return angle_study(d, phs, rule, blk, cons)


MA2_CASES = ["simplex-element", "recursive-element", "simplex-global", "simplex-3phase"]
MA2_DIRECTIONS = ["density", "angle", "both"]
_MA2_FAIL = ("MA2 FAIL: central FD with the pre-registered h = 1e-6 differs from the gradient by 2.43e-6 relative "
             "(compliance, > 1e-6); round-off: the FD error is 7.2e-8 at h = 1e-4 and grows for smaller h, Richardson "
             "FD (h = 1e-3, 5e-4) agrees to 6.1e-9 (test_ma2_posthoc_richardson)")
MA2_PARAMS = [pytest.param(c, dr, marks=pytest.mark.xfail(strict=True, reason=_MA2_FAIL))
              if (c, dr) == ("simplex-global", "density") else pytest.param(c, dr)
              for dr in MA2_DIRECTIONS for c in MA2_CASES]


def _ma2_fd(case: str, direction: str) -> tuple[Study, np.ndarray, np.ndarray, Any]:
    st, emb = _ma2_study(case)
    rng = np.random.default_rng(13)
    x = rng.uniform(0.2, 0.9, st.n_var)
    dv = rng.normal(size=st.n_var)
    off = emb.layout["angle"].offset
    if direction == "density":
        dv[off:] = 0.0
    elif direction == "angle":
        dv[:off] = 0.0
    return st, x, dv, st.evaluate(x)


def _central(st: Study, x: np.ndarray, dv: np.ndarray, h: float) -> np.ndarray:
    p, m = st.evaluate(x + h * dv, gradients=False), st.evaluate(x - h * dv, gradients=False)
    return np.concatenate([[(p.f0 - m.f0) / (2 * h)], (p.g - m.g) / (2 * h)])


@pytest.mark.parametrize(("case", "direction"), MA2_PARAMS)
def test_ma2_gradients_finite_difference(case: str, direction: str) -> None:
    st, x, dv, ev = _ma2_fd(case, direction)
    fd = _central(st, x, dv, 1e-6)
    ad = np.concatenate([[float(ev.df0 @ dv)], ev.dg @ dv])
    for i, (f, a) in enumerate(zip(fd, ad, strict=True)):
        assert abs(f - a) <= 1e-6 * abs(f), (i, f, a)


@pytest.mark.parametrize("direction", MA2_DIRECTIONS)
@pytest.mark.parametrize("case", MA2_CASES)
def test_ma2_posthoc_richardson(case: str, direction: str) -> None:
    """POST HOC (after MA2 failed for simplex-global / density): Richardson FD (4 D(h/2) − D(h)) / 3 with h = 1e-3,
    which removes the h² term and stays clear of the round-off that dominates at h = 1e-6; same threshold."""
    st, x, dv, ev = _ma2_fd(case, direction)
    fd = (4 * _central(st, x, dv, 5e-4) - _central(st, x, dv, 1e-3)) / 3
    ad = np.concatenate([[float(ev.df0 @ dv)], ev.dg @ dv])
    for i, (f, a) in enumerate(zip(fd, ad, strict=True)):
        assert abs(f - a) <= 1e-6 * abs(f), (i, f, a)


def test_ma2b_taylor_remainder_small_steps() -> None:
    st, _ = _ma2_study("simplex-element")
    rng = np.random.default_rng(13)
    x = rng.uniform(0.2, 0.9, st.n_var)
    dv = rng.normal(size=st.n_var)
    ev = st.evaluate(x)
    hs = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])
    r = [abs(st.evaluate(x + hh * dv, gradients=False).f0 - ev.f0 - hh * float(ev.df0 @ dv)) for hh in hs]
    assert float(np.polyfit(np.log(hs), np.log(r), 1)[0]) >= 1.9


# ====================================================================== MA3: start vector
def test_ma3_default_start_layout() -> None:
    d = Domain((12, 6))
    start = math.radians(20.0)
    blk = AngleBlock("element", start=start, rmin=1.5)
    phs = [matrix(), fibre()]
    st, emb = angle_study(d, phs, "simplex", blk, [Constraint(PhaseVolume("matrix"), 0.25),
                                                   Constraint(PhaseVolume("fibre"), 0.15)])
    x0 = st.default_start()
    n = d.n_design
    assert st.n_var == len(x0) == 2 * n + n
    assert np.array_equal(x0[:2 * n], phase_start(2, n, [0.25, 0.15]))
    assert np.array_equal(x0[2 * n:], np.full(n, blk.s_of(start)))
    st2, _ = angle_study(d, phs, "simplex", blk, [Constraint(PhaseVolume(), 0.4)])
    x1 = st2.default_start()
    tau = 1.0 - (1.0 - 0.4) ** (1.0 / 2)
    assert np.array_equal(x1[:2 * n], np.full(2 * n, tau))
    assert np.array_equal(x1[2 * n:], np.full(n, blk.s_of(start)))
    with pytest.raises(ValueError, match="entries"):
        emb.reduce(np.zeros(n))                         # one field's worth for two fields


# ====================================================================== MA4: YAML example
def test_ma4_yaml_example_runs() -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    setup = build_setup(raw, overrides=["run.max_iter=20"])
    res = setup.study.run(max_iter=20)
    hist = res.history
    assert all(np.all(np.isfinite(np.asarray(v, float))) for k, v in hist.items()
               if isinstance(v, list) and v and isinstance(v[0], (int, float)))
    assert res.objective < hist["f0"][0]
    for c in setup.study.constraints:
        assert res.constraint_values[c.key] <= c.upper * (1 + 1e-2), c.key


# ====================================================================== MA5: fail-closed
def _two_phase_angle_cfg(**kw: Any) -> dict[str, Any]:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw.update(copy.deepcopy(kw))
    return raw


def test_ma5a_density_false_with_several_phases_is_refused() -> None:
    chain = [{"type": "embed", "density": False, "angle": {"rmin": 2.0}}]
    with pytest.raises(ConfigError, match="density false and 2 phases is not supported"):
        build_setup(_two_phase_angle_cfg(chain=chain))
    with pytest.raises(ValueError, match="density false with 2 phases"):
        OrientedEmbed(Domain((6, 4)), AngleBlock("element"), density=False, n_fields=2)


def test_ma5b_oc_is_refused() -> None:
    with pytest.raises(ConfigError, match="angle block with optimizer oc"):
        build_setup(_two_phase_angle_cfg(optimizer={"type": "oc"},
                                         constraints=[{"response": {"type": "phase_volume"}, "upper": 0.4}]))
    d = Domain((12, 6))
    st, _ = angle_study(d, [matrix(), fibre()], "simplex", AngleBlock("element"), [Constraint(PhaseVolume(), 0.4)])
    with pytest.raises(ValueError, match="does not handle several phases"):
        Study(d, st.physics, tip_load(d), st.chain, st.interp, Compliance(), [Constraint(PhaseVolume(), 0.4)], OC())


def test_ma5c_volume_response_is_refused() -> None:
    with pytest.raises(ConfigError, match="only Compliance, PhaseVolume, Mass, Cost"):
        build_setup(_two_phase_angle_cfg(constraints=[{"response": "volume", "upper": 0.4}]))
    d = Domain((12, 6))
    with pytest.raises(ValueError, match="only Compliance, PhaseVolume, Mass, Cost"):
        angle_study(d, [matrix(), fibre()], "simplex", AngleBlock("element"), [Constraint(Volume(), 0.4)])


def test_ma5d_oriented_embed_must_be_first() -> None:
    d = Domain((12, 6))
    emb = OrientedEmbed(d, AngleBlock("element"), n_fields=2)
    tp = TensorPhases(d, [matrix(), fibre()])
    tp.attach_angle(emb)
    te = TensorElasticity(d, left_fixed(d))
    te.bind(tp)
    with pytest.raises(ValueError, match="chain may hold only Embed or OrientedEmbed first"):
        Study(d, te, tip_load(d), Chain([Embed(d, 2), emb]), tp, Compliance(),
              [Constraint(PhaseVolume(), 0.4)], MMA())
