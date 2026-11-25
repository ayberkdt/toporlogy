"""PR-11: connectivity ties placed together with their mirror images in a symmetric problem.

Roadmap §2.3: on the Ideathon bracket (symmetric about x = 500) the frozen tie runs on one side only,
e_sym(keep) = 0.224 (228 cells). With ``connectivity.symmetry`` (or a ``symmetry`` link in the chain) the frozen
region is the mirror closure of the tie and the interface cells: measured 24 Nov 2026 on the bracket,
e_sym(keep) 0.224 -> 0 (tie cells 114 -> 228).
"""
from __future__ import annotations

import numpy as np

from toporlogy import Domain
from toporlogy.design.manufacturing import mirror_closure, mirror_partner
from toporlogy.optimization.connectivity import one_body
from toporlogy.optimization.diagnostics import field_symmetry


def _box(shape, *slices):
    m = np.zeros(shape, bool)
    for s in slices:
        m[s] = True
    return m


def test_mirror_closure_is_the_union_of_the_images():
    d = Domain((10, 6), h=1.0)
    m = _box(d.shape, np.s_[1, 2])
    one = mirror_closure(d, m, [(0, 5.0)])
    assert one[1, 2] and one[8, 2] and one.sum() == 2
    two = mirror_closure(d, m, [(0, 5.0), (1, 3.0)])
    assert two.sum() == 4 and two[1, 3] and two[8, 3]
    p = mirror_partner(d, 0, 5.0).reshape(d.shape)
    assert p[1, 2] == np.ravel_multi_index((8, 2), d.shape)


def _two_pad_problem():
    """Interface "base": both bottom corners (symmetric about x = 10); "top": the top centre. The tie from the
    base group to the top starts at one corner only: an asymmetric frozen region."""
    shape = (20, 10)
    d = Domain(shape, h=1.0)
    base = _box(shape, np.s_[0:2, 0], np.s_[18:20, 0])
    top = _box(shape, np.s_[9:11, 9])
    return d, {"base": base, "top": top}


def test_one_body_ties_follow_the_mirror_planes():
    d, ifc = _two_pad_problem()
    plain = one_body(d, ifc, ties=1, constraint=None)
    sym = one_body(d, ifc, ties=1, constraint=None, symmetry=[(0, 10.0)])
    e_plain = field_symmetry(d, {"keep": plain.domain.keep.astype(float)}, "x", 10.0)["fields"]["keep"]["e_sym"]
    e_sym = field_symmetry(d, {"keep": sym.domain.keep.astype(float)}, "x", 10.0)["fields"]["keep"]["e_sym"]
    assert e_plain > 0.1 and e_sym == 0.0
    assert (sym.domain.keep >= plain.domain.keep).all()           # the closure only adds cells
    print(f"\n  e_sym(keep): one-sided tie {e_plain:.3f}, mirrored {e_sym:.3f}")


def test_yaml_symmetry_option_and_symmetry_link_both_mirror_the_ties():
    from pathlib import Path

    import yaml

    from toporlogy.configuration.build import build_setup
    ex = Path(__file__).resolve().parents[2] / "examples" / "split_connectivity2d.yaml"
    raw = yaml.safe_load(ex.read_text(encoding="utf-8"))
    raw["post"] = []
    base = build_setup(raw)
    assert "tie_symmetry" not in base.info
    explicit = build_setup({**raw, "connectivity": {**raw["connectivity"], "symmetry": {"x": "mid"}}})
    linked = build_setup({**raw, "chain": [{"type": "symmetry", "planes": {"x": "mid"}}] + list(raw["chain"])})
    for s in (explicit, linked):
        d = s.study.domain
        mid = float(s.info["tie_symmetry"][0][1])
        assert s.info["tie_symmetry"][0][0] == "x"
        assert field_symmetry(d, {"keep": d.keep.astype(float)}, "x", mid)["fields"]["keep"]["e_sym"] == 0.0
