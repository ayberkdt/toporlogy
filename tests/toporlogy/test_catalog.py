"""The scientific catalogs (``toporlogy.configuration.catalog``): one source with the registry and the references,
valid schema, traceable evidence, generated docs up to date (08 §3 and §5)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import toporlogy.configuration.build  # noqa: F401  (registers the built-in components)
from toporlogy.configuration import catalog as C
from toporlogy.configuration.registry import REGISTRY
from toporlogy.fem.elements import ELEMENTS_2D, ELEMENTS_3D
from toporlogy.references import REFERENCES

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
KEY = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d{4}[a-z]?$")


def test_schema_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", C.SCHEMA_VERSION)


@pytest.mark.parametrize("entries", [C.ELEMENTS, C.PHYSICS, C.SOLVERS, C.BENCHMARKS, C.MATERIALS, C.RULES],
                         ids=["elements", "physics", "solvers", "benchmarks", "materials", "rules"])
def test_ids_are_unique(entries: tuple) -> None:
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_physics_and_solvers_are_the_registry_ones() -> None:
    assert sorted(p.id for p in C.PHYSICS) == REGISTRY.names("physics")
    assert sorted(C.PHYS) == REGISTRY.names("physics")
    assert sorted(s.id for s in C.SOLVERS if s.registry) == REGISTRY.names("solver")


def test_every_registered_response_has_a_status() -> None:
    known = set(C.RESPONSE_MATRIX) | set(C.PHYSICS_INDEPENDENT) | set(C.COMPOSITE_RESPONSES)
    registered = set(REGISTRY.names("response"))
    assert not registered - known, f"responses without a catalog status: {sorted(registered - known)}"
    assert not known - registered, f"catalog responses that are not registered: {sorted(known - registered)}"
    assert not set(C.RESPONSE_MATRIX) & set(C.PHYSICS_INDEPENDENT)


def test_elements_match_the_element_library() -> None:
    ids = {e.id for e in C.ELEMENTS}
    assert set(ELEMENTS_2D) | set(ELEMENTS_3D) <= ids
    physics = set(REGISTRY.names("physics"))
    for e in C.ELEMENTS:
        assert set(e.physics) <= physics | {"post:reverify"}, e.id
    for p in C.PHYSICS:
        assert set(p.elements) <= ids, p.id


def test_reference_keys_exist() -> None:
    for e in C.ELEMENTS:
        missing = [k for k in e.references if k not in REFERENCES]
        assert not missing, f"element {e.id}: unknown reference keys {missing}"
    for b in C.BENCHMARKS:
        token = b.reference.split()[0]
        if KEY.match(token):
            assert token in REFERENCES, f"benchmark {b.id}: unknown reference key {token}"


def _cells() -> list[tuple[str, C.Cell]]:
    return [(f"{r} x {p}", c) for r, row in C.RESPONSE_MATRIX.items() for p, c in row.items()]


def test_statuses_and_their_obligations() -> None:
    for name, c in _cells():
        assert c.status in C.STATUSES, name
        if c.status in (C.SUPPORTED, C.EXPERIMENTAL):
            assert c.evidence, f"{name}: {c.status} needs evidence"
        if c.status == C.UNSUPPORTED:
            assert c.reason and c.enforced in ("component", "catalog", "first_evaluation"), name
        if c.status == C.EXPERIMENTAL:
            assert c.reason, f"{name}: say what is partial"
    for r in C.RULES:
        assert r.status in C.STATUSES, r.id
        if r.status in (C.SUPPORTED, C.EXPERIMENTAL):
            assert r.evidence, f"rule {r.id}: {r.status} needs evidence"
        if r.status == C.UNSUPPORTED:
            assert r.reason and r.enforced, f"rule {r.id}"
    for b in C.BENCHMARKS:
        assert b.kind in C.BENCHMARK_KINDS, b.id
        assert b.runtime.split()[0] in ("fast", "slow"), b.id


def test_evidence_tests_exist() -> None:
    """Every test node id named as evidence or benchmark is a test function in that file."""
    for test_id in sorted(set(C._iter_evidence(C.as_dict()))):
        path, _, fn = test_id.partition("::")
        f = REPO / path
        assert f.is_file(), f"{test_id}: no file {path}"
        assert re.search(rf"^def {re.escape(fn)}\(", f.read_text(encoding="utf-8"), re.M), f"{test_id}: no function"


def test_materials_are_synthetic_with_a_source() -> None:
    """No real material value without a traceable source in this slice (08 §3: do not invent data)."""
    for m in C.MATERIALS:
        assert m.kind == "synthetic", f"{m.id}: real material records need source, page, table and licence"
        assert m.source and m.units, m.id


def test_registered_does_not_mean_supported() -> None:
    """Combinations nobody tested stay NOT VERIFIED (e.g. robust compliance with the multigrid physics)."""
    assert C.status("elasticity_gmg", "robust_compliance").status == C.NOT_VERIFIED
    assert C.status("nonlinear_elasticity", "stress").status == C.NOT_VERIFIED
    assert C.status("elasticity", "volume").status == C.SUPPORTED


def test_generated_catalog_is_current() -> None:
    data = C.as_dict()
    j = DOCS / "vv_katalog.json"
    m = DOCS / "vv_katalog.md"
    stale = "docs/vv_katalog.* is stale: python -m toporlogy.configuration.catalog --write docs"
    assert json.loads(j.read_text(encoding="utf-8")) == data, stale
    assert m.read_text(encoding="utf-8") == C.render_markdown(data), stale
