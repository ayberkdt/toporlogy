"""The strict-xfail and skip ledger ``docs/vv_kutugu.json`` matches the test suite (08 §2: register, catch XPASS).

* every xfail marker in the suite is strict, so an unexpected pass (XPASS) fails the run;
* the ledger lists exactly the collected strict xfails (slow ones included), each with reason, model/backend, risk,
  resolution condition and an existing record;
* the ledger lists exactly the skip sites in the test sources (``pytest.skip``, ``importorskip``, ``skipif``);
* ``docs/vv_kutugu.md`` is the rendering of the JSON.

The xfail list comes from the conftest hook (``_verification.COLLECTED``, filled before ``-m`` deselection); when this
session did not collect every test file (e.g. only this file was selected), a ``--collect-only`` subprocess fills it.

    python tests/toporlogy/test_catalog_ledger.py --write       # re-render docs/vv_kutugu.md after editing the JSON
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

import _verification as V

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEDGER = REPO / "docs" / "vv_kutugu.json"
RENDERED = REPO / "docs" / "vv_kutugu.md"
FIELDS = ("nodeid", "kind", "model_backend", "reason", "risk", "resolution", "record", "slow")
SKIP_SITE = re.compile(r"pytest\.skip\(|importorskip\(|mark\.skipif\(|mark\.skip\(")
SELF = {"conftest.py", "_verification.py", Path(__file__).name}


def ledger() -> dict[str, Any]:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _test_files() -> set[str]:
    return {p.relative_to(REPO).as_posix() for p in HERE.glob("test_*.py")}


def collected() -> list[dict[str, Any]]:
    """Collected items of the whole suite with their markers (this session's, or a collect-only subprocess)."""
    have = {r["nodeid"].split("::")[0] for r in V.COLLECTED}
    if _test_files() <= have:
        return list(V.COLLECTED)
    with tempfile.TemporaryDirectory() as d:
        dump = Path(d) / "collected.json"
        env = dict(os.environ, TOPORLOGY_COLLECT_DUMP=str(dump))
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
        p = subprocess.run([sys.executable, "-m", "pytest", str(HERE), "--collect-only", "-q", "-p", "no:randomly",
                            "-p", "no:cacheprovider"], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        assert dump.is_file(), f"collection failed (exit {p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
        return list(json.loads(dump.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def items() -> list[dict[str, Any]]:
    return collected()


def test_every_xfail_is_strict(items: list[dict[str, Any]]) -> None:
    loose = [r["nodeid"] for r in items if any(not x["strict"] for x in r["xfail"])]
    assert not loose, f"non-strict xfail hides an XPASS; make it strict=True and add it to the ledger: {loose}"


def test_ledger_lists_exactly_the_collected_strict_xfails(items: list[dict[str, Any]]) -> None:
    found = {r["nodeid"] for r in items if r["xfail"]}
    listed = [e["nodeid"] for e in ledger()["xfail"]]
    assert len(listed) == len(set(listed)), "duplicate ledger entries"
    assert not found - set(listed), f"xfail tests missing from docs/vv_kutugu.json: {sorted(found - set(listed))}"
    assert not set(listed) - found, f"ledger entries that are no xfail test (fixed or renamed? remove or rename the " \
                                    f"entry): {sorted(set(listed) - found)}"


def test_ledger_entries_are_complete(items: list[dict[str, Any]]) -> None:
    data = ledger()
    slow = {r["nodeid"]: r["slow"] for r in items}
    for e in data["xfail"]:
        missing = [f for f in FIELDS if f not in e or (e[f] == "" and f != "slow")]
        assert not missing, f"{e.get('nodeid')}: missing {missing}"
        assert e["risk"] in data["risk_levels"], f"{e['nodeid']}: risk {e['risk']!r}"
        assert (REPO / e["record"]).is_file(), f"{e['nodeid']}: record {e['record']} does not exist"
        if e["nodeid"] in slow:
            assert e["slow"] == slow[e["nodeid"]], f"{e['nodeid']}: slow flag is {e['slow']}, the test's is " \
                                                   f"{slow[e['nodeid']]}"


def skip_sites() -> set[tuple[str, str]]:
    out = set()
    for p in sorted(HERE.glob("*.py")):
        if p.name in SELF:
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if SKIP_SITE.search(line):
                out.add((p.relative_to(REPO).as_posix(), line.strip()))
    return out


def test_ledger_lists_exactly_the_skip_sites() -> None:
    listed = {(s["file"], s["site"]) for s in ledger()["skips"]}
    found = skip_sites()
    assert not found - listed, f"skip sites missing from docs/vv_kutugu.json: {sorted(found - listed)}"
    assert not listed - found, f"ledger skip entries without that source line: {sorted(listed - found)}"


def render(data: dict[str, Any]) -> str:
    n_slow = sum(e["slow"] for e in data["xfail"])
    L = ["# Strict xfail ve atlama kütüğü (üretilmiş dosya; kaynak `docs/vv_kutugu.json`)", "",
         "`tests/toporlogy/test_catalog_ledger.py` bu kütüğü toplanan testlerle karşılaştırır: her xfail strict "
         "(XPASS koşuyu kırar), kütük toplanan strict xfail listesiyle ve test kaynaklarındaki atlama noktalarıyla "
         "birebir aynı. Beklenen başarısızlık PASS sayılmaz; atlanan bir arka ucun testi NOT VERIFIED'dır. "
         "Yeniden üretim: `python tests/toporlogy/test_catalog_ledger.py --write`.", "",
         f"Toplam {len(data['xfail'])} strict xfail ({n_slow} tanesi yavaş pakette), {len(data['skips'])} atlama "
         "noktası.", "", "Risk düzeyleri: " + "; ".join(f"**{k}** {v}" for k, v in data["risk_levels"].items()) + ".",
         "", "## 1. Strict xfail", "", "| Test | Tür | Model / arka uç | Neden | Risk | Çözüm koşulu | Kayıt | Yavaş |",
         "|:---|:---|:---|:---|:---|:---|:---|:---|"]
    for e in data["xfail"]:
        L.append(f"| `{e['nodeid'].removeprefix('tests/toporlogy/')}` | {e['kind']} | {e['model_backend']} | "
                 f"{e['reason']} | {e['risk']} | {e['resolution']} | {e['record']} | {'evet' if e['slow'] else '—'} |")
    L += ["", "## 2. Atlama noktaları", "", "| Dosya | Nokta | Arka uç | Neyi kapsar | Atlanınca |",
          "|:---|:---|:---|:---|:---|"]
    for s in data["skips"]:
        site = s["site"].replace("|", "\\|")
        L.append(f"| {s['file'].removeprefix('tests/toporlogy/')} | `{site}` | {s['backend']} | {s['covers']} | "
                 f"{s['status_when_skipped']} |")
    return "\n".join(L) + "\n"


def test_rendered_ledger_is_current() -> None:
    assert RENDERED.read_text(encoding="utf-8") == render(ledger()), \
        "docs/vv_kutugu.md is stale: python tests/toporlogy/test_catalog_ledger.py --write"


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit(f"usage: python {Path(__file__).name} --write")
    RENDERED.write_text(render(ledger()), encoding="utf-8", newline="\n")
    print(f"wrote {RENDERED}")


# ---------------------------------------------------------------------------------------------- risk coverage matrix
def test_every_test_file_is_classified() -> None:
    import _vv_layers as R
    files = {p.name for p in HERE.glob("test_*.py")}
    assert not files - set(R.FILE_LAYERS), f"add these test files to _vv_layers.FILE_LAYERS: {sorted(files - set(R.FILE_LAYERS))}"
    assert not set(R.FILE_LAYERS) - files, f"stale entries in _vv_layers.FILE_LAYERS: {sorted(set(R.FILE_LAYERS) - files)}"
    for f, layers in R.FILE_LAYERS.items():
        assert layers and set(layers) <= set(R.LAYERS), f


def test_rendered_risk_matrix_is_current() -> None:
    import _vv_layers as R
    assert R.OUT.read_text(encoding="utf-8") == R.render(), \
        "docs/vv_risk_kapsam.md is stale: python tests/toporlogy/_vv_layers.py --write"
