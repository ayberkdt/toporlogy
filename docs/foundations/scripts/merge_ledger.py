"""Fold docs/foundations/ledger_additions/*.json into the strict-xfail ledger docs/vv_kutugu.json.

    python docs/foundations/scripts/merge_ledger.py
    python tests/toporlogy/test_catalog_ledger.py --write      # then re-render docs/vv_kutugu.md

Entries are keyed by nodeid: a new nodeid is appended; an existing one must be identical (a changed entry is an
error, not an overwrite). The addition files stay as the per-slice record.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "docs" / "vv_kutugu.json"
FIELDS = ("nodeid", "kind", "model_backend", "reason", "risk", "resolution", "record", "slow")

d = json.loads(LEDGER.read_text(encoding="utf-8"))
have = {e["nodeid"]: e for e in d["xfail"]}
added = 0
for f in sorted((ROOT / "docs" / "foundations" / "ledger_additions").glob("*.json")):
    for e in json.loads(f.read_text(encoding="utf-8")):
        if tuple(sorted(e)) != tuple(sorted(FIELDS)):
            raise SystemExit(f"{f.name}: entry {e.get('nodeid')} has fields {sorted(e)}, expected {sorted(FIELDS)}")
        old = have.get(e["nodeid"])
        if old is None:
            d["xfail"].append({k: e[k] for k in FIELDS})
            have[e["nodeid"]] = e
            added += 1
        elif {k: old[k] for k in FIELDS} != {k: e[k] for k in FIELDS}:
            raise SystemExit(f"{f.name}: {e['nodeid']} differs from the ledger entry; edit one place deliberately")
LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"added {added}, total {len(d['xfail'])}")
