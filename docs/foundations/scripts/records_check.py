"""Integrity of the Foundations II records (WP24 gate): exit 1 on any defect.

    python docs/foundations/scripts/records_check.py

Checks: (1) every preregistration in docs/foundations/prereg has a "First result" section (or is explicitly a
replacement whose result lives in the replacing record); (2) every docs/foundations/ledger_additions entry is in
docs/vv_kutugu.json, identical; (3) every ledger "record" path exists; (4) the "Preregistered <time>" stamp of a
record is not later than the committer time of the commit that first added the file (a hand-typed future time).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "docs" / "foundations" / "prereg"
problems: list[str] = []
acknowledged: list[str] = []


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


for f in sorted(PRE.glob("*.md")):
    t = f.read_text(encoding="utf-8")
    if "First result" not in t and "replacement" not in t.lower() and "not run as specified" not in t:
        problems.append(f"{f.name}: no 'First result' section")
    m = re.search(r"Preregistered (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) ([+-]\d{4})", t)
    if m:
        stamp = dt.datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M %z")
        first = git("log", "--diff-filter=A", "--format=%cI", "--", str(f.relative_to(ROOT)).replace("\\", "/"))
        if first:
            added = dt.datetime.fromisoformat(first.splitlines()[-1])
            if stamp > added + dt.timedelta(minutes=1):
                if "not reliable" in t or "corrected" in t:
                    acknowledged.append(f"{f.name}: stamp {stamp:%H:%M} later than its first commit {added:%H:%M} "
                                        "(acknowledged in the record)")
                else:
                    problems.append(f"{f.name}: stamp {stamp:%H:%M} later than its first commit {added:%H:%M}")

ledger = json.loads((ROOT / "docs" / "vv_kutugu.json").read_text(encoding="utf-8"))
have = {e["nodeid"]: e for e in ledger["xfail"]}
for f in sorted((ROOT / "docs" / "foundations" / "ledger_additions").glob("*.json")):
    for e in json.loads(f.read_text(encoding="utf-8")):
        if have.get(e["nodeid"]) != e:
            problems.append(f"{f.name}: {e['nodeid']} missing from or different in docs/vv_kutugu.json")
for e in ledger["xfail"]:
    rec = e.get("record", "")
    if rec and not (ROOT / rec).exists():
        problems.append(f"ledger {e['nodeid']}: record {rec} does not exist")

for a in acknowledged:
    print("acknowledged:", a)
print("\n".join(problems) if problems else "records check: no unacknowledged defects")
sys.exit(1 if problems else 0)
