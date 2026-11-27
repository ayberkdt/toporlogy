"""Render docs/foundations/STATUS.md from docs/foundations/status.json (the single source of the live tracker).

    python docs/foundations/scripts/status.py            # refresh head/branch/updated_at and render
    python docs/foundations/scripts/status.py --check    # exit 1 if STATUS.md is stale

The JSON is written atomically (temp file + rename). Actual HEAD and branch are read from git, the time from the
clock; nothing here is typed by hand.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JS = ROOT / "docs" / "foundations" / "status.json"
MD = ROOT / "docs" / "foundations" / "STATUS.md"


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def render(s: dict) -> str:
    L = ["# Foundations II — live status", "",
         f"Actual HEAD: `{s['head']}`  ", f"Branch / dirty state: `{s['branch']}` / {s['dirty']}  ",
         f"Updated at: {s['updated_at']}  ", f"Environment / resource budget: {s['environment']}  ",
         f"Last successful integrated checkpoint: {s['last_verified_checkpoint']}  ",
         f"Current unit: {s['current']}", "",
         "| ID | Priority | Dependencies | State | Evidence / SHA | Blocker / next action |", "|---|---|---|---|---|---|"]
    for u in s["units"]:
        L.append(f"| {u['id']} | {u['priority']} | {', '.join(u['depends_on']) or '—'} | {u['state']} | "
                 f"{u.get('evidence') or '—'} | {u.get('next_action') or '—'} |")
    L += ["", "## Next executable units", ""] + [f"{i + 1}. {t}" for i, t in enumerate(s["next"])]
    L += ["", "## Historical failed hypotheses and unresolved risks", ""] + [f"- {t}" for t in s["failed"]]
    L += ["", "## Changes to public API / physical conventions", ""] + [f"- {t}" for t in s["api_changes"]]
    L += ["", "## Tests actually run (with exit code)", ""] + [f"- {t}" for t in s["runs"]]
    L += ["", "## Resource and permission blockers", ""] + [f"- {t}" for t in s["blockers"]]
    return "\n".join(L) + "\n"


def main() -> int:
    s = json.loads(JS.read_text(encoding="utf-8"))
    if "--check" in sys.argv:
        return 0 if MD.read_text(encoding="utf-8") == render(s) else 1
    s["head"] = _git("rev-parse", "--short", "HEAD")
    s["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    s["dirty"] = "dirty" if _git("status", "--porcelain") else "clean"
    s["updated_at"] = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    tmp = JS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, JS)
    MD.write_text(render(s), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
