"""Before / after measurement of the speed-ups (docs/olcumler/hizlandirma_onkayit.md): every row is profiled with
the base code and with this tree's code back to back, each in its own process (``hizlandirma_profil.py``).

    git archive 10daa03 src | tar -x -C <base>          # the base code
    python docs/olcumler/hizlandirma_olcum.py <base> docs/olcumler/hizlandirma_sonrasi.json [--rows small large]

The base run gets ``PYTHONPATH=<base>/src``, the new run ``PYTHONPATH=<repo>/src``; rows with ``new_only`` (the opt-in
Cholesky) run only with the new code. Results are appended to the output JSON (one row per run, ``code`` = "base" or
"new").
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "docs" / "olcumler" / "hizlandirma_profil.py"
CHOL = "physics.solver={type: direct, method: cholesky}"
PHYS = "physics.type=elasticity"          # examples without a physics section (mbb2d)

# (tag, example, overrides, iters, repeats, new_only)
ROWS = {
    "small": [(ex, ex, [], 5, 3, False) for ex in (
        "cantilever3d", "eigenfrequency2d", "lbracket_bore3d", "mbb2d", "multiload3d", "split_connectivity2d",
        "symmetric_am3d", "buckling_column2d", "inverter2d", "mbb_simpl2d", "stress_al_lbracket2d",
        "stress_lbracket2d")],
    "large": [
        ("mbb2d_300x100", "mbb2d", ["domain.shape=[300,100]"], 5, 3, False),
        ("cantilever3d_40x20x10", "cantilever3d", ["domain.shape=[40,20,10]"], 3, 2, False),
        ("multiload3d_40x16x12", "multiload3d", ["domain.shape=[40,16,12]"], 3, 2, False),
        ("mbb2d_600x200", "mbb2d", ["domain.shape=[600,200]"], 3, 2, False),
    ],
    "cholesky": [
        ("mbb2d_300x100_chol", "mbb2d", ["domain.shape=[300,100]", PHYS, CHOL], 5, 3, True),
        ("cantilever3d_40x20x10_chol", "cantilever3d", ["domain.shape=[40,20,10]", PHYS, CHOL], 3, 2, True),
        ("multiload3d_40x16x12_chol", "multiload3d", ["domain.shape=[40,16,12]", PHYS, CHOL], 3, 2, True),
        ("mbb2d_600x200_chol", "mbb2d", ["domain.shape=[600,200]", PHYS, CHOL], 3, 2, True),
        ("cantilever3d_chol", "cantilever3d", [PHYS, CHOL], 5, 3, True),
        ("multiload3d_chol", "multiload3d", [PHYS, CHOL], 5, 3, True),
        ("symmetric_am3d_chol", "symmetric_am3d", [PHYS, CHOL], 5, 3, True),
        ("mbb2d_chol", "mbb2d", [PHYS, CHOL], 5, 3, True),
    ],
}


def run(code: str, src: Path, out: Path, tag: str, ex: str, sets: list[str], iters: int, rep: int) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    tmp = out.with_suffix(f".{code}.tmp.json")
    tmp.unlink(missing_ok=True)
    cmd = [sys.executable, str(PROFILE), str(tmp), "--only", ex, "--tag", tag, "--iters", str(iters), "--repeats",
           str(rep)] + [a for s in sets for a in ("--set", s)]
    t = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)  # noqa: S603
    print(f"{code:4s} {p.stdout.strip()} ({time.perf_counter() - t:.0f} s)", flush=True)
    if p.returncode != 0:
        print(p.stderr[-3000:], flush=True)
    data = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"rows": [], "environments": {}}
    if tmp.is_file():
        part = json.loads(tmp.read_text(encoding="utf-8"))
        for r in part["rows"]:
            r["code"] = code
            data["rows"].append(r)
        data["environments"][code] = part["environment"]
        tmp.unlink()
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="folder holding the base code's src/")
    ap.add_argument("out")
    ap.add_argument("--rows", nargs="*", default=list(ROWS))
    ap.add_argument("--new-first", action="store_true", help="run the new code before the base code in every row "
                    "(a second pass in the other order shows how much the order and the load drift matter)")
    a = ap.parse_args()
    out = Path(a.out)
    for group in a.rows:
        for tag, ex, sets, iters, rep, new_only in ROWS[group]:
            order = ["new"] if new_only else (["new", "base"] if a.new_first else ["base", "new"])
            for code in order:
                src = REPO / "src" if code == "new" else Path(a.base) / "src"
                run(code, src, out, tag, ex, sets, iters, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
