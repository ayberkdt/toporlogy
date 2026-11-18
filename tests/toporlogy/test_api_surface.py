"""G0: the public API and the component listing must survive the migration (docs/gelistirme_plani.md §5.1, §5.2).

``baseline/api_surface.json``: for every public toporlogy module, its public names and what they are (the
``__qualname__`` of classes and functions, ``<type>`` for other objects). A move keeps the old module path working
(thin re-export) and the object the same, so every recorded name must still import from its recorded module and
keep its fingerprint; new names and new class attributes are allowed. ``baseline/list.txt``: the exact output of ``python -m toporlogy list``.

    python tests/toporlogy/test_api_surface.py --write          # after an intended API change
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SURFACE = HERE / "baseline" / "api_surface.json"
LISTING = HERE / "baseline" / "list.txt"


def _public_modules() -> list[str]:
    import toporlogy
    names = ["toporlogy"]
    for m in pkgutil.walk_packages(toporlogy.__path__, "toporlogy."):
        if any(part.startswith("_") for part in m.name.split(".")[1:]):
            continue                                        # private modules (_mma_core, _gmsh_worker, references._*)
        names.append(m.name)
    return sorted(names)


def _describe(obj: Any) -> Any:
    """A fingerprint that survives a move but tells two same-named objects apart (geometry.Shape vs
    post.primitives.Shape): function -> "qualname(parameter names)"; class -> {"class": qualname, "attrs": public
    attribute names}; data -> "<type>". None for modules and objects defined outside toporlogy."""
    if inspect.ismodule(obj):
        return None
    if inspect.isclass(obj) or inspect.isfunction(obj):
        if not (getattr(obj, "__module__", "") or "").startswith("toporlogy"):
            return None
        if inspect.isclass(obj):
            return {"class": obj.__qualname__, "attrs": sorted(a for a in dir(obj) if not a.startswith("_"))}
        try:
            params = ",".join(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            params = "?"
        return f"{obj.__qualname__}({params})"
    return f"<{type(obj).__name__}>"


def _compatible(want: Any, got: Any) -> bool:
    """Same function signature / data type; for a class the same qualname and every recorded attribute still there
    (new methods are additions, not breaks; a missing one or another class behind the name is a break)."""
    if isinstance(want, dict):
        return isinstance(got, dict) and got["class"] == want["class"] and set(want["attrs"]) <= set(got["attrs"])
    return got == want


def surface() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _public_modules():
        try:
            mod = importlib.import_module(name)
        except ImportError as e:                            # optional extra missing: recorded, compared only if present
            out[name] = {"__optional__": f"{type(e).__name__}: {e}"}
            continue
        public = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]
        entries = {}
        for n in sorted(public):
            d = _describe(getattr(mod, n, None))
            if d is not None:
                entries[n] = d
        out[name] = entries
    return out


def listing() -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    p = subprocess.run([sys.executable, "-m", "toporlogy", "list"], env=env, capture_output=True, text=True,
                       encoding="utf-8", timeout=300)
    assert p.returncode == 0, p.stderr
    return p.stdout.replace("\r\n", "\n")


def _load(path: Path) -> str:
    # the snapshots are tracked in git: a missing file is a deleted safety net, not a reason to skip
    assert path.is_file(), f"{path} is missing: restore it from git or run python {Path(__file__).name} --write"
    return path.read_text(encoding="utf-8")


def test_recorded_public_names_still_import_and_are_the_same_objects() -> None:
    stored = json.loads(_load(SURFACE))
    problems = []
    for name, entries in stored.items():
        if "__optional__" in entries:
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            problems.append(f"{name}: no longer importable ({e})")
            continue
        for n, want in entries.items():
            if not hasattr(mod, n):
                problems.append(f"{name}.{n}: missing")
                continue
            got = _describe(getattr(mod, n))
            if not _compatible(want, got):
                if isinstance(want, dict) and isinstance(got, dict) and got["class"] == want["class"]:
                    got = {"missing attrs": sorted(set(want["attrs"]) - set(got["attrs"]))}
                problems.append(f"{name}.{n}: was {want if not isinstance(want, dict) else want['class']}, now {got}")
    assert not problems, "\n".join(problems)


def test_component_listing_unchanged() -> None:
    want = _load(LISTING)
    got = listing()
    if got != want:
        import difflib
        diff = "".join(difflib.unified_diff(want.splitlines(True), got.splitlines(True), "list.txt", "now", n=1))
        pytest.fail(f"python -m toporlogy list changed:\n{diff[:4000]}")


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit(f"usage: python {Path(__file__).name} --write")
    SURFACE.parent.mkdir(exist_ok=True)
    SURFACE.write_text(json.dumps(surface(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    LISTING.write_text(listing(), encoding="utf-8", newline="\n")
    print(f"wrote {SURFACE} and {LISTING}")


# Submodules that were attributes of the package right after `import toporlogy` before the G5 moves (the imports of
# toporlogy/__init__.py at ff1db52): `import toporlogy; toporlogy.study.Study` must keep working.
PACKAGE_MODULE_ATTRS = ("solvers", "chain", "checks", "domain", "fem", "interp", "optim", "record", "responses",
                        "study", "bc", "connectivity", "dynamics", "geometry", "lengthscale", "manufacturing", "robust")


def test_old_submodules_are_attributes_after_import_toporlogy() -> None:
    """In a fresh process (other tests' imports would set the attributes): each name above is an attribute of the
    package and is the module that ``import toporlogy.<name>`` gives (for G5 aliases: the moved module)."""
    code = "\n".join([
        "import importlib",
        "import toporlogy as T",
        f"names = {PACKAGE_MODULE_ATTRS!r}",
        "missing = [n for n in names if not hasattr(T, n)]",
        "wrong = [n for n in names if hasattr(T, n) and getattr(T, n) is not importlib.import_module(f'toporlogy.{n}')]",
        "print(missing, wrong)",
    ])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "[] []", f"(missing, not the same module): {p.stdout.strip()}"
