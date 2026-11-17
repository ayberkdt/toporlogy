"""G0: layer rules for internal imports and the PARDISO-before-torch import order (docs/gelistirme_plani.md §4.3,
§5.1-§5.2).

The internal import graph is read from the source with ``ast`` (module-level and function-level imports). Each
module belongs to a layer by name prefix; old and new (post-migration) names map to the same layer, so the rules
survive the moves. A layer may import itself and the layers listed for it. Known violations are listed in
``EXCEPTIONS`` and must shrink, never grow.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "toporlogy"

# prefix (relative to toporlogy) -> layer; the longest matching prefix wins
LAYER_OF = {
    "references": "base", "domain": "base", "core": "base",
    "geometry": "geometry",
    "solvers": "solvers",
    "fem": "physics", "physics": "physics",
    "chain": "design", "interp": "design", "manufacturing": "design", "design": "design",
    "boundary": "boundary",
    "responses": "optimization", "study": "optimization", "robust": "optimization", "lengthscale": "optimization",
    "connectivity": "optimization", "dynamics": "optimization", "checks": "optimization", "optim": "optimization",
    "optimization": "optimization", "record": "optimization",
    "post": "postprocessing", "postprocessing": "postprocessing", "verification": "postprocessing",
    "configuration": "configuration",
    "registry": "compat", "config": "compat", "bc": "compat",  # compatibility modules after G2-G4 (re-exports)
    "cli": "execution", "__main__": "execution", "execution": "execution",
    "": "package",                                          # toporlogy/__init__.py
}
ALLOWED = {
    "base": set(),
    "geometry": {"base"},
    "solvers": {"base"},
    "physics": {"base", "solvers"},
    "design": {"base"},
    "boundary": {"base", "geometry", "physics", "solvers", "optimization"},   # bc: BodyForceCompliance is a response
    "optimization": {"base", "geometry", "design", "physics", "solvers", "boundary"},
    "postprocessing": {"base", "geometry", "physics", "solvers"},
    "configuration": {"base", "geometry", "solvers", "physics", "design", "boundary", "optimization",
                      "postprocessing", "package"},
    "execution": {"base", "geometry", "solvers", "physics", "design", "boundary", "optimization", "postprocessing",
                  "configuration", "package", "compat"},
    # old module paths kept for users: they may re-export anything, but inside the package only the top layer may
    # still import them (new code imports the new modules)
    "compat": {"base", "geometry", "solvers", "physics", "design", "boundary", "optimization", "postprocessing",
               "configuration", "execution", "package"},
    "package": {"base", "geometry", "solvers", "physics", "design", "boundary", "optimization", "postprocessing",
                "compat"},                                  # toporlogy.bc stays an attribute of the package
}
# (importer, imported) known violations; docs/gelistirme_plani.md §4.3. Remove entries as they are fixed.
EXCEPTIONS = {
    ("fem.modal", "dynamics"),                              # physics -> optimization (lazy, eigenvalue cache)
}


def _module(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def layer(mod: str) -> str:
    best = ""
    for prefix in LAYER_OF:
        if prefix and (mod == prefix or mod.startswith(prefix + ".")) and len(prefix) > len(best):
            best = prefix
    if not best and mod:
        raise KeyError(f"module {mod!r} has no layer: add its prefix to LAYER_OF")
    return LAYER_OF[best]


def _files() -> list[Path]:
    return [p for p in sorted(SRC.rglob("*.py")) if "__pycache__" not in p.parts]


def edges() -> set[tuple[str, str]]:
    """(importer, imported) module pairs inside toporlogy, names relative to the package ("" = toporlogy)."""
    modules = {_module(p) for p in _files()}

    def strip(name: str) -> str | None:
        if name == "toporlogy" or name.startswith("toporlogy."):
            return name.removeprefix("toporlogy").lstrip(".")
        return None

    out = set()
    for p in _files():
        mod = _module(p)
        pkg = mod if p.name == "__init__.py" else mod.rpartition(".")[0]
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [t for t in (strip(a.name) for a in node.names) if t is not None]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".") if pkg else []
                    base = base[:len(base) - node.level + 1]
                    full: str | None = ".".join(base + ([node.module] if node.module else []))
                else:
                    full = strip(node.module or "")
                if full is None:
                    continue
                for a in node.names:                        # "from X import y": y is a module or a name in X
                    sub = f"{full}.{a.name}" if full else a.name
                    targets.append(sub if sub in modules else full)
            out.update((mod, t) for t in targets if t != mod)
    return out


def test_every_module_has_a_layer() -> None:
    for p in _files():
        layer(_module(p))


def test_internal_imports_follow_the_layer_rules() -> None:
    bad = []
    for a, b in sorted(edges()):
        la, lb = layer(a), layer(b)
        if la != lb and lb not in ALLOWED[la] and (a, b) not in EXCEPTIONS:
            bad.append(f"{a} ({la}) -> {b} ({lb})")
    assert not bad, "imports against the layer rules (docs/gelistirme_plani.md §4.3):\n" + "\n".join(bad)


def test_exceptions_are_still_needed() -> None:
    present = edges()
    stale = sorted(e for e in EXCEPTIONS if e not in present)
    assert not stale, f"fixed, remove from EXCEPTIONS: {stale}"


def test_importing_toporlogy_does_not_load_torch_before_pardiso() -> None:
    """__init__ loads MKL PARDISO before torch (two libiomp5md.dll copies abort the process otherwise). In a fresh
    process, importing toporlogy must never leave the loader saying that torch came first."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    code = "import toporlogy; from toporlogy.solvers import pardiso; print(pardiso.state()['reason'])"
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert not p.stdout.strip().startswith("torch already imported"), p.stdout
