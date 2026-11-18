"""Layer rules for internal imports, no alias names, and the PARDISO-before-torch import order
(docs/gelistirme_plani.md §4.3, §5.1-§5.2).

The internal import graph is read from the source with ``ast`` (module-level and function-level imports). Each
module belongs to a layer by name prefix. A layer may import itself and the layers listed for it. Known violations
are listed in ``EXCEPTIONS`` and must shrink, never grow.

Every module, class and YAML component has one name: no module that stands in for another (``sys.modules``
aliases or re-export shims), no second name bound to a class, no constructor registered under two names.
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
    "references": "base", "core": "base",
    "geometry": "geometry",
    "solvers": "solvers",
    "fem": "physics",
    "design": "design",
    "boundary": "boundary",
    "optimizers": "optimization", "optimization": "optimization",
    "post": "postprocessing",
    "configuration": "configuration",
    "__main__": "execution", "execution": "execution",
    "": "package",                                          # toporlogy/__init__.py
}
ALLOWED = {
    "base": set(),
    "geometry": {"base"},
    "solvers": {"base"},
    "physics": {"base", "solvers"},
    "design": {"base"},
    "boundary": {"base", "geometry", "physics", "solvers", "optimization"},   # BodyForceCompliance is a response
    "optimization": {"base", "geometry", "design", "physics", "solvers", "boundary"},
    "postprocessing": {"base", "geometry", "physics", "solvers"},
    "configuration": {"base", "geometry", "solvers", "physics", "design", "boundary", "optimization",
                      "postprocessing", "package"},
    "execution": {"base", "geometry", "solvers", "physics", "design", "boundary", "optimization", "postprocessing",
                  "configuration", "package"},
    "package": {"base", "solvers", "physics", "design", "optimization", "execution"},   # flattens the core names
}
# (importer, imported) known violations; docs/gelistirme_plani.md §4.3. Remove entries as they are fixed.
EXCEPTIONS = {
    ("fem.modal", "optimization.dynamics"),                 # physics -> optimization (lazy, eigenvalue cache)
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


# ---------------------------------------------------------------------------------------------- one name each
def test_no_module_stands_in_for_another() -> None:
    """No ``sys.modules[__name__] = ...`` alias modules and no module that only re-exports another one."""
    bad = []
    for p in _files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) and ast.unparse(t.value) == "sys.modules"
                                                    for t in node.targets):
                bad.append(f"{_module(p)}: assigns sys.modules")
        body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        if p.name != "__init__.py" and body and all(isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                                                    for n in body):
            names = {t.id for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            if names <= {"__all__"}:
                bad.append(f"{_module(p)}: only imports (a re-export shim)")
    assert not bad, bad


def test_no_second_name_for_a_class_or_function() -> None:
    """``Sphere = Ball`` at module level, or ``from x import Implicit as ImplicitShape``: the same object under two
    public names. Module aliases (``import numpy as np``) and private names (``as _x``) are not names of the API."""
    bad = []
    for p in _files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        defined = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for n in tree.body:
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name) and n.value.id in defined):
                bad.append(f"{_module(p)}: {n.targets[0].id} = {n.value.id}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.asname and not a.asname.startswith("_") and a.asname != a.name and a.name[:1].isupper():
                        bad.append(f"{_module(p)}: {a.name} as {a.asname}")
    assert not bad, bad


def test_no_component_has_two_registry_names() -> None:
    """A YAML type name is the only name of its constructor: ``ball`` / ``sphere`` / ``circle`` for one function
    made three spellings of one component."""
    import collections

    from toporlogy.configuration import REGISTRY
    names = collections.defaultdict(list)
    for kind, entries in REGISTRY._entries.items():
        for name, e in entries.items():
            names[(kind, e.target if isinstance(e.target, str) else id(e.target))].append(name)
    shared = [(kind, v) for (kind, _), v in names.items() if len(v) > 1]
    assert not shared, shared


def test_no_public_name_is_defined_in_two_modules() -> None:
    """One concept, one name: two public classes or functions with the same name in different modules (two
    ``Cylinder``, two ``read_stl``, two ``volume``) are either one implementation written twice or two concepts that
    need different names. Private modules (``_mma_core``, ``_gmsh_worker``, ``references._*``) do not count."""
    homes: dict[str, list[str]] = {}
    for p in _files():
        mod = _module(p)
        if any(part.startswith("_") for part in mod.split(".")):
            continue
        for n in ast.parse(p.read_text(encoding="utf-8")).body:
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and not n.name.startswith("_"):
                homes.setdefault(n.name, []).append(mod)
    twice = {name: mods for name, mods in homes.items() if len(mods) > 1}
    assert not twice, twice


def test_every_signed_distance_solid_is_a_geometry_shape() -> None:
    """One hierarchy for one concept: every class in the package with an ``sdf`` method derives from
    ``toporlogy.geometry.Shape`` (post primitives used to have a second, unrelated base class)."""
    import importlib
    import inspect
    import pkgutil

    import toporlogy
    from toporlogy.geometry import Shape
    bad = []
    for m in pkgutil.walk_packages(toporlogy.__path__, "toporlogy."):
        if any(part.startswith("_") for part in m.name.split(".")[1:]):
            continue                                         # __main__ runs the CLI; private modules
        try:
            mod = importlib.import_module(m.name)
        except ImportError:                                  # optional extra missing
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__ and "sdf" in vars(cls) and not issubclass(cls, Shape):
                bad.append(f"{cls.__module__}.{cls.__qualname__}")
    assert not bad, bad
