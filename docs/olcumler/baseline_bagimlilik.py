"""Import graph of the core modules (00_MASTER "Başlangıç denetimi": Study, EvalContext, Domain, Elasticity,
ElasticityGMG, NonlinearElasticity, MultiPhaseSIMP, registry, references, T10, post).

Static (AST) imports between toporlogy modules, module level and function level (lazy) kept apart; plus every
``toporlogy`` module that imports one of the core modules. Prints JSON:

    python docs/olcumler/baseline_bagimlilik.py docs/olcumler/baseline_bagimlilik.json
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CORE = {
    "Study": "toporlogy.optimization.study", "EvalContext": "toporlogy.optimization.responses",
    "Domain": "toporlogy.core.domain", "Elasticity": "toporlogy.fem.elastic",
    "ElasticityGMG": "toporlogy.fem.elastic_gmg", "NonlinearElasticity": "toporlogy.fem.nonlinear",
    "MultiPhaseSIMP": "toporlogy.design.phases", "registry": "toporlogy.configuration.registry",
    "build": "toporlogy.configuration.build", "references": "toporlogy.references",
    "material": "toporlogy.core.material", "elements": "toporlogy.fem.elements",
    "T10 (verify)": "toporlogy.post.verify", "post pipeline": "toporlogy.post.pipeline",
    "post steps": "toporlogy.configuration.post_steps", "buckling": "toporlogy.fem.buckling",
    "modal": "toporlogy.fem.modal", "stress": "toporlogy.optimization.stress",
}


def module_name(p: Path) -> str:
    rel = p.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve(mod: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = mod.split(".")[: -node.level] if not mod.endswith("__init__") else mod.split(".")
    return ".".join(base + ([node.module] if node.module else []))


def imports(p: Path) -> tuple[set[str], set[str]]:
    """(module-level, inside functions) toporlogy imports of one file."""
    mod = module_name(p)
    tree = ast.parse(p.read_text(encoding="utf-8"))
    top: set[str] = set()
    lazy: set[str] = set()

    def visit(node: ast.AST, inside: bool) -> None:
        for ch in ast.iter_child_nodes(node):
            names: list[str] = []
            if isinstance(ch, ast.ImportFrom):
                base = resolve(mod, ch)
                names = [base] + [f"{base}.{a.name}" for a in ch.names]
            elif isinstance(ch, ast.Import):
                names = [a.name for a in ch.names]
            for n in names:
                if n.startswith("toporlogy"):
                    (lazy if inside else top).add(n)
            visit(ch, inside or isinstance(ch, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda))

    visit(tree, False)
    return top, lazy


def known(names: set[str], modules: set[str]) -> set[str]:
    """Reduce imported names to module names that exist (``a.b.C`` → ``a.b``)."""
    out = set()
    for n in names:
        while n and n not in modules:
            n = n.rpartition(".")[0]
        if n:
            out.add(n)
    return out


def main() -> int:
    files = sorted((SRC / "toporlogy").rglob("*.py"))
    modules = {module_name(p) for p in files}
    graph = {}
    for p in files:
        top, lazy = imports(p)
        m = module_name(p)
        graph[m] = {"module_level": sorted(known(top, modules) - {m}), "lazy": sorted(known(lazy, modules) - {m})}
    core = {}
    for label, m in CORE.items():
        core[label] = {"module": m, "imports": graph[m]["module_level"], "imports_lazy": graph[m]["lazy"],
                       "imported_by": sorted(k for k, v in graph.items() if m in v["module_level"]),
                       "imported_by_lazy": sorted(k for k, v in graph.items() if m in v["lazy"])}
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"git_sha": sha, "core": core, "graph": graph}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for label, row in core.items():
        print(f"{label:16s} imports {len(row['imports'])}+{len(row['imports_lazy'])} lazy, "
              f"imported by {len(row['imported_by'])}+{len(row['imported_by_lazy'])} lazy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
