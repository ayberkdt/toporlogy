"""Risk coverage matrix (08 §1): test layers x library areas, generated from the test sources.

Each test file is classified by hand into the layers of 08 §1 (``FILE_LAYERS``); the library areas it exercises are
read from its imports (``toporlogy.<area>``). A cell counts the test files (and test functions) of that layer that
import that area. It is a map of where tests exist, not a measure of success: a parametrised test counts once, an
empty cell is an untested pair, and a full cell says nothing about the strength of its checks
(``tests/toporlogy/test_catalog_ledger.py::test_every_test_file_is_classified`` keeps the map complete).

    python tests/toporlogy/_vv_layers.py --write          # docs/vv_risk_kapsam.md
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "docs" / "vv_risk_kapsam.md"

LAYERS = {
    "U": "unit / property",
    "MP": "material point",
    "EL": "element patch / quadrature",
    "AS": "assembly / BC / reactions",
    "AN": "analytical / MMS",
    "IS": "independent solver / implementation",
    "OG": "optimization / KKT / gradient",
    "E2E": "end-to-end geometry",
    "IC": "integration / compatibility",
    "PF": "performance",
}
AREAS = ("core", "geometry", "boundary", "design", "fem", "solvers", "optimization", "optimizers", "configuration",
         "execution", "post", "references")

FILE_LAYERS: dict[str, tuple[str, ...]] = {
    "test_aggregates.py": ("U", "OG"), "test_api_surface.py": ("IC",), "test_architecture.py": ("IC",),
    "test_baseline.py": ("IC",), "test_baseline_policy.py": ("IC",), "test_bc.py": ("AS",),
    "test_benchmark.py": ("OG",), "test_buckling.py": ("EL", "AN", "OG"), "test_buckling_physics.py": ("AN", "OG"),
    "test_catalog.py": ("IC",), "test_catalog_ledger.py": ("IC",), "test_cli.py": ("IC",),
    "test_closed_forms.py": ("U", "AN"), "test_compatibility.py": ("IC",), "test_compose.py": ("U", "OG"),
    "test_config.py": ("IC",), "test_connectivity.py": ("U", "OG", "E2E"), "test_core.py": ("U", "EL", "AS"),
    "test_deriving_methods.py": ("OG", "IC"), "test_element_orientation.py": ("MP", "EL", "OG", "E2E"),
    "test_fault_injection.py": ("IC",), "test_fem_mms.py": ("AN",), "test_fem_mms_extended.py": ("AN",),
    "test_frame_invariance.py": ("U", "MP"), "test_fs2019_column.py": ("AN", "IS"), "test_gcmma.py": ("OG",),
    "test_geometry.py": ("U", "E2E"), "test_gmg.py": ("IS", "OG", "PF"), "test_gradients.py": ("OG",),
    "test_heat.py": ("EL", "AN", "OG"), "test_independent_buckling.py": ("IS",),
    "test_independent_matrices.py": ("IS", "EL"), "test_independent_stress.py": ("IS", "MP"),
    "test_independent_vv.py": ("IS", "AN"), "test_invariants.py": ("U",), "test_kinematic_contract.py": ("AS",),
    "test_kkt_estimate.py": ("OG",), "test_lengthscale.py": ("U", "OG"), "test_manufacturing.py": ("U", "OG"),
    "test_material.py": ("MP", "EL"), "test_material_config.py": ("MP", "IC"), "test_mechanisms.py": ("AS",),
    "test_merged_physics.py": ("IC",), "test_method_components.py": ("IC",), "test_mma.py": ("OG",),
    "test_modal.py": ("EL", "AN", "OG"), "test_multi_material.py": ("MP", "AN", "OG"),
    "test_nonlinear.py": ("MP", "AN", "OG"), "test_nonlinear_orientation.py": ("MP", "OG"),
    "test_onebody.py": ("OG", "E2E"), "test_ordering.py": ("AS", "PF"), "test_pde_filter.py": ("U", "OG"),
    "test_post_cantilever.py": ("E2E",), "test_post_checks.py": ("E2E",), "test_post_pipeline.py": ("E2E",),
    "test_post_sdf.py": ("U", "E2E"), "test_post_verify.py": ("EL", "E2E"), "test_published_values.py": ("IS", "AN"),
    "test_registry_features.py": ("IC",), "test_regression.py": ("IS", "OG"), "test_research_examples.py": ("IC",),
    "test_residual.py": ("AS",), "test_result_contract.py": ("IC", "OG"), "test_robust.py": ("OG",),
    "test_sdf_contract.py": ("E2E",), "test_simpl.py": ("OG",), "test_skfem_crosscheck.py": ("IS", "EL"),
    "test_strength.py": ("MP",), "test_stress.py": ("AN", "OG"), "test_study_options.py": ("IC",),
    "test_sweep_connectivity.py": ("E2E",), "test_sweep_post.py": ("E2E",), "test_symmetric_ties.py": ("E2E",),
    "test_symmetry.py": ("U", "E2E"), "test_t10_convergence.py": ("AN", "E2E"), "test_taylor_remainder.py": ("OG",),
    "test_timoshenko_cantilever.py": ("AN",), "test_anisotropy.py": ("U", "MP"), "test_linear_elastic.py": ("U", "MP"),
    "test_tensor.py": ("U",), "test_stress_recovery.py": ("EL", "AN"), "test_stress_singularity.py": ("AN",),
    "test_stress_lbracket.py": ("OG", "E2E"),
    "test_stress_zero_density.py": ("U", "OG"),
    "test_vtu.py": ("U", "IC"),
    "test_units.py": ("U", "IC"),
    "test_material_catalog.py": ("MP", "IC"),
    "test_homogenization.py": ("AN", "OG"),
    "test_thermoelastic.py": ("AN", "AS", "OG"),
    "test_mass_properties.py": ("U", "AN", "OG"),
    "test_implicit.py": ("U", "OG"),
    "test_mode_tracking.py": ("U", "AN"),
    "test_sweep_resume.py": ("IC",),
    "test_tsai_wu_admissibility.py": ("MP",),
    "test_auxetic_cell.py": ("OG", "E2E"),
    "test_laminate.py": ("U", "MP", "AN"),
    "test_uq.py": ("U", "AN", "OG"),
    "test_pareto.py": ("AN", "OG"),
    "test_identification.py": ("MP", "OG"),
    "test_rom.py": ("IS", "AN"),
}


def areas_of(path: Path) -> set[str]:
    """Library areas a test file imports (``toporlogy.<area>``; ``import toporlogy`` alone counts for none)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        for n in names:
            parts = n.split(".")
            if parts[0] == "toporlogy" and len(parts) > 1 and parts[1] in AREAS:
                out.add(parts[1])
    return out


def n_tests(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))


def matrix() -> dict[str, dict[str, tuple[int, int]]]:
    """layer -> area -> (test files, test functions)."""
    m = {lay: {a: (0, 0) for a in AREAS} for lay in LAYERS}
    for p in sorted(HERE.glob("test_*.py")):
        layers = FILE_LAYERS[p.name]
        k = n_tests(p)
        for a in areas_of(p):
            for lay in layers:
                f, t = m[lay][a]
                m[lay][a] = (f + 1, t + k)
    return m


def render() -> str:
    m = matrix()
    L = ["# Risk kapsam matrisi: test katmanı × kütüphane alanı (üretilmiş dosya)", "",
         "Kaynak: `tests/toporlogy/_vv_layers.py` (dosya → katman sınıflaması elle, dosya → alan içe aktarmalardan). "
         "Yeniden üretim: `python tests/toporlogy/_vv_layers.py --write`. Hücre: o katmandaki ve o alanı içe aktaran "
         "test dosyası sayısı / test fonksiyonu sayısı. Bu bir harita, başarı ölçütü değil: parametreli test bir "
         "sayılır, dolu hücre denetimin gücünü söylemez, boş hücre (—) o çiftte test olmadığını söyler.", "",
         "| Katman | " + " | ".join(AREAS) + " |", "|:---|" + "---:|" * len(AREAS)]
    for lay, name in LAYERS.items():
        cells = [f"{f}/{t}" if f else "—" for f, t in (m[lay][a] for a in AREAS)]
        L.append(f"| {lay} {name} | " + " | ".join(cells) + " |")
    L += ["", "Boş hücreler (açık risk adayı; her biri bir risk değildir, örn. `references` için çoğu katman anlamsız):",
          ""]
    for lay, name in LAYERS.items():
        empty = [a for a in AREAS if not m[lay][a][0]]
        if empty:
            L.append(f"- **{lay}** ({name}): {', '.join(empty)}")
    L += ["", "Katman sınıflaması (dosya → katmanlar):", ""]
    for f, lays in sorted(FILE_LAYERS.items()):
        L.append(f"- `{f}`: {', '.join(lays)}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit(f"usage: python {Path(__file__).name} --write")
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
