"""MB1-MB5: KUBC and SUBC on a general-mesh cell around the periodic value (docs/foundations/prereg/MB_mesh_bounds.md,
thresholds committed in c57ce8d before this file was written). Synthetic, dimensionless."""
from __future__ import annotations

import numpy as np
import pytest

from test_mesh_homogenization import NU, _grid_match, rel, skew2

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.homogenization import homogenize_elasticity
from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.fem.mesh.mesh import rectangle

D = D_plane_stress(NU)


@pytest.mark.parametrize("case", ["skew_T6", "square_Q4"])
def test_mb1_homogeneous_cell(case: str) -> None:
    m, L = skew2("T6") if case == "skew_T6" else (rectangle(4, 3, 1.0, 0.8, "Q4"), np.diag([1.0, 0.8]))
    for bc in ("kubc", "subc"):
        h = homogenize_mesh_elasticity(m, D, L, bc=bc)
        print(f"\n  MB1 {case} {bc}: {rel(h.C, D):.1e}")
        assert rel(h.C, D) <= 1e-10


def _cells():
    m, L = skew2("T6", (4, 4))
    E = np.random.default_rng(5).uniform(0.2, 2.0, m.n_cells)
    m2 = rectangle(10, 10, 1.0, 1.0, "T3")
    E2 = np.ones(m2.n_cells)
    E2[np.random.default_rng(2).permutation(m2.n_cells)[: m2.n_cells // 2]] = 10.0
    return [("skew_T6", m, L, E), ("two_phase_T3", m2, np.eye(2), E2)]


@pytest.mark.parametrize("i", [0, 1])
def test_mb2_mb3_loewner_order(i: int) -> None:
    name, m, L, E = _cells()[i]
    C = {bc: homogenize_mesh_elasticity(m, D, L, E, bc=bc).C for bc in ("subc", "periodic", "kubc")}
    s = float(np.abs(C["periodic"]).max())
    lo = float(np.linalg.eigvalsh(C["periodic"] - C["subc"]).min())
    hi = float(np.linalg.eigvalsh(C["kubc"] - C["periodic"]).min())
    gap = float(np.linalg.eigvalsh(C["kubc"] - C["subc"]).min())
    print(f"\n  MB2/3 {name}: PBC-SUBC {lo / s:.3e}, KUBC-PBC {hi / s:.3e}, KUBC-SUBC {gap / s:.3e}")
    assert lo >= -1e-10 * s and hi >= -1e-10 * s
    assert gap >= 1e-6 * s


def test_mb4_against_the_regular_grid() -> None:
    m, d, idx = _grid_match(12, 10)
    Eg = np.random.default_rng(3).uniform(0.1, 2.0, d.n_active)
    Em = np.empty(m.n_cells)
    Em[idx] = Eg
    for bc in ("kubc", "subc"):
        g = homogenize_elasticity(d, Eg, NU, bc=bc)
        h = homogenize_mesh_elasticity(m, D, np.diag([12.0, 10.0]), Em, bc=bc)
        print(f"\n  MB4 {bc}: {rel(h.C, g.C):.1e}")
        assert rel(h.C, g.C) <= 1e-10


def test_mb5_refusals() -> None:
    m, L = skew2("Q4")
    with pytest.raises(ValueError, match="bc must be"):
        homogenize_mesh_elasticity(m, D, L, bc="mixed")
    for bc in ("kubc", "subc"):
        with pytest.raises(ValueError, match="periodic"):
            homogenize_mesh_elasticity(m, D, L, bc=bc, sensitivities=True)


# ---------------------------------------------------------------------------------------------- MB6
@pytest.mark.parametrize("element", ["H8", "T4"])
def test_mb6_three_dimensional_bounds(element: str) -> None:
    from test_mesh_homogenization import skew3

    from toporlogy.fem.elements import D_iso
    m, L = skew3(element)
    D3 = D_iso(NU)
    for bc in ("kubc", "subc"):
        h = homogenize_mesh_elasticity(m, D3, L, bc=bc)
        print(f"\n  MB6 {element} homogeneous {bc}: {rel(h.C, D3):.1e}")
        assert rel(h.C, D3) <= 1e-10
    E = np.random.default_rng(4).uniform(0.2, 2.0, m.n_cells)
    C = {bc: homogenize_mesh_elasticity(m, D3, L, E, bc=bc).C for bc in ("subc", "periodic", "kubc")}
    s = float(np.abs(C["periodic"]).max())
    lo = float(np.linalg.eigvalsh(C["periodic"] - C["subc"]).min())
    hi = float(np.linalg.eigvalsh(C["kubc"] - C["periodic"]).min())
    print(f"  MB6 {element} random: PBC-SUBC {lo / s:.3e}, KUBC-PBC {hi / s:.3e}")
    assert lo >= -1e-10 * s and hi >= -1e-10 * s
