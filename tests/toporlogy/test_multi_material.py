"""Several solid phases + void (toporlogy.design.phases): the checks pre-registered in docs/olcumler/cok_malzeme.md §0
(2026-11-25 17:56:53 +0300, before any of them ran). Ids V1a … V7 as in that table.

Results are printed (``-s``) and copied into docs/olcumler/cok_malzeme.md / .json by docs/olcumler/cok_malzeme.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.design.phases import MultiPhaseSIMP, Phase, phase_labels, phase_report, phase_start
from toporlogy.optimization.responses import Cost, Mass, PhaseVolume
from toporlogy.optimizers.gcmma import GCMMA
from toporlogy.optimizers.simpl import SiMPL
from toporlogy.post.export import export_phase_surfaces, voxel_surface

RESULTS: dict[str, dict] = {}            # filled by the tests, read by docs/olcumler/cok_malzeme.py


def _record(key: str, **kw) -> None:
    RESULTS[key] = kw
    print(f"\n  {key}: " + ", ".join(f"{k} = {v}" for k, v in kw.items()))


def _mbb_bc(d: T.Domain) -> tuple[np.ndarray, np.ndarray]:
    """Half MBB beam (as tests/toporlogy/test_mma.py): x fixed on the left edge, y at the bottom-right node, load −1
    (y) at the top-left node."""
    nelx, nely = d.shape
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    roller = d.nodes_where(lambda X: (X[:, 0] == nelx) & (X[:, 1] == 0))
    top_left = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == nely))
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(roller, 1)])
    F = d.load_vector()[:, 0]
    F[d.dofs(top_left, 1)] = -1.0
    return fixed, F


# ============================================================================================ V1: one phase = SIMP
def test_v1a_one_phase_modulus_is_simp_bitwise():
    d = T.Domain((7, 5))
    rng = np.random.default_rng(0)
    for x in (torch.as_tensor(rng.random(d.shape)), torch.as_tensor((rng.random(d.shape) > 0.5).astype(float))):
        for E0, p, emin in ((1.0, 3.0, 1e-9), (210e3, 4.0, 1e-6)):
            a = T.SIMP(d, E0=E0, penal=p, emin_ratio=emin)(x)
            b = MultiPhaseSIMP(d, [Phase("steel", E0)], penal=p, emin_ratio=emin)(x)
            assert torch.equal(a, b)
    _record("V1a", result="bit-identical (torch.equal), 2 designs x 2 parameter sets")


def _mbb_study(interp_fn, optimizer, n=(30, 10)) -> T.Study:
    d = T.Domain(n)
    fixed, F = _mbb_bc(d)
    return T.Study(d, T.Elasticity(d, fixed, element="Q4"), F, T.standard_chain(d, 1.5, projection=False),
                   interp_fn(d), T.Compliance(), [T.Constraint(T.Volume(), 0.5)], optimizer)


def test_v1b_one_phase_oc_run_is_simp_bitwise():
    r1 = _mbb_study(lambda d: T.SIMP(d, penal=3.0, emin_ratio=1e-9), T.OC()).run(max_iter=20, tol=0.0)
    r2 = _mbb_study(lambda d: MultiPhaseSIMP(d, [{"name": "a", "E": 1.0}], penal=3.0, emin_ratio=1e-9),
                    T.OC()).run(max_iter=20, tol=0.0)
    assert r1.history["f0"] == r2.history["f0"]
    assert np.array_equal(r1.xphys, r2.xphys) and r1.objective == r2.objective
    _record("V1b", result="bit-identical f0 history (20) and final xphys", f_final=r1.objective)


_YAML = {"name": "mbb", "domain": {"type": "box", "shape": [30, 10]},
         "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}, "dofs": "x"},
                      {"type": "fixed", "where": {"type": "point", "at": ["max", "min"]}, "dofs": "y"}],
         "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["min", "max"]},
                                    "force": [0, -1]}]}],
         "constraints": [{"response": "volume", "upper": 0.5}], "run": {"max_iter": 15}}


def _config_run(raw: dict) -> T.Result:
    from toporlogy.configuration.build import build_setup
    st = build_setup(raw)
    return st.study.run(max_iter=int(st.config["run"]["max_iter"]), tol=float(st.config["run"]["tol"]))


def test_v1c_one_phase_yaml_is_default_bitwise():
    r1 = _config_run(_YAML)
    raw = dict(_YAML, material={"E0": 1.0, "phases": [{"name": "solid", "E": 1.0}]},
               interpolation="multiphase_simp")
    r2 = _config_run(raw)
    assert np.array_equal(r1.xphys, r2.xphys) and r1.objective == r2.objective
    _record("V1c", result="bit-identical final xphys and objective (YAML, OC, 15 iterations)", f_final=r1.objective)


# ============================================================================================ V2: two phases, same E
def test_v2a_same_e_reduces_to_simp():
    d = T.Domain((6, 4))
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.random(d.shape))
    simp = T.SIMP(d, E0=2.0, penal=3.0, emin_ratio=1e-9)(x)
    mp = MultiPhaseSIMP(d, [Phase("a", 2.0), Phase("b", 2.0)], penal=3.0, emin_ratio=1e-9)
    z = torch.zeros_like(x)
    err = max(float(((mp(torch.stack([z, x])) - simp).abs() / simp).max()),
              float(((mp(torch.stack([x, z])) - simp).abs() / simp).max()))
    assert err <= 1e-14
    _record("V2a", max_rel_error=f"{err:.1e}")


def test_v2b_same_e_phase_swap_keeps_compliance():
    d = T.Domain((12, 6))
    fixed, F = _mbb_bc(d)
    phys = T.Elasticity(d, fixed, element="Q4")
    mp = MultiPhaseSIMP(d, [Phase("a", 1.0, 1.0, 1.0), Phase("b", 1.0, 3.0, 2.0)], emin_ratio=1e-9)
    rng = np.random.default_rng(2)
    solid = torch.as_tensor((rng.random(d.shape) > 0.3).astype(float))
    zero = torch.zeros_like(solid)
    act = torch.as_tensor(d.active_elements)
    c = [float((torch.as_tensor(F) * phys.solve(mp(f).reshape(-1)[act], F[:, None])[:, 0]).sum())
         for f in (torch.stack([solid, zero]), torch.stack([zero, solid]))]
    err = abs(c[0] - c[1]) / abs(c[0])
    assert err <= 1e-14
    _record("V2b", compliance=c[0], rel_difference=f"{err:.1e}")


def test_v2c_same_e_two_phase_optimum_matches_single_material():
    from toporlogy.optimizers.mma import MMA
    n = (60, 20)
    d = T.Domain(n)
    fixed, F = _mbb_bc(d)
    single = T.Study(d, T.Elasticity(d, fixed, element="Q4"), F, T.standard_chain(d, 1.5, projection=False),
                     T.SIMP(d, penal=3.0, emin_ratio=1e-9), T.Compliance(), [T.Constraint(T.Volume(), 0.5)], MMA())
    r1 = single.run(max_iter=150, tol=0.0)
    chain = T.Chain([T.Embed(d, n_fields=2), T.DensityFilter(d, 1.5), T.Passive(d)])
    two = T.Study(d, T.Elasticity(d, fixed, element="Q4"), F, chain,
                  MultiPhaseSIMP(d, [Phase("a", 1.0), Phase("b", 1.0)], penal=3.0, emin_ratio=1e-9),
                  T.Compliance(), [T.Constraint(PhaseVolume(), 0.5)], MMA())
    r2 = two.run(max_iter=150, tol=0.0)
    rel = abs(r2.objective - r1.objective) / r1.objective
    _record("V2c", single=r1.objective, two_phase=r2.objective, rel_difference=f"{rel:.2e}",
            solid_two_phase=r2.constraint_values["solid_volume"], feasible=(r1.feasible, r2.feasible))
    assert r1.feasible and r2.feasible
    assert rel <= 0.03


# ============================================================================================ V3: two-material bars
def _bar(nx: int, ny: int, layout: str, EA: float, EB: float, nu: float) -> tuple[float, float]:
    """FE compliance and closed form of a two-material bar pulled along x (thickness 1, h = 1).

    parallel: phase A in the lower half (y < ny/2), B above; consistent nodal forces of the exact piecewise
    uniform traction σ_k = E_k ε. series (ν must be 0): A for x < nx/2, B beyond, uniform traction."""
    d = T.Domain((nx, ny))
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    origin = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == 0))
    phys = T.Elasticity(d, np.concatenate([d.dofs(left, 0), d.dofs(origin, 1)]), nu=nu, element="Q4")
    F = d.load_vector()[:, 0]
    yc = d.element_centers()[:, 1].reshape(d.shape)
    xc = d.element_centers()[:, 0].reshape(d.shape)
    inA = (yc < ny / 2) if layout == "parallel" else (xc < nx / 2)
    mp = MultiPhaseSIMP(d, [Phase("A", EA), Phase("B", EB)], emin_ratio=1e-9)
    tA = torch.as_tensor(np.ones(d.shape))
    tB = torch.as_tensor((~inA).astype(float))
    E = mp(torch.stack([tA, tB]))
    if layout == "parallel":
        eps = 1.0
        for j in range(ny):                                   # right-edge segment j (y from j to j + 1)
            sig = (EA if j < ny / 2 else EB) * eps
            for yn in (j, j + 1):
                node = d.nodes_where(lambda X, yn=yn: (X[:, 0] == nx) & (X[:, 1] == yn))
                F[d.dofs(node, 0)] += sig * 0.5
        P = float(F.sum())
        exact = P ** 2 * nx / (EA * ny / 2 + EB * ny / 2)
    else:
        for j in range(ny):
            for yn in (j, j + 1):
                node = d.nodes_where(lambda X, yn=yn: (X[:, 0] == nx) & (X[:, 1] == yn))
                F[d.dofs(node, 0)] += 0.5
        P = float(F.sum())
        exact = P ** 2 * (nx / 2 / EA + nx / 2 / EB) / ny
    act = torch.as_tensor(d.active_elements)
    U = phys.solve(E.reshape(-1)[act], F[:, None])[:, 0]
    return float((torch.as_tensor(F) * U).sum()), exact


@pytest.mark.parametrize("layout,nu", [("parallel", 0.3), ("series", 0.0)])
def test_v3_two_material_bar_closed_form(layout, nu):
    c, exact = _bar(8, 4, layout, 1.0, 3.0, nu)
    rel = abs(c - exact) / exact
    _record(f"V3_{layout}", fe=c, closed_form=exact, rel_error=f"{rel:.1e}", nu=nu)
    assert rel <= 1e-10


# ============================================================================================ V4: gradients
def _domain_2d() -> T.Domain:
    shape = (10, 6)
    mask = np.ones(shape, bool)
    mask[6:, 4:] = False
    mask[3:5, 2:4] = False
    keep = np.zeros(shape, bool)
    keep[-1, :2] = True
    void = np.zeros(shape, bool)
    void[1, 5] = True
    rng = np.random.default_rng(3)
    frac = np.where(mask, 0.6 + 0.4 * rng.random(shape), 0.0)
    return T.Domain(shape, 1.0, frac=frac, keep=keep, void=void, thickness=0.7)


def _phase_study() -> T.Study:
    from toporlogy.optimizers.mma import MMA
    d = _domain_2d()
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    F = d.load_vector()[:, 0]
    F[d.dofs(d.nodes_where(lambda X: (X[:, 0] == 10) & (X[:, 1] == 2)), 1)] = -1.0
    chain = T.Chain([T.Embed(d, n_fields=3), T.DensityFilter(d, 1.5), T.HeavisideProjection(2.0, 0.5), T.Passive(d)])
    mp = MultiPhaseSIMP(d, [Phase("a", 1.0, 1.0, 3.0), Phase("b", 2.0, 2.0, 2.0), Phase("c", 4.0, 3.0, 1.0)])
    cons = [T.Constraint(PhaseVolume("b"), 0.2), T.Constraint(PhaseVolume(), 0.5), T.Constraint(Mass(), 40.0),
            T.Constraint(Cost(), 30.0), T.Constraint(PhaseVolume(["a", "c"], weighted=True), 0.3)]
    return T.Study(d, T.Elasticity(d, d.dofs(left)), F, chain, mp, T.Compliance(), cons, MMA())


def test_v4_gradients_finite_difference():
    st = _phase_study()
    rng = np.random.default_rng(4)
    x = 0.2 + 0.6 * rng.random(st.n_var)
    r1, r2 = (T.study_gradcheck(st, x, h=h, n_dirs=5, seed=0) for h in (1e-3, 2e-3))
    err = 0.0
    for a, b in zip(r1["directions"], r2["directions"], strict=True):
        fd = (4 * a["fd"] - b["fd"]) / 3
        err = max(err, abs(fd - a["ad"]) / max(abs(fd), abs(a["ad"]), 1e-300))
    errs = {"compliance": err}
    for i, c in enumerate(st.constraints):
        errs[c.key] = T.study_gradcheck(st, x, which=i, n_dirs=5, seed=0)["max_rel_error"]
    _record("V4", **{k: f"{v:.1e}" for k, v in errs.items()})
    assert max(errs.values()) <= 1e-6


# ============================================================================================ V5: published setup
@pytest.mark.slow
def test_v5_gangl_2026_sec61_reduced_runs():
    """Gangl et al. 2026 (arXiv:2605.11994) Sec. 6.1, 2-D: Ω = (0, 3) × (0, 1), left edge clamped, f = (0, −1) on
    the disc |x − (2.9, 0.5)| ≤ 0.05, ν = 0.3, E = (1e-6 void, 1, 3, 5), p = 3, ∫η_2 ≤ 0.18, ∫η_3, ∫η_4 ≤ 0.36
    (absolute, |Ω| = 3: fractions 0.06, 0.12, 0.12). The paper gives no compliance value (sub-agent reading of the
    HTML: NOT FOUND) → NOT VERIFIED; only that the constraints hold is checked. Differences: 96 × 32 elements
    (paper h = 2^-8), cone density filter with R = 2√3 ε = 0.06 (paper: PDE filter with ε = 0.06/(2√3); the
    R ↔ ε relation from memory [H]), the disc load spread equally over the nodes inside it, MMA."""
    from toporlogy.optimizers.mma import MMA
    nx, ny = 96, 32
    h = 1.0 / ny
    d = T.Domain((nx, ny), h=h)
    left = d.nodes_where(lambda X: np.abs(X[:, 0]) < 1e-9)
    disc = d.nodes_where(lambda X: np.hypot(X[:, 0] - 2.9, X[:, 1] - 0.5) <= 0.05 + 1e-9)
    F = d.load_vector()[:, 0]
    F[d.dofs(disc, 1)] = -np.pi * 0.05 ** 2 / len(disc)
    chain = T.Chain([T.Embed(d, n_fields=3), T.DensityFilter(d, 0.06 / h), T.Passive(d)])
    mp = MultiPhaseSIMP(d, [Phase("E1", 1.0), Phase("E3", 3.0), Phase("E5", 5.0)], penal=3.0, emin_ratio=1e-6 / 5)
    cons = [T.Constraint(PhaseVolume("E1"), 0.06), T.Constraint(PhaseVolume("E3"), 0.12),
            T.Constraint(PhaseVolume("E5"), 0.12)]
    st = T.Study(d, T.Elasticity(d, d.dofs(left)), F, chain, mp, T.Compliance(), cons, MMA())
    r = st.run(max_iter=200, tol=1e-3)
    rep = phase_report(mp, r.xphys, d)
    _record("V5", compliance=r.objective, iterations=r.iterations, termination=r.termination,
            max_violation=r.max_violation, volumes={k: v["volume_fraction"] for k, v in rep["phases"].items()},
            grey=rep["grey_fraction"], comparison="NOT VERIFIED (no value in the source)")
    assert r.max_violation <= 1e-3


# ============================================================================================ V6: rejections
def test_v6_rejections():
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.registry import ConfigError
    from toporlogy.optimizers.mma import MMA
    st = _phase_study()
    d, F = st.domain, st.F[:, 0]
    two = [Phase("a", 1.0), Phase("b", 2.0)]

    def study(opt, cons, chain=None, interp=None):
        return T.Study(d, st.physics, F, chain or T.Chain([T.Embed(d, n_fields=2), T.DensityFilter(d, 1.5),
                                                           T.Passive(d)]),
                       interp or MultiPhaseSIMP(d, two), T.Compliance(), cons, opt)

    for opt in (T.OC(), SiMPL()):
        with pytest.raises(ValueError, match="does not handle several phases"):
            study(opt, [T.Constraint(PhaseVolume(), 0.4)])
    with pytest.raises(ValueError, match="only Compliance, PhaseVolume"):
        study(MMA(), [T.Constraint(T.Volume(), 0.4)])
    with pytest.raises(ValueError, match="embed link has 1"):
        study(MMA(), [T.Constraint(PhaseVolume(), 0.4)], chain=T.standard_chain(d, 1.5))
    from toporlogy.design.manufacturing import AMFilter
    with pytest.raises(ValueError, match="chain may hold only"):
        study(MMA(), [T.Constraint(PhaseVolume(), 0.4)],
              chain=T.Chain([T.Embed(d, n_fields=2), AMFilter(d), T.Passive(d)]))
    with pytest.raises(ValueError, match="increasing E"):
        MultiPhaseSIMP(d, [Phase("b", 2.0), Phase("a", 1.0)])
    ctx = T.EvalContext(d, torch.zeros((2, *d.shape), dtype=torch.float64), torch.zeros(d.shape))
    with pytest.raises(ValueError, match="one design field"):
        T.Volume()(ctx)
    GCMMA_ok = study(GCMMA(), [T.Constraint(PhaseVolume(), 0.4)])
    assert GCMMA_ok.n_var == 2 * d.n_design
    raw = dict(_YAML, material={"phases": [{"name": "a", "E": 1.0}, {"name": "b", "E": 2.0}]})
    with pytest.raises(ConfigError, match="multiphase_simp"):
        build_setup(raw)
    raw = dict(raw, interpolation="multiphase_simp", optimizer="mma",
               constraints=[{"response": {"type": "phase_volume"}, "upper": 0.4}], post=["measure_lengthscale"])
    with pytest.raises(ConfigError, match="phase_report"):
        build_setup(raw)
    _record("V6", result="OC, SiMPL, Volume, field-count mismatch, AM filter, decreasing E, missing "
                         "multiphase_simp and a single-field post step raise; GCMMA accepted")


# ============================================================================================ V7: YAML and export
def test_v7_yaml_end_to_end(tmp_path: Path):
    from toporlogy.configuration.build import build_setup
    from toporlogy.execution.runner import run_setup
    raw = dict(_YAML, material={"phases": [{"name": "soft", "E": 1.0, "density": 1.0, "cost": 1.0},
                                           {"name": "stiff", "E": 3.0, "density": 2.0, "cost": 5.0}]},
               interpolation="multiphase_simp", optimizer="mma", run={"max_iter": 15},
               constraints=[{"response": {"type": "phase_volume", "phases": "soft"}, "upper": 0.2},
                            {"response": {"type": "phase_volume", "phases": "stiff"}, "upper": 0.2},
                            {"response": "mass", "upper": 250.0}],
               post=["phase_report"])
    s = run_setup(build_setup(raw), tmp_path)
    vals = {c["name"]: c["value"] for c in s["constraints"]}
    ph = s["phases"]["phases"]
    errs = [abs(ph[n]["volume_fraction"] - vals[f"volume_{n}"]) / vals[f"volume_{n}"] for n in ("soft", "stiff")]
    errs.append(abs(s["phases"]["mass"] - vals["mass"]) / vals["mass"])
    assert max(errs) <= 1e-12
    assert (tmp_path / "phases.npz").exists() and (tmp_path / "phases.json").exists()
    assert s["post"]["phase_report"]["phases"]["soft"]["volume_fraction"] == ph["soft"]["volume_fraction"]
    _record("V7_yaml", max_rel_error=f"{max(errs):.1e}", summary=s["post"]["phase_report"]["summary"])


def _signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _closed(V: np.ndarray, F: np.ndarray) -> bool:
    from collections import Counter
    key = [tuple(np.round(v, 9)) for v in V]
    cnt: Counter = Counter()
    for f in F:
        for i in range(3):
            cnt[(key[f[i]], key[f[(i + 1) % 3]])] += 1
    return all(cnt[(b, a)] == n for (a, b), n in cnt.items())


def test_v7_phase_surfaces_closed_and_exact_volume(tmp_path: Path):
    rng = np.random.default_rng(5)
    lab = rng.integers(0, 3, size=(6, 5, 4))
    lab[0, 0, 0] = -1
    h, worst = 0.25, 0.0
    files = export_phase_surfaces(lab, ["a", "b"], tmp_path, h, origin=[1.0, 2.0, 3.0])
    assert sorted(files) == ["a", "b"]
    for i in (1, 2):
        V, F = voxel_surface(lab == i, h, [1.0, 2.0, 3.0])
        assert _closed(V, F)
        exact = (lab == i).sum() * h ** 3
        worst = max(worst, abs(_signed_volume(V, F) - exact) / exact)
    assert worst <= 1e-12
    _record("V7_surface", closed=True, max_rel_volume_error=f"{worst:.1e}")


# ============================================================================================ helpers (post hoc)
def test_phase_start_and_labels():
    """Post hoc (not in the pre-registration): the start design hits the given phase fractions; labels."""
    t = phase_start(3, 4, [0.06, 0.12, 0.12])
    d = T.Domain((2, 2))
    mp = MultiPhaseSIMP(d, [Phase("a", 1.0), Phase("b", 3.0), Phase("c", 5.0)])
    eta = mp.fractions(torch.as_tensor(t.reshape(3, 2, 2))).numpy()
    np.testing.assert_allclose(eta[1:, 0, 0], [0.06, 0.12, 0.12], rtol=1e-14)
    f = torch.zeros((3, 2, 2), dtype=torch.float64)
    f[2, 0, 0] = 1.0
    f[0, 1, 1] = 1.0
    lab = phase_labels(mp, f.numpy(), d.active)
    assert lab.tolist() == [[3, 0], [0, 1]]
