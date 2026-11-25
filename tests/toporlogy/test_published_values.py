"""PR-09 continued: toporlogy against numbers PRINTED in published papers (not only against codes or closed forms).

tests/toporlogy/test_independent_vv.py compares toporlogy with transcriptions of top88 and top3d; what it left open
(docs/yol_haritasi_2026-11-18.md, PR-09 status and §6) is stress and buckling. Both are covered here by a published
value at a fixed, fully specified design, so that neither side's optimiser enters:

* **Buckling** — Ferrari, Sigmund & Guest (2021), "Topology Optimization with linearized buckling criteria in 250
  lines of Matlab", arXiv:2101.02973v1, §5.2 (p. 12): "The fundamental BLF is λ1 = 3.15 for the initial, solid
  structure" of the wall-reinforcement problem (320 × 320). The code that produced it is printed in its Appendix C;
  tests/toporlogy/published_codes.py (:class:`TopBuck250`) transcribes its analysis part.
* **Stress** — Deng, Vulimiri & To (2021), "An efficient 146-line 3D sensitivity analysis code of stress-based
  topology optimization written in MATLAB", arXiv:2104.01210v1, §6.2: for the 200 × 60 × 1 cantilever "the p-norm
  stress value decreases from initial 33.17 to 3.31". Its Appendix A is transcribed in :func:`deng_pnorm`.

Every tolerance below was written down before the first run (pre-registration file of 2026-11-25 14:37 +03:00 in the
session notes, summarised in docs/olcumler/yayimlanmis_degerler.md §1); none was changed after seeing a result. Each
docstring says why the two sides can differ at all. Measured numbers are printed (run with ``-s``).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from published_codes import DENG_B, Top3D, TopBuck250, deng_D, deng_pnorm, lk_H8
from toporlogy import MMA, OC, SIMP, Constraint, Domain, Elasticity, Study, Volume, standard_chain
from toporlogy.fem.buckling import LinearBuckling, gauss_operators
from toporlogy.fem.elements import D_iso, element_matrix
from toporlogy.optimization.aggregates import pnorm
from toporlogy.optimization.local import Aggregated
from toporlogy.optimization.stress import VonMises, centroid_stress, von_mises_matrix

SEED = 20261125
BLF_WALL_PUBLISHED = 3.15          # Ferrari, Sigmund & Guest 2021, §5.2, p. 12
PNORM_CANTILEVER_PUBLISHED = 33.17  # Deng, Vulimiri & To 2021, §6.2


# ================================================================================ helpers: MATLAB layout -> toporlogy
def _maps_2d(ref: TopBuck250, d: Domain) -> tuple[np.ndarray, np.ndarray]:
    """``dof[k]``: toporlogy DOF of the reference's 0-based DOF k; ``el[e]``: toporlogy element of MATLAB element e.

    MATLAB numbers nodes column-major from the top-left node and elements column-major from the top-left element;
    toporlogy numbers both x-major from the bottom-left. Both put the x component first.
    """
    n0 = np.arange((ref.nelx + 1) * (ref.nely + 1))
    ix, iy = n0 // (ref.nely + 1), ref.nely - n0 % (ref.nely + 1)
    tn = d.node_number(np.column_stack([ix, iy]))
    dof = np.empty(ref.nDof, np.int64)
    dof[0::2], dof[1::2] = 2 * tn, 2 * tn + 1
    e0 = np.arange(ref.nEl)
    el = (e0 // ref.nely) * ref.nely + (ref.nely - 1 - e0 % ref.nely)
    return dof, el


def _toporlogy_2d(ref: TopBuck250) -> tuple[Domain, Elasticity, np.ndarray, np.ndarray, np.ndarray]:
    """The reference's mesh, supports and load built in toporlogy (Q4, plane stress, unit thickness)."""
    d = Domain((ref.nelx, ref.nely), h=ref.Lx / ref.nelx)
    dof, el = _maps_2d(ref, d)
    F = d.load_vector()
    F[dof, 0] = ref.F
    m = Elasticity(d, dof[ref.fixed], nu=ref.nu, element="Q4")
    return d, m, F, dof, el


def _moduli(ref: TopBuck250, xPhys: np.ndarray, el: np.ndarray, pK: float, pG: float) -> tuple[np.ndarray, np.ndarray]:
    """topBuck250's interpolations (lines 114, 143) in toporlogy element order: E = Emin + x^pK (E0 − Emin) for K and
    E_G = E0 x^pG (no lower bound) for the stress stiffness."""
    E, EG = np.empty(ref.nEl), np.empty(ref.nEl)
    E[el] = ref.Emin + xPhys ** pK * (ref.E0 - ref.Emin)
    EG[el] = ref.E0 * xPhys ** pG
    return E, EG


def _permuted(A: sp.spmatrix, dof: np.ndarray, n: int) -> sp.csr_matrix:
    """Sparse matrix in the reference's DOF numbering → toporlogy's numbering."""
    C = A.tocoo()
    return sp.csr_matrix((C.data, (dof[C.row], dof[C.col])), shape=(n, n))


# ================================================================================ 1. buckling: topBuck250
def test_topbuck250_element_matrix_and_wall_setup():
    """Setup checks (exact by construction, no tolerance to pre-register beyond rounding):

    * topBuck250's Ke from the lists c1, c2 equals toporlogy's Q4 matrix to 1e-15 (both are the bilinear plane-stress
      element; toporlogy's is typed from top88, topBuck250's from a different listing, lines 18-25);
    * the wall's initial design (lines 9, 93-96, 101-111: volfrac = 1 because prSel{1}(1) = 'V') is crisp: every
      element is 1 except the opening P0, which is 0. The filter cannot blur it because every active element is at
      least nelx/40 = 8 elements away from P0 (P1 frame of thickness 8) and the cone has radius rmin = 3;
    * the load sums to the printed total |q| = 1e-2 and acts on the 321 left-edge nodes; 36 bottom nodes are clamped.
    """
    ref = TopBuck250(320, 320, 1.0, setup="wall", rmin=3.0)
    assert np.max(np.abs(ref.Ke0 - element_matrix(2, "Q4", 0.3))) <= 1e-15
    xPhys = ref.physical(ref.initial_x(1.0))
    solid = np.ones(ref.nEl, bool)
    solid[ref.pasV] = False
    assert np.max(np.abs(xPhys[solid] - 1.0)) <= 1e-12 and np.max(np.abs(xPhys[~solid])) <= 1e-12
    assert len(ref.pasV) == 193 * 129 and len(ref.lcDof) == 321
    assert abs(ref.F.sum() - 1e-2) <= 1e-15 and len(ref.fixed) == 2 * 36


@pytest.mark.parametrize("design", ["solid", "random"])
def test_topbuck250_column_fixed_design(design):
    """topBuck250's own column setup (lines 64-69) on a 60 × 30 mesh, Lx = 2, pK = pG = 3; no published number, this
    is the code-to-code check that makes the slow wall test interpretable.

    Designs: x = 1 everywhere (pasS solid), and a seeded random x in [0.2, 1] on the active set; both go through the
    transcription's filter and projection (rmin 4, β 2, η 0.5, Neumann filter BC) and the same physical density is
    handed to toporlogy (toporlogy's own filter has a different boundary rule and is not under test here).

    Pre-registered: compliance |Δc|/c ≤ 1e-9; centroid stress max|Δσ|/max|σ| ≤ 1e-9; stress stiffness against the
    transcription switched to Gauss-point stress max|ΔK_G|/max|K_G| ≤ 1e-10; the lowest four λ against it ≤ 1e-8
    relative. The published (centroid-stress) λ are printed next to toporlogy's, without a bound.
    Why they can differ: rounding (solvers SuperLU/PARDISO vs SuperLU, summation order); ARPACK tolerance in the two
    eigen solvers (toporlogy: generalised symmetric Lanczos with tol 1e-10; reference: non-symmetric ARPACK on K⁻¹G).
    """
    out = {}
    for mode in ("centroid", "gauss"):
        ref = TopBuck250(60, 30, 2.0, setup="column", rmin=4.0, stress_at=mode)
        x = ref.initial_x(1.0)
        if design == "random":
            x[ref.act] = np.random.default_rng(SEED).uniform(0.2, 1.0, len(ref.act))
        xPhys = ref.physical(x)
        out[mode] = ref.analyse(xPhys, 3.0, 3.0, n_eig=4)
    d, m, F, dof, el = _toporlogy_2d(ref)
    E, EG = _moduli(ref, out["gauss"]["xPhys"], el, 3.0, 3.0)
    U = m.solve_numpy(E, F)[:, 0]
    c_rel = abs(float(F[:, 0] @ U) - out["gauss"]["c"]) / out["gauss"]["c"]
    s_T = centroid_stress(d, torch.as_tensor(U), 0.3, 1.0).detach().numpy()[:, :, 0]
    s_R = out["gauss"]["sGP"]
    s_rel = float(np.max(np.abs(s_T[el] - s_R)) / np.max(np.abs(s_R)))
    lb = LinearBuckling(m)
    KG_T = lb.geometric_matrix(EG, U)
    KG_R = _permuted(out["gauss"]["G"], dof, d.ndof)[m.free][:, m.free]
    kg_rel = float(abs(KG_T - KG_R).max() / abs(KG_R).max())
    mu_T, _ = lb.modes(E, EG, U, k=4)
    lam_T = 1 / mu_T[:4]
    lam_G = 1 / out["gauss"]["mu"][:4]
    lam_C = 1 / out["centroid"]["mu"][:4]
    lam_rel = float(np.max(np.abs(lam_T - lam_G) / lam_G))
    print(f"\n  topBuck250 column {design}: c_rel={c_rel:.2e} sigma_rel={s_rel:.2e} KG_rel={kg_rel:.2e} "
          f"lambda_rel(gauss)={lam_rel:.2e}\n    lambda toporlogy {np.round(lam_T, 6)}\n    lambda published "
          f"code (centroid stress) {np.round(lam_C, 6)}; rel. diff {np.round((lam_T - lam_C) / lam_C, 5)}")
    assert c_rel <= 1e-9
    assert s_rel <= 1e-9
    assert kg_rel <= 1e-10
    assert lam_rel <= 1e-8


@pytest.fixture(scope="module")
def wall() -> dict[str, float]:
    """λ1 of the solid wall (Ferrari, Sigmund & Guest 2021, §5.2): the transcription as published (centroid stress
    in G), the transcription with Gauss-point stress, and toporlogy. About 160 s; used by slow tests only.

    Setup (all read from the paper): 320 × 320, Lx = 1, E0 = 1, Emin = 1e-6, ν = 0.3, pK = pG = 3 at loop 1 (the call
    ``topBuck250(320,320,3,3,2,'N',0.5,2,[0.1,0.7,1.2],500,1,3,12,160,{['V','C','B'],[2.5,1.05]})``), x = 1 on A and
    P1, 0 on the opening P0; the bottom nodes of columns 1:9, 120:128, 257:265, 313:321 clamped; an x-load
    1e-2/320 on every left-edge node, end nodes halved.
    """
    lam: dict[str, float] = {}
    for mode in ("centroid", "gauss"):
        ref = TopBuck250(320, 320, 1.0, setup="wall", rmin=3.0, stress_at=mode)
        xPhys = ref.physical(ref.initial_x(1.0))
        lam[mode] = 1.0 / float(ref.analyse(xPhys, 3.0, 3.0, n_eig=12)["mu"][0])
    d, m, F, dof, el = _toporlogy_2d(ref)
    E, EG = _moduli(ref, xPhys, el, 3.0, 3.0)
    U = m.solve_numpy(E, F)[:, 0]
    mu_T, _ = LinearBuckling(m).modes(E, EG, U, k=6)
    lam["toporlogy"] = 1.0 / float(mu_T[0])
    print(f"\n  wall, solid design: published {BLF_WALL_PUBLISHED}; transcription (centroid) {lam['centroid']:.6f}; "
          f"transcription (gauss) {lam['gauss']:.6f}; toporlogy {lam['toporlogy']:.6f}; centroid vs gauss "
          f"{(lam['gauss'] - lam['centroid']) / lam['centroid']:.2e}")
    return lam


@pytest.mark.slow
def test_topbuck250_wall_toporlogy_equals_the_code(wall):
    """B3, pre-registered: the transcription with Gauss-point stress against toporlogy, |Δλ1|/λ1 ≤ 1e-8 (same
    formulation; rounding and the two eigen solvers' tolerances only). First run 2026-11-25: 8.9e-11, PASS."""
    assert abs(wall["toporlogy"] - wall["gauss"]) / wall["gauss"] <= 1e-8


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="published 3.15 NOT reproduced: the transcribed code gives 1.303")
def test_topbuck250_wall_solid_blf_against_published_value(wall):
    """The published number: λ1 = 3.15 for the initial, solid wall (Ferrari, Sigmund & Guest 2021, §5.2, p. 12).

    Pre-registered:
    B1 transcription as published (centroid stress in G): |λ1 − 3.15| ≤ 0.0051 (printed rounding 0.005 + solver).
    B2 toporlogy (``LinearBuckling``: stress at each Gauss point, its documented formulation):
       |λ1 / 3.15 − 1| ≤ 1e-2. Why it can differ: the rounding (1.6e-3) and G built from Gauss-point instead of
       centroid stress; the size of the latter was not known before the run.

    Record of the first run (2026-11-25, 162 s): both FAILED. Transcription (centroid) 1.303072, toporlogy 1.201863
    (= the Gauss-point transcription to 8.9e-11); the published code's centroid stress gives a λ1 7.8 % above the
    Gauss-point one on this mesh (column: 0.01-0.5 %; why the wall differs more was not investigated). The paper's own code, transcribed, misses the
    paper's number by a factor 2.42 under the printed call. The gap is not in toporlogy; it lies in the unprinted
    setup, the printed value, or an error left in the transcription (it could not be run in MATLAB: licence error),
    and these were not told apart. The value does NOT verify toporlogy's buckling. The column checks above (code to code, 1e-13)
    do not depend on this. Post-hoc probes, none adopted: the load reversed (−x) 0.863; every bottom node clamped
    2.33; the first 40 λ of the printed setup run 1.303 … 3.196 (λ39 = 3.156, no reason to take it). An independent
    re-read of the transcription against the paper found no translation error, and the authors' downloadable file
    (topopt.mek.dtu.dk, topbuck250_finalsamo.m, read, not run) has the same analysis lines; it differs from the
    printed listing only elsewhere (argument order ocPar/maxit, ocUpdate's restartAs argument). Unexplained.
    """
    assert abs(wall["centroid"] - BLF_WALL_PUBLISHED) <= 0.0051
    assert abs(wall["toporlogy"] / BLF_WALL_PUBLISHED - 1.0) <= 1e-2


# ================================================================================ 2. stress: Deng, Vulimiri & To
def _deng_pair() -> tuple[Top3D, Domain, Elasticity, np.ndarray, np.ndarray]:
    """§6.2's cantilever: 200 × 60 × 1, volfrac 0.3, pl = 3, filter radius 2.5 (the FE is top3d's, see deng_pnorm),
    and the same problem in toporlogy (H8, face x = 0 clamped, −1 in y on the two nodes of the edge x = 200, y = 0)."""
    ref = Top3D(200, 60, 1, 0.3, 3.0, 2.5)
    d = Domain((200, 60, 1))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    tip = d.nodes_where(lambda X: (X[:, 0] == 200) & (X[:, 1] == 0))
    F = d.load_vector()
    F[d.dofs(tip, 1), 0] = -1.0
    m = Elasticity(d, fixed, nu=0.3, element="H8")
    xyz = ref.node_xyz(np.arange(201 * 61 * 2))
    tn = d.node_number(xyz)
    dof = (3 * tn[:, None] + np.arange(3)).ravel()
    return ref, d, m, F, dof


def _deng_start(ref: Top3D) -> np.ndarray:
    """ASSUMPTION (pre-registered): "initial" = x = volfrac = 0.3 everywhere; the paper does not print the driver.
    Its filter (line 4 of stress_minimize, top3d's H / Hs) leaves a constant field constant."""
    return ref.physical(np.full(ref.shape_f, ref.volfrac))


def test_deng_printed_strain_matrix_is_one_gauss_point():
    """S3 (first part): the B printed in lines 126-144 is the strain matrix at ONE of the 2×2×2 Gauss points of the
    unit hexahedron, not at the centroid. Pre-registered: it equals exactly one of toporlogy's eight Gauss-point
    matrices (``fem.buckling.gauss_operators``, rows mapped from the paper's εx, εy, εz, γxy, γyz, γzx to toporlogy's
    xx, yy, zz, yz, xz, xy) to ≤ 1e-5 (printing to 5 significant digits), and the paper's element stiffness is
    top3d's (equal to ``lk_H8`` by reading; checked here against toporlogy's H8 to 1e-12)."""
    _, B, _ = gauss_operators(3)
    paper = DENG_B[[0, 1, 2, 4, 5, 3]]
    err = np.array([np.max(np.abs(Bg - paper)) for Bg in B])
    print(f"\n  printed B vs Gauss points: {np.array2string(err, precision=2)}")
    assert np.sum(err <= 1e-5) == 1
    assert np.max(np.abs(lk_H8(0.3) - element_matrix(3, "H8", 0.3))) <= 1e-12


@pytest.fixture(scope="module")
def deng() -> dict[str, float]:
    """Everything measured on §6.2's cantilever at the assumed start x = 0.3 (one FE solve per side)."""
    ref, d, m, F, dof = _deng_pair()
    xPhys = _deng_start(ref)
    r = deng_pnorm(ref, xPhys, q=0.5, p=10.0)
    xg = np.ascontiguousarray(np.asarray(xPhys)[::-1].transpose(1, 0, 2))      # (nely, nelx, nelz) -> (nx, ny, nz)
    E = 1e-9 + xg.ravel() ** 3 * (1 - 1e-9)
    U = m.solve_numpy(E, F)[:, 0]
    U_R = np.asarray(r["U"])
    A = m.matrix(E)
    K = (A + A.T - sp.diags(A.diagonal())).tocsr()                           # full reduced stiffness
    b = F[m.free, 0]
    U_Rt = np.zeros(d.ndof)
    U_Rt[dof] = U_R
    _, B, _ = gauss_operators(3)
    g = int(np.argmin([np.max(np.abs(Bg - DENG_B[[0, 1, 2, 4, 5, 3]])) for Bg in B]))
    sig = xg.ravel()[:, None] ** 0.5 * (U[d.edof] @ (D_iso(0.3) @ B[g]).T)       # h = 1, E0 = 1
    vm = torch.sqrt(torch.einsum("ev,vw,ew->e", *(torch.as_tensor(a) for a in (sig, von_mises_matrix(3), sig))))
    study = Study(d, m, F, standard_chain(d, 2.5, projection=False), SIMP(d, penal=3.0, emin_ratio=1e-9),
                  Aggregated(VonMises(d, sigma_lim=1.0, nu=0.3, E0=1.0, form="ratio", q=0.5), "pnorm", 10.0),
                  [Constraint(Volume(), 0.3)], OC())
    out = {"pn_R": float(r["pnorm"]), "pn_T": float(pnorm(vm, p=10.0)), "g": float(g),
           "U_rel": float(np.max(np.abs(U[dof] - U_R)) / np.max(np.abs(U_R))),
           "res_T": float(np.linalg.norm(K @ U[m.free] - b) / np.linalg.norm(b)),
           "res_R": float(np.linalg.norm(K @ U_Rt[m.free] - b) / np.linalg.norm(b)),
           "pn_centroid": float(next(iter(study.evaluate_density(xg).values.values())))}
    print(f"\n  Deng cantilever, x = 0.3: published {PNORM_CANTILEVER_PUBLISHED}; transcription {out['pn_R']:.6f}; "
          f"toporlogy at Gauss point {g} {out['pn_T']:.6f}; toporlogy VonMises (centroid) {out['pn_centroid']:.6f}; "
          f"U rel {out['U_rel']:.2e}; residuals in toporlogy's K: toporlogy {out['res_T']:.1e}, transcription "
          f"{out['res_R']:.1e}")
    return out


def test_deng_toporlogy_stress_equals_the_published_code(deng):
    """S3 (second part), pre-registered: toporlogy's stress at the Gauss point the paper evaluates (exact B from
    ``gauss_operators``, ``D_iso``, ``von_mises_matrix``) aggregated by ``aggregates.pnorm`` against the
    transcription with the printed B: ≤ 2e-4 relative (the printed B carries 5 significant digits).
    Also: the paper's elasticity matrix is toporlogy's ``D_iso`` with the shear rows in the paper's order (exact)."""
    rel = abs(deng["pn_T"] - deng["pn_R"]) / deng["pn_R"]
    print(f"\n  toporlogy vs transcription: {rel:.2e}")
    assert rel <= 2e-4
    assert np.allclose(deng_D(0.3), D_iso(0.3)[[0, 1, 2, 5, 3, 4]][:, [0, 1, 2, 5, 3, 4]], atol=1e-15)


@pytest.mark.xfail(strict=True, reason="pre-registered bound 1e-9 FAILED on the first run (9.7e-8): "
                                       "cond(K) ~ 5e9, see the docstring")
def test_deng_displacements_agree_to_rounding(deng):
    """S2, pre-registered: max|ΔU| / max|U| ≤ 1e-9 between toporlogy's H8 solve and the transcription (SciPy
    ``spsolve`` on top3d's ``(K + K')/2``).

    Record of the first run (2026-11-25): FAILED, 9.68e-8. Cause measured afterwards (post hoc, scratch probe; the
    bound was not changed): the 200 × 60 × 1 cantilever, one element thick, has cond(K_ff) ≈ 5.4e9, so a forward
    error of ~1e-7 is what backward-stable solves give; both relative residuals are ~1e-11 (next test), and against
    an iteratively refined solution toporlogy is off by 7.6e-9, the transcription by 5.1e-8. The bound ignored
    conditioning (it was carried over from the 24 × 8 × 4 Top3D case of test_independent_vv.py). Strict xfail: if
    this ever passes, the record here is stale.
    """
    assert deng["U_rel"] <= 1e-9


def test_deng_displacements_are_backward_stable(deng):
    """Post hoc (written after S2 failed, not pre-registered): both displacement fields satisfy toporlogy's reduced
    system to a relative residual ≤ 1e-10 (measured 5.1e-12 and 1.1e-11), i.e. the S2 gap is conditioning, not a
    different problem."""
    assert deng["res_T"] <= 1e-10 and deng["res_R"] <= 1e-10


@pytest.mark.xfail(strict=True, reason="published 33.17 NOT reproduced: the transcribed code gives 68.45 at x = 0.3")
def test_deng_initial_pnorm_against_published_value(deng):
    """The published number: p-norm 33.17 at the start of §6.2's optimisation (pl = 3, q = 0.5, p = 10).

    Pre-registered: S1 the transcription with the printed B at x = 0.3: |pn − 33.17| ≤ 0.01 (printed rounding 0.005
    + the printed B's 5 digits, ~1e-4 relative); S3 toporlogy at the same Gauss point: |pn − 33.17| ≤ 0.006.
    S4, printed without a bound: toporlogy's ``VonMises`` response (centroid stress, ratio form, q = 0.5, σ_lim = 1)
    with the same p-norm (a different evaluation point, not the same quantity).

    Record of the first run (2026-11-25): both FAILED. Transcription 68.447505, toporlogy 68.447056, centroid 52.711.
    Because the paper's own code, transcribed, misses the paper's number by a factor 2.06 while toporlogy matches the
    code to 6.6e-6, the gap is not in toporlogy; it lies in a setup the paper does not print (its MMA driver and
    initial design), the printed value, or an error left in the transcription; these were not told apart. Hypotheses tried afterwards (post hoc, scratch probe; none adopted): load on one node only → 17.26;
    uniform x = 0.4 → 33.34, 0.5 → 19.09, 1 → 3.374. The published value therefore does NOT verify toporlogy's stress.

    Explained afterwards (2026-11-25, post hoc): 33.17 is the value after the FIRST MMA step (move 0.1), not at x0;
    one step from x0 = 0.3 gives 33.130 (−0.12 %), and the L-bracket's 56.23 is matched the same way (+0.84 %), see
    :func:`test_deng_cantilever_published_value_is_after_the_first_mma_step`. The pre-registered reading ("initial" =
    x0) was wrong, so this test stays a strict xfail; toporlogy's code is not at fault.
    """
    assert abs(deng["pn_R"] - PNORM_CANTILEVER_PUBLISHED) <= 0.01
    assert abs(deng["pn_T"] - PNORM_CANTILEVER_PUBLISHED) <= 0.006


# ================================================================================ 3. Deng et al.: what "initial" means
PNORM_LBRACKET_PUBLISHED = 56.23    # Deng, Vulimiri & To 2021, §6.3, 2-D L-bracket 200 × 200 × 1


def _lbracket(ref: Top3D) -> np.ndarray:
    """§6.3 / §6.4 L-bracket (2-D: nelz = 1) on the transcription: top face clamped (``fixed_node`` code of §6.4),
    −1 in y on the 3 × (nelz + 1) nodes x = nelx, y = nely/2 … nely/2 − 2 (``loaddof1..3``), passive set
    ``x_t(1:nelx/2, nely/2:end, :) = 1e-4``. Returns the passive mask in MATLAB element order."""
    nelx, nely, nelz = ref.nelx, ref.nely, ref.nelz
    layer = (nely + 1) * (nelx + 1)
    fn = np.array([n + j * layer for j in range(nelz + 1) for n in range(1, layer - nely + 1, nely + 1)])
    il, jl, kl = np.meshgrid(nelx, 0, np.arange(nelz + 1))
    loadnid = (kl * layer + il * (nely + 1) + (nely + 1 - jl)).reshape(-1, order="F")
    b = 3 * loadnid - 1 - nelx // 2 * 3
    F = np.zeros(3 * layer * (nelz + 1))
    np.add.at(F, np.union1d(np.union1d(b, b + 3), b + 6) - 1, -1.0)
    ref.F = F
    ref.fixeddofs = np.unique(np.concatenate([3 * fn - 2, 3 * fn - 1, 3 * fn]) - 1)
    ref.freedofs = np.setdiff1d(np.arange(F.size), ref.fixeddofs)
    passive = np.zeros(ref.shape_f, bool)
    passive[: nelx // 2, nely // 2 - 1:, :] = True
    return ref._col(passive)


def _after_one_mma_step(ref: Top3D, passive: np.ndarray) -> tuple[float, float]:
    """p-norm of the transcription (printed B) at x0 = 0.3 and after ONE MMA step with move 0.1 (§6.2: "the moving
    limit of MMA optimizer is set as 0.1"), the way stress_minimize presents the problem: filter, passive elements
    set to 1e-4 after the filter, objective = p-norm, constraint mean(x̄) − 0.3 ≤ 0, filtered sensitivities.

    The step is toporlogy's MMA (Svanberg's 2007 coefficients; dual by L-BFGS-B, not Svanberg's subsolv), the
    sensitivity toporlogy's autograd of its own Gauss-point stress (equal to the transcription to 6.6e-6)."""
    nelx, nely, nelz = ref.nelx, ref.nely, ref.nelz
    d = Domain((nelx, nely, nelz))
    tn = d.node_number(ref.node_xyz(np.arange(ref.F.size // 3)))
    dof = (3 * tn[:, None] + np.arange(3)).ravel()
    F = d.load_vector()
    F[dof, 0] = ref.F
    m = Elasticity(d, dof[ref.fixeddofs], nu=0.3, element="H8")
    e = np.arange(ref.nele)
    el = ((e // nely) % nelx * nely + (nely - 1 - e % nely)) * nelz + e // (nely * nelx)   # MATLAB -> toporlogy
    _, B, _ = gauss_operators(3)
    S = torch.as_tensor(D_iso(0.3) @ B[7])                                      # the printed point (+g, +g, +g)
    V = torch.as_tensor(von_mises_matrix(3))
    edof = torch.as_tensor(d.edof[el])
    idx = (torch.as_tensor(el),)

    def phys(x: np.ndarray) -> np.ndarray:
        xp = ref.H @ x / ref.Hs
        xp[passive] = 1e-4
        return xp

    def f_df(x: np.ndarray) -> tuple[float, np.ndarray]:
        xp = torch.as_tensor(phys(x)).requires_grad_(True)
        E = torch.zeros(ref.nele, dtype=torch.float64).index_put(idx, 1e-9 + xp ** 3 * (1 - 1e-9))
        U = m.solve(E, F)[:, 0]
        sig = xp[:, None] ** 0.5 * (U[edof] @ S.T)
        f = pnorm(torch.sqrt(torch.einsum("ev,vw,ew->e", sig, V, sig)), p=10.0)
        (gt,) = torch.autograd.grad(f, xp)
        gx = gt.numpy().copy()
        gx[passive] = 0.0
        return float(f), ref.H @ (gx / ref.Hs)

    x0 = np.full(ref.nele, 0.3)
    f0, df0 = f_df(x0)
    dv = ref.H @ (np.full(ref.nele, 1.0 / ref.nele) / ref.Hs)
    x1 = MMA(move=0.1, normalise=None).step(x0, f0, df0, np.array([np.mean(phys(x0)) - 0.3]), dv[None, :])
    pn = [float(deng_pnorm(ref, ref._field(phys(x)), 0.5, 10.0)["pnorm"]) for x in (x0, x1)]
    return pn[0], pn[1]


def test_deng_cantilever_published_value_is_after_the_first_mma_step():
    """POST HOC (written after S1 failed; not pre-registered, the bound below was set after seeing 0.12 %).

    Hypothesis from the failed S1: the paper's "initial" p-norm is the value recorded after the first MMA update,
    not at x0. The transcription at x0 = 0.3 gives 68.45 (S1); after one MMA step with the printed move 0.1 it
    gives 33.130 against the printed 33.17 (−0.12 %). The same reading explains the 2-D L-bracket (slow test below,
    +0.84 %). Bound: 0.5 % — the MMA here is not the authors' (subproblem solver, c, d not printed)."""
    ref = Top3D(200, 60, 1, 0.3, 3.0, 2.5)
    pn0, pn1 = _after_one_mma_step(ref, np.zeros(ref.nele, bool))
    print(f"\n  Deng cantilever: x0 {pn0:.4f}; after one MMA step {pn1:.4f}; published {PNORM_CANTILEVER_PUBLISHED} "
          f"({pn1 / PNORM_CANTILEVER_PUBLISHED - 1:+.2%})")
    assert abs(pn1 / PNORM_CANTILEVER_PUBLISHED - 1) <= 5e-3


@pytest.mark.slow
def test_deng_lbracket_published_value_is_after_the_first_mma_step():
    """POST HOC, same hypothesis on the 2-D L-bracket (§6.3: "decreases from the initial 56.23"); setup read from
    §6.4 (see :func:`_lbracket`; where the passive set is applied is not printed: here after the filter, x0 = 0.3
    everywhere). Measured: x0 117.91, after one MMA step 56.70 (+0.84 %). Bound 2 %, set after seeing the value."""
    ref = Top3D(200, 200, 1, 0.3, 3.0, 2.5)
    pn0, pn1 = _after_one_mma_step(ref, _lbracket(ref))
    print(f"\n  Deng L-bracket: x0 {pn0:.4f}; after one MMA step {pn1:.4f}; published {PNORM_LBRACKET_PUBLISHED} "
          f"({pn1 / PNORM_LBRACKET_PUBLISHED - 1:+.2%})")
    assert abs(pn1 / PNORM_LBRACKET_PUBLISHED - 1) <= 2e-2
