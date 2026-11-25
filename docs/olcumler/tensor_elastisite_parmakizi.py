"""Fingerprint of the tensor / elasticity functions that the tensor-core work rewires (record R1 of
``tensor_elastisite.md``).

    python docs/olcumler/tensor_elastisite_parmakizi.py --write     # once, on the base commit 1d1f027 (before)
    python docs/olcumler/tensor_elastisite_parmakizi.py             # after: compares, prints max differences

Every array is produced from fixed inputs (seeded RNG) through the public (or, for fem.nonlinear, module-private)
functions as they are called today; the comparison reports for each entry whether it is bit-for-bit equal and the
largest relative difference. Names that no longer exist are reported, not skipped silently.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
OUT = HERE / "tensor_elastisite_oncesi.json"      # *.npz is not tracked (.gitignore)


def _ortho() -> list:
    from toporlogy.core.material import Orthotropic
    return [Orthotropic(40.0, 9.0, 9.0, 4.0, 4.0, 3.2, 0.28, 0.28, 0.4),
            Orthotropic(1.0, 0.5, 0.25, 0.3, 0.2, 0.1, 0.2, 0.1, 0.3),
            Orthotropic.isotropic(7.0, 0.27), Orthotropic.isotropic(1.0, -0.4)]


def fingerprint() -> dict[str, np.ndarray]:
    import torch

    from toporlogy.core import material as M
    from toporlogy.design import interp
    from toporlogy.fem import elements as el
    from toporlogy.fem import nonlinear as nl
    from toporlogy.optimization import strength as st
    from toporlogy.optimization.stress import von_mises_matrix
    from toporlogy.post.verify import von_mises

    rng = np.random.default_rng(20261126)
    out: dict[str, np.ndarray] = {}
    nus = [-0.9, -0.4, 0.0, 0.2, 0.3, 0.45, 0.49]
    out["D_iso"] = np.array([el.D_iso(v) for v in nus])
    out["D_plane_stress"] = np.array([el.D_plane_stress(v) for v in nus])
    out["D_plane_strain"] = np.array([el.D_plane_strain(v) for v in nus])
    out["ke_h8"] = el.ke_h8(0.3, "H8IM")[0]
    out["ke_q4"] = el.ke_q4(0.3)
    Rs = [np.eye(3), M.rotation_about_z(30.0), M.rotation_from_axes([1, 2, 3], [0, 1, -1]),
          M.rotation_from_axes([0.3, -1, 0.2], [1, 0, 0])]
    for k, m in enumerate(_ortho()):
        out[f"ortho{k}_S"] = m.compliance()
        out[f"ortho{k}_C"] = np.array([m.stiffness(R) for R in Rs])
        C = m.stiffness()
        out[f"ortho{k}_rot"] = np.array([M.rotate_stiffness(C, R) for R in Rs])
        out[f"ortho{k}_ps"] = np.array([M.plane_stress(m.stiffness(R)) for R in Rs])
        out[f"ortho{k}_pe"] = np.array([M.plane_strain(m.stiffness(R)) for R in Rs])
        out[f"ortho{k}_row"] = np.array([M.out_of_plane_row(m.stiffness(R)) for R in Rs])
        out[f"ortho{k}_unit2"] = np.array([M.unit_elasticity(m, 2, p, Rs[1]) for p in M.PLANE_MODES])
        out[f"ortho{k}_field"] = M.unit_elasticity_field(m, 2, "stress", M.rotations_about_z([0, 15, 15, 90]))
    s = rng.standard_normal((40, 6))
    st_t = torch.as_tensor(s)
    out["rot_stress"] = st.rotate_stress_to_material(st_t, Rs[2]).numpy()
    out["rot_stress_rows"] = st.rotate_stress_to_material(st_t, np.array([Rs[k % 4] for k in range(40)])).numpy()
    out["von_mises_eq"] = st.von_mises_equivalent(st_t, 1e-6).numpy()
    out["tresca_eq"] = st.tresca_equivalent(st_t).numpy()
    iso = M.Strength(1.0, 1.7, safety_factor=1.2)
    ortho = M.OrthotropicStrength(1.5, 1.2, 0.05, 0.2, 0.08)
    for kind in st.CRITERIA:
        c = st.make_criterion(kind, ortho if kind == "tsai_wu" else iso, Rs[3] if kind == "tsai_wu" else None)
        out[f"crit_{kind}"] = c.index(st_t).numpy()
    out["expand_plane"] = st.expand_plane(torch.as_tensor(s[:, :3]), torch.as_tensor(s[:, 3])).numpy()
    out["vm_matrix2"], out["vm_matrix3"] = von_mises_matrix(2), von_mises_matrix(3)
    out["verify_vm"] = von_mises(s)
    for dim in (2, 3):
        H = 0.3 * rng.standard_normal((25, dim, dim))
        out[f"nl_green{dim}"] = nl._green(H)
        out[f"nl_voigt{dim}"] = nl._voigt(nl._green(H), dim)
        out[f"nl_detm1_{dim}"] = nl._det_minus_one(H)
        out[f"nl_detm1_small{dim}"] = nl._det_minus_one(1e-9 * H)
    H3 = 0.2 * rng.standard_normal((25, 3, 3))
    S, C, psi = nl.neo_hooke_response(nl._green(H3), nl._det_minus_one(H3), 1.3, 0.4, True)
    out["nh_S"], out["nh_C"], out["nh_psi"] = S, C, psi
    out["simp_penal"] = np.array([[interp.simp_admissible_penal(v, d) for v in nus] for d in (2, 3)])
    x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    out["hs_moduli"] = np.array([[torch.stack(interp.hashin_shtrikman_moduli(x, 2.0, v, d)).numpy() for v in nus]
                                 for d in (2, 3)])
    return out


def main() -> int:
    got = fingerprint()
    if "--write" in sys.argv:
        rec = {k: {"shape": list(np.shape(v)),
                   "sha256": hashlib.sha256(np.ascontiguousarray(v, float).tobytes()).hexdigest(),
                   "values": np.asarray(v, float).ravel().tolist()} for k, v in got.items()}
        OUT.write_text(json.dumps(rec, indent=0) + chr(10), encoding="utf-8")
        print(f"wrote {OUT} ({len(got)} arrays)")
        return 0
    ref = {k: np.array(v["values"], float).reshape(v["shape"]) for k, v in json.loads(OUT.read_text("utf-8")).items()}
    worst = 0.0
    for k in sorted(set(ref) | set(got)):
        if k not in got or k not in ref:
            print(f"{k:24s} MISSING in {'now' if k not in got else 'record'}")
            worst = np.inf
            continue
        a, b = ref[k], got[k]
        same = a.shape == b.shape and np.array_equal(a, b)
        rel = 0.0 if same else float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
        worst = max(worst, rel)
        print(f"{k:24s} {'bitwise' if same else f'rel {rel:.3e}'}")
    print(f"largest relative difference: {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
