"""Component constructors: design.
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.helpers import _axis, _coord, _point
from toporlogy.configuration.registry import BuildContext, register


def register_design() -> None:
    """Register the design components."""
    from toporlogy.design import manufacturing as MF
    length = "length"

    # ---------------------------------------------------------------- chain links (manufacturing)
    def symmetry_link(ctx: BuildContext, planes: Any) -> Any:
        """Mirror tying about planes {x: 30} or [[x, 30], [y, mid]] (coordinate: number, min, mid, max); first
        link of the chain."""
        items = list(planes.items()) if isinstance(planes, dict) else [tuple(p) for p in planes]
        d = ctx.domain
        return MF.SymmetryTying(d, [(_axis(a, d.dim), _coord(c, _axis(a, d.dim), d)) for a, c in items])

    def pattern(ctx: BuildContext, kind: str = "translational", offset: list | None = None, n: int | None = None,
                center: list | None = None, axis: Any = "z") -> Any:
        """Pattern repetition: translational (offset vector) or cyclic (n sectors about center[, axis])."""
        d = ctx.domain
        if kind == "translational":
            if offset is None:
                raise ValueError("translational pattern needs offset")
            return MF.PatternTying.translational(d, offset)
        if kind == "cyclic":
            if n is None or center is None:
                raise ValueError("cyclic pattern needs n and center")
            return MF.PatternTying.cyclic(d, n, _point(center, d, "center"), axis)
        raise ValueError(f'kind: translational | cyclic, got "{kind}"')

    def group_average(ctx: BuildContext) -> Any:
        """Re-ties the filtered field to the groups of the chain's tying link (put right after density_filter)."""
        tying = next((lk for lk in ctx.links if isinstance(lk, MF.Tying)), None)
        if tying is None:
            raise ValueError("group_average needs a tying link (symmetry, extrusion, pattern) first in the chain")
        return MF.GroupAverage(tying)

    register("link", "am_filter", MF.AMFilter, context=("domain",), types={"build": None},
             units={"base_coordinate": length})
    register("link", "casting", MF.Casting, context=("domain",), types={"direction": None},
             units={"parting": length})
    register("link", "symmetry", symmetry_link, types={"planes": None}, units={"planes": length})
    register("link", "extrusion", MF.ExtrusionTying, context=("domain",), types={"axis": None})
    register("link", "pattern", pattern, types={"axis": None}, units={"offset": length, "center": length})
    register("link", "group_average", group_average)
    # ---------------------------------------------------------------- several phases (material.phases)
    def phase_report_step(ctx: BuildContext, surfaces: bool = True) -> Any:
        """Several phases: label field (0 void, i phase i, -1 outside) and fractions to phases.npz, per-phase
        volume fraction / volume / mass / cost and grey share to phases.json; in 3-D one voxel-boundary STL per
        phase (surfaces true). Information only: no verdict."""
        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            from pathlib import Path

            import numpy as np
            import torch

            from toporlogy.design.phases import phase_labels, phase_report
            from toporlogy.post.export import export_phase_surfaces, write_json
            interp, d = study.interp, study.domain
            if not hasattr(interp, "fractions"):
                raise ValueError("phase_report needs interpolation multiphase_simp (material.phases)")
            rep = phase_report(interp, res.xphys, d)
            lab = phase_labels(interp, res.xphys, d.active)
            with torch.no_grad():
                eta = interp.fractions(torch.as_tensor(np.asarray(res.xphys, float))).numpy()
            np.savez_compressed(Path(folder) / "phases.npz", labels=lab, fractions=eta,
                                names=np.array(rep["names"]))
            files = {"npz": str(Path(folder) / "phases.npz")}
            if surfaces and d.dim == 3:
                files.update(export_phase_surfaces(lab, rep["names"][1:], folder, d.h, d.origin))
            files["json"] = write_json(Path(folder) / "phases.json", rep)
            rep["files"] = files
            vols = ", ".join(f"{n} {v['volume_fraction']:.4f}" for n, v in rep["phases"].items())
            rep["summary"] = (f"{vols} (volume fraction); mass {rep['mass']:.6g}, cost {rep['cost']:.6g}, "
                              f"grey {rep['grey_fraction']:.3f}")
            return rep
        return step

    register("post", "phase_report", phase_report_step)
    register("link", "helmholtz_filter", "toporlogy.design.pde_filter:HelmholtzFilter", context=("domain",),
             types={"rmin": float}, units={"rmin": "elements"})
