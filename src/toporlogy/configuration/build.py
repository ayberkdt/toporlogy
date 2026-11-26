"""Building a study from a resolved config: ``Setup``, boundary conditions, ``build``, ``build_setup``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toporlogy.configuration.registry import BuildContext, ConfigError, REGISTRY, join
from toporlogy.configuration.resolve import _robust_settings, apply_overrides, read_config, resolve


# ============================================================================================== build
@dataclass
class Setup:
    """A built study with what the CLI prints about it."""

    study: Any
    config: dict[str, Any]                     # resolved
    fixed: np.ndarray
    cases: list[dict[str, Any]]                # name, weight, resultant, n_nodes
    constraint_names: list[str]
    post: list[tuple[str, Callable[..., Any]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bc: Any = None                             # toporlogy.boundary.BoundaryConditions
    bc_report: Any = None                      # its CheckReport (no errors: those raise ConfigError)
    interfaces: dict[str, int] = field(default_factory=dict)      # connectivity interface name → elements
    info: dict[str, Any] = field(default_factory=dict)            # robust settings, ties, bores, notes

    @property
    def run_options(self) -> dict[str, Any]:
        return dict(self.config["run"])

    def bc_summary(self) -> str:
        """Load and support tables of :meth:`toporlogy.boundary.BoundaryConditions.summary` (its last line, the
        verdict for the Python API, left out: the CLI prints the check itself)."""
        if self.bc is None:
            return ""
        lines = [ln for ln in self.bc.summary().splitlines() if not ln.startswith("Check ")]
        if not self.config.get("load_cases"):            # modal analysis: the placeholder case has no load
            first = next((i for i, ln in enumerate(lines) if ln.startswith("Supports of")), 0)
            lines = [ln.replace("Supports of modal:", "Supports:") for ln in lines[first:]]
        return "\n".join(lines)


def _resultant(f: np.ndarray, dim: int) -> list[float]:
    return [float(v) for v in f.reshape(-1, dim).sum(axis=0)]


def _as_supports(obj: Any, dim: int, name: str) -> list[Any]:
    """A built support: a bc.Support, or fixed DOF numbers (components that predate bc) → Fixed per direction."""
    from toporlogy.boundary import Fixed, NodeIds, Support
    if isinstance(obj, Support):
        return [obj]
    dofs = np.unique(np.asarray(obj, np.int64).reshape(-1))
    if not dofs.size:
        raise ValueError("support fixes no DOF")
    return [Fixed(NodeIds(dofs[dofs % dim == k] // dim), [k], name=f"{name} {'xyz'[k]}")
            for k in range(dim) if np.any(dofs % dim == k)]


def _as_load(obj: Any, dim: int, ndof: int, name: str) -> Any:
    """A built load: a bc.Load, or a load vector (ndof,) (components that predate bc) → NodalForces."""
    from toporlogy.boundary import Load, NodalForces, NodeIds
    if isinstance(obj, Load):
        return obj
    f = np.asarray(obj, float).reshape(-1)
    if f.shape != (ndof,):
        raise ValueError(f"load vector has {f.size} entries, the domain {ndof} DOFs")
    fn = f.reshape(-1, dim)
    nodes = np.flatnonzero(np.any(fn != 0, axis=1))
    if not nodes.size:
        raise ValueError("load is zero")
    return NodalForces(NodeIds(nodes), fn[nodes], name=name)


def _supports(ctx: BuildContext, specs: Sequence[dict[str, Any]], path: str, prefix: str) -> list[Any]:
    from toporlogy.boundary import EmptySelectionError, part_info
    out = []
    info = part_info(ctx.domain)
    for i, s in enumerate(specs):
        sp = join(path, i)
        built = REGISTRY.build("support", s, ctx, sp)
        default = f"{prefix}{i + 1} {s['type']}"
        objs = _as_supports(built, ctx.domain.dim, default)
        for o in objs:
            if o is built and s.get("name") is None:
                o.name = default
            try:
                o.directions(info)
                o.springs(info)
            except (EmptySelectionError, ValueError) as e:
                raise ConfigError(f"support {s['type']}: {e}", sp) from None
        out += objs
    return out


def _loads(ctx: BuildContext, specs: Sequence[dict[str, Any]], path: str, prefix: str) -> list[Any]:
    from toporlogy.boundary import EmptySelectionError, part_info
    out = []
    info = part_info(ctx.domain)
    for j, ld in enumerate(specs):
        lp = join(path, j)
        default = f"{prefix}{j + 1} {ld['type']}"
        o = _as_load(REGISTRY.build("load", ld, ctx, lp), ctx.domain.dim, ctx.domain.ndof, default)
        if ld.get("name") is None:
            o.name = default
        try:
            o.apply(info)
        except (EmptySelectionError, ValueError) as e:
            raise ConfigError(f"load {ld['type']}: {e}", lp) from None
        out.append(o)
    return out


def _boundary_conditions(domain: Any, supports: list[Any], cases: list[Any], combos: list[Any]) -> Any:
    from toporlogy.boundary import BoundaryConditions
    try:
        return BoundaryConditions(domain, supports, cases, combos)
    except ValueError as e:
        raise ConfigError(str(e), "load_cases") from None


def _default_interfaces(bcs: Any) -> dict[str, np.ndarray]:
    """One interface per support and per load (bc node sets → elements touching them); body forces left out."""
    from toporlogy.boundary import BodyForce
    from toporlogy.optimization.connectivity import elements_touching_nodes
    d = bcs.domain
    out: dict[str, np.ndarray] = {}
    multi = len(bcs.cases) > 1
    for i, c in enumerate(bcs.cases):
        if i == 0 or c.supports is not None:
            for name, _kind, nodes, _rows in bcs.constraints(i).per_support:
                key = name if c.supports is None or not multi else f"{c.name}.{name}"
                if key not in out and len(nodes):
                    out[key] = elements_touching_nodes(d, nodes)
        for ld, a in zip(c.loads, bcs.applied[c.name], strict=True):
            if isinstance(ld, BodyForce) or getattr(ld, "_no_interface", False) or not len(a.nodes):
                continue
            key = a.name if not multi else f"{c.name}.{a.name}"
            out.setdefault(key, elements_touching_nodes(d, a.nodes))
    return {k: m for k, m in out.items() if m.any()}


def _explicit_interfaces(ctx: BuildContext, specs: dict[str, Any]) -> dict[str, np.ndarray]:
    from toporlogy.boundary import EmptySelectionError
    from toporlogy.optimization.connectivity import elements_touching_nodes
    from toporlogy.configuration.helpers import as_selector
    out = {}
    for name, spec in specs.items():
        path = join("connectivity.interfaces", name)
        try:
            nodes = as_selector(REGISTRY.build("selector", spec, ctx, path)).select(ctx.domain)
        except EmptySelectionError as e:
            raise ConfigError(f"selector: {e}", path) from None
        out[name] = elements_touching_nodes(ctx.domain, nodes)
    return out


def _check_bc(bcs: Any, cases: list[dict[str, Any]]) -> Any:
    """bc.check(); errors → ConfigError at supports / load_cases.<i>[.supports]."""
    rep = bcs.check()
    if rep.errors:
        names = [c["name"] for c in cases]
        errs = []
        for it in rep.errors:
            k = names.index(it.case) if it.case in names else None
            if it.code in ("rigid-body", "overlapping-supports", "over-constrained", "support-on-void"):
                path = join(join("load_cases", k), "supports") if k is not None and \
                    cases[k].get("supports") is not None else "supports"
            else:
                path = join("load_cases", k) if k is not None else "load_cases"
            errs.append((path, f"{it.code}: {it.message}"))
        raise ConfigError(errs)
    return rep


def _tie_planes(spec: Any, chain: list[Any], d: Any) -> list[tuple[int, float]]:
    """Mirror planes for the connectivity ties: ``connectivity.symmetry`` if given, else those of a ``symmetry``
    link in the chain (tied variables cannot keep a one-sided frozen tie), else none."""
    from toporlogy.configuration.helpers import _axis, _coord
    if spec is None:
        link = next((lk for lk in chain if isinstance(lk, dict) and str(lk.get("type", "")).lower() == "symmetry"),
                    None)
        spec = None if link is None else link.get("planes")
    if spec is None:
        return []
    items = list(spec.items()) if isinstance(spec, dict) else [tuple(p) for p in spec]
    return [(_axis(a, d.dim), _coord(c, _axis(a, d.dim), d)) for a, c in items]


def build(resolved: dict[str, Any]) -> Setup:
    """Construct every component of a resolved config (see :func:`resolve`) into a :class:`Setup`."""
    from toporlogy.boundary import AllNodes, Combination, LoadCase
    from toporlogy.design.chain import Chain, DensityFilter, HeavisideProjection
    from toporlogy.design.manufacturing import GroupAverage, Tying, validate_chain
    from toporlogy.configuration.helpers import SpreadForce
    from toporlogy.optimization.study import Constraint, Study

    r = resolved
    from toporlogy.configuration.schema import NO_LOAD_PHYSICS
    no_load = r["physics"]["type"] in NO_LOAD_PHYSICS
    combos_r = r.get("combinations") or []
    weights = [float(c["weight"]) for c in r["load_cases"]] + [float(c["weight"]) for c in combos_r]
    ctx = BuildContext(config=r, material=dict(r["material"]), run=dict(r["run"]),
                       n_cases=max(len(weights), 1), case_weights=weights or [1.0])
    B = REGISTRY.build
    info: dict[str, Any] = {"notes": []}
    d = ctx.domain = B("domain", r["domain"], ctx, "domain")

    rb = r.get("robust")
    if rb is not None:
        from toporlogy.optimization.robust import RobustSet
        if rb.get("min_solid") is not None:
            s = _robust_settings(rb, d.h, d.dim)
            info["robust"] = {k: float(v) for k, v in s.items()}
            eta_e, eta_d = s["eta_e"], s["eta_d"]
        else:
            eta_e = 0.75 if rb.get("eta_e") is None else rb["eta_e"]
            eta_d = rb.get("eta_d")
            info["robust"] = {"eta_e": eta_e, "eta_d": 1 - eta_e if eta_d is None else eta_d}
        try:
            ctx.robust = RobustSet(eta_e, eta_d, rb.get("eta_i"), rb.get("beta"))
        except ValueError as e:
            raise ConfigError(str(e), "robust") from None

    # ---------------------------------------------------------------- boundary conditions
    default_sup = _supports(ctx, r["supports"], "supports", "S")
    lcs = []
    for i, c in enumerate(r["load_cases"]):
        cp = join("load_cases", i)
        many = len(r["load_cases"]) > 1
        sup = None if c.get("supports") is None else _supports(ctx, c["supports"], join(cp, "supports"),
                                                               f"{c['name']}.S" if many else "S")
        lcs.append(LoadCase(c["name"], _loads(ctx, c["loads"], join(cp, "loads"), "L"), sup))
    if not lcs:                                         # modal / transient heat analysis without static loads
        dummy = SpreadForce(AllNodes(), np.zeros(d.dim), "no static load")
        dummy._no_interface = True
        lcs = [LoadCase("modal", [dummy])]
    combos = [Combination(cb["name"], dict(cb["factors"])) for cb in combos_r]
    bcs = _boundary_conditions(d, default_sup, lcs, combos)

    cb = r.get("connectivity")
    plan = None
    if cb is not None and cb.get("interfaces"):
        ctx.bc = bcs
        ctx.interfaces = _explicit_interfaces(ctx, cb["interfaces"])
    else:
        ctx.interfaces = _default_interfaces(bcs)
    if cb is not None:
        from toporlogy.optimization.connectivity import one_body
        if len(ctx.interfaces) < 2:
            raise ConfigError(f"connectivity needs at least 2 interfaces, found {len(ctx.interfaces)}",
                              "connectivity.interfaces")
        opts = dict(cb.get("options") or {})
        if cb["constraint"] == "coupling":
            opts.setdefault("nu", ctx.material["nu"])
        d0 = d
        if cb["ties"]:
            # tie_regions returns the tie WITHOUT the interface cells: an interface cell that is not frozen can go
            # void and cut the tie (measured on examples/split_connectivity2d.yaml: 2 bodies). Freeze them.
            from toporlogy.optimization.connectivity import with_keep
            pads = np.logical_or.reduce(list(ctx.interfaces.values())) & d.active
            info["frozen_interface_cells"] = int((pads & ~d.keep).sum())
            d0 = with_keep(d, pads)
        try:
            planes = _tie_planes(cb.get("symmetry"), r.get("chain") or [], d)
            if planes:
                info["tie_symmetry"] = [["xyz"[k], c] for k, c in planes]
            plan = one_body(d0, ctx.interfaces, ties=cb["ties"], constraint=cb["constraint"], eta=cb["eta"],
                            bound=cb["bound"], sink=cb["sink"], threshold=cb["threshold"],
                            repair_section=cb["repair_section"], symmetry=planes, **opts)
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            raise ConfigError(f"connectivity: {e}", "connectivity") from None
        ctx.interfaces = plan.interfaces
        if plan.ties is not None:
            # plan.ties also holds the interface cells (one_body freezes them with the ties) and may overlap keep;
            # those are counted in frozen_interface_cells / the domain's keep, so count the new tie cells only
            info["ties"] = int((plan.ties & ~d0.keep).sum())
            d = ctx.domain = plan.domain
            bcs = _boundary_conditions(d, default_sup, lcs, combos)
    ctx.bc = bcs
    rep = _check_bc(bcs, r["load_cases"])
    if no_load:
        F = d.load_vector(1)
        cases: list[dict[str, Any]] = []
    else:
        F = bcs.F("all")
        cases = []
        for i, name in enumerate(bcs.names):
            if not np.isfinite(F[:, i]).all():
                where = join("load_cases", i) if i < len(r["load_cases"]) else \
                    join("combinations", i - len(r["load_cases"]))
                raise ConfigError("load case has a load that is not finite (NaN or Inf)", where)
            if not np.any(F[:, i]):
                path = join("load_cases", i) if i < len(r["load_cases"]) else \
                    join("combinations", i - len(r["load_cases"]))
                raise ConfigError("load case has zero load", path)
            cases.append({"name": name, "weight": weights[i], "resultant": _resultant(F[:, i], d.dim),
                          "n_nodes": int(np.any(F[:, i].reshape(-1, d.dim) != 0, axis=1).sum()),
                          "combination": i >= len(r["load_cases"])})
    ctx.n_cases = F.shape[1]
    if no_load:
        ctx.case_weights = [1.0]
    ctx.fixed = np.unique(np.concatenate([bcs.constraints(i).fixed for i in range(len(bcs.cases))]))
    warnings = []
    loaded_fixed = np.intersect1d(np.flatnonzero(np.any(F != 0, axis=1)), ctx.fixed)
    if loaded_fixed.size:
        warnings.append(f"{loaded_fixed.size} loaded DOF(s) are also fixed; their load does no work")

    # ---------------------------------------------------------------- physics, chain, responses
    ctx.solver = None
    ctx.physics = B("physics", r["physics"], ctx, "physics")
    for i, lk in enumerate(r["chain"]):
        ctx.links.append(B("link", lk, ctx, join("chain", i)))
    ctx.chain = Chain(ctx.links)
    try:
        validate_chain(ctx.chain)
    except ValueError as e:
        raise ConfigError(str(e), "chain") from None
    if any(isinstance(lk, Tying) for lk in ctx.links) and not any(isinstance(lk, GroupAverage) for lk in ctx.links):
        warnings.append("the chain has a tying link but no group_average after density_filter; the filtered field "
                        "is not tied exactly near box faces")
    ctx.interp = B("interpolation", r["interpolation"], ctx, "interpolation")
    objective = ctx.objective = B("response", r["objective"], ctx, "objective")
    constraints, cnames = ctx.constraints, []
    for i, c in enumerate(r["constraints"]):
        cp = join("constraints", i)
        resp = B("response", c["response"], ctx, join(cp, "response"))
        if c.get("name"):
            resp.name = c["name"]
        if resp.name == objective.name or resp.name in cnames:
            raise ConfigError(f'response name "{resp.name}" is used twice; set {cp}.name', cp)
        upper = c.get("upper")
        if upper is None:
            upper = getattr(resp, "config_upper", None)
            if upper is None:
                raise ConfigError(f"response {c['response']['type']} has no default bound; set upper",
                                  join(cp, "upper"))
        from toporlogy.optimization.responses import Volume
        if isinstance(resp, Volume) and not 0.0 < float(upper) <= 1.0:
            raise ConfigError(f"volume is a fraction of the part: upper must be in (0, 1], got {upper}",
                              join(cp, "upper"))
        scale = c.get("scale") if c.get("scale") is not None else getattr(resp, "config_scale", None)
        cnames.append(resp.name)
        constraints.append(Constraint(resp, float(upper), scale))
    if plan is not None:
        for con in plan.constraints:
            if con.key in cnames or con.key == objective.name:
                raise ConfigError(f'constraint name "{con.key}" is used twice', "connectivity")
            cnames.append(con.key)
            constraints.append(con)
    if ctx.robust is not None:
        if ctx.chain.find(DensityFilter) is None:
            raise ConfigError("robust realisations need a density_filter link", "chain")
        if ctx.robust.beta is None and ctx.chain.find(HeavisideProjection) is None:
            raise ConfigError("robust realisations need a heaviside_projection link (or robust.beta)", "chain")
    optimizer = B("optimizer", r["optimizer"], ctx, "optimizer")
    if not constraints and getattr(optimizer, "min_constraints", 1) > 0:
        raise ConfigError(f"optimizer {r['optimizer']['type']} needs at least one constraint", "constraints")
    schedule = None if r["schedule"] is None else B("schedule", r["schedule"], ctx, "schedule")
    if r["schedule"] is not None and r["schedule"]["type"] == "beta_continuation" and \
            ctx.chain.find(HeavisideProjection) is None:
        warnings.append("schedule beta_continuation finds no heaviside_projection link; beta stays fixed")
    post = [(p["type"], B("post", p, ctx, join("post", i))) for i, p in enumerate(r["post"])]
    phases = ctx.material.get("phases")
    if phases and r["interpolation"]["type"] != "multiphase_simp":
        raise ConfigError("material.phases needs interpolation multiphase_simp", "interpolation")
    if phases and len(phases) > 1:
        other = [p["type"] for p in r["post"] if p["type"] != "phase_report"]
        if other:                     # these steps read one density field (toporlogy.design.phases)
            raise ConfigError(f"with several phases only the phase_report post step is available, got {other}",
                              "post")
    from toporlogy.configuration.catalog import check_build
    check_build(r, ctx.physics, [objective] + [c.response for c in constraints])    # catalog's fail-closed rules
    try:
        study = Study(d, ctx.physics, F, ctx.chain, ctx.interp, objective, constraints, optimizer, schedule,
                      move_beta_ref=r["run"].get("move_beta_ref"))
    except ValueError as e:
        raise ConfigError(str(e), "constraints" if "constraint" in str(e) else "") from e
    study.config = r
    bores = ctx.extra.get("bores_found")
    if bores:
        from toporlogy.configuration.helpers import _describe_bores
        info["bores"] = _describe_bores(bores)
    info["notes"] += ctx.extra.get("notes", [])
    return Setup(study, r, ctx.fixed, cases, cnames, post, warnings, bcs, rep,
                 {k: int(m.sum()) for k, m in (ctx.interfaces or {}).items()}, info)


def build_setup(raw: dict[str, Any], overrides: Sequence[str] | Sequence[tuple[str, Any]] = ()) -> Setup:
    """:func:`apply_overrides` → :func:`resolve` → :func:`build`."""
    return build(resolve(apply_overrides(raw, overrides)))


def study_from_dict(d: dict[str, Any], overrides: Sequence[str] = ()) -> Any:
    """Study from a raw config mapping; the resolved config is attached as ``study.config``."""
    return build_setup(d, overrides).study


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Any:
    """Study from a YAML/JSON file (``study.config`` = resolved config; run options in ``study.config["run"]``)."""
    return study_from_dict(read_config(path), overrides)
