"""Connectivity of a density design: detection (gate), prevention during optimisation, repair.

A part must be ONE body, and every functional interface region (supports, load pads, other keep regions, passed as
named element masks) must belong to it. Compliance minimisation with several supports and several load points can
legally produce separate load paths (each load to its own supports) joined by nothing or by a grey film that vanishes
on thresholding. This module gives

Detection
    :func:`analyse_connectivity` → :class:`ConnectivityReport`: connected components of the thresholded design
    (face contact by default), body volumes, interface → body map, unattached interfaces, enclosed voids, minimum
    cross-section ("neck") between every pair of interfaces by an exact max-flow / min-cut on the element graph,
    and robustness checks at other thresholds and after an erosion by one element. Verdict PASS / FRAGILE / FAIL.

Prevention (differentiable responses for :class:`toporlogy.study.Study`, use with MMA — they are constraints in
addition to the volume bound)
    :class:`ConductionConnectivity` — virtual heat conduction: solid conducts, void insulates; one interface is the
    heat sink (T = 0), every other interface receives unit heat in its own load case. Its thermal resistance
    ``R_i = q_iᵀ T_i`` stays O(path length / section) when a solid path to the sink exists and grows like 1/k_min
    when it does not. Constraint: aggregate of ``R_i / R_i,ref`` ≤ bound.
    :class:`CouplingCompliance` — auxiliary structural cases: one interface held, unit forces in every axis
    direction on each other interface (the supports of the main problem released); aggregate compliance ratio ≤
    bound forces a stiff path from every interface to the held one.

    Why a single sink / a single held group: with sinks on *all* supports, a design made of two bodies — each load
    on its own support — has every interface connected to *a* sink, so the constraint is satisfied by exactly the
    failure it should prevent. Connecting every interface to one reference interface is equivalent to "all
    interfaces in one body" (connectivity is transitive). The same holds for the coupling cases.

Frozen ties (no optimisation)
    :func:`tie_regions` — keep-mask of minimal cross-section along shortest paths through the design domain that
    connect all interface groups (Prim-like growth of a Steiner tree); :func:`with_keep` gives the domain with the
    tie frozen solid. The optimiser may thicken it.

Repair
    :func:`bridge` — adds minimal-volume bridges between the bodies along density-weighted shortest paths (multi-
    source Dijkstra; cheap through solid, expensive through void) with a given section, optionally drops floating
    islands. Re-verification (:func:`analyse_connectivity` on the result, re-analysis of stiffness) is the caller's job.

Literature: the virtual-temperature idea for connectivity is due to Liu, Li, Chen, Tong & Cheng (2015, Front. Mech.
Eng. 10:126-137; enclosed voids: void conducts, solid insulates) and Li, Chen, Liu & Tong (2016, Struct. Multidisc.
Optim. 54:971-984; "virtual scalar field", connectivity as a maximum-temperature constraint). Only the abstracts and
the publisher landing page were read in this session; the formulation here (single sink, per-interface load cases,
resistance ratios, p-norm, robust eroded realisation) is this module's own and is NOT claimed to reproduce theirs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy import ndimage
from scipy.sparse import csgraph

from toporlogy.chain import DensityFilter, HeavisideProjection, Passive
from toporlogy.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.heat import ConductivityInterp, HeatConduction, element_nodes
from toporlogy.responses import EvalContext, Response

__all__ = ["analyse_connectivity", "ConnectivityReport", "Neck", "ThresholdCheck", "min_cut_cells",
           "elements_touching_nodes", "label_bodies", "ConductionConnectivity", "CouplingCompliance",
           "tie_regions", "with_keep", "bridge", "BridgeResult", "split_benchmark", "SplitBenchmark"]


# ====================================================================================================== grid helpers
def _structure(dim: int, connectivity: str | int) -> np.ndarray:
    """Neighbourhood: "face" (6 / 4 neighbours), "edge" (18; in 2-D = face), "vertex" (26 / 8); or the count."""
    ranks = {3: {"face": 1, "edge": 2, "vertex": 3, 6: 1, 18: 2, 26: 3},
             2: {"face": 1, "edge": 1, "vertex": 2, 4: 1, 8: 2}}[dim]
    key = connectivity.lower() if isinstance(connectivity, str) else int(connectivity)
    if key not in ranks:
        raise ValueError(f"connectivity for {dim}-D: one of {sorted(map(str, ranks))}")
    return ndimage.generate_binary_structure(dim, ranks[key])


def _offsets(structure: np.ndarray) -> list[tuple[int, ...]]:
    """Lexicographically positive neighbour offsets of a 3^dim structure (each undirected pair once)."""
    c = np.array(structure.shape) // 2
    out = []
    for idx in zip(*np.nonzero(structure), strict=True):
        o = tuple(int(i - k) for i, k in zip(idx, c, strict=True))
        if any(o) and o > tuple([0] * len(o)):
            out.append(o)
    return out


def _neighbour_pairs(mask: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat element numbers (p, q) of all neighbour pairs with both cells in ``mask`` (each pair once)."""
    ids = np.arange(mask.size).reshape(mask.shape)
    P, Q = [], []
    for o in _offsets(structure):
        src = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(o, mask.shape, strict=True))
        dst = tuple(slice(max(0, d), n - max(0, -d)) for d, n in zip(o, mask.shape, strict=True))
        m = mask[src] & mask[dst]
        P.append(ids[src][m])
        Q.append(ids[dst][m])
    return np.concatenate(P), np.concatenate(Q)


def label_bodies(solid: np.ndarray, connectivity: str | int = "face") -> tuple[np.ndarray, np.ndarray]:
    """Connected components of ``solid``: labels (0 = void, 1… = bodies by decreasing cell count) and cell counts."""
    solid = np.asarray(solid, bool)
    lab, n = ndimage.label(solid, structure=_structure(solid.ndim, connectivity))
    if n == 0:
        return lab, np.zeros(0, np.int64)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    order = np.argsort(-sizes, kind="stable")
    remap = np.zeros(n + 1, np.int64)
    remap[order + 1] = np.arange(1, n + 1)
    return remap[lab], sizes[order]


def elements_touching_nodes(domain: Domain, nodes: Sequence[int], active_only: bool = True) -> np.ndarray:
    """Bool element mask of the elements having at least one of ``nodes`` as a corner (e.g. support regions)."""
    hit = np.zeros(domain.nn, bool)
    hit[np.asarray(nodes, np.int64)] = True
    m = hit[element_nodes(domain)].any(axis=1).reshape(domain.shape)
    return m & domain.active if active_only else m


def _interfaces(domain: Domain, interfaces: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not interfaces:
        raise ValueError("at least one interface region is required")
    out = {}
    for name, m in interfaces.items():
        m = np.asarray(m, bool)
        if m.shape != domain.shape:
            raise ValueError(f"interface {name!r}: mask shape {m.shape}, domain {domain.shape}")
        m = m & domain.active
        if not m.any():
            raise ValueError(f"interface {name!r} has no element in the part")
        out[str(name)] = m
    return out


def _nodes_of(domain: Domain, mask: np.ndarray) -> np.ndarray:
    e = np.flatnonzero((np.asarray(mask, bool) & domain.active).ravel())
    return np.unique(element_nodes(domain, e))


def _cell_volume(domain: Domain) -> float:
    return domain.h ** domain.dim * (domain.thickness if domain.dim == 2 else 1.0)


def _cell_area(domain: Domain) -> float:
    return domain.h ** (domain.dim - 1) * (domain.thickness if domain.dim == 2 else 1.0)


# ====================================================================================================== min cut
def min_cut_cells(solid: np.ndarray, a: np.ndarray, b: np.ndarray, connectivity: str | int = "face",
                  frozen: np.ndarray | None = None) -> tuple[int | None, np.ndarray]:
    """Minimum number of ``solid`` cells whose removal disconnects region ``a`` from region ``b``.

    Exact (max-flow / min-cut with vertex splitting, Dinic's algorithm of :func:`scipy.sparse.csgraph.maximum_flow`).
    Cells of ``a``, ``b`` and ``frozen`` cannot be removed. Returns ``(n_cells, cut)`` with ``cut`` the flat element
    numbers of one minimum cut; ``(0, [])`` if a and b are not connected, ``(None, [])`` if no cut exists (a and b
    touch, or are joined through frozen cells only).
    """
    solid = np.asarray(solid, bool)
    a, b = np.asarray(a, bool) & solid, np.asarray(b, bool) & solid
    if not a.any() or not b.any():
        return 0, np.zeros(0, np.int64)
    st = _structure(solid.ndim, connectivity)
    lab, _ = ndimage.label(solid, structure=st)
    common = np.intersect1d(np.unique(lab[a]), np.unique(lab[b]))
    if common.size == 0:
        return 0, np.zeros(0, np.int64)
    sub = np.isin(lab, common)
    if (a & b).any():
        return None, np.zeros(0, np.int64)
    idx = np.flatnonzero(sub.ravel())
    N = len(idx)
    pos = np.full(solid.size, -1, np.int64)
    pos[idx] = np.arange(N)
    p, q = _neighbour_pairs(sub, st)
    i, j = pos[p], pos[q]
    A, B = pos[np.flatnonzero(a.ravel() & sub.ravel())], pos[np.flatnonzero(b.ravel() & sub.ravel())]
    big = N + 1
    cap = np.ones(N, np.int64)
    unc = a | b if frozen is None else (a | b | np.asarray(frozen, bool))
    cap[pos[np.flatnonzero(unc.ravel() & sub.ravel())]] = big
    # nodes: in_k = k, out_k = N + k, S = 2N, T = 2N + 1, S0 = 2N + 2 (S0 → S with capacity big bounds the total
    # flow by big, so int32 capacities cannot overflow)
    S, T, S0 = 2 * N, 2 * N + 1, 2 * N + 2
    rows = np.concatenate([np.arange(N), N + i, N + j, np.full(len(A), S), N + B, [S0]])
    cols = np.concatenate([N + np.arange(N), j, i, A, np.full(len(B), T), [S]])
    caps = np.concatenate([cap, np.full(2 * len(i), big), np.full(len(A), big), np.full(len(B), big), [big]])
    nv = 2 * N + 3
    C = sp.csr_matrix((caps.astype(np.int32), (rows, cols)), shape=(nv, nv))
    C.sum_duplicates()
    C.sort_indices()
    res = csgraph.maximum_flow(C, S0, T, method="dinic")
    if res.flow_value >= big:
        return None, np.zeros(0, np.int64)
    R = (C - res.flow).tocsr()
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    reach = np.zeros(nv, bool)
    reach[csgraph.breadth_first_order(R, S0, directed=True, return_predecessors=False)] = True
    cut_local = np.flatnonzero(reach[:N] & ~reach[N:2 * N])
    return int(res.flow_value), idx[cut_local]


# ====================================================================================================== detection
@dataclass
class Neck:
    """Minimum cross-section between two interfaces of one body (``cells`` None: they touch, nothing to cut)."""

    a: str
    b: str
    cells: int | None
    area: float | None
    where: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.int64))


@dataclass
class ThresholdCheck:
    """Connectivity of one robustness realisation (another threshold, or eroded by one element)."""

    label: str
    n_bodies: int
    connected: bool
    unattached: list[str]


@dataclass
class ConnectivityReport:
    """Result of :func:`analyse_connectivity`. Bodies are numbered 1… by decreasing volume; 0 = void."""

    threshold: float
    connectivity: str | int
    labels: np.ndarray = field(repr=False)
    n_bodies: int
    body_cells: list[int]
    body_volume: list[float]
    interface_bodies: dict[str, list[int]]
    interface_solid_fraction: dict[str, float]
    unattached: list[str]
    main_body: int
    connected: bool
    floating: list[int]
    enclosed_voids: list[int]
    voids_matter: bool
    void_policy: str
    necks: list[Neck]
    checks: list[ThresholdCheck]
    passed: bool
    fragile: bool
    verdict: str
    reasons: list[str]

    @property
    def ok(self) -> bool:
        """True only for verdict PASS (one body holding every interface, robust under all checks)."""
        return self.verdict == "PASS"

    @property
    def min_neck(self) -> Neck | None:
        cut = [n for n in self.necks if n.cells is not None]
        return min(cut, key=lambda n: n.cells) if cut else None

    def summary(self) -> str:
        """Readable verdict: first line = verdict, last line = result + one next step."""
        L = [f"Connectivity: {self.verdict} - {self.n_bodies} bod{'y' if self.n_bodies == 1 else 'ies'}"
             f" (threshold {self.threshold:g}, {self.connectivity} contact)"]
        if self.reasons:
            L.append("  why:            " + "; ".join(self.reasons))
        shown = [f"#{k + 1} {c} cells ({v:.4g})" for k, (c, v) in
                 enumerate(zip(self.body_cells[:5], self.body_volume[:5], strict=True))]
        if self.n_bodies > 5:
            shown.append(f"+{self.n_bodies - 5} more")
        L.append("  bodies:         " + (", ".join(shown) if shown else "none"))
        L.append("  interfaces:     " + ", ".join(
            f"{k} {'/'.join(f'#{b}' for b in v) if v else 'NOT ATTACHED'}" for k, v in self.interface_bodies.items()))
        if self.voids_matter:                        # 2-D closed holes are through holes: not shown
            L.append(f"  enclosed voids: {len(self.enclosed_voids)}"
                     + (f" ({sum(self.enclosed_voids)} cells)" if self.enclosed_voids else ""))
        m = self.min_neck
        if m is not None:
            L.append(f"  narrowest neck: {m.a}-{m.b} {m.cells} cells (area {m.area:.4g})")
        if self.checks:
            L.append("  robustness:     " + "; ".join(
                f"{c.label} {c.n_bodies} bod{'y' if c.n_bodies == 1 else 'ies'}, "
                f"{'connected' if c.connected else 'SPLIT'}" for c in self.checks))
        nxt = {"PASS": "use the design; re-run this check after any post-processing.",
               "FRAGILE": "thicken the weak links (tie_regions, a connectivity constraint, or a larger bridge section) "
                          "and check again.",
               "FAIL": "repair with bridge() or re-optimise with a connectivity constraint, then check again."}
        L.append(f"Result: {self.verdict}. Next: {nxt[self.verdict]}")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.summary()


def _attachment(lab: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[dict[str, list[int]], list[str], int, bool]:
    ib = {k: sorted(int(v) for v in np.unique(lab[m]) if v > 0) for k, m in masks.items()}
    unattached = [k for k, v in ib.items() if not v]
    counts: dict[int, int] = {}
    for v in ib.values():
        for b in v:
            counts[b] = counts.get(b, 0) + 1
    main = min(counts, key=lambda b: (-counts[b], b)) if counts else 0
    connected = not unattached and all(v == [main] for v in ib.values())
    return ib, unattached, main, connected


def _enclosed_voids(solid: np.ndarray) -> list[int]:
    """Cell counts of void components (face-connected, whole box) that touch no box face."""
    lab, n = ndimage.label(~solid, structure=_structure(solid.ndim, "face"))
    if n == 0:
        return []
    border = set()
    for ax in range(solid.ndim):
        for side in (0, -1):
            border |= set(np.unique(np.take(lab, side, axis=ax)).tolist())
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return sorted((int(sizes[k]) for k in range(1, n + 1) if k not in border), reverse=True)


def analyse_connectivity(domain: Domain, xphys: np.ndarray, interfaces: Mapping[str, Any], threshold: float = 0.5,
                         connectivity: str | int = "face", robust_thresholds: Sequence[float] | None = (0.3, 0.7),
                         erosion: bool = True, necks: bool = True, voids: str = "fail") -> ConnectivityReport:
    """Connectivity gate of a density design.

    Parameters
    ----------
    domain : Domain
    xphys : array of ``domain.shape``
        Physical density; solid = ``xphys ≥ threshold`` on the part.
    interfaces : mapping name → bool element mask
        Functional regions that must belong to the single body (supports: see :func:`elements_touching_nodes`).
    connectivity : "face" (default; printing and FE need face contact), "edge", "vertex" or 6 / 18 / 26 (4 / 8).
    robust_thresholds : thresholds of the robustness checks (with ``erosion``: the nominal solid eroded by one
        element, face neighbourhood; interface and keep cells do not erode, the box border counts as free surface).
    necks : compute the minimum cross-section between every pair of interfaces (max-flow / min-cut; interface and
        keep cells cannot be cut).
    voids : "fail" — enclosed voids (3-D: void components, face-connected, not reaching the box border; trapped
        powder / unremovable core) make the verdict FAIL; "warn" — reported only. In 2-D closed holes are through
        holes of the plate and never fail.

    Verdict
    -------
    FAIL: nominally not one body, or an interface not attached to it, or (3-D, ``voids="fail"``) enclosed voids.
    FRAGILE: nominally PASS, but at least one robustness realisation leaves the interfaces in more than one body
    (floating islands in a realisation are reported but not counted). PASS otherwise.
    """
    if voids not in ("fail", "warn"):
        raise ValueError('voids: "fail" | "warn"')
    x = np.asarray(xphys, float).reshape(domain.shape)
    masks = _interfaces(domain, interfaces)
    solid = (x >= threshold) & domain.active
    lab, sizes = label_bodies(solid, connectivity)
    n = len(sizes)
    vol = ndimage.sum_labels(domain.frac, lab, index=np.arange(1, n + 1)) * _cell_volume(domain) if n else []
    ib, unattached, main, connected = _attachment(lab, masks)
    has_iface = {b for v in ib.values() for b in v}
    floating = [b for b in range(1, n + 1) if b not in has_iface]
    sfrac = {k: float(solid[m].mean()) for k, m in masks.items()}
    voids_matter = domain.dim == 3
    enc = _enclosed_voids(solid)
    reasons = []
    if n == 0:
        reasons.append("no solid cell")
    if unattached:
        reasons.append("not attached: " + ", ".join(unattached))
    if not connected and not unattached:
        reasons.append(f"interfaces spread over {len(has_iface)} bodies")
    if floating:
        reasons.append(f"{len(floating)} floating bod{'y' if len(floating) == 1 else 'ies'} without interface")
    if voids_matter and enc and voids == "fail":
        reasons.append(f"{len(enc)} enclosed void(s)")
    passed = connected and n == 1 and not (voids_matter and enc and voids == "fail")
    neck_list: list[Neck] = []
    frozen = domain.keep | np.logical_or.reduce(list(masks.values()))
    if necks and connected:
        names = list(masks)
        for ia in range(len(names)):
            for jb in range(ia + 1, len(names)):
                c, where = min_cut_cells(solid, masks[names[ia]], masks[names[jb]], connectivity, frozen)
                neck_list.append(Neck(names[ia], names[jb], c, None if c is None else c * _cell_area(domain), where))
    checks: list[ThresholdCheck] = []
    for t in robust_thresholds or ():
        s = (x >= t) & domain.active
        lb, sz = label_bodies(s, connectivity)
        _, un, _, con = _attachment(lb, masks)
        checks.append(ThresholdCheck(f"t={t:g}", len(sz), con, un))
    if erosion:
        st = _structure(domain.dim, "face")
        s = ndimage.binary_erosion(solid, structure=st, border_value=0) | (solid & frozen)
        lb, sz = label_bodies(s, connectivity)
        _, un, _, con = _attachment(lb, masks)
        checks.append(ThresholdCheck("eroded", len(sz), con, un))
    fragile = passed and any(not c.connected for c in checks)
    if fragile:
        reasons.append("split in: " + ", ".join(c.label for c in checks if not c.connected))
    verdict = "PASS" if passed and not fragile else ("FRAGILE" if passed else "FAIL")
    return ConnectivityReport(float(threshold), connectivity, lab, n, [int(s) for s in sizes],
                              [float(v) for v in vol], ib, sfrac, unattached, main, connected, floating, enc,
                              voids_matter, voids, neck_list, checks, passed, fragile, verdict, reasons)


# ====================================================================================================== responses
def _aggregate(r: torch.Tensor, kind: str, p: float) -> torch.Tensor:
    if r.numel() == 0:
        raise ValueError("no load case")
    if kind == "pnorm":
        m = r.detach().abs().max().clamp_min(1e-300)
        return m * ((r / m) ** p).sum() ** (1.0 / p)
    if kind == "ks":
        m = r.detach().max()
        return m + torch.log(torch.exp(p * (r - m)).sum()) / p
    if kind == "mean":
        return r.mean()
    raise ValueError('aggregate: "pnorm" | "ks" | "mean"')


def _solid_design(domain: Domain) -> torch.Tensor:
    """x̄ = 1 on the part except void regions (the stiffest / best-conducting admissible design)."""
    return torch.as_tensor((domain.active & ~domain.void).astype(float), dtype=torch.float64)


class _RatioResponse(Response):
    """Shared part: per-case raw values → ratios to a reference → aggregate."""

    aggregate_kind = "pnorm"

    def __init__(self, domain: Domain, aggregate: str, p: float, reference: Any, eta: float | None) -> None:
        self.domain = domain
        self.eta = None if eta is None else float(eta)
        if self.eta is not None and not 0.0 < self.eta < 1.0:
            raise ValueError("0 < eta < 1 is required")
        self._passive = Passive(domain)
        if aggregate not in ("pnorm", "ks", "mean"):
            raise ValueError('aggregate: "pnorm" | "ks" | "mean"')
        self.aggregate_kind, self.p = aggregate, float(p)
        if isinstance(reference, str):
            if reference not in ("solid", "first"):
                raise ValueError('reference: "solid", "first", None or per-case values')
            self.reference_mode, self.ref = reference, None
        elif reference is None:
            self.reference_mode, self.ref = "none", None
        else:
            self.reference_mode, self.ref = "given", np.asarray(reference, float).reshape(-1)

    def field(self, ctx: EvalContext) -> torch.Tensor:
        """Density field the response sees: x̄, or with ``eta`` the eroded realisation — the filtered field (output
        of the chain's :class:`DensityFilter`) projected with threshold η at the chain's current β, passive regions
        applied."""
        if self.eta is None:
            return ctx.xphys
        xt = ctx.field_after(DensityFilter)
        proj = ctx.chain.find(HeavisideProjection)
        beta = proj.beta if proj is not None else 1.0
        return self._passive(HeavisideProjection(beta, self.eta)(xt))

    def raw(self, ctx: EvalContext) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def _reference(self, ctx: EvalContext, raw: torch.Tensor) -> np.ndarray:
        if self.ref is None:
            if self.reference_mode == "none":
                self.ref = np.ones(raw.numel())
            elif self.reference_mode == "first":
                self.ref = raw.detach().numpy().copy()
            else:
                self.ref = self.solid_reference(ctx)
        if self.ref.size != raw.numel():
            raise ValueError(f"{self.ref.size} reference values for {raw.numel()} cases")
        return self.ref

    def solid_reference(self, ctx: EvalContext) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def ratios(self, ctx: EvalContext) -> torch.Tensor:
        """Per-case value / reference (the reference is fixed after the first call)."""
        if self.reference_mode == "solid" and self.ref is None:
            self.ref = self.solid_reference(ctx)          # before the differentiable solve (shares the factor)
        r = self.raw(ctx)
        return r / torch.as_tensor(self._reference(ctx, r), dtype=r.dtype)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return _aggregate(self.ratios(ctx), self.aggregate_kind, self.p)


class ConductionConnectivity(_RatioResponse):
    """Virtual-conduction connectivity measure (constraint ``value ≤ bound``).

    One interface (``sink``, default the first) is held at T = 0. Every other interface i is a load case with unit
    total heat spread evenly over its nodes (sink nodes excluded); ``R_i = q_iᵀ T_i`` is its thermal resistance to the
    sink. Conductivity ``k = (k_min + x̄^p (1 − k_min)) · frac`` (:class:`~toporlogy.fem.heat.ConductivityInterp`).
    The value is the p-norm (or KS / mean) of ``R_i / R_i,ref``.

    Parameters
    ----------
    domain : Domain
    interfaces : mapping name → bool element mask
    sink : name of the reference interface (T = 0)
    kmin_ratio, penal : conductivity interpolation. Small k_min makes a cut-off interface heat up by ~1/k_min;
        penal > 1 makes grey material a poor conductor (a grey film cannot satisfy the constraint, so the
        connection survives thresholding).
    aggregate, p : "pnorm" (default, p = 8), "ks" (p = ρ) or "mean".
    eta : None → conductivity of x̄ (the study's physical density). A float (e.g. 0.6-0.7) → conductivity of the
        *eroded* realisation: the filtered field (output of the chain's :class:`DensityFilter`) projected with
        threshold η at the chain's current β, then passive regions applied. Requiring the eroded design to conduct
        makes the nominal one connected with margin (robust to thresholding and to a thinner print).
    reference : "solid" (default: value of the all-solid admissible design, computed once), "first" (value at the
        first evaluation, i.e. the start design), None (raw resistances) or per-case values.
    solver : linear solver of the conduction model (default DirectSolver).

    Calibration (see the module notes and the test report): for a connected 0/1 design the ratios are
    O(1/volume fraction × path tortuosity), for a disconnected one O(1/k_min). Any bound in that gap separates them;
    with grey start designs the ratio is O(1/x̄^p), so the start may be infeasible — MMA restores feasibility.
    """

    name = "conduction_connectivity"

    def __init__(self, domain: Domain, interfaces: Mapping[str, Any], sink: str | None = None, *,
                 kmin_ratio: float = 1e-3, penal: float = 3.0, aggregate: str = "pnorm", p: float = 8.0,
                 eta: float | None = None, reference: Any = "solid", solver: Any = None) -> None:
        super().__init__(domain, aggregate, p, reference, eta)
        masks = _interfaces(domain, interfaces)
        names = list(masks)
        if len(names) < 2:
            raise ValueError("need at least two interfaces (a sink and a source)")
        self.sink = names[0] if sink is None else str(sink)
        if self.sink not in masks:
            raise KeyError(f"sink {self.sink!r} is not an interface")
        self.interp = ConductivityInterp(domain, penal=penal, kmin_ratio=kmin_ratio)
        sink_nodes = _nodes_of(domain, masks[self.sink])
        self.physics = HeatConduction(domain, sink_nodes, solver)
        cols, self.sources = [], []
        for k in names:
            if k == self.sink:
                continue
            nodes = np.setdiff1d(_nodes_of(domain, masks[k]), sink_nodes)
            if nodes.size == 0:
                continue                                    # lies inside the sink: connected by definition
            q = np.zeros(domain.nn)
            q[nodes] = 1.0 / nodes.size
            cols.append(q)
            self.sources.append(k)
        if not cols:
            raise ValueError("every interface lies inside the sink")
        self.Q = np.stack(cols, axis=1)
        self._active = torch.as_tensor(domain.active_elements, dtype=torch.long)

    def resistances(self, xfield: torch.Tensor) -> torch.Tensor:
        """``R_i`` (n_sources,) of a density field (differentiable)."""
        k = self.interp(xfield).reshape(-1)[self._active]
        T = self.physics.solve(k, self.Q)
        return (torch.as_tensor(self.Q, dtype=T.dtype) * T).sum(dim=0)

    def raw(self, ctx: EvalContext) -> torch.Tensor:
        return self.resistances(self.field(ctx))

    def solid_reference(self, ctx: EvalContext | None = None) -> np.ndarray:
        with torch.no_grad():
            return self.resistances(_solid_design(self.domain)).numpy().copy()

    def ratios_of(self, xphys: np.ndarray) -> dict[str, float]:
        """Ratios per source interface for a given physical density (no gradients; for calibration / reports)."""
        if self.ref is None and self.reference_mode == "solid":
            self.ref = self.solid_reference()
        with torch.no_grad():
            r = self.resistances(torch.as_tensor(np.asarray(xphys, float).reshape(self.domain.shape)))
        ref = self.ref if self.ref is not None else np.ones(r.numel())
        return dict(zip(self.sources, (r.numpy() / ref).tolist(), strict=True))


class CouplingCompliance(_RatioResponse):
    """Auxiliary coupling cases: one interface clamped, unit forces on every other interface.

    ``held`` (default the first interface) has all DOFs of its nodes fixed — the supports of the main problem are
    NOT used. For every other interface and every direction in ``directions`` (default all axes) one load case
    applies a unit total force spread evenly over the interface nodes. Each case is a compliance ``F_cᵀ U_c``
    (self-adjoint: no extra solve for the gradient). The value is the p-norm (or KS / mean) of the ratios to the
    reference. The element moduli are the study's ``ctx.E`` (same interpolation as the main problem); with ``eta``
    they are the interpolation of the eroded realisation (see :class:`ConductionConnectivity`), which makes the
    connection survive thresholding.

    The forces are deliberately NOT self-equilibrated among the loaded interfaces: a set that is equilibrated by
    itself does no work on a rigid motion, so an island holding all loaded interfaces but floating free of the held
    one would still be stiff against it, and the constraint would not see the gap. Here the held interface takes the
    reactions, so every loaded interface needs a stiff path to it.

    ``reference``: "solid" (default: E of the all-solid admissible design through ``ctx.interp``, computed at the
    first call), "first", None or per-case values.
    """

    name = "coupling_compliance"

    def __init__(self, domain: Domain, interfaces: Mapping[str, Any], held: str | None = None, *,
                 directions: Sequence[int] | None = None, nu: float = 0.3, element: str | None = None,
                 aggregate: str = "pnorm", p: float = 8.0, eta: float | None = None, reference: Any = "solid",
                 solver: Any = None) -> None:
        super().__init__(domain, aggregate, p, reference, eta)
        masks = _interfaces(domain, interfaces)
        names = list(masks)
        if len(names) < 2:
            raise ValueError("need at least two interfaces (one held, one loaded)")
        self.held = names[0] if held is None else str(held)
        if self.held not in masks:
            raise KeyError(f"held interface {self.held!r} is not an interface")
        held_nodes = _nodes_of(domain, masks[self.held])
        self.physics = Elasticity(domain, domain.dofs(held_nodes), nu=nu, element=element, solver=solver)
        dirs = list(range(domain.dim)) if directions is None else [int(d) for d in directions]
        cols, self.cases = [], []
        for k in names:
            if k == self.held:
                continue
            nodes = np.setdiff1d(_nodes_of(domain, masks[k]), held_nodes)
            if nodes.size == 0:
                continue
            for d in dirs:
                f = np.zeros(domain.ndof)
                f[domain.dim * nodes + d] = 1.0 / nodes.size
                cols.append(f)
                self.cases.append((k, d))
        if not cols:
            raise ValueError("every interface lies inside the held one")
        self.F = np.stack(cols, axis=1)
        self._active = torch.as_tensor(domain.active_elements, dtype=torch.long)

    def compliances(self, E: torch.Tensor) -> torch.Tensor:
        """Per-case compliance for an element modulus grid (differentiable)."""
        U = self.physics.solve(E.reshape(-1)[self._active], self.F)
        return (torch.as_tensor(self.F, dtype=U.dtype) * U).sum(dim=0)

    def raw(self, ctx: EvalContext) -> torch.Tensor:
        if self.eta is None:
            return self.compliances(ctx.E)
        if ctx.interp is None:
            raise ValueError("eta needs ctx.interp (the study interpolation)")
        return self.compliances(ctx.interp(self.field(ctx)))

    def solid_reference(self, ctx: EvalContext) -> np.ndarray:
        if ctx.interp is None:
            raise ValueError('reference="solid" needs ctx.interp (the study interpolation)')
        with torch.no_grad():
            return self.compliances(ctx.interp(_solid_design(self.domain))).numpy().copy()


# ====================================================================================================== paths, ties
def _path_graph(allowed: np.ndarray, cost: np.ndarray) -> sp.csr_matrix:
    """Directed face-neighbour graph on all box cells; edge p→q weighs cost[q] (> 0) and exists if both allowed."""
    p, q = _neighbour_pairs(allowed, _structure(allowed.ndim, "face"))
    c = cost.ravel()
    n = allowed.size
    return sp.csr_matrix((np.concatenate([c[q], c[p]]), (np.concatenate([p, q]), np.concatenate([q, p]))),
                         shape=(n, n))


def _nearest_path(G: sp.csr_matrix, tree: np.ndarray, targets: list[np.ndarray]) -> tuple[int, np.ndarray] | None:
    """Shortest path from any cell of ``tree`` to the nearest target set: (target index, path cells) or None."""
    dist, pred, _ = csgraph.dijkstra(G, directed=True, indices=np.flatnonzero(tree.ravel()),
                                     return_predecessors=True, min_only=True)
    best, bk, bt = np.inf, -1, -1
    for k, t in enumerate(targets):
        cells = np.flatnonzero(t.ravel())
        if cells.size == 0:
            continue
        i = cells[np.argmin(dist[cells])]
        if dist[i] < best:
            best, bk, bt = float(dist[i]), k, int(i)
    if not np.isfinite(best):
        return None
    path = [bt]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return bk, np.asarray(path[::-1], np.int64)


def _edge_cells(allowed: np.ndarray, section: int) -> np.ndarray:
    """Allowed cells closer than ``section // 2`` (chessboard) to a non-allowed cell or the box border: a path there
    would be clipped below the requested section, so path costs penalise them."""
    r = int(section) // 2
    if r == 0:
        return np.zeros(allowed.shape, bool)
    padded = np.pad(allowed, 1, constant_values=False)
    dist = ndimage.distance_transform_cdt(padded, metric="chessboard")
    inner = tuple(slice(1, -1) for _ in range(allowed.ndim))
    return allowed & (dist[inner] <= r)


def _thicken(path_mask: np.ndarray, section: int, allowed: np.ndarray) -> np.ndarray:
    s = int(section)
    if s < 1:
        raise ValueError("section ≥ 1 (cells) is required")
    out = path_mask if s == 1 else ndimage.binary_dilation(path_mask, structure=np.ones((s,) * path_mask.ndim, bool))
    return (out | path_mask) & allowed


def tie_regions(domain: Domain, groups: Mapping[str, Any], section: int = 1, avoid: np.ndarray | None = None,
                keep_cost: float = 1e-3) -> np.ndarray:
    """Keep-mask of minimal ties connecting all interface groups through the design domain.

    Prim-like Steiner tree: start from the first group; repeatedly add the shortest face-connected path (Dijkstra,
    cost 1 per cell, ``keep_cost`` through cells already frozen solid) from the current tree to the nearest
    unconnected group. Paths avoid cells outside the part, ``domain.void`` and ``avoid``. Each path is thickened to a
    ``section`` × ``section`` (× ``section``) box cross-section (clipped by the allowed cells; new cells within
    ``section // 2`` of a free boundary — box border, outside the part, void — cost one unit more, so paths keep
    their full section where they can).

    Returns the bool element mask of the ties (group cells not included); freeze it with :func:`with_keep`.
    Raises RuntimeError if a group cannot be reached.
    """
    masks = _interfaces(domain, groups)
    allowed = domain.active & ~domain.void
    if avoid is not None:
        allowed &= ~np.asarray(avoid, bool)
    for k, m in masks.items():
        if not (m & allowed).any():
            raise ValueError(f"group {k!r} lies entirely in excluded cells")
    cost = np.where(domain.keep, float(keep_cost), 1.0 + _edge_cells(allowed, section))
    G = _path_graph(allowed, cost)
    names = list(masks)
    tree = masks[names[0]] & allowed
    remaining = names[1:]
    paths = np.zeros(domain.shape, bool)
    while remaining:
        found = _nearest_path(G, tree, [masks[k] & allowed for k in remaining])
        if found is None:
            raise RuntimeError(f"groups {remaining} cannot be reached from {names[0]!r} through the design domain")
        k, cells = found
        pm = np.zeros(domain.ne, bool)
        pm[cells] = True
        pm = pm.reshape(domain.shape)
        paths |= pm
        tree = tree | pm | (masks[remaining[k]] & allowed)
        remaining.pop(k)
    group_cells = np.logical_or.reduce(list(masks.values()))
    return _thicken(paths, section, allowed) & ~group_cells


def with_keep(domain: Domain, extra_keep: np.ndarray) -> Domain:
    """Copy of ``domain`` with ``extra_keep`` added to the frozen-solid region (removed from void)."""
    extra = np.asarray(extra_keep, bool).reshape(domain.shape)
    return Domain(domain.shape, domain.h, origin=domain.origin, frac=domain.frac, keep=domain.keep | extra,
                  void=domain.void & ~extra, thickness=domain.thickness)


# ====================================================================================================== repair
@dataclass
class BridgeResult:
    """Output of :func:`bridge`. Masks are element grids of ``domain.shape``."""

    xphys: np.ndarray                 # repaired density (added cells = 1, dropped islands = 0)
    solid: np.ndarray                 # repaired thresholded design (one body holding every interface)
    added: np.ndarray                 # cells set solid by bridges
    removed: np.ndarray               # cells of dropped floating islands
    forced: np.ndarray                # interface cells that were void and were set solid
    n_bridges: int
    added_cells: int
    added_volume: float
    removed_cells: int
    paths: list[np.ndarray] = field(default_factory=list)   # flat cell numbers of every path (before thickening)


def bridge(domain: Domain, xphys: np.ndarray, interfaces: Mapping[str, Any], threshold: float = 0.5,
           section: int = 2, connectivity: str | int = "face", void_cost: float = 1.0, solid_cost: float = 1e-3,
           drop_islands: bool = True, binary: bool = False) -> BridgeResult:
    """Repair: join every body holding an interface into one body with minimal-volume bridges.

    The thresholded design is grown from the body of the first interface: each step adds the cheapest face-connected
    path (multi-source Dijkstra from the current body) to the nearest body that holds an interface and is not yet
    joined. Cell cost = ``solid_cost`` for solid cells, ``solid_cost + void_cost · (1 − x̄)`` otherwise (grey is
    cheaper than void; new cells within ``section // 2`` of a free boundary cost ``void_cost`` more, so the
    thickened bridge is not clipped), so the path follows existing material and adds as little as possible. Paths
    stay in the part and out of ``domain.void``; each is thickened to a ``section``-wide box section and set
    solid. Interface cells that are void are set solid first (``forced``). With ``drop_islands`` bodies holding no
    interface are removed afterwards (bodies crossed by a bridge are kept: they became part of the body).

    The result is one body (``connectivity`` sense) containing every interface, checked here; the call raises
    RuntimeError if a body cannot be reached. It is NOT re-analysed: stiffness, stress and :func:`analyse_connectivity`
    of the repaired design (other thresholds, erosion) are the caller's job.

    ``binary=True`` returns ``xphys`` as the 0/1 repaired design (then any threshold gives the same body).
    """
    x = np.clip(np.asarray(xphys, float).reshape(domain.shape), 0.0, 1.0)
    masks = _interfaces(domain, interfaces)
    allowed = domain.active & ~domain.void
    iface = np.logical_or.reduce(list(masks.values())) & allowed
    solid = (x >= threshold) & domain.active
    forced = iface & ~solid
    solid = solid | forced
    names = list(masks)
    added = np.zeros(domain.shape, bool)
    paths: list[np.ndarray] = []
    first = masks[names[0]] & solid
    edge = _edge_cells(allowed, section)

    while True:
        lab, _ = label_bodies(solid, connectivity)
        root = int(np.bincount(lab[first]).argmax()) if first.any() else 0
        need = sorted({int(b) for m in masks.values() for b in np.unique(lab[m & solid]) if b > 0} - {root})
        if not need:
            break
        cost = np.where(solid, solid_cost, solid_cost + void_cost * (1.0 - x) + void_cost * edge)
        G = _path_graph(allowed, cost)
        found = _nearest_path(G, lab == root, [lab == b for b in need])
        if found is None:
            raise RuntimeError(f"bodies {need} cannot be reached through the design domain")
        _, cells = found
        paths.append(cells)
        pm = np.zeros(domain.ne, bool)
        pm[cells] = True
        grow = _thicken(pm.reshape(domain.shape), section, allowed)
        added |= grow & ~solid
        solid = solid | grow

    lab, _ = label_bodies(solid, connectivity)
    root = int(np.bincount(lab[first]).argmax())
    removed = np.zeros(domain.shape, bool)
    if drop_islands:
        removed = solid & (lab != root)
        solid = solid & ~removed
    # guarantee (deterministic check of what this function promises)
    lab, sizes = label_bodies(solid, connectivity)
    bodies = {int(b) for m in masks.values() for b in np.unique(lab[m & solid]) if b > 0}
    if len(bodies) != 1 or (drop_islands and len(sizes) != 1):
        raise RuntimeError(f"bridge failed: {len(sizes)} bodies, interfaces in {sorted(bodies)}")
    if binary:
        xr = solid.astype(float)
    else:
        xr = x.copy()
        xr[added | forced] = 1.0
        xr[removed] = 0.0
    n_add = int(added.sum())
    return BridgeResult(xr, solid, added, removed, forced, len(paths), n_add,
                        float((domain.frac * added).sum() * _cell_volume(domain)), int(removed.sum()), paths)


# ====================================================================================================== benchmark
@dataclass
class SplitBenchmark:
    """A load case that splits under plain compliance minimisation (two load pads, each near its own support)."""

    domain: Domain
    fixed: np.ndarray                 # fixed DOFs of the main problem
    loads: np.ndarray                 # (ndof, 1)
    interfaces: dict[str, np.ndarray]


def split_benchmark(dim: int = 2, n: int = 24, coupling: float = 0.02, h: float = 1.0) -> SplitBenchmark:
    """Two-span benchmark that falls apart into two bodies without a connectivity measure.

    Box ``(2.5 n) × n`` (2-D, y up) or ``(2.5 n) × (n/3) × n`` (3-D, z up). Two support pads on the bottom face near
    the two ends (all DOFs of their nodes fixed); two load pads at mid-height, each above-inside its own support,
    loaded downward with unit total force, plus a weak self-equilibrated pull ``±coupling`` along x that tries to
    separate the pads. Every pad (support rows and load blocks) is a keep region and an interface: names
    ``support_a``, ``support_b``, ``load_a``, ``load_b``.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    n = int(n)
    nx = int(round(2.5 * n))
    shape = (nx, n) if dim == 2 else (nx, max(2, n // 3), n)
    up = dim - 1                                             # vertical axis
    ex = np.indices(shape)
    pad = max(2, n // 6)                                     # load-pad edge (elements)
    sw = max(2, n // 4)                                      # support-pad width (elements)
    xa0, xb0 = n // 6, nx - n // 6 - sw                      # support pads [x0, x0 + sw)
    sup_a = (ex[0] >= xa0) & (ex[0] < xa0 + sw) & (ex[up] == 0)
    sup_b = (ex[0] >= xb0) & (ex[0] < xb0 + sw) & (ex[up] == 0)
    la0, lb0 = xa0 + (sw - pad) // 2, xb0 + (sw - pad) // 2   # load pads above the support pads
    z0 = n // 2 - pad // 2
    load_a = (ex[0] >= la0) & (ex[0] < la0 + pad) & (ex[up] >= z0) & (ex[up] < z0 + pad)
    load_b = (ex[0] >= lb0) & (ex[0] < lb0 + pad) & (ex[up] >= z0) & (ex[up] < z0 + pad)
    keep = sup_a | sup_b | load_a | load_b
    d = Domain(shape, h, keep=keep)
    ifc = {"support_a": sup_a, "support_b": sup_b, "load_a": load_a, "load_b": load_b}
    fixed_nodes = np.concatenate([np.intersect1d(_nodes_of(d, m), d.nodes_where(lambda X: X[:, up] < 1e-9 * h))
                                  for m in (sup_a, sup_b)])
    fixed = d.dofs(fixed_nodes)
    F = d.load_vector(1)
    for m, sx in ((load_a, -1.0), (load_b, 1.0)):
        nodes = _nodes_of(d, m)
        F[d.dim * nodes + up, 0] -= 1.0 / nodes.size
        F[d.dim * nodes + 0, 0] += sx * coupling / nodes.size
    return SplitBenchmark(d, fixed, F, ifc)


# ====================================================================================================== one-call plan
@dataclass
class OneBodyPlan:
    """Everything needed to keep a design in one body, from :func:`one_body`.

    Use ``plan.domain`` (the input domain with the minimal ties frozen) to build the chain and physics, add
    ``plan.constraints`` to the study, and call ``plan.finish(xphys)`` on the result (gate + optional repair).
    """

    domain: Domain
    interfaces: dict[str, np.ndarray]
    ties: np.ndarray | None
    constraints: list[Any]
    threshold: float = 0.5
    repair_section: int = 2

    def gate(self, xphys: np.ndarray, **kw: Any) -> ConnectivityReport:
        """:func:`analyse_connectivity` with the plan's interfaces (thresholds 0.3/0.7 and erosion included)."""
        return analyse_connectivity(self.domain, xphys, self.interfaces, threshold=kw.pop("threshold", self.threshold),
                                    **kw)

    def finish(self, xphys: np.ndarray, repair: bool = True,
               **kw: Any) -> tuple[np.ndarray, ConnectivityReport, BridgeResult | None]:
        """Gate the design; if it is not PASS and ``repair``, join the bodies with :func:`bridge` and gate again;
        realisations that still split (a stricter threshold, the eroded solid) get section-3 bridges (x̄ = 1), up to
        three rounds. ``BridgeResult.added`` holds all added cells.

        Returns (design, final report, bridge result or None). A repaired design must be re-verified (stiffness,
        stress) by the caller: bridges are geometric, not optimised.
        """
        rep = self.gate(xphys, **kw)
        if rep.verdict == "PASS" or not repair:
            return np.asarray(xphys, float), rep, None
        br = bridge(self.domain, xphys, self.interfaces, threshold=self.threshold, section=self.repair_section)
        x = br.xphys
        rep = self.gate(x, **kw)
        # A bridge of section ≤ 2 vanishes under the one-element erosion of the gate, and bridge() adds nothing to a
        # design that is one body at the nominal threshold but splits at a stricter realisation (FRAGILE). Repair
        # each split realisation (threshold t > nominal, or the eroded solid) with section-3 paths set to x̄ = 1:
        # such a path keeps a one-cell core after the erosion and is solid at every threshold.
        for _ in range(3):
            if rep.verdict == "PASS":
                break
            added = np.zeros(self.domain.shape, bool)
            for c in rep.checks:
                if c.connected:
                    continue
                if c.label == "eroded":
                    solid = (x >= self.threshold) & self.domain.active
                    frozen = self.domain.keep | np.logical_or.reduce(list(self.interfaces.values()))
                    er = ndimage.binary_erosion(solid, structure=_structure(self.domain.dim, "face"),
                                                border_value=0) | (solid & frozen)
                    b2 = bridge(self.domain, er.astype(float), self.interfaces, section=3, drop_islands=False)
                else:
                    b2 = bridge(self.domain, x, self.interfaces, threshold=float(c.label[2:]), section=3,
                                drop_islands=False)
                added |= b2.added | b2.forced
            if not added.any():
                break
            x = x.copy()
            x[added] = 1.0
            br.added = br.added | (added & ~br.added)
            br.added_cells = int(br.added.sum())
            br.added_volume = float((self.domain.frac * br.added).sum() * _cell_volume(self.domain))
            br.solid = (x >= self.threshold) & self.domain.active
            br.xphys = x
            rep = self.gate(x, **kw)
        return x, rep, br


def one_body(domain: Domain, interfaces: Mapping[str, Any], *, ties: int | None = 1,
             constraint: str | None = "conduction", eta: float | None = 0.7, bound: float | None = None,
             sink: str | None = None, threshold: float = 0.5, repair_section: int = 2,
             **response_kw: Any) -> OneBodyPlan:
    """Layered protection against a design that falls apart (defaults from the measured comparison in
    ``tests/toporlogy/test_connectivity.py``: only frozen ties survived the one-element erosion test; the conduction and
    coupling constraints are robust to the 0.3/0.7 thresholds only when they see the eroded field, ``eta`` ≈ 0.7).

    interfaces : name → bool element mask, or name → node numbers (converted to the elements touching them).
    ties : section (elements) of frozen minimal ties between the interfaces; None/0 = no ties.
    constraint : "conduction" (virtual heat conduction, bound default 20 with k_min 1e-3), "coupling" (auxiliary
        structural case, bound default 2.5 × first value) or None.
    """
    from toporlogy.study import Constraint
    masks: dict[str, np.ndarray] = {}
    for name, v in interfaces.items():
        a = np.asarray(v)
        if a.dtype == bool and a.shape == domain.shape:
            masks[str(name)] = a & domain.active
        else:
            masks[str(name)] = elements_touching_nodes(domain, a.astype(np.int64))
    masks = _interfaces(domain, masks)
    tie_mask = None
    dom = domain
    if ties:
        tie_mask = tie_regions(domain, masks, section=int(ties))
        # freeze the interface cells too: a tie ending on an interface cell that the optimiser empties is cut
        # (measured on the splitting benchmark: interface cells at x̄ = 0.014 left 2 bodies despite the tie)
        for m in masks.values():
            tie_mask = tie_mask | m
        dom = with_keep(domain, tie_mask)
    cons: list[Any] = []
    if constraint == "conduction":
        resp = ConductionConnectivity(dom, masks, sink=sink, eta=eta, **response_kw)
        cons.append(Constraint(resp, 20.0 if bound is None else float(bound), name="connectivity"))
    elif constraint == "coupling":
        response_kw.setdefault("reference", "first")
        resp = CouplingCompliance(dom, masks, held=sink, eta=eta, **response_kw)
        cons.append(Constraint(resp, 2.5 if bound is None else float(bound), name="connectivity"))
    elif constraint is not None:
        raise ValueError("constraint: conduction | coupling | None")
    return OneBodyPlan(dom, masks, tie_mask, cons, threshold, repair_section)


__all__ += ["OneBodyPlan", "one_body"]
