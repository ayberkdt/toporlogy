"""Load cases, combinations and ``BoundaryConditions`` (assembly, physics, check, summary).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from scipy.spatial import cKDTree
from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.geometry import Shape
from typing import Any

import numpy as np

from toporlogy.boundary.report import CheckReport, Issue, _describe_mode, _format_loads, _rb_basis, _table
from toporlogy.boundary.constrained import (
    ConstrainedElasticity, ConstraintSet, PerCaseSupports, _constraint_key, _resolve,
)
from toporlogy.boundary.loads import Applied, Load
from toporlogy.boundary.part import EmptySelectionError, part_info
from toporlogy.boundary.supports import Support


# =============================================================================================== cases and container
@dataclass
class LoadCase:
    """Named load case; ``supports`` None = the container's default supports."""

    name: str
    loads: Sequence[Load]
    supports: Sequence[Support] | None = None


@dataclass
class Combination:
    """Linear combination of load cases: column = Σ factor × case column (cases must share their supports)."""

    name: str
    factors: dict[str, float]


class BoundaryConditions:
    """Supports, named load cases and combinations on a domain; builds ``F``, the physics and the checks.

    Parameters
    ----------
    domain : Domain
    supports : default supports of every case without its own
    cases : load cases (names unique)
    combinations : optional linear combinations; :meth:`F` then returns cases followed by combinations
    """

    def __init__(self, domain: Domain, supports: Sequence[Support], cases: Sequence[LoadCase],
                 combinations: Sequence[Combination] = ()) -> None:
        self.domain = domain
        self.info = part_info(domain)
        self.supports = list(supports)
        self.cases = list(cases)
        self.combinations = list(combinations)
        names = [c.name for c in self.cases] + [c.name for c in self.combinations]
        if len(set(names)) != len(names):
            raise ValueError(f"load case / combination names must be unique: {names}")
        if not self.cases:
            raise ValueError("at least one load case is required")
        self.applied: dict[str, list[Applied]] = {}
        for c in self.cases:
            out = []
            for ld in c.loads:
                try:
                    out.append(ld.apply(self.info))
                except EmptySelectionError as exc:
                    raise EmptySelectionError(f"load case {c.name!r}, load {ld.name!r}: {exc}") from None
            self.applied[c.name] = out
        # support sets, identified by what they constrain on this part (fixed DOFs, local frames, springs), not by
        # the identity of the support objects: equal supports built separately are one set (one factorisation, and
        # combinations of their cases are valid)
        self._sets: list[tuple[bytes, list[Support]]] = []
        self._case_set: list[int] = []
        self._cs: dict[int, ConstraintSet] = {}
        resolved: dict[tuple[int, ...], ConstraintSet] = {}
        for c in self.cases:
            sup = list(c.supports) if c.supports is not None else self.supports
            if not sup:
                raise ValueError(f"load case {c.name!r} has no supports")
            ids = tuple(id(s) for s in sup)
            if ids not in resolved:
                try:
                    resolved[ids] = _resolve(sup, self.info)
                except EmptySelectionError as exc:
                    raise EmptySelectionError(f"supports of case {c.name!r}: {exc}") from None
            cs = resolved[ids]
            key = _constraint_key(cs)
            for k, (kk, _) in enumerate(self._sets):
                if kk == key:
                    self._case_set.append(k)
                    break
            else:
                self._sets.append((key, sup))
                self._cs[len(self._sets) - 1] = cs
                self._case_set.append(len(self._sets) - 1)
        idx = {c.name: i for i, c in enumerate(self.cases)}
        for cb in self.combinations:
            miss = [n for n in cb.factors if n not in idx]
            if miss:
                raise ValueError(f"combination {cb.name!r}: unknown case(s) {miss}")
            sets = {self._case_set[idx[n]] for n in cb.factors}
            if len(sets) > 1:
                raise ValueError(f"combination {cb.name!r} mixes cases with different supports - superposition "
                                 "is only valid for one support set")

    # ------------------------------------------------------------------ assembly
    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cases] + [c.name for c in self.combinations]

    def case_vector(self, name: str) -> np.ndarray:
        """(ndof,) load vector of one case."""
        d = self.domain
        f = np.zeros(d.ndof)
        for a in self.applied[name]:
            np.add.at(f, (d.dim * a.nodes[:, None] + np.arange(d.dim)).ravel(), a.forces.ravel())
        return f

    def F(self, which: str = "all") -> np.ndarray:
        """(ndof, n) load array: "cases", "combinations" or "all" (cases then combinations)."""
        cols = [self.case_vector(c.name) for c in self.cases]
        base = np.column_stack(cols)
        idx = {c.name: i for i, c in enumerate(self.cases)}
        comb = [sum(f * base[:, idx[n]] for n, f in cb.factors.items()) for cb in self.combinations]
        if which == "cases":
            return base
        if which == "combinations":
            if not comb:
                raise ValueError("no combinations defined")
            return np.column_stack(comb)
        if which != "all":
            raise ValueError("which: 'all' | 'cases' | 'combinations'")
        return np.column_stack([*cols, *comb])

    def _column_sets(self, which: str = "all") -> list[int]:
        idx = {c.name: i for i, c in enumerate(self.cases)}
        sets = list(self._case_set)
        comb = [self._case_set[idx[next(iter(cb.factors))]] for cb in self.combinations]
        return sets if which == "cases" else (comb if which == "combinations" else sets + comb)

    def constraints(self, case: str | int = 0) -> ConstraintSet:
        """Resolved supports of a load case (name or index)."""
        k = case if isinstance(case, int) else [c.name for c in self.cases].index(case)
        return self._cs[self._case_set[k]]

    def fixed_dofs(self, case: str | int = 0) -> np.ndarray:
        """Globally fixed DOFs of a case (oblique constraints and springs are not included: see ``constraints``)."""
        return self.constraints(case).fixed

    def _make(self, cs: ConstraintSet, nu: float, element: str | None, solver: Any,
              elasticity: np.ndarray | None = None) -> Elasticity:
        if cs.needs_extension:
            return ConstrainedElasticity(self.domain, cs, nu=nu, element=element, solver=solver,
                                         elasticity=elasticity)
        return Elasticity(self.domain, cs.fixed, nu=nu, element=element, solver=solver, elasticity=elasticity)

    def physics_per_support_set(self, nu: float = 0.3, element: str | None = None,
                                solver_factory: Callable[[], Any] | None = None,
                                which: str = "all", elasticity: np.ndarray | None = None,
                                ) -> list[tuple[Elasticity, np.ndarray]]:
        """One physics per distinct support set with the column indices of ``F(which)`` it solves; ``elasticity``
        is the unit-modulus D of a non-isotropic material (see :class:`~toporlogy.fem.elastic.Elasticity`)."""
        sets = self._column_sets(which)
        out = []
        for k in sorted(set(sets)):
            cols = np.flatnonzero(np.asarray(sets) == k)
            out.append((self._make(self._cs[k], nu, element, solver_factory() if solver_factory else None,
                                   elasticity), cols))
        return out

    def physics(self, nu: float = 0.3, element: str | None = None,
                solver_factory: Callable[[], Any] | None = None, which: str = "all",
                elasticity: np.ndarray | None = None) -> Any:
        """Physics for ``Study``: Elasticity / ConstrainedElasticity, or PerCaseSupports for several support sets."""
        groups = self.physics_per_support_set(nu, element, solver_factory, which, elasticity)
        return groups[0][0] if len(groups) == 1 else PerCaseSupports(groups)

    # ------------------------------------------------------------------ non-design regions
    def bc_nodes(self, supports: bool = True, loads: bool = True) -> np.ndarray:
        parts = []
        if supports:
            for cs in self._cs.values():
                parts += [p[2] for p in cs.per_support]
        if loads:
            for lst in self.applied.values():
                parts += [a.nodes for a in lst if a.kind != "body force"]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def keep_mask(self, thickness: float | None = None, supports: bool = True, loads: bool = True) -> np.ndarray:
        """Frozen pads / rings: active elements whose centre lies within ``thickness`` (≥ 2 h, default 2 h) of a
        support or load node. Returns a bool array of ``domain.shape`` (pass as ``Domain(keep=…)``)."""
        d = self.domain
        t = max(2.0 * d.h if thickness is None else float(thickness), 2.0 * d.h)
        nodes = self.bc_nodes(supports, loads)
        out = np.zeros(d.ne, bool)
        if len(nodes):
            act = d.active_elements
            dist, _ = cKDTree(self.info.X[nodes]).query(d.element_centers()[act], distance_upper_bound=t + 1e-9 * d.h)
            out[act[dist <= t + 1e-9 * d.h]] = True
        return out.reshape(d.shape)

    # ------------------------------------------------------------------ checks
    def _rigid_modes(self, cs: ConstraintSet) -> list[tuple[int, list[str]]]:
        """Per face-connected component: (number of nodes, descriptions of unrestrained rigid-body modes)."""
        from scipy import ndimage
        d, info = self.domain, self.info
        lab, ncomp = ndimage.label(d.active)
        lab_act = lab.ravel()[d.active_elements]
        cn, cd = cs.constrained_directions(d.dim)
        out = []
        for k in range(1, ncomp + 1):
            nodes = np.unique(info.elem_nodes[lab_act == k])
            inset = np.zeros(d.nn, bool)
            inset[nodes] = True
            X = info.X[nodes]
            c = X.mean(axis=0)
            L = max(float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))), d.h)
            rows = []
            m = inset[cn] if len(cn) else np.zeros(0, bool)
            if m.any():
                rows.append(np.einsum("ia,iak->ik", cd[m], _rb_basis(info.X[cn[m]], c, L)))
            if len(cs.spring_nodes):
                ms = inset[cs.spring_nodes]
                if ms.any():
                    K = cs.spring_K[ms]
                    tr = np.maximum(np.trace(K, axis1=1, axis2=2), 1e-300)
                    Phi = _rb_basis(info.X[cs.spring_nodes[ms]], c, L)
                    rows.append(np.einsum("iab,ibk->iak", K / tr[:, None, None], Phi).reshape(-1, Phi.shape[2]))
            nrb = 6 if d.dim == 3 else 3
            if rows:
                C = np.concatenate(rows)
                _, s, Vt = np.linalg.svd(C, full_matrices=True)
                rank = int((s > 1e-8 * max(s[0], 1e-300)).sum())
                N = Vt[rank:]
            else:
                N = np.eye(nrb)
            descr = []
            if len(N):
                # readable basis: reduced row echelon form of the null space
                R = N.copy()
                piv_row = 0
                for col in range(nrb):
                    if piv_row >= len(R):
                        break
                    p = piv_row + int(np.argmax(np.abs(R[piv_row:, col])))
                    if abs(R[p, col]) < 1e-9:
                        continue
                    R[[piv_row, p]] = R[[p, piv_row]]
                    R[piv_row] /= R[piv_row, col]
                    for r_ in range(len(R)):
                        if r_ != piv_row:
                            R[r_] -= R[r_, col] * R[piv_row]
                    piv_row += 1
                descr = [_describe_mode(th, c, L, d.dim) for th in R]
            out.append((len(nodes), descr))
        return out

    def check(self, rtol: float = 1e-9) -> CheckReport:
        """All checks (listed in the :mod:`toporlogy.boundary` notes); the report lists errors, warnings and infos."""
        rep = CheckReport()
        d, info = self.domain, self.info
        dim = d.dim
        add = rep.issues.append
        for ci, c in enumerate(self.cases):
            cs = self._cs[self._case_set[ci]]
            fvec = self.case_vector(c.name)
            for a in self.applied[c.name]:
                Fi = a.intended_force
                scale: float | np.floating[Any] = np.linalg.norm(Fi) + (np.linalg.norm(a.intended_moment) / info.L
                                              if a.intended_moment is not None else 0.0)
                scale = max(scale, 1e-300)
                eF = np.linalg.norm(a.force - Fi) / scale
                if eF > rtol:
                    add(Issue("error", "resultant-force", f"{a.name}: applied {a.force} != intended {Fi} "
                              f"(rel. {eF:.2e})", c.name))
                if a.intended_moment is not None:
                    eM = np.linalg.norm(a.moment(X=info.X) - a.intended_moment) / info.L / scale
                    if eM > rtol:
                        add(Issue("error", "resultant-moment", f"{a.name}: applied moment {a.moment(X=info.X)} != "
                                  f"intended {a.intended_moment} (rel. {eM:.2e})", c.name))
                dead = a.nodes[~info.node_active[a.nodes]]
                if len(dead):
                    add(Issue("error", "inactive-node", f"{a.name}: {len(dead)} node(s) are not part of the active "
                              "domain - the solver drops their forces", c.name))
                if len(a.nodes) == 1:
                    add(Issue("warning", "single-node-load", f"{a.name}: applied on one node (singular stress, grey "
                              "load blob); use PointLoad/SurfaceForce", c.name))
                touched_void = a.nodes[np.isin(a.nodes, self._void_nodes())]
                if len(touched_void) and a.kind != "body force":
                    add(Issue("warning", "load-on-void", f"{a.name}: {len(touched_void)} node(s) only touch void "
                              "(forced empty) elements", c.name))
                for note in a.notes:
                    lvl = "info" if note.startswith("info: ") else "warning"
                    add(Issue(lvl, "load-note", f"{a.name}: {note.removeprefix('info: ')}", c.name))
            # loads on constrained directions (the solver removes them)
            cn, cd = cs.constrained_directions(dim)
            if len(cn):
                fnod = fvec.reshape(d.nn, dim)
                comp = np.einsum("ia,ia->i", cd, fnod[cn])
                lost = (comp[:, None] * cd).sum(axis=0)
                tot = np.abs(fvec).sum()
                if tot > 0 and np.abs(comp).sum() > 1e-12 * tot:
                    add(Issue("warning", "load-on-support", f"{np.abs(comp).sum() / tot:.1%} of the nodal load acts "
                              f"on constrained directions and is taken by the support (resultant lost {lost})",
                              c.name))
        for k in range(len(self._sets)):
            cs = self._cs[k]
            cases = [c.name for c, s in zip(self.cases, self._case_set, strict=True) if s == k]
            tag = ", ".join(cases)
            for nn_, modes in self._rigid_modes(cs):
                if modes:
                    add(Issue("error", "rigid-body", f"{len(modes)} unrestrained rigid-body mode(s) of a component "
                              f"with {nn_} nodes: " + "; ".join(modes), tag))
            if len(cs.multiply_constrained):
                add(Issue("warning", "overlapping-supports", f"{len(cs.multiply_constrained)} node(s) belong to more "
                          f"than one support ({cs.redundant_rows} redundant constraint row(s)); check that this "
                          "is intended", tag))
            ns = int(self.info.surface.sum())
            ncon = len(np.unique(np.concatenate([p[2] for p in cs.per_support])))
            if ncon > 0.5 * ns:
                add(Issue("warning", "over-constrained", f"supports cover {ncon / ns:.0%} of the surface nodes - "
                          "a stiff, over-constrained model", tag))
            for name, _kind, nodes, _ in cs.per_support:
                if np.isin(nodes, self._void_nodes()).any():
                    add(Issue("warning", "support-on-void", f"{name}: nodes that only touch void elements", tag))
        return rep

    def _void_nodes(self) -> np.ndarray:
        d = self.domain
        if not d.void.any():
            return np.zeros(0, np.int64)
        act = d.active_elements
        en = self.info.elem_nodes
        solidish = ~d.void.ravel()[act]
        good = np.zeros(d.nn, bool)
        good[en[solidish].ravel()] = True
        return np.flatnonzero(self.info.node_active & ~good)

    # ------------------------------------------------------------------ summary
    def summary(self, rtol: float = 1e-9) -> str:
        """Human-readable tables: per load case the loads (intended vs applied resultant, moment about the load's
        reference point, relative error), per support set the supports; last line: verdict and next step."""
        info = self.info
        rep = self.check(rtol)
        lines: list[str] = []
        for c in self.cases:
            rows, errs = [], []
            for a in self.applied[c.name]:
                F, Fi = a.force, a.intended_force
                M = a.moment(X=info.X)
                scale = max(np.linalg.norm(Fi) + (np.linalg.norm(a.intended_moment) / info.L
                                                  if a.intended_moment is not None else 0.0), 1e-300)
                e = np.linalg.norm(F - Fi) / scale
                if a.intended_moment is not None:
                    e = max(e, np.linalg.norm(M - a.intended_moment) / info.L / scale)
                errs.append(e)
                rows.append([a.name, a.kind, len(a.nodes), Fi, F, M, e])
            if not rows:                                   # a case without loads (zero load vector)
                lines += [f'Load case "{c.name}": no loads', ""]
                continue
            ok = max(errs) <= rtol
            lines.append(f'Load case "{c.name}": {len(rows)} load(s), resultants '
                         f'{"match" if ok else "DO NOT match"} the intended values (max rel. error {max(errs):.1e})')
            lines.append(_format_loads(rows, info.L))
            lines.append("")
        for k, (_, sup) in enumerate(self._sets):
            cs = self._cs[k]
            cases = [c.name for c, s in zip(self.cases, self._case_set, strict=True) if s == k]
            free = sum(len(m) for _, m in self._rigid_modes(cs))
            lines.append(f"Supports of {', '.join(cases)}: {len(sup)} support(s), "
                         f"unrestrained rigid-body modes: {free}")
            rows2 = []
            for name, kind, nodes, nrows in cs.per_support:
                rows2.append([name, kind, len(nodes), nrows])
            lines.append(_table(["Support", "Kind", "Nodes", "Constraint rows"], rows2, "llrr"))
            lines.append(f"  fixed DOFs {len(cs.fixed)}; oblique-frame nodes {len(cs.frames)}; spring nodes "
                         f"{len(cs.spring_nodes)}")
            lines.append("")
        verdict = "OK" if rep.ok else "NOT OK"
        nxt = "print(bc.check()) for the messages" if rep.issues else "build the physics with bc.physics()"
        lines.append(f"Check {verdict}: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s) - next: {nxt}.")
        return "\n".join(lines)


def clearance_mask(domain: Domain, shape: Shape, threshold: float = 0.5) -> np.ndarray:
    """Void mask for a clearance volume (bolt head, tool access): active elements more than ``threshold`` covered
    by ``shape``. Pass as ``Domain(void=…)``."""
    from toporlogy.geometry import coverage
    return (coverage(shape, domain) > threshold) & domain.active
