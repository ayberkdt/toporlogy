"""Manufacturing constraints as chain links (docs/mimari_plan.md §3): AM overhang filter, variable tying
(symmetry, extrusion, pattern repetition) and a single-pull casting / milling link.

Valid order in a chain (:func:`manufacturing_chain` builds it, :func:`validate_chain` checks it)::

    [Tying: Symmetry | Extrusion | Pattern]  z → design vector      (first link only; Study optimises z)
    Embed                                    design vector → grid
    DensityFilter
    [GroupAverage]                           re-ties the filtered field (only with a tying link)
    [AMFilter] [Casting]                     act on the filtered field x̃ (grid → grid)
    [HeavisideProjection]
    Passive                                  keep = 1, void = 0, outside = 0

Why this order: a tying link maps a reduced vector to the design vector, so it must be first (``Study`` reads its
``n_in`` / ``reduce``). The AM filter and the casting link must see the *smoothed* field (Langelaar 2017 places the
AM filter after the density filter) and must come before the projection — the projection is a monotone map, so it
keeps a monotone (castable) column monotone and does not create unsupported material where the AM filter left
none. Passive comes last so frozen regions are exact.

Masked (non-box) domains, convention of this module:

* elements outside the part and ``void`` elements are *not printed / not material*: in the AM filter their printed
  density is exactly 0, so they support nothing; in the casting link they are 0;
* ``keep`` elements are *printed solid*: printed density exactly 1 (they support the layer above); whether a keep
  region itself is supported is not something the design can change, so the filter does not restrict it
  (:func:`overhang_map` can still report it).

Gradients come from autograd (all links are plain torch code); each link has a finite-difference test.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as tF
from scipy.sparse.csgraph import connected_components

from toporlogy.references import cite
from toporlogy.design.chain import Chain, DensityFilter, Embed, HeavisideProjection, Passive
from toporlogy.core.domain import Domain

__all__ = ["parse_direction", "AMFilter", "overhang_map", "overhang_violations",
           "Tying", "Symmetry", "Extrusion", "Pattern", "GroupAverage", "combine_tying",
           "Casting", "undercut_map", "undercuts",
           "manufacturing_chain", "validate_chain"]

_AXES = "xyz"


# ================================================================================================== helpers
def parse_direction(spec: Any, dim: int) -> tuple[int, int]:
    """Direction spec → (axis, sign). Accepts ``"+z"``, ``"-x"``, ``"y"`` (= +y), an axis int, or (axis, ±1)."""
    if isinstance(spec, str):
        s = spec.strip().lower()
        sign = -1 if s.startswith("-") else 1
        s = s.lstrip("+-")
        if len(s) != 1 or s not in _AXES:
            raise ValueError(f"direction {spec!r}: expected one of ±x, ±y, ±z")
        axis = _AXES.index(s)
    elif isinstance(spec, (tuple, list)):
        axis, sign = int(spec[0]), int(np.sign(spec[1]))
    else:
        axis, sign = int(spec), 1
    if not 0 <= axis < dim or sign not in (-1, 1):
        raise ValueError(f"direction {spec!r} is not valid in {dim}-D")
    return axis, sign


def _axis_index(a: Any, dim: int) -> int:
    k = _AXES.index(a.lower()) if isinstance(a, str) else int(a)
    if not 0 <= k < dim:
        raise ValueError(f"axis {a!r} is not valid in {dim}-D")
    return k


def _to_layers(g: torch.Tensor | np.ndarray, axis: int, sign: int):
    """Move ``axis`` to 0 and flip so that layer 0 is the first one in the direction (+: index 0)."""
    if isinstance(g, torch.Tensor):
        g = torch.movedim(g, axis, 0)
        return torch.flip(g, (0,)) if sign < 0 else g
    g = np.moveaxis(g, axis, 0)
    return g[::-1] if sign < 0 else g


def _from_layers(g: torch.Tensor | np.ndarray, axis: int, sign: int):
    if isinstance(g, torch.Tensor):
        g = torch.flip(g, (0,)) if sign < 0 else g
        return torch.movedim(g, 0, axis)
    g = g[::-1] if sign < 0 else g
    return np.moveaxis(g, 0, axis)


def _support_sum(a: torch.Tensor) -> torch.Tensor:
    """Sum over the support stencil of a layer (centre + face neighbours inside the box): 3 cells in 2-D (a is 1-D),
    5 cells in 3-D (a is 2-D)."""
    s = a.clone()
    for ax in range(a.dim()):
        n = a.shape[ax]
        if n < 2:
            continue
        lo = a.narrow(ax, 0, n - 1)          # neighbour at index − 1 for indices 1..n−1
        hi = a.narrow(ax, 1, n - 1)          # neighbour at index + 1 for indices 0..n−2
        zero = torch.zeros_like(a.narrow(ax, 0, 1))
        s = s + torch.cat([zero, lo], dim=ax) + torch.cat([hi, zero], dim=ax)
    return s


def _support_sum_fast(a: torch.Tensor) -> torch.Tensor:
    """Same as :func:`_support_sum` with one zero-pad and slices (fewer kernel launches; the AM recursion)."""
    if a.dim() == 1:
        p = tF.pad(a[None, None], (1, 1))[0, 0]
        return p[:-2] + p[1:-1] + p[2:]
    p = tF.pad(a[None, None], (1, 1, 1, 1))[0, 0]
    return p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _support_sum_np(a: np.ndarray) -> np.ndarray:
    """Numpy :func:`_support_sum` (zero outside the box)."""
    p = np.pad(a, 1)
    if a.ndim == 1:
        return p[:-2] + p[1:-1] + p[2:]
    return p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _support_count(shape: tuple[int, ...]) -> np.ndarray:
    return _support_sum(torch.ones(shape, dtype=torch.float64)).numpy()


def _support_any(a: np.ndarray) -> np.ndarray:
    """Boolean: any cell of the support stencil true (numpy, a is one layer)."""
    return _support_sum(torch.as_tensor(a.astype(float))).numpy() > 0.5


def _default_build(build: Any, dim: int) -> Any:
    """Default build direction: +z in 3-D, +y in 2-D."""
    return ("+z" if dim == 3 else "+y") if build is None else build


def _masks(domain: Domain | None, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(keep, off) boolean grids; off = outside the part or void."""
    if domain is None:
        return np.zeros(shape, bool), np.zeros(shape, bool)
    if tuple(domain.shape) != tuple(shape):
        raise ValueError(f"field shape {shape} does not match the domain {domain.shape}")
    return domain.keep.copy(), (~domain.active) | domain.void


# ================================================================================================== AM filter
@cite("langelaar_2017", note="Eq. (3)-(8): smin/smax layer recursion; 3-D form with the five-cell support region")
class AMFilter(torch.nn.Module):
    """Layer-by-layer additive-manufacturing filter (overhang ≤ 45° on the square grid), 2-D and 3-D.

    The grid is printed layer by layer in the build direction ``build`` (±x, ±y[, ±z]; default +z in 3-D, +y in
    2-D). An element's printed density is ξ = smin(x̃, Ξ) with Ξ = smax of the printed densities of its support
    cells in the layer below: 3 cells in 2-D (below-left, below, below-right), 5 in 3-D (below + its four face
    neighbours); fewer at the box edge. ``smax(v) = (Σ vⱼ^P)^(1/Q)``, ``Q = P + ln N / ln ξ₀`` (Eq. 5–6),
    ``smin(x, Ξ) = ½(x + Ξ − √((x − Ξ)² + ε) + √ε)`` (Eq. 3). Defaults are the paper's: P = 40, ε = 1e-4,
    ξ₀ = 0.5 (p. 875).

    Base plate: layers ``0 .. base_layer`` (counted from the start of the build direction) are printed as designed.
    ``base_layer = 0`` is a plate at the domain boundary. Alternatively ``base_coordinate`` (a coordinate along the
    build axis) puts the plate in the layer that contains it.

    Masks (module convention): outside-part and void elements get printed density 0 (support nothing); keep
    elements get printed density 1 (printed solid, support the layer above). Pass ``masked=False`` to ignore the
    masks (identical to the old 2-D/3-D implementations, which treat every grid cell as design).

    Because Q < P a fully dense support row gives Ξ = N^(1/Q) > 1, so printed densities can exceed 1 by a fraction
    of a per cent; that is the paper's operator.

    On the CPU the forward pass runs in numpy and the backward pass is the hand-written adjoint (layer recursion
    run backwards, as in Langelaar 2017 Eq. 10–14), several times faster than autograd through the layer loop;
    ``autograd=True`` (or a non-CPU tensor) uses the pure-torch path :meth:`forward_torch`. Both are tested
    against each other and against finite differences.
    """

    def __init__(self, domain: Domain, build: Any = None, base_layer: int = 0, base_coordinate: float | None = None,
                 P: float = 40.0, eps: float = 1e-4, xi0: float = 0.5, masked: bool = True,
                 autograd: bool = False) -> None:
        super().__init__()
        self.shape = tuple(domain.shape)
        self.dim = domain.dim
        self.axis, self.sign = parse_direction(_default_build(build, domain.dim), domain.dim)
        nlay = self.shape[self.axis]
        if base_coordinate is not None:
            t = (float(base_coordinate) - domain.origin[self.axis]) / domain.h
            k = int(np.clip(math.floor(t), 0, nlay - 1))
            base_layer = k if self.sign > 0 else nlay - 1 - k
        if not 0 <= int(base_layer) < nlay:
            raise ValueError(f"base_layer must lie in [0, {nlay - 1}]")
        self.base_layer = int(base_layer)
        self.P, self.eps, self.xi0 = float(P), float(eps), float(xi0)
        keep, off = _masks(domain, self.shape) if masked else _masks(None, self.shape)
        self.register_buffer("keep", torch.as_tensor(np.ascontiguousarray(_to_layers(keep, self.axis, self.sign))))
        self.register_buffer("off", torch.as_tensor(np.ascontiguousarray(_to_layers(off, self.axis, self.sign))))
        self.register_buffer("free", (~self.keep & ~self.off).double())
        self.register_buffer("keepf", self.keep.double())
        self.masked = bool(self.keep.any() or self.off.any())
        n = _support_count(self.keep.shape[1:])
        q = self.P + np.log(n) / math.log(self.xi0)
        self.register_buffer("invq", torch.as_tensor(1.0 / q))
        self._free_np, self._keep_np = self.free.numpy().copy(), self.keepf.numpy().copy()
        self._invq_np = 1.0 / q
        self.autograd = bool(autograd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu" and not self.autograd:
            return _AMFunction.apply(x, self)
        return self.forward_torch(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-torch forward (autograd backward); used off the CPU and as a cross-check of the numpy path."""
        a = _to_layers(x, self.axis, self.sign)
        se = math.sqrt(self.eps)
        invq = self.invq.to(x.dtype)
        free, keep = self.free.to(x.dtype), self.keepf.to(x.dtype)
        if self.masked:
            a = a * free + keep                  # keep → 1, outside / void → 0
        layers = list(a[: self.base_layer + 1].unbind(0))
        for k in range(self.base_layer + 1, a.shape[0]):
            Xi = torch.clamp_min(_support_sum_fast(layers[-1] ** self.P), 1e-300) ** invq
            ak = a[k]
            xi = 0.5 * (ak + Xi + se - torch.hypot(ak - Xi, torch.tensor(se, dtype=x.dtype)))
            if self.masked:
                xi = xi * free[k] + keep[k]
            layers.append(xi)
        return _from_layers(torch.stack(layers), self.axis, self.sign)

    # ---------------------------------------------------------------- numpy path with the exact adjoint
    def _np_forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        a = np.ascontiguousarray(_to_layers(x, self.axis, self.sign))
        free, keep = self._free_np, self._keep_np
        if self.masked:
            a = a * free + keep
        xi = np.empty_like(a)
        xi[: self.base_layer + 1] = a[: self.base_layer + 1]
        se, P = math.sqrt(self.eps), self.P
        cache = []
        for k in range(self.base_layer + 1, a.shape[0]):
            s = np.maximum(_support_sum_np(xi[k - 1] ** P), 1e-300)
            Xi = s ** self._invq_np
            d = a[k] - Xi
            root = np.sqrt(d * d + self.eps)
            v = 0.5 * (a[k] + Xi - root + se)
            xi[k] = v * free[k] + keep[k] if self.masked else v
            cache.append((s, d / root))
        return xi, cache, a

    def _np_backward(self, xi: np.ndarray, cache: list, g: np.ndarray) -> np.ndarray:
        lam = np.array(_to_layers(g, self.axis, self.sign), dtype=float)
        free = self._free_np
        dx = np.zeros_like(lam)
        b, P, iq = self.base_layer, self.P, self._invq_np
        for k in range(lam.shape[0] - 1, b, -1):
            s, t = cache[k - b - 1]
            gr = lam[k] * free[k] if self.masked else lam[k]
            dx[k] = gr * 0.5 * (1.0 - t)
            gs = gr * 0.5 * (1.0 + t) * iq * s ** (iq - 1.0)
            gs = np.where(s > 1e-300, gs, 0.0)
            lam[k - 1] += _support_sum_np(gs) * P * xi[k - 1] ** (P - 1.0)    # the stencil is symmetric
        dx[: b + 1] = lam[: b + 1]
        if self.masked:
            dx *= free
        return np.ascontiguousarray(_from_layers(dx, self.axis, self.sign))


class _AMFunction(torch.autograd.Function):
    """Numpy forward of :class:`AMFilter` with the hand-written adjoint (Langelaar 2017, Eq. 10–14 form)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, f: AMFilter) -> torch.Tensor:  # type: ignore[override]
        xi, cache, _ = f._np_forward(x.detach().double().numpy())
        ctx.f, ctx.xi, ctx.cache, ctx.dtype = f, xi, cache, x.dtype
        return torch.from_numpy(np.ascontiguousarray(_from_layers(xi, f.axis, f.sign))).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, None]:  # type: ignore[override]
        dx = ctx.f._np_backward(ctx.xi, ctx.cache, g.detach().double().numpy())
        return torch.from_numpy(dx).to(ctx.dtype), None


def overhang_map(xphys: np.ndarray, build: Any = None, domain: Domain | None = None, threshold: float = 0.5,
                 base_layer: int = 0, include_keep: bool = False) -> np.ndarray:
    """Boolean grid of unsupported solid elements: solid (> ``threshold``) in a layer above ``base_layer`` with no
    solid cell in its support stencil (3 cells in 2-D, 5 in 3-D) in the layer below.

    With ``domain``: outside/void cells are never solid, keep cells always solid; keep cells are reported only if
    ``include_keep`` (the design cannot change them).
    """
    x = np.asarray(xphys, float)
    dim = x.ndim
    axis, sign = parse_direction(_default_build(build, dim), dim)
    keep, off = _masks(domain, x.shape)
    solid = ((x > threshold) | keep) & ~off
    g = _to_layers(solid, axis, sign)
    out = np.zeros_like(g)
    for k in range(base_layer + 1, g.shape[0]):
        out[k] = g[k] & ~_support_any(g[k - 1])
    if not include_keep:
        out &= ~_to_layers(keep, axis, sign)
    return np.ascontiguousarray(_from_layers(out, axis, sign))


def overhang_violations(xphys: np.ndarray, build: Any = None, domain: Domain | None = None,
                        threshold: float = 0.5, base_layer: int = 0, include_keep: bool = False) -> int:
    """Number of unsupported solid elements (:func:`overhang_map`)."""
    return int(overhang_map(xphys, build, domain, threshold, base_layer, include_keep).sum())


# ================================================================================================== tying
def _cell_of(domain: Domain, pts: np.ndarray) -> np.ndarray:
    """Element (box) number of the cell containing each point (nearest cell centre), −1 outside the box."""
    idx = np.floor((np.asarray(pts, float) - domain.origin) / domain.h).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(domain.shape)), axis=1)
    out = np.full(len(idx), -1, np.int64)
    if ok.any():
        out[ok] = np.ravel_multi_index(tuple(idx[ok].T), domain.shape)
    return out


class Tying(torch.nn.Module):
    """Variable tying z → design vector: design elements are partitioned into groups that share one variable.

    ``maps`` is a list of (ne,) int arrays: ``maps[m][e]`` = box element related to element e (−1: none / outside
    the box). The groups are the equivalence classes of the relation "e ~ maps[m][e]" restricted to design
    elements: the transitive closure over *all* relations, computed as connected components of the relation
    graph (the same partition a union-find over every pair gives). The classes are therefore closed even when a
    nearest-cell map is not an involution (rotations, several planes, masks).

    * ``forward(z)`` = ``z[group]`` (design vector); its adjoint sums the gradient over each group;
    * ``reduce(x)`` = group means; ``expand(reduce(x))`` is the orthogonal projection onto tied designs
      (P (PᵀP)⁻¹ Pᵀ with the 0/1 membership matrix P);
    * ``n_in`` = number of groups.

    Cells whose partner is not a design element (outside the box or the part, keep, void) are listed in
    :attr:`unpaired` (design element numbers per category). They are *not* forced to anything: they keep the
    relations they do have (often none, then they are a free single variable). A problem is exactly invariant under
    the tying only if the part mask, keep and void regions are themselves invariant; :attr:`unpaired` shows where
    they are not.
    """

    def __init__(self, domain: Domain, maps: Sequence[np.ndarray], unpaired: Mapping[str, np.ndarray] | None = None,
                 kind: str = "tying") -> None:
        super().__init__()
        self.domain, self.kind = domain, kind
        self.maps = [np.asarray(m, np.int64).reshape(domain.ne) for m in maps]
        de = domain.design_elements
        pos = np.full(domain.ne, -1, np.int64)
        pos[de] = np.arange(len(de))
        rows, cols = [], []
        for m in self.maps:
            p = m[de]
            ok = p >= 0
            j = np.full(len(de), -1, np.int64)
            j[ok] = pos[p[ok]]
            ok = j >= 0
            rows.append(np.flatnonzero(ok))
            cols.append(j[ok])
        r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        n = len(de)
        graph = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
        _, lab = connected_components(graph, directed=True, connection="weak")
        # number groups by their first (smallest) design index → deterministic
        _, first = np.unique(lab, return_index=True)
        order = np.empty(len(first), np.int64)
        order[np.argsort(first)] = np.arange(len(first))
        group = order[lab]
        self.n_in = int(len(first))
        self.group_np = group
        self.counts = np.bincount(group, minlength=self.n_in).astype(float)
        self.register_buffer("group", torch.as_tensor(group, dtype=torch.long))
        self.unpaired = dict(unpaired) if unpaired is not None else self._unpaired_from_maps()
        # relations e → j between design elements for which no map gives j → e: the closure makes them harmless
        r_e, c_e = de[r], de[c]
        key = r_e * domain.ne + c_e
        self.non_mutual = int(np.sum(~np.isin(c_e * domain.ne + r_e, key)))

    def _unpaired_from_maps(self) -> dict[str, np.ndarray]:
        d = self.domain
        de = d.design_elements
        cats = {"outside_box": np.zeros(len(de), bool), "outside_part": np.zeros(len(de), bool),
                "keep": np.zeros(len(de), bool), "void": np.zeros(len(de), bool)}
        act, kp, vd = d.active.ravel(), d.keep.ravel(), d.void.ravel()
        for m in self.maps:
            p = m[de]
            cats["outside_box"] |= p < 0
            q = np.where(p >= 0, p, 0)
            inb = p >= 0
            cats["outside_part"] |= inb & ~act[q]
            cats["keep"] |= inb & kp[q]
            cats["void"] |= inb & vd[q]
        return {k: de[v] for k, v in cats.items()}

    # ---------------------------------------------------------------- maps
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[self.group]

    def expand(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float)[self.group_np]

    def reduce(self, x_design: Any) -> np.ndarray:
        """Group means of a design vector (n_design,) → z (n_in,)."""
        x = x_design.detach().numpy() if isinstance(x_design, torch.Tensor) else np.asarray(x_design, float)
        return np.bincount(self.group_np, weights=x, minlength=self.n_in) / self.counts

    def project(self, x_design: np.ndarray) -> np.ndarray:
        """Orthogonal projection of a design vector onto the tied subspace (= expand(reduce(x)))."""
        return self.expand(self.reduce(x_design))

    def report(self) -> dict[str, Any]:
        sizes = np.bincount(self.counts.astype(np.int64))
        return {"kind": self.kind, "n_design": int(self.domain.n_design), "n_groups": self.n_in,
                "group_sizes": {int(s): int(c) for s, c in enumerate(sizes) if c},
                "unpaired": {k: int(len(v)) for k, v in self.unpaired.items()},
                "non_mutual_relations": self.non_mutual}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_design={self.domain.n_design}, n_groups={self.n_in})"


def combine_tying(*links: Tying) -> Tying:
    """One tying link whose groups are the closure of all relations of ``links`` (same domain), e.g. symmetry +
    extrusion. ``unpaired`` lists are merged."""
    if not links:
        raise ValueError("combine_tying needs at least one link")
    d = links[0].domain
    if any(lk.domain is not d for lk in links):
        raise ValueError("all links must use the same Domain object")
    maps = [m for lk in links for m in lk.maps]
    cats: dict[str, np.ndarray] = {}
    for lk in links:
        for k, v in lk.unpaired.items():
            cats[k] = np.union1d(cats.get(k, np.zeros(0, np.int64)), v)
    return Tying(d, maps, cats, kind="+".join(lk.kind for lk in links))


class Symmetry(Tying):
    """Mirror symmetry about one or more planes ``axis = coordinate``.

    ``planes``: mapping {axis: coordinate} (axis "x"/"y"/"z" or 0/1/2) or a sequence of (axis, coordinate). The
    plane may lie anywhere (not only on a grid line or cell centre): the partner of an element is the cell that
    contains its mirrored centre (nearest cell centre). Groups are the closure over all planes (so with two planes
    the four mirror images of a cell share one variable).
    """

    def __init__(self, domain: Domain, planes: Mapping[Any, float] | Sequence[tuple[Any, float]]) -> None:
        items = list(planes.items()) if isinstance(planes, Mapping) else list(planes)
        if not items:
            raise ValueError("at least one plane is required")
        self.planes = [(_axis_index(a, domain.dim), float(c)) for a, c in items]
        X = domain.element_centers()
        maps = []
        for k, c in self.planes:
            Y = X.copy()
            Y[:, k] = 2.0 * c - X[:, k]
            maps.append(_cell_of(domain, Y))
        super().__init__(domain, maps, kind="symmetry")


class Extrusion(Tying):
    """Constant cross-section along ``axis``: all design elements of a grid line parallel to ``axis`` share one
    variable.

    Lines that also contain outside-part, keep or void cells cannot be a true extrusion; their design elements are
    listed in :attr:`unpaired` (by category) and stay tied among themselves.
    """

    def __init__(self, domain: Domain, axis: Any) -> None:
        k = _axis_index(axis, domain.dim)
        self.axis = k
        shape = domain.shape
        lines = _to_layers(np.arange(domain.ne).reshape(shape), k, 1)          # (n_k, rest...)
        des = _to_layers(domain.design, k, 1)
        has = des.any(axis=0)
        first_layer = np.argmax(des, axis=0)                                    # first design cell of each line
        first = np.take_along_axis(lines, first_layer[None], axis=0)[0]
        target = np.where(has, first, -1)
        m = np.broadcast_to(target, lines.shape)
        mp = np.empty(domain.ne, np.int64)
        mp[lines.ravel()] = m.ravel()
        cats = {}
        de_mask = domain.design
        for name, bad in (("outside_part", ~domain.active), ("keep", domain.keep), ("void", domain.void)):
            line_bad = _to_layers(bad, k, 1).any(axis=0)
            hit = np.broadcast_to(line_bad, lines.shape)
            sel = _from_layers(np.ascontiguousarray(hit), k, 1) & de_mask
            cats[name] = np.flatnonzero(sel.ravel())
        cats["outside_box"] = np.zeros(0, np.int64)
        super().__init__(domain, [mp], cats, kind="extrusion")


class Pattern(Tying):
    """Pattern repetition by nearest-cell mapping, closed by the same equivalence closure as :class:`Symmetry`.

    Use the constructors :meth:`translational` (offset vector in coordinates; every cell is related to the cell at
    ± offset, so all copies inside the box share variables) and :meth:`cyclic` (``n`` sectors around an axis:
    every cell is related to the cells of its centre rotated by ± 2π/n).
    """

    def __init__(self, domain: Domain, maps: Sequence[np.ndarray], kind: str) -> None:
        super().__init__(domain, maps, kind=kind)

    @classmethod
    def translational(cls, domain: Domain, offset: Sequence[float]) -> Pattern:
        t = np.asarray(offset, float).reshape(domain.dim)
        if not np.any(t):
            raise ValueError("offset must be non-zero")
        X = domain.element_centers()
        p = cls(domain, [_cell_of(domain, X + t), _cell_of(domain, X - t)], kind="pattern-translational")
        p.offset = t
        return p

    @classmethod
    def cyclic(cls, domain: Domain, n: int, center: Sequence[float], axis: Any = 2) -> Pattern:
        """``n`` ≥ 2 sectors about the line through ``center`` parallel to ``axis`` (3-D) or the point ``center``
        (2-D; ``axis`` ignored)."""
        if int(n) < 2:
            raise ValueError("n ≥ 2 is required")
        X = domain.element_centers()
        c = np.asarray(center, float).reshape(domain.dim)
        if domain.dim == 2:
            u, v = 0, 1
        else:
            k = _axis_index(axis, 3)
            u, v = [(1, 2), (2, 0), (0, 1)][k]
        maps = []
        for th in (2 * math.pi / n, -2 * math.pi / n):
            Y = X.copy()
            du, dv = X[:, u] - c[u], X[:, v] - c[v]
            Y[:, u] = c[u] + math.cos(th) * du - math.sin(th) * dv
            Y[:, v] = c[v] + math.sin(th) * du + math.cos(th) * dv
            maps.append(_cell_of(domain, Y))
        p = cls(domain, maps, kind="pattern-cyclic")
        p.n_sectors, p.center = int(n), c
        return p


class GroupAverage(torch.nn.Module):
    """Grid → grid: replace the value of every design element by the mean over its :class:`Tying` group.

    Placed right after the density filter. Tying the design vector alone does not make the *filtered* field tied
    where the filter is not equivariant under the tying map: the filter is normalised by its in-box weight sum, so
    near the box faces a constant-along-a-line design does not stay constant (extrusion), the copies of a pattern
    at the two ends of the box differ, and a mirror plane off the box mid-plane is not respected. Averaging the
    filtered field over the same groups is a linear orthogonal projection (exact, differentiable) that restores the
    tying; for a mirror plane on the mid-plane of a symmetric domain it only removes rounding noise. Non-design
    elements are passed through.
    """

    def __init__(self, tying: Tying) -> None:
        super().__init__()
        d = tying.domain
        self.shape, self.n = tuple(d.shape), tying.n_in
        self.register_buffer("idx", torch.as_tensor(d.design_elements, dtype=torch.long))
        self.register_buffer("group", tying.group.clone())
        self.register_buffer("counts", torch.as_tensor(tying.counts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.reshape(-1)
        v = g[self.idx]
        s = torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, self.group, v)
        mean = s / self.counts.to(x.dtype)
        return g.index_put((self.idx,), mean[self.group]).reshape(self.shape)


# ================================================================================================== casting
class Casting(torch.nn.Module):
    """Single pull direction (casting / 2.5-D milling, no undercuts) as a cumulative smooth maximum.

    One-sided (``parting=None``): the tool is withdrawn in ``direction`` and the parting surface is the box face at
    the start of that direction. Along every grid line parallel to the direction the output is

        x̂_k = (Σ_{j ≥ k} x_j^p)^(1/p)          (j runs from k away from the parting surface)

    a p-norm cumulative maximum. Two-sided (``parting`` = coordinate along the axis): cells on each side use the
    cells farther from the parting plane on their own side (the two halves of the mould are pulled apart).

    Properties (exact, not asymptotic): the output is non-increasing away from the parting surface along every
    line (a sum over nested sets of non-negative terms), so a threshold at *any* level gives a design without
    undercuts; x̂_k ≥ max_{j≥k} x_j; for a line of N cells x̂ ≤ N^(1/p)·max (overshoot above 1 at most N^(1/p) − 1,
    e.g. 5 % for N = 8, p = 40; a following projection maps it back near 1).

    Masks: keep cells enter as 1 (and are 1), void / outside cells are 0. A void cell, or an outside cell that lies
    *after* the first part cell of a line (counted from the parting surface), blocks the tool: every design cell
    beyond it on that line is forced to 0 (anything there would be an undercut). Outside cells before the first
    part cell of a line (the part starts further in) do not block. A keep cell beyond a blocking cell is an
    inconsistent specification (an undercut no design can remove); keep wins, the cell stays 1, and
    :func:`undercut_map` reports the resulting undercut.

    Source: the casting constraint as a projection along the pull direction is the approach of Gersborg &
    Andreasen (2011) and Vatanabe et al. (2016); neither paper was read in this session (publisher pages refused
    the download), so this p-norm form is this module's own construction, not their equations.
    """

    def __init__(self, domain: Domain, direction: Any = "+z", parting: float | None = None, p: float = 40.0) -> None:
        super().__init__()
        self.shape = tuple(domain.shape)
        self.axis, self.sign = parse_direction(direction, domain.dim)
        self.p = float(p)
        if not self.p >= 1:
            raise ValueError("p ≥ 1 is required")
        n = self.shape[self.axis]
        self.parting = None if parting is None else float(parting)
        if self.parting is not None:
            centres = domain.origin[self.axis] + domain.h * (np.arange(n) + 0.5)
            upper = centres > self.parting          # cells on the + side of the plane
        else:
            upper = np.ones(n, bool) if self.sign > 0 else np.zeros(n, bool)
        self.upper = upper
        keep, off = _masks(domain, self.shape)
        allowed = self._allowed(domain, upper)
        self.register_buffer("keep", torch.as_tensor(keep))
        self.register_buffer("zero", torch.as_tensor(off | ~allowed))

    def _allowed(self, domain: Domain, upper: np.ndarray) -> np.ndarray:
        """Cells not shadowed by a blocking (void / outside) cell between them and the parting surface."""
        outside = np.moveaxis(~domain.active, self.axis, 0)
        void = np.moveaxis(domain.void, self.axis, 0)
        ac = np.moveaxis(domain.active, self.axis, 0)
        allowed = np.ones_like(ac)
        for idx in (np.flatnonzero(upper), np.flatnonzero(~upper)[::-1]):
            if not len(idx):
                continue
            seen_part = np.zeros(ac.shape[1:], bool)
            blocked = np.zeros(ac.shape[1:], bool)
            for k in idx:                       # walk away from the parting surface
                blocked |= (seen_part & outside[k]) | void[k]
                seen_part |= ac[k]
                allowed[k] = ~blocked
        return np.moveaxis(allowed, 0, self.axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        v = torch.where(self.keep, one, torch.where(self.zero, zero, x))
        # log domain: log x̂_k = (1/p) logcumsumexp(p log x_j) — no under/overflow of x^p and a finite gradient for
        # tiny densities; inputs below 1e-200 count as 1e-200 (zero gradient there)
        lx = self.p * torch.log(torch.clamp_min(torch.movedim(v, self.axis, 0), 1e-200))
        ku = int(np.argmax(self.upper)) if self.upper.any() else lx.shape[0]   # first cell on the + side
        parts = []
        if ku > 0:                               # − side: cumulative from the far (−) end up to k
            parts.append(torch.logcumsumexp(lx[:ku], 0))
        if ku < lx.shape[0]:                     # + side: cumulative from the far (+) end down to k
            parts.append(torch.flip(torch.logcumsumexp(torch.flip(lx[ku:], (0,)), 0), (0,)))
        L = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
        out = torch.exp(L / self.p)
        out = torch.movedim(out, 0, self.axis)
        return torch.where(self.keep, one, torch.where(self.zero, zero, out))


def undercut_map(xphys: np.ndarray, direction: Any = "+z", domain: Domain | None = None,
                 parting: float | None = None, threshold: float = 0.5) -> np.ndarray:
    """Boolean grid of undercut cells: empty cells (≤ ``threshold``, or void / outside the part) that lie after the
    first part cell of their line (seen from the parting surface) and have a solid cell farther from the parting
    surface on the same line. With ``domain``: keep cells are solid, void / outside cells empty."""
    x = np.asarray(xphys, float)
    axis, sign = parse_direction(direction, x.ndim)
    keep, off = _masks(domain, x.shape)
    act = np.ones(x.shape, bool) if domain is None else domain.active
    solid = ((x > threshold) | keep) & ~off
    n = x.shape[axis]
    if parting is None:
        upper = np.ones(n, bool) if sign > 0 else np.zeros(n, bool)
    else:
        o = 0.0 if domain is None else domain.origin[axis]
        h = 1.0 if domain is None else domain.h
        upper = (o + h * (np.arange(n) + 0.5)) > float(parting)
    S = np.moveaxis(solid, axis, 0)
    A = np.moveaxis(act, axis, 0)
    out = np.zeros_like(S)
    for idx in (np.flatnonzero(upper), np.flatnonzero(~upper)[::-1]):
        if not len(idx):
            continue
        far = np.zeros(S.shape[1:], bool)
        for k in idx[::-1]:                     # from the far end toward the parting surface: "solid beyond"
            out[k] = ~S[k] & far
            far |= S[k]
        seen = np.zeros(S.shape[1:], bool)
        for k in idx:                           # drop cells before the first part cell of the line
            seen |= A[k]
            out[k] &= seen
    return np.ascontiguousarray(np.moveaxis(out, 0, axis))


def undercuts(xphys: np.ndarray, direction: Any = "+z", domain: Domain | None = None,
              parting: float | None = None, threshold: float = 0.5) -> int:
    """Number of undercut cells (:func:`undercut_map`)."""
    return int(undercut_map(xphys, direction, domain, parting, threshold).sum())


# ================================================================================================== composition
_ORDER = {"tying": 0, "embed": 1, "filter": 2, "regroup": 3, "mfg": 4, "projection": 5, "passive": 6}


def _rank(link: torch.nn.Module) -> int | None:
    if isinstance(link, Tying):
        return _ORDER["tying"]
    if isinstance(link, Embed):
        return _ORDER["embed"]
    if isinstance(link, DensityFilter):
        return _ORDER["filter"]
    if isinstance(link, GroupAverage):
        return _ORDER["regroup"]
    if isinstance(link, (AMFilter, Casting)):
        return _ORDER["mfg"]
    if isinstance(link, HeavisideProjection):
        return _ORDER["projection"]
    if isinstance(link, Passive):
        return _ORDER["passive"]
    return None


def validate_chain(chain: Chain | Iterable[torch.nn.Module]) -> None:
    """Raise ``ValueError`` if the known links are out of the order tying → Embed → filter → [GroupAverage] →
    AM / casting → projection → Passive, a tying link is not the first link, or there is more than one tying link
    (combine them with :func:`combine_tying`). Unknown link types are ignored."""
    links = list(chain.links if isinstance(chain, Chain) else chain)
    ranks = [(i, _rank(lk), type(lk).__name__) for i, lk in enumerate(links)]
    ties = [i for i, r, _ in ranks if r == _ORDER["tying"]]
    if len(ties) > 1:
        raise ValueError("more than one tying link: combine them with combine_tying()")
    if ties and ties[0] != 0:
        raise ValueError("a tying link must be the first link of the chain")
    last = -1
    for _, r, name in ranks:
        if r is None:
            continue
        if r < last:
            raise ValueError(f"{name} is out of order (valid: tying → Embed → DensityFilter → GroupAverage → "
                             f"AMFilter/Casting → HeavisideProjection → Passive)")
        last = r
    if any(r == _ORDER["mfg"] for _, r, _ in ranks) and not any(r == _ORDER["filter"] for _, r, _ in ranks):
        raise ValueError("AMFilter / Casting need a DensityFilter before them")
    if any(r == _ORDER["regroup"] for _, r, _ in ranks) and not ties:
        raise ValueError("GroupAverage needs a tying link at the start of the chain")


def manufacturing_chain(domain: Domain, rmin: float, *, tying: Tying | Sequence[Tying] | None = None,
                        am: AMFilter | Mapping[str, Any] | None = None,
                        casting: Casting | Mapping[str, Any] | None = None, projection: bool = True,
                        beta: float = 1.0, eta: float = 0.5, regroup: bool = True) -> Chain:
    """[tying] → Embed → DensityFilter(rmin) → [GroupAverage] → [AMFilter] → [Casting] → [HeavisideProjection] →
    Passive.

    ``regroup`` (default True) inserts :class:`GroupAverage` after the filter when a tying link is given, so the
    physical density is tied exactly (see :class:`GroupAverage`). The AM filter and the casting link act along one
    axis; they keep a tying exact only if the tying map commutes with them (mirror planes and translations
    perpendicular to the build / pull axis, extrusion along an axis perpendicular to it).

    ``am`` / ``casting``: a link or the keyword arguments of :class:`AMFilter` / :class:`Casting`. Several tying
    links are combined into one (:func:`combine_tying`). Using AM and casting together is allowed but the two
    operators do not commute; the AM filter is applied first.
    """
    links: list[torch.nn.Module] = []
    t = None
    if tying is not None:
        t = tying if isinstance(tying, Tying) else combine_tying(*tying)
        links.append(t)
    links += [Embed(domain), DensityFilter(domain, rmin)]
    if t is not None and regroup:
        links.append(GroupAverage(t))
    if am is not None:
        links.append(am if isinstance(am, AMFilter) else AMFilter(domain, **dict(am)))
    if casting is not None:
        links.append(casting if isinstance(casting, Casting) else Casting(domain, **dict(casting)))
    if projection:
        links.append(HeavisideProjection(beta, eta))
    links.append(Passive(domain))
    chain = Chain(links)
    validate_chain(chain)
    return chain
