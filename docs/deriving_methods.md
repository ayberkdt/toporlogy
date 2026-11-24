# Deriving a new method

How to get from an idea to a result that can be trusted, using the building blocks of the library. Every example in
this guide is executed by `tests/toporlogy/test_deriving_methods.py`; the code blocks below are shortened copies of
it. Where a number is quoted it was measured on 21 Nov 2026 on small 2-D grids
([`olcumler/yontemler.md`](olcumler/yontemler.md) states the environment and the limits of those measurements).

## 0. The contract

A method enters the library, or a comparison, only through the same gates as the built-in components:

1. **Forward computation only.** Derivatives come from autograd and the adjoint of the linear solve. If a
   hand-written derivative seems necessary, first try to write the quantity as a torch expression of `ctx.E`,
   `ctx.xphys` and `ctx.U` (buckling load factors are done that way: `optimization/buckling.py`).
2. **Finite-difference gate, pre-registered at 1e-6** (`optimization.diagnostics.gradcheck`,
   `study_gradcheck`). Through the linear solve use the Richardson form of the tests (h = 1e-3 and 2h): the plain
   central difference at h = 1e-6 has a noise floor of 1e-5…1e-4 there (`tests/toporlogy/test_gradients.py`).
3. **A closed form or an invariant**, not only an optimisation that "looks right": a patch test, a bound, a
   conservation law, a known optimum.
4. **Comparison on the 0/1 design with the solves counted** (`optimization.benchmark`), against a tuned standard
   method at the same resolution and from the same start.
5. **Say what was not read and not measured.** A formula taken from a paper that was not read says so in its
   docstring; a parameter tuned on one problem says so.

## 1. A new local criterion (ten lines) gives two methods

Anything defined per element — stress, strain energy density, temperature, damage — is a `LocalField`: a class with
`__call__(ctx) -> tensor` whose entries are feasible when ≤ 0.

```python
class EnergyDensity(LocalField):
    name, needs = "energy_density", frozenset({"u"})

    def __init__(self, w_lim):
        self.w_lim = float(w_lim)

    def __call__(self, ctx):
        m, d = ctx.physics, ctx.domain
        ue = ctx.U[torch.as_tensor(m.edof)]                                   # (elements, local dofs, cases)
        e = torch.einsum("eac,ab,ebc->ec", ue, torch.as_tensor(m.KE), ue)
        E = ctx.E.reshape(-1)[torch.as_tensor(m.elements)]
        vol = d.h ** d.dim if d.dim == 3 else d.h ** 2 * d.thickness
        return 0.5 * m.geom * E[:, None] * e / vol / self.w_lim - 1.0
```

Both standard strategies now exist for it, with gradients:

```python
Constraint(Aggregated(field, "ks", 10.0), 0.0)                 # one global constraint (MMA / GCMMA)
AugmentedLagrangian(Volume(), field)                           # local constraints in the objective
```

The test checks both gradients (≤ 1e-6) and the invariant that the field sums to half the compliance. Which
aggregate: `optimization/aggregates.py` lists six with their bounds; the upper-bound forms are conservative with a
gap that grows with the number of entries, the lower-bound and induced forms need `adaptive=` to be used as a
constraint. What to expect from the two strategies on a stress problem is measured in `olcumler/yontemler.md` §3 —
including that the augmented Lagrangian needed its settings changed before it worked.

## 2. A new global response (one line)

```python
worst = Functional(lambda ctx: ks((torch.as_tensor(ctx.F) * ctx.U).sum(dim=0), 0.5), "worst_case", {"u"})
```

is the worst-case compliance over the load cases (a KS upper bound of it). After 15 iterations on the two-case
cantilever of the test the worst case is 131.5, against 136.3 for the design that minimises the sum. `WeightedSum`,
`Ratio` and `Shifted` combine responses; `Displacement` gives `Lᵀu` (mechanism output, deflection bound).

Things that go wrong here, all of which the gates catch:

- a quantity that is scale-dependent inside `ks` (ρ has the unit 1/value: normalise first);
- a `max`, `abs` or `where` at a point where two branches meet (not differentiable there; the finite-difference
  gate passes at random interior points and the optimiser then chatters — prefer the smooth aggregates);
- state that changes between evaluations (an adaptive scale, a reference taken at the first call): the response is
  then a different function per iteration. Allowed, but freeze it for the gradient check and report it.

## 3. A new chain link, and why the gate matters

A link is a `torch.nn.Module` from an element grid to an element grid. A volume-preserving projection chooses its
threshold η at every call so that the mean is unchanged. The tempting implementation finds η by bisection on the
detached field and applies the projection with that constant: the value is right, **the gradient is wrong**, because
η depends on the field. The test implements both; the shortcut fails the gate by more than 1e-3, the version that
adds the implicit-function term passes at 1e-6:

```python
eta = eta0 - (proj(x, eta0).mean() - x.mean()) / d_mean_proj_d_eta        # first-order exact dependence of η on x
return proj(x, eta)
```

General rule: whenever a forward computation *solves* for something (a threshold, a multiplier, a shift, an
eigenvector), either differentiate through the solve's defining equation as above, or prove that the term drops out
(as for a Rayleigh quotient at its eigenvector).

A link that maps a smaller variable vector to the design vector (symmetry, extrusion, a parametrisation by basis
functions) declares `n_in` and `reduce`; it must be the first link (`docs/registry.md`, "Chain ends"). This is the
entry point for reparameterisation experiments (literature note G-6).

## 4. A new optimizer

`Optimizer.step(x, f0, df0, g, dg, constraint_fn)` returns the next design in the box [0, 1]. Two optional
services of the study:

- `needs_constraint_fn = True`: `constraint_fn(x) -> g` without gradients, through the chain (only for constraints
  that need no state solve; OC, SiMPL);
- `needs_value_fn = True`: `self.value_fn(x) -> (f0, g)` without gradients, one state solve per call (line
  searches, conservative inner iterations; GCMMA, SiMPL). The study counts these in `settings["inner_evaluations"]`.

`kkt_multipliers()` lets the study report a KKT residual; without it the result has none. A thirty-line projected
gradient method in the test runs in the study unchanged. Compared on equal terms (cantilever 36 × 12, 81 state
solves each, same start, 0/1 design at exactly 50 % volume):

| | Grey compliance | Binary compliance | rel. | M_nd |
|:---|---:|---:|---:|---:|
| projected gradient | 305.8 | 1.311e5 | 1.31 | 0.75 |
| OC | 245.4 | 1.001e5 | 1.00 | 0.47 |
| MMA | 243.4 | 1.001e5 | 1.00 | 0.46 |

Read the table the way any such table must be read: the binary compliances are 400 times the grey ones. These
designs are still half grey (M_nd 0.46–0.75: no projection, 80 iterations), so thresholding changes them into
something else; the likely cause of the factor is that the thresholded designs are barely connected, which was not
examined. The comparison is fair between the rows, and it says the new method is 31 % worse than the standard ones
at the same budget; it says nothing about converged designs. A claim of superiority needs converged, crisp designs,
several problems and starts, and the criteria written down before the runs.

## 5. A protocol for a method claim

1. Write the hypothesis and the pass criterion first (as `docs/olcumler/robustness.md` does: "pre-registered").
2. Fix the problem set, the resolution, the start designs and the solve budget for *all* methods.
3. Baselines: OC or MMA with standard settings *and* continuation, not a deliberately weak configuration.
4. Report for every run: binary objective, constraint values on the binary design, M_nd, total state solves,
   termination reason, KKT residual when available (`compare_runs`, `Result`).
5. Report the failures and the runs that did not converge; do not average them away.
6. For anything geometric or stress-related, check the 0/1 design and, where a tet mesher is available, the
   body-fitted model (`post.verify`): the grey design's numbers are not the part's numbers (measured gap for
   stress: 2.5 % to 71 % on one problem).

## 6. Where the open problems are

[`literatur_2026-11-21.md`](literatur_2026-11-21.md) §3 lists eight gaps with the building blocks the library
already has for each and a first measurable step. The two that use the most distinctive part of this repository
(the post-processing chain with tet re-analysis) are the density-to-geometry gap and boundary-accurate stress on a
fixed grid.
