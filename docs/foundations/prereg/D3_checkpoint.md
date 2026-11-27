# D3 — Checkpointed history adjoint for the viscoelastic march (preregistration)

Preregistered 2026-11-27 04:03 +0300 (clock), before the change and the tests. Base `92cd73e`.
Change: `ViscoelasticStructure.response(w, checkpoint_every=None)`: with an integer k the time march is split into
segments of k steps, each wrapped in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, so the forward pass
keeps only the segment boundary states and the backward pass recomputes one segment at a time (WP20 D3). Default
None = the current behaviour.

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| D3a | gradient of J(t_s) with k ∈ {1, 4, 12} equals the gradient without checkpointing (VO2 cantilever set-up, 12 steps) | no checkpoint | rel 1e-12 |
| D3b | bytes of the tensors saved for backward during the forward pass (counted exactly with `torch.autograd.graph.saved_tensors_hooks`) with k = 4 are at most half of those without checkpointing | the count without checkpointing | ratio ≤ 0.5 |
| D3c | cost: the number of linear solves (forward calls of the step solve) with checkpointing is ≤ 2 × the number without (one recomputation) | counted | ≤ 2× |
| D3d | default (checkpoint_every=None) objective and gradient are bitwise those of the code before the change (VO1/VO2 values reproduced exactly: same function body) | the earlier response | `==` / `array_equal` |

Run: `python -m pytest tests/toporlogy/test_checkpoint_adjoint.py -q -p no:randomly`.
