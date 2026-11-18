"""Connected bodies of a boolean voxel field, shared by the optimisation (connectivity) and the post-processing
(topology, island removal)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["neighbourhood", "label_bodies"]


def neighbourhood(dim: int, connectivity: str | int) -> np.ndarray:
    """Neighbourhood: "face" (6 / 4 neighbours), "edge" (18; in 2-D = face), "vertex" (26 / 8); or the count."""
    ranks = {3: {"face": 1, "edge": 2, "vertex": 3, 6: 1, 18: 2, 26: 3},
             2: {"face": 1, "edge": 1, "vertex": 2, 4: 1, 8: 2}}[dim]
    key = connectivity.lower() if isinstance(connectivity, str) else int(connectivity)
    if key not in ranks:
        raise ValueError(f"connectivity for {dim}-D: one of {sorted(map(str, ranks))}")
    return ndimage.generate_binary_structure(dim, ranks[key])


def label_bodies(solid: np.ndarray, connectivity: str | int = "face") -> tuple[np.ndarray, np.ndarray]:
    """Connected components of ``solid``: labels (0 = void, 1… = bodies by decreasing cell count) and cell counts."""
    solid = np.asarray(solid, bool)
    lab, n = ndimage.label(solid, structure=neighbourhood(solid.ndim, connectivity))
    if n == 0:
        return lab, np.zeros(0, np.int64)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    order = np.argsort(-sizes, kind="stable")
    remap = np.zeros(n + 1, np.int64)
    remap[order + 1] = np.arange(1, n + 1)
    return remap[lab], sizes[order]
