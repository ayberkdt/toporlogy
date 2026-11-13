"""tests/topo: import aerokit.topo before anything imports torch.

MKL PARDISO must be loaded before torch, otherwise two libiomp5md.dll copies abort the process (see
aerokit/topo/__init__.py). Optional native packages (pypardiso + MKL, gmsh) may live in a separate folder; set
``AEROKIT_PYLIB`` to it (several folders separated by ``os.pathsep``). Without them the tests fall back to SuperLU
and skip the gmsh-based checks.
"""
import os
import sys

for _p in os.environ.get("AEROKIT_PYLIB", "D:/tmp_ideathon/pylib").split(os.pathsep):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# frozen copies of earlier drivers used as regression oracles (tests/topo/topo_oracles)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import aerokit.topo  # noqa: E402,F401  (side effect: PARDISO before torch)
