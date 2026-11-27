"""Critical point exactly at an accepted path point (docs/foundations/prereg/R07c_locate_zero.md, R7c3; R7c1/R7c2
are the promoted test_bifurcation A1/A2 and the existing files)."""
from __future__ import annotations

import pytest

from toporlogy.solvers.continuation import find_critical, locate_critical

from test_bifurcation import CF, DENSE, Polynomial, _primary


class _Shifted:
    """Wraps a spectrum and shifts every eigenvalue position by +2: the endpoint values (0, −1) become (2, 1), both
    positive and none near zero (a +1 shift would move the second endpoint onto zero)."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, lin):
        s = self.inner(lin)

        class _S:
            index = s.index

            def position(self, j):
                f, v = s.position(j)
                return f + 2.0, v
        return _S()


def test_r7c3_endpoint_outside_tolerance_still_raises() -> None:
    m = Polynomial()
    path = _primary(m, CF.ds)
    i = find_critical(path)[0]
    cp = locate_critical(m, path, i, DENSE)                         # the accepted point is the critical one
    assert cp.converged and cp.evaluations == 0
    with pytest.raises(ValueError, match="not bracketed"):
        locate_critical(m, path, i, _Shifted(DENSE))
