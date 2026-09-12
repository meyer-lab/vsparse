"""Shared Hypothesis strategies for property-based tests across the suite.

The underlying compress/decompress/matmul/select kernels are
@numba.njit(cache=True); the first call in a process pays JIT compilation cost
that has nothing to do with the example being tested, so every property test
built on these strategies should also apply `_slow_first_call` (below) to
give Hypothesis room instead of tripping its default deadline/"too slow"
health check.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

MAX_SIDE = 25

slow_first_call = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def dense_matrices(draw, *, max_side: int = MAX_SIDE, low: int = 0, high: int = 4) -> np.ndarray:
    """A dense float64 matrix with plenty of structural zeros, of a shrinkable
    shape (including empty axes)."""
    shape = (
        draw(st.integers(0, max_side)),
        draw(st.integers(0, max_side)),
    )
    values = draw(arrays(dtype=np.float64, shape=shape, elements=st.integers(low, high).map(float)))
    zero_mask = draw(arrays(dtype=bool, shape=shape, elements=st.booleans()))
    values[zero_mask] = 0.0
    return values


def signed_dense_matrices(*, max_side: int = MAX_SIDE):
    """A dense_matrices() variant with negative values too, for max/min tests."""
    return dense_matrices(max_side=max_side, low=-5, high=5)


def axis_key(data: st.DataObject, n: int):
    """One valid, possibly-duplicating/negative index selector for an axis of
    length `n`: a full slice, a general slice, a fancy int list, or a boolean
    mask. Never a bare int -- that collapses a dimension and is exercised by
    dedicated unit tests instead."""
    kind = data.draw(st.sampled_from(["full", "slice", "fancy", "bool"]))
    if kind == "full":
        return slice(None)
    if kind == "slice":
        start = data.draw(st.one_of(st.none(), st.integers(-n - 2, n + 2)))
        stop = data.draw(st.one_of(st.none(), st.integers(-n - 2, n + 2)))
        step = data.draw(st.sampled_from([None, 1, 2, 3, -1, -2]))
        return slice(start, stop, step)
    if kind == "fancy":
        if n == 0:
            return []
        return data.draw(st.lists(st.integers(-n, n - 1), min_size=0, max_size=2 * n))
    return data.draw(arrays(dtype=bool, shape=n, elements=st.booleans()))
