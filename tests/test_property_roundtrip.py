"""Prototype: property-based tests via Hypothesis, alongside the existing
seeded-rng fixtures in conftest.py.

These cover the same kind of ground as the manual `rng.integers(...)` fuzzing
used elsewhere (e.g. test_chunked_transpose.py), but let Hypothesis choose
shapes/values/densities and shrink any failure to a minimal example instead of
us hand-picking a handful of cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from vsparse import VCSCArray, VCSRArray

# The underlying compress/decompress/matmul kernels are @numba.njit(cache=True);
# the first call in a process pays JIT compilation cost that has nothing to do
# with the example being tested, so give Hypothesis room instead of tripping
# its default deadline/"too slow" health check.
_slow_first_call = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

_MAX_SIDE = 25


@st.composite
def dense_matrices(draw, *, max_side: int = _MAX_SIDE) -> np.ndarray:
    """A dense float64 matrix with plenty of structural zeros, of a shrinkable
    shape (including empty axes)."""
    shape = (
        draw(st.integers(0, max_side)),
        draw(st.integers(0, max_side)),
    )
    values = draw(arrays(dtype=np.float64, shape=shape, elements=st.integers(0, 4).map(float)))
    zero_mask = draw(arrays(dtype=bool, shape=shape, elements=st.booleans()))
    values[zero_mask] = 0.0
    return values


def _scipy_for(vcls, dense: np.ndarray):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@_slow_first_call
@given(dense=dense_matrices())
def test_from_scipy_toarray_roundtrips(vcls, dense):
    """Compressing and decompressing must reproduce the original matrix exactly,
    for any shape/density/zero-pattern -- not just the handful conftest picks."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))

    np.testing.assert_array_equal(v.toarray(), dense)
    assert v.shape == dense.shape
    assert v.nnz == int(np.count_nonzero(dense))


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@_slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_matmul_matches_dense_reference(vcls, dense, data):
    """v @ B must agree with the dense reference for any compatible B, at any
    shape Hypothesis manages to construct (including empty axes)."""
    n_cols = dense.shape[1]
    p = data.draw(st.integers(0, 4))
    b = data.draw(arrays(dtype=np.float64, shape=(n_cols, p), elements=st.floats(-10, 10)))

    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(v @ b, dense @ b, atol=1e-8)
