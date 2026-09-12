"""Property-based tests via Hypothesis for VCSC/VCSR construction and round-trip.

These cover the same kind of ground as the manual `rng.integers(...)` fuzzing
used elsewhere (e.g. test_chunked_transpose.py), but let Hypothesis choose
shapes/values/densities and shrink any failure to a minimal example instead of
us hand-picking a handful of cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import dense_matrices, slow_first_call
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from vsparse import VCSCArray, VCSRArray


def _scipy_for(vcls, dense: np.ndarray):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_from_scipy_toarray_roundtrips(vcls, dense):
    """Compressing and decompressing must reproduce the original matrix exactly,
    for any shape/density/zero-pattern -- not just the handful conftest picks."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))

    np.testing.assert_array_equal(v.toarray(), dense)
    assert v.shape == dense.shape
    assert v.nnz == int(np.count_nonzero(dense))
    np.testing.assert_array_equal(v.to_scipy().toarray(), dense)
    other = VCSRArray if vcls is VCSCArray else VCSCArray
    np.testing.assert_array_equal(
        (v.to_csr() if other is VCSCArray else v.to_csc()).toarray(), dense
    )


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_matmul_matches_dense_reference(vcls, dense, data):
    """v @ B must agree with the dense reference for any compatible B, at any
    shape Hypothesis manages to construct (including empty axes)."""
    n_cols = dense.shape[1]
    p = data.draw(st.integers(0, 4))
    b = data.draw(arrays(dtype=np.float64, shape=(n_cols, p), elements=st.floats(-10, 10)))

    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(v @ b, dense @ b, atol=1e-8)


def test_value_compression_deduplicates():
    """Repeated values within a major slice collapse to one stored entry."""
    dense = np.zeros((10, 10))
    dense[:, 0] = 3.0  # ten repeats of the same value in one column
    dense[0, 1] = 7.0
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    assert v.nnz == 11
    assert v.n_unique == 2  # one unique value per nonempty column
