"""Edge-case and error-path tests for indexing on VCSCArray/VCSRArray.

General (row-key, col-key) indexing correctness against a dense reference --
including negative/fancy/boolean/duplicate keys -- is covered by property-based
tests in test_property_indexing.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _as_dense(result):
    return result.toarray() if hasattr(result, "toarray") else np.asarray(result)


def test_single_row_key_selects_rows(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    result = v[0]
    np.testing.assert_allclose(_as_dense(result).reshape(-1), dense[0].reshape(-1))


def test_major_axis_out_of_bounds_raises(dense, vcls):
    """Verify that out-of-bounds single or array integer indices raise IndexError."""
    v = vcls.from_scipy(sp.csr_array(dense))
    n = dense.shape[1] if vcls is VCSCArray else dense.shape[0]

    # Single integer out of bounds (positive and negative)
    with pytest.raises(IndexError, match="out of bounds"):
        _ = v[:, n] if vcls is VCSCArray else v[n, :]
    with pytest.raises(IndexError, match="out of bounds"):
        _ = v[:, -n - 1] if vcls is VCSCArray else v[-n - 1, :]

    # Array of indices with out-of-bounds element
    with pytest.raises(IndexError, match="out of bounds"):
        _ = v[:, [n]] if vcls is VCSCArray else v[[n], :]


def test_major_axis_boolean_length_mismatch_raises(dense, vcls):
    """Verify that boolean index mask with mismatched length raises IndexError."""
    v = vcls.from_scipy(sp.csr_array(dense))
    n = dense.shape[1] if vcls is VCSCArray else dense.shape[0]
    invalid_mask = np.array([True] * (n + 1), dtype=bool)

    with pytest.raises(IndexError, match="boolean index does not match"):
        _ = v[:, invalid_mask] if vcls is VCSCArray else v[invalid_mask, :]


def test_major_axis_empty_selection(dense, vcls):
    """Verify that indexing with an empty list yields an empty VCSC/VCSR array with correct shape."""
    v = vcls.from_scipy(sp.csr_array(dense))
    if vcls is VCSCArray:
        sub = v[:, []]
        assert sub.shape == (dense.shape[0], 0)
    else:
        sub = v[[], :]
        assert sub.shape == (0, dense.shape[1])
    assert isinstance(sub, vcls)
    assert sub.nnz == 0
    assert sub.n_unique == 0


def test_invalid_tuple_indexing_dimensions_raises(dense, vcls):
    """Verify that tuple keys with length != 2 raise IndexError."""
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(IndexError, match="arrays are 2-D"):
        _ = v[0, 0, 0]
