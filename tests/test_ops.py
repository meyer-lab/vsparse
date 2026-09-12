"""Error-path and regression tests for scalar/matmul ops on VCSCArray/VCSRArray.

Numeric correctness against a dense reference (scalar mul/div, neg, transpose,
log1p, matvec/matmat) is covered by property-based tests in test_property_ops.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _make(vcls, dense):
    return vcls.from_scipy(sp.csr_array(dense))


def test_matvec_dimension_mismatch_raises(dense, vcls):
    """Verify that dimension mismatch in matrix-vector product raises ValueError."""
    v = _make(vcls, dense)
    with pytest.raises(ValueError, match="not aligned"):
        v @ np.ones(dense.shape[1] + 1)


def test_unsupported_scalar_operands_raise(dense, vcls):
    """Non-scalar operands are now elementwise: non-broadcastable shapes/bad types raise, they don't silently no-op."""
    v = _make(vcls, dense)
    if dense.shape != (1, 1):  # a (1, 1) array broadcasts against any shape
        bad_shape = np.ones((dense.shape[0] + 3, dense.shape[1] + 3))
        with pytest.raises(ValueError, match="inconsistent shapes"):
            _ = v * bad_shape
        with pytest.raises(ValueError, match="inconsistent shapes"):
            _ = v / bad_shape
    with pytest.raises(TypeError):
        _ = v * {"a": 1}
    with pytest.raises(TypeError):
        _ = v / {"a": 1}


def test_matmat_dimension_mismatch_raises(dense, vcls):
    """Verify that dimension mismatches in 2-D matrix products raise ValueError."""
    v = _make(vcls, dense)
    # Right matrix multiplication dimension mismatch
    with pytest.raises(ValueError, match="not aligned"):
        v @ np.ones((dense.shape[1] + 2, 4))

    # Left vector multiplication dimension mismatch
    with pytest.raises(ValueError, match="not aligned"):
        np.ones(dense.shape[0] + 2) @ v

    # Left matrix multiplication dimension mismatch
    with pytest.raises(ValueError, match="not aligned"):
        np.ones((4, dense.shape[0] + 2)) @ v


def test_unsupported_matmul_operands_raise(dense, vcls):
    """Verify that >2D array operands in matmul raise TypeError."""
    v = _make(vcls, dense)
    arr_3d = np.ones((dense.shape[1], 2, 2))
    with pytest.raises(TypeError):
        _ = v @ arr_3d
    with pytest.raises(TypeError):
        _ = arr_3d @ v


def test_matmul_does_not_overflow_narrow_value_dtype(vcls):
    """Regression test for #43: matvec/matmat must not wrap modulo a narrow
    stored dtype (e.g. uint16 counts) the way ``values.dtype``-sized
    accumulators used to. ``sum()`` already got this right; the products
    should agree with it instead of silently wrapping.
    """
    n_rows, n_cols = 2000, 5
    dense = np.full((n_rows, n_cols), 40, dtype=np.uint16)  # col totals = 80000 > uint16 max
    v = _make(vcls, dense)
    assert v.dtype == np.uint16

    ones_rows = np.ones(n_rows, dtype=np.float64)
    expected_cols = np.ravel(v.sum(axis=0))
    np.testing.assert_array_equal(ones_rows @ v, expected_cols)

    ones_mat = np.ones((3, n_rows), dtype=np.float64)
    np.testing.assert_array_equal(ones_mat @ v, np.tile(expected_cols, (3, 1)))

    dense2 = np.zeros((3, 2000), dtype=np.uint16)
    dense2[0, :] = 40  # row 0 total = 80000 > uint16 max
    v2 = _make(vcls, dense2)
    ones_cols = np.ones(2000, dtype=np.float64)
    expected_rows = np.ravel(v2.sum(axis=1))
    np.testing.assert_array_equal(v2 @ ones_cols, expected_rows)

    ones_mat2 = np.ones((2000, 3), dtype=np.float64)
    np.testing.assert_array_equal(v2 @ ones_mat2, np.tile(expected_rows, (3, 1)).T)
