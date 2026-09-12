"""Edge-case tests for both-axes indexing, plus getnnz/count_nonzero and astype.

General (row-key, col-key) indexing correctness against a dense reference is
covered by property-based tests in test_property_indexing.py; getnnz/
count_nonzero/astype don't vary with the index-shape space that targets, so
they stay as fixture-driven tests here.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


# -- both-axes indexing edge cases --------------------------------------------


def test_minor_axis_empty_selection(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    if vcls is VCSCArray:
        sub = v[[], :]
        assert sub.shape == (0, dense.shape[1])
    else:
        sub = v[:, []]
        assert sub.shape == (dense.shape[0], 0)
    assert isinstance(sub, vcls)
    assert sub.nnz == 0
    assert sub.n_unique == 0


def test_both_full_slice_returns_copy(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    sub = v[:, :]
    assert isinstance(sub, vcls)
    assert sub is not v
    np.testing.assert_allclose(sub.toarray(), dense)


def test_both_int_still_returns_scalar(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    val = v[0, 0]
    assert np.isscalar(val) or isinstance(val, np.generic)
    assert val == dense[0, 0]


# -- getnnz / count_nonzero ----------------------------------------------------


def test_getnnz_overall(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    assert v.getnnz() == np.count_nonzero(dense)


def test_getnnz_per_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    np.testing.assert_array_equal(v.getnnz(axis=0), np.count_nonzero(dense, axis=0))
    np.testing.assert_array_equal(v.getnnz(axis=1), np.count_nonzero(dense, axis=1))


def test_getnnz_invalid_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(ValueError):
        v.getnnz(axis=2)


def test_count_nonzero(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    assert v.count_nonzero() == np.count_nonzero(dense)


def test_count_nonzero_excludes_explicit_zero_values():
    """count_nonzero must exclude stored-but-zero values, unlike nnz/getnnz."""
    dense = np.array([[1.0, 0.0], [0.0, 2.0]])
    for vcls in (VCSCArray, VCSRArray):
        v = vcls.from_scipy(sp.csr_array(dense))
        # Manually inject an explicit zero into the stored (unique) values,
        # which nnz/getnnz still counts as a stored element.
        zeroed_values = v.values.copy()
        zeroed_values[0] = 0.0
        v2 = vcls(v.shape, v.major_ptr, zeroed_values, v.value_ptr, v.indices)
        assert v2.getnnz() == v.nnz
        assert v2.count_nonzero() < v2.getnnz()


# -- astype ---------------------------------------------------------------------


def test_astype_casts_values(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    out = v.astype(np.float32)
    assert isinstance(out, vcls)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.toarray(), dense.astype(np.float32))
    # original is untouched
    assert v.dtype == dense.dtype


def test_astype_no_copy_same_dtype_returns_self(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    out = v.astype(v.dtype, copy=False)
    assert out is v


def test_astype_copy_true_same_dtype_returns_new_object(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    out = v.astype(v.dtype, copy=True)
    assert out is not v
    np.testing.assert_allclose(out.toarray(), dense)
