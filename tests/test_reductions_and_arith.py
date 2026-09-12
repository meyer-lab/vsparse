"""Edge-case and error-path tests for per-axis sum/mean/max/min and elementwise
arithmetic on VCSCArray/VCSRArray.

General numeric correctness against a dense reference is covered by
property-based tests in test_property_reductions_arith.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def test_sum_invalid_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(ValueError):
        v.sum(axis=2)


def test_max_min_invalid_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(ValueError):
        v.max(axis=2)
    with pytest.raises(ValueError):
        v.min(axis=2)


def test_max_min_all_negative_column():
    """A column of only negative values must still report 0 if it has a structural zero."""
    dense = np.array([[-1.0, -2.0], [0.0, -3.0]])
    for vcls in (VCSCArray, VCSRArray):
        v = vcls.from_scipy(sp.csr_array(dense))
        np.testing.assert_allclose(v.max(axis=0), dense.max(axis=0))
        np.testing.assert_allclose(v.min(axis=0), dense.min(axis=0))


def test_max_min_fully_dense_negative():
    """A fully-dense negative row/column must not spuriously include 0."""
    dense = np.array([[-1.0, -2.0], [-4.0, -3.0]])
    for vcls in (VCSCArray, VCSRArray):
        v = vcls.from_scipy(sp.csr_array(dense))
        assert v.max() == pytest.approx(-1.0)
        np.testing.assert_allclose(v.max(axis=0), dense.max(axis=0))
        np.testing.assert_allclose(v.max(axis=1), dense.max(axis=1))


def test_add_sub_zero_scalar(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    np.testing.assert_allclose((v + 0).toarray(), dense)
    np.testing.assert_allclose((v - 0).toarray(), dense)
    np.testing.assert_allclose((0 - v).toarray(), -dense)


def test_add_nonzero_scalar_raises(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(NotImplementedError):
        v + 5
    with pytest.raises(NotImplementedError):
        v - 5
    with pytest.raises(NotImplementedError):
        5 - v
