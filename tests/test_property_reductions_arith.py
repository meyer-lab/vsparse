"""Property-based tests for per-axis sum/mean/max/min and elementwise
arithmetic on VCSCArray/VCSRArray, against a dense reference."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import dense_matrices, signed_dense_matrices, slow_first_call
from hypothesis import given

from vsparse import VCSCArray, VCSRArray


def _make(vcls, dense):
    return vcls.from_scipy(sp.csr_array(dense))


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_sum_and_mean_per_axis(vcls, dense):
    v = _make(vcls, dense)
    assert v.sum() == pytest.approx(dense.sum())
    np.testing.assert_allclose(v.sum(axis=0), dense.sum(axis=0))
    np.testing.assert_allclose(v.sum(axis=1), dense.sum(axis=1))
    if dense.size:
        assert v.mean() == pytest.approx(dense.mean())
        np.testing.assert_allclose(v.mean(axis=0), dense.mean(axis=0))
        np.testing.assert_allclose(v.mean(axis=1), dense.mean(axis=1))


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=signed_dense_matrices())
def test_max_and_min_per_axis(vcls, dense):
    """Extrema have to fold in the implicit zeros the layout never stores."""
    if dense.size == 0:
        return
    v = _make(vcls, dense)
    assert v.max() == pytest.approx(dense.max())
    assert v.min() == pytest.approx(dense.min())
    np.testing.assert_allclose(v.max(axis=0), dense.max(axis=0))
    np.testing.assert_allclose(v.max(axis=1), dense.max(axis=1))
    np.testing.assert_allclose(v.min(axis=0), dense.min(axis=0))
    np.testing.assert_allclose(v.min(axis=1), dense.min(axis=1))


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_add_sub_vcs_vcs(vcls, dense):
    v = _make(vcls, dense)
    other_dense = dense * 2
    other = _make(vcls, other_dense)

    added = v + other
    assert isinstance(added, vcls)
    np.testing.assert_allclose(added.toarray(), dense + other_dense)

    subbed = v - other
    assert isinstance(subbed, vcls)
    np.testing.assert_allclose(subbed.toarray(), dense - other_dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_add_sub_dense(vcls, dense):
    v = _make(vcls, dense)
    other_dense = np.ones_like(dense)

    np.testing.assert_allclose(np.asarray(v + other_dense), dense + other_dense)
    np.testing.assert_allclose(np.asarray(other_dense + v), other_dense + dense)
    np.testing.assert_allclose(np.asarray(v - other_dense), dense - other_dense)
    np.testing.assert_allclose(np.asarray(other_dense - v), other_dense - dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_multiply_elementwise(vcls, dense):
    v = _make(vcls, dense)
    other_dense = dense + 1  # avoid trivially all-zero result
    other = _make(vcls, other_dense)

    prod = v.multiply(other)
    assert isinstance(prod, vcls)
    np.testing.assert_allclose(prod.toarray(), dense * other_dense)

    prod_star = v * other
    assert isinstance(prod_star, vcls)
    np.testing.assert_allclose(prod_star.toarray(), dense * other_dense)

    prod_dense = v.multiply(other_dense)
    assert isinstance(prod_dense, vcls)
    np.testing.assert_allclose(prod_dense.toarray(), dense * other_dense)
