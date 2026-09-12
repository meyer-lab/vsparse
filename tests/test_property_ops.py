"""Property-based tests for elementwise scalar ops, transpose, log1p, and
matvec/matmat on VCSCArray/VCSRArray, against a dense reference."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import dense_matrices, slow_first_call
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from vsparse import VCSCArray, VCSRArray


def _make(vcls, dense):
    return vcls.from_scipy(sp.csr_array(dense))


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), scalar=st.sampled_from([2.0, -1.5, 0.0]))
def test_scalar_mul(vcls, dense, scalar):
    v = _make(vcls, dense)
    np.testing.assert_allclose((v * scalar).toarray(), dense * scalar)
    np.testing.assert_allclose((scalar * v).toarray(), dense * scalar)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_scalar_div(vcls, dense):
    v = _make(vcls, dense)
    np.testing.assert_allclose((v / 2.0).toarray(), dense / 2.0)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_neg(vcls, dense):
    v = _make(vcls, dense)
    np.testing.assert_allclose((-v).toarray(), -dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_transpose(vcls, dense):
    v = _make(vcls, dense)
    vt = v.T
    np.testing.assert_allclose(vt.toarray(), dense.T)
    assert vt.shape == dense.T.shape
    vtt = vt.T
    assert type(vtt) is type(v)
    np.testing.assert_allclose(vtt.toarray(), dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_log1p(vcls, dense):
    v = _make(vcls, dense)
    np.testing.assert_allclose(v.log1p().toarray(), np.log1p(dense))


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_matvec_and_matmat(vcls, dense, data):
    v = _make(vcls, dense)
    n_rows, n_cols = dense.shape
    p = data.draw(st.integers(0, 4))

    x = data.draw(arrays(dtype=np.float64, shape=n_cols, elements=st.floats(-10, 10)))
    np.testing.assert_allclose(v @ x, dense @ x, atol=1e-8)

    y = data.draw(arrays(dtype=np.float64, shape=n_rows, elements=st.floats(-10, 10)))
    np.testing.assert_allclose(y @ v, y @ dense, atol=1e-8)

    b = data.draw(arrays(dtype=np.float64, shape=(n_cols, p), elements=st.floats(-10, 10)))
    np.testing.assert_allclose(v @ b, dense @ b, atol=1e-8)

    c = data.draw(arrays(dtype=np.float64, shape=(p, n_rows), elements=st.floats(-10, 10)))
    np.testing.assert_allclose(c @ v, c @ dense, atol=1e-8)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_scalar_mul_zero_returns_empty_like(vcls, dense):
    """Multiplying by 0 produces an empty-like array preserving shape and dtype."""
    v = _make(vcls, dense)
    v0 = v * 0
    assert isinstance(v0, vcls)
    assert v0.shape == v.shape
    assert v0.dtype == v.dtype
    assert v0.nnz == 0
    assert v0.n_unique == 0
    np.testing.assert_allclose(v0.toarray(), np.zeros(v.shape))
