"""Property-based tests for the norm_sq/slice_norms/to_scipy_sparse trio.

These give ``VCSCArrayNormalized``/``VCSRArrayNormalized`` the pieces a
duck-typed ``parafac2`` backend needs beyond ``__matmul__``/``__rmatmul__``
(see https://github.com/meyer-lab/parafac2's ``parafac2.utils`` module
docstring): a squared-Frobenius-norm reduction, a per-condition-group version
of the same, and a way to materialize the underlying sparse term as a real
scipy array plus its external mean correction. Each is checked directly
against a dense/numpy reference built from :meth:`toarray`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import dense_matrices, slow_first_call
from hypothesis import assume, given
from hypothesis import strategies as st

from vsparse import RECIPES, VCSCArray, VCSRArray


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


@st.composite
def condition_groups(draw, *, n_rows: int):
    """A random condition index (in ``[0, n_cond)``) for each of ``n_rows`` rows."""
    if n_rows == 0:
        return np.zeros(0, dtype=np.int64), 1
    n_cond = draw(st.integers(1, max(1, n_rows)))
    idxs = draw(st.lists(st.integers(0, n_cond - 1), min_size=n_rows, max_size=n_rows))
    return np.asarray(idxs, dtype=np.int64), n_cond


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@pytest.mark.parametrize("recipe", sorted(RECIPES))
@slow_first_call
@given(dense=dense_matrices())
def test_norm_sq_matches_dense_reference(vcls, recipe, dense):
    assume(dense.sum() > 0)
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized(recipe)
    expected = float(np.sum(nv.toarray() ** 2))
    assert nv.norm_sq() >= -1e-6
    np.testing.assert_allclose(nv.norm_sq(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_slice_norms_matches_dense_reference(vcls, dense, data):
    assume(dense.sum() > 0)
    idxs, n_cond = data.draw(condition_groups(n_rows=dense.shape[0]))

    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()
    ref = nv.toarray()
    expected = np.array([np.linalg.norm(ref[idxs == i]) for i in range(n_cond)])

    np.testing.assert_allclose(nv.slice_norms(idxs, n_cond), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@pytest.mark.parametrize("recipe", sorted(RECIPES))
@slow_first_call
@given(dense=dense_matrices())
def test_to_scipy_sparse_plus_means_reconstructs_toarray(vcls, recipe, dense):
    assume(dense.sum() > 0)
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized(recipe)

    sparse = nv.to_scipy_sparse()
    assert sparse.dtype == np.float64
    assert sparse.shape == dense.shape
    assert isinstance(sparse, sp.csc_array if vcls is VCSCArray else sp.csr_array)

    reconstructed = sparse.toarray() - nv.means
    np.testing.assert_allclose(reconstructed, nv.toarray(), atol=1e-6)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_to_scipy_sparse_has_the_same_sparsity_pattern_as_the_raw_array(vcls, dense):
    assume(dense.sum() > 0)
    raw = _scipy_for(vcls, dense)
    nv = vcls.from_scipy(raw).normalized()
    assert nv.to_scipy_sparse().nnz == raw.nnz
