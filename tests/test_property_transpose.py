"""Property-based tests for vsparse._construct.transpose_major: direct
VCSC<->VCSR storage regrouping, via _VCSBase._transpose_major()."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import dense_matrices, slow_first_call
from hypothesis import example, given

from vsparse import VCSCArray, VCSRArray


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@example(dense=np.zeros((6, 5)))
@given(dense=dense_matrices())
def test_transpose_major_matches_dense(vcls, dense):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    dual = v._transpose_major()

    other_cls = VCSRArray if vcls is VCSCArray else VCSCArray
    assert isinstance(dual, other_cls)
    assert dual.shape == dense.shape
    np.testing.assert_allclose(dual.toarray(), dense)
    # Same logical nonzeros, so nnz must match; unique-value counts may differ per axis.
    assert dual.nnz == v.nnz


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_transpose_major_is_involutive(vcls, dense):
    """Transposing twice returns to the original format with the same matrix."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    back = v._transpose_major()._transpose_major()
    assert type(back) is type(v)
    np.testing.assert_allclose(back.toarray(), dense)
