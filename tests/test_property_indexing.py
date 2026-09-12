"""Property-based tests for general (row-key, col-key) indexing on
VCSCArray/VCSRArray, against dense reference indexing.

Each axis key is independently one of: a full slice, a general slice, a fancy
int list (with duplicates and negative indices), or a boolean mask -- never a
bare int, since that collapses a dimension and is exercised by dedicated unit
tests in test_indexing.py / test_general_indexing_nnz_astype.py instead. This
generalizes the fixed hand-picked cases previously spread across
test_indexing.py, test_general_indexing_nnz_astype.py, and the manual
25-iteration fuzz loop in test_select_minor_fanout.py -- including the
duplicate-index "fan out" regression those covered.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import axis_key, dense_matrices, slow_first_call
from hypothesis import given
from hypothesis import strategies as st

from vsparse import VCSCArray, VCSRArray


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_general_indexing_matches_dense(vcls, dense, data):
    row_key = axis_key(data, dense.shape[0])
    col_key = axis_key(data, dense.shape[1])

    v = vcls.from_scipy(_scipy_for(vcls, dense))
    result = v[row_key, col_key]
    expected = dense[row_key, :][:, col_key]

    assert isinstance(result, vcls)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result.toarray(), expected)
