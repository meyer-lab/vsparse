"""The CPU matmul kernels split work by nonzeros and hoist the transform out of value groups.

Both are internal to the kernels, so the tests check what they must preserve: the products on matrices with
heavy-tailed slice lengths, for every recipe, direction and thread count, and the chunking helper itself.
"""

from __future__ import annotations

from itertools import pairwise

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import RECIPES, VCSCArray, VCSRArray
from vsparse._vcs_matmul import _nnz_balanced_bounds


def _skewed_counts(n_rows: int = 300, n_cols: int = 80, seed: int = 0) -> np.ndarray:
    """Integer counts with a few genes in nearly every cell and long runs of rare ones, placed at the end.

    The long columns are contiguous, so an equal-count split of the columns gives one chunk most of the work.
    """
    rng = np.random.default_rng(seed)
    p = np.concatenate([np.full(n_cols // 8, 0.9), np.full(n_cols - n_cols // 8, 0.03)])[::-1]
    dense = rng.geometric(0.5, size=(n_rows, n_cols)) * (rng.random((n_rows, n_cols)) < p)
    dense[:, 0] = 1  # every cell has at least one count
    return dense.astype(np.float64)


def _reference(dense: np.ndarray, recipe: str) -> np.ndarray:
    """The normalized matrix from the view itself, densified; the products below test only the kernels."""
    return VCSRArray.from_scipy(sp.csr_array(dense)).normalized(recipe).toarray()


@pytest.mark.parametrize("vcls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("threads", [1, 3, 8])
def test_products_match_dense_on_skewed_counts(vcls, recipe, threads):
    dense = _skewed_counts()
    arr = sp.csr_array(dense) if vcls is VCSRArray else sp.csc_array(dense)
    nv = vcls.from_scipy(arr).normalized(recipe)
    ref = _reference(dense, recipe)
    rng = np.random.default_rng(1)
    B, L = rng.normal(size=(dense.shape[1], 5)), rng.normal(size=(4, dense.shape[0]))
    previous = numba.get_num_threads()
    # numba refuses more threads than the machine launched with (4 on CI runners).
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    try:
        np.testing.assert_allclose(nv @ B, ref @ B, atol=1e-9)
        np.testing.assert_allclose(L @ nv, L @ ref, atol=1e-9)
        np.testing.assert_allclose(nv @ B[:, 0], ref @ B[:, 0], atol=1e-9)
        np.testing.assert_allclose(L[0] @ nv, L[0] @ ref, atol=1e-9)
    finally:
        numba.set_num_threads(previous)


@pytest.mark.parametrize("vcls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("nchunks", [1, 2, 7, 64, 1000])
def test_bounds_cover_every_slice_and_balance_the_nonzeros(vcls, nchunks):
    dense = _skewed_counts(n_rows=200, n_cols=120)
    arr = vcls.from_scipy(sp.csr_array(dense) if vcls is VCSRArray else sp.csc_array(dense))
    bounds = _nnz_balanced_bounds(arr.major_ptr, arr.value_ptr, nchunks)
    assert bounds.shape == (nchunks + 1,)
    assert bounds[0] == 0 and bounds[-1] == arr.n_major
    assert np.all(np.diff(bounds) >= 0)
    per_slice = np.diff(arr.value_ptr[arr.major_ptr])
    held = (
        np.add.reduceat(per_slice, bounds[:-1].clip(max=arr.n_major - 1))
        if arr.n_major
        else np.zeros(0)
    )
    held = np.where(np.diff(bounds) > 0, held, 0)
    assert held.sum() == per_slice.sum()
    # No chunk exceeds its share by more than the longest slice (a cut lands on a slice boundary).
    assert held.max() <= per_slice.sum() / nchunks + per_slice.max()


def test_bounds_of_an_empty_array():
    major_ptr = np.zeros(4, dtype=np.int64)
    bounds = _nnz_balanced_bounds(major_ptr, np.zeros(1, dtype=np.int64), 5)
    assert bounds[0] == 0 and bounds[-1] == 3 and np.all(np.diff(bounds) >= 0)


def test_equal_count_split_would_be_unbalanced_here():
    """Guards the premise: on the skewed matrix an equal-count split of the columns is badly unbalanced."""
    dense = _skewed_counts(n_rows=400, n_cols=160)
    per_col = (dense != 0).sum(axis=0)
    equal = [c.sum() for c in np.array_split(per_col, 8)]
    assert max(equal) > 2 * per_col.sum() / 8
    arr = VCSCArray.from_scipy(sp.csc_array(dense))
    bounds = _nnz_balanced_bounds(arr.major_ptr, arr.value_ptr, 8)
    balanced = [per_col[a:b].sum() for a, b in pairwise(bounds)]
    assert max(balanced) < max(equal)
