"""Tests for zero-copy block slicing (#77) and fast fancy/combined selection (#78)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from vsparse import VCSCArray, VCSRArray


def _make_array(cls, n_rows=60, n_cols=40, density=0.15, seed=0):
    rng = np.random.default_rng(seed)
    mat = sp.random_array((n_rows, n_cols), density=density, format="lil", random_state=rng)
    mat[:3] = 0
    mat[20:23] = 0
    mat[-2:] = 0
    mat[:, 5:8] = 0
    csr = mat.tocsr()
    csr.data = rng.integers(1, 10, size=csr.nnz).astype(np.float64)
    return cls.from_scipy(csr)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray], ids=["vcsr", "vcsc"])
def test_major_slice_is_zero_copy_and_exact(cls):
    arr = _make_array(cls)
    n_major = arr.n_major
    for s, e in [(0, n_major), (0, 0), (5, 25), (10, 10), (-10, n_major), (3, -5)]:
        sl = arr.major_slice(s, e)
        expected = arr[s:e] if cls is VCSRArray else arr[:, s:e]
        np.testing.assert_array_equal(sl.toarray(), expected.toarray())
        if sl.nnz > 0:
            assert np.shares_memory(sl.indices, arr.indices)
            assert np.shares_memory(sl.values, arr.values)

    with pytest.raises(IndexError):
        arr.major_slice(10, 5)
    with pytest.raises(IndexError):
        arr.major_slice(0, n_major + 1)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray], ids=["vcsr", "vcsc"])
def test_nnz_blocks_cover_axis_and_respect_budget(cls):
    arr = _make_array(cls, n_rows=80, n_cols=50, density=0.2)
    for max_nnz in (1, 10, 50, 200, 100_000):
        blocks = arr.nnz_blocks(max_nnz)
        assert blocks[0][0] == 0
        assert blocks[-1][1] == arr.n_major
        total_nnz = 0
        for i, (s, e) in enumerate(blocks):
            assert s < e
            if i > 0:
                assert s == blocks[i - 1][1]
            blk = arr.major_slice(s, e)
            total_nnz += blk.nnz
            if e - s > 1:
                assert blk.nnz <= max_nnz
        assert total_nnz == arr.nnz

    with pytest.raises(ValueError, match="max_nnz must be positive"):
        arr.nnz_blocks(0)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray], ids=["vcsr", "vcsc"])
def test_normalized_major_slice_and_iter_blocks(cls):
    arr = _make_array(cls, n_rows=50, n_cols=35, density=0.25, seed=1)
    view = arr.normalized("parafac2")
    full_dense = view.toarray()

    blocks = list(view.iter_blocks(max_nnz=40))
    assert len(blocks) > 1
    parts = [b.toarray() for b in blocks]
    axis = 0 if cls is VCSRArray else 1
    np.testing.assert_allclose(np.concatenate(parts, axis=axis), full_dense, atol=1e-12)

    # Verify stats sharing on major_slice
    b0 = blocks[0]
    if cls is VCSRArray:
        assert np.shares_memory(b0.gene_scale, view.gene_scale)
        assert np.shares_memory(b0.col_mean, view.col_mean)
        assert np.shares_memory(b0.col_post_scale, view.col_post_scale)
    else:
        assert np.shares_memory(b0.row_scale, view.row_scale)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray], ids=["vcsr", "vcsc"])
def test_combined_select_matches_scipy(cls):
    arr = _make_array(cls, n_rows=45, n_cols=35, density=0.2, seed=2)
    ref = arr.toarray()
    rng = np.random.default_rng(5)

    for rows, cols in [
        (rng.random(45) < 0.4, rng.random(35) < 0.5),
        (np.array([40, 2, 2, 0, 21]), np.array([30, 5, 5, 1, 0])),
        (np.array([], dtype=np.int64), np.array([1, 2, 3])),
        (np.array([0, 1, 2]), np.array([], dtype=np.int64)),
        (slice(5, 30, 2), slice(None, None, -3)),
    ]:
        got = arr.select(rows, cols)
        expected = ref[rows][:, cols]
        np.testing.assert_array_equal(got.toarray(), expected)
        np.testing.assert_array_equal(arr[rows, cols].toarray(), expected)


@settings(max_examples=25, deadline=None)
@given(
    n_rows=st.integers(1, 25),
    n_cols=st.integers(1, 25),
    density=st.floats(0.0, 0.5),
    seed=st.integers(0, 10_000),
)
def test_select_and_major_slice_hypothesis(n_rows, n_cols, density, seed):
    rng = np.random.default_rng(seed)
    mat = sp.random_array((n_rows, n_cols), density=density, format="csr", random_state=rng)
    if mat.nnz:
        mat.data = rng.integers(1, 6, size=mat.nnz).astype(np.float64)
    ref = mat.toarray()

    for cls in (VCSRArray, VCSCArray):
        arr = cls.from_scipy(mat)
        r_idx = rng.integers(0, n_rows, size=rng.integers(0, n_rows + 3))
        c_idx = rng.integers(0, n_cols, size=rng.integers(0, n_cols + 3))
        np.testing.assert_array_equal(arr.select(r_idx, c_idx).toarray(), ref[r_idx][:, c_idx])

        s = int(rng.integers(0, arr.n_major + 1))
        e = int(rng.integers(s, arr.n_major + 1))
        expected_sl = ref[s:e] if cls is VCSRArray else ref[:, s:e]
        np.testing.assert_array_equal(arr.major_slice(s, e).toarray(), expected_sl)
