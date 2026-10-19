"""``iter_packed`` and the cursor underneath it (``vsparse._stream``).

Block ``k`` has to equal ``load_packed(path)[lo:hi]``, so that is what most of
these assert. The rest force the cursor's window machinery to run many times
over a tiny fixture -- a window that ends mid-row, a read too small for the
elements it was sized for -- since on real files those paths run once in a
million rows and would otherwise never be exercised.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from vsparse import _stream, iter_packed, load_packed


@pytest.fixture
def dense25(rng):
    dense = (rng.random((25, 40)) < 0.35) * rng.integers(1, 30, (25, 40))
    dense[7] = 0  # an empty row owns no bytes at all
    dense[11] = rng.integers(1, 30, 40)  # a full one
    return dense.astype(np.float32)


@pytest.fixture
def path(ivcs_file, dense25):
    obs = pd.DataFrame({"grp": [f"c{i % 4}" for i in range(25)], "depth": dense25.sum(1)})
    return ivcs_file(dense25, obs=obs)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 24, 25, 30])
def test_blocks_reassemble_the_matrix(path, dense25, chunk_size):
    blocks = list(iter_packed(path, chunk_size=chunk_size))
    assert len(blocks) == -(-25 // chunk_size)
    np.testing.assert_array_equal(np.vstack([b.X.toarray() for b in blocks]), dense25)


@pytest.mark.parametrize("chunk_size", [1, 4, 25])
def test_each_block_equals_the_same_in_memory_slice(path, chunk_size):
    full = load_packed(path)
    for i, block in enumerate(iter_packed(path, chunk_size=chunk_size)):
        want = full[i * chunk_size : (i + 1) * chunk_size]
        np.testing.assert_array_equal(block.X.toarray(), want.X.toarray())
        pd.testing.assert_frame_equal(block.obs, want.obs)
        pd.testing.assert_frame_equal(block.var, want.var)


def test_blocks_carry_no_statistics_sidecar(path):
    """``col_sums`` and friends are whole-matrix quantities; a block's are not the file's."""
    block = next(iter(iter_packed(path, chunk_size=5)))
    assert block.X._stats is None


def test_a_wide_matrix_needs_more_bytes_than_elements(ivcs_file, rng):
    """Column gaps over 127 take a second varint byte, so the sized read comes up short."""
    dense = ((rng.random((30, 5000)) < 0.01) * rng.integers(1, 9, (30, 5000))).astype(np.float32)
    assert dense.sum(axis=0).max() > 0
    p = ivcs_file(dense)
    with h5py.File(p, "r") as f:
        px = _stream.open_packed(f["X"])
        assert px.packed.shape[0] > int(px.elem_ptr[-1])  # multi-byte varints really are present
    np.testing.assert_array_equal(
        np.vstack([b.X.toarray() for b in iter_packed(p, chunk_size=4)]), dense
    )


@pytest.mark.parametrize("window_elems", [1, 2, 5, 40, 1 << 20])
def test_a_row_subset_spans_however_many_windows(path, dense25, window_elems):
    """``read_rows`` carries only (row, byte) across windows, at any window size."""
    rows = np.array([0, 1, 7, 8, 11, 24])
    with h5py.File(path, "r") as f:
        px = _stream.open_packed(f["X"])
        got = _stream.read_rows(px, rows, window_elems=window_elems)
    np.testing.assert_array_equal(got.toarray(), dense25[rows])


def test_an_all_zero_matrix_streams(ivcs_file):
    """Every row is empty, so no row owns any bytes and the stream is empty."""
    p = ivcs_file(np.zeros((6, 4), dtype=np.float32))
    blocks = list(iter_packed(p, chunk_size=4))
    np.testing.assert_array_equal(np.vstack([b.X.toarray() for b in blocks]), np.zeros((6, 4)))


def test_a_file_with_no_rows_yields_nothing(ivcs_file):
    assert list(iter_packed(ivcs_file(np.zeros((0, 4), dtype=np.float32)))) == []


def test_chunk_size_must_be_positive(path):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        next(iter(iter_packed(path, chunk_size=0)))


def test_ivcsc_files_cannot_be_streamed_by_row(ivcs_file, dense25):
    p = ivcs_file(dense25, format="csc")
    with pytest.raises(NotImplementedError, match="stored IVCSC"):
        next(iter(iter_packed(p)))
