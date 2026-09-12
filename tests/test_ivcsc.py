from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from vsparse import _ivcsc
from vsparse._base import VCSCArray


def test_pack_unpack_roundtrip(csc):
    v = VCSCArray.from_scipy(csc)
    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    assert packed.dtype == np.uint8
    unpacked = _ivcsc.unpack_indices(v.value_ptr, packed, v.indices.dtype)

    # Order within a (major-slice, value) group is not preserved (packing
    # sorts each group), so compare group-wise as sets rather than arrays.
    for g in range(v.value_ptr.shape[0] - 1):
        start, end = v.value_ptr[g], v.value_ptr[g + 1]
        assert set(unpacked[start:end].tolist()) == set(v.indices[start:end].tolist())


def test_pack_unpack_empty():
    value_ptr = np.zeros(1, dtype=np.int64)
    indices = np.empty(0, dtype=np.int32)
    packed = _ivcsc.pack_indices(value_ptr, indices)
    assert packed.shape == (0,)
    unpacked = _ivcsc.unpack_indices(value_ptr, packed, np.dtype(np.int32))
    assert unpacked.shape == (0,)


def test_pack_unpack_large_indices():
    # Force multi-byte varints.
    dense = np.zeros((300, 2))
    dense[10, 0] = 1.0
    dense[299, 0] = 1.0
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    unpacked = _ivcsc.unpack_indices(v.value_ptr, packed, v.indices.dtype)
    np.testing.assert_array_equal(np.sort(unpacked), np.sort(v.indices))


def test_unpack_parallel_matches_serial():
    # Large + repetitive enough to exercise the parallel decode path for
    # real (multiple chunks, chunk boundaries landing mid-group), not just
    # fall back to the serial path via the size threshold.
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 5, size=(8000, 800)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.7] = 0.0
    v = VCSCArray.from_scipy(sp.csc_array(dense))

    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    assert packed.nbytes > _ivcsc._PARALLEL_MIN_BYTES, (
        "test data too small to hit the parallel path"
    )

    out = np.empty(v.indices.shape[0], dtype=v.indices.dtype)
    _ivcsc._unpack(v.value_ptr, packed, out)

    for n_chunks in (2, 5, 16):
        parallel_out = np.empty(v.indices.shape[0], dtype=v.indices.dtype)
        _ivcsc._unpack_parallel(v.value_ptr, packed, parallel_out, n_chunks)
        np.testing.assert_array_equal(parallel_out, out)

    # And the public entry point picks the parallel path automatically.
    assert _ivcsc._num_chunks(packed.shape[0]) > 1
    auto = _ivcsc.unpack_indices(v.value_ptr, packed, v.indices.dtype)
    np.testing.assert_array_equal(auto, out)
