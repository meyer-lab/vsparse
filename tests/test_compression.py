"""Tests for compression helper options, codec dispatch, and string column exclusion."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from vsparse import _compression


def test_h5_dataset_kwargs_options():
    """Verify h5_dataset_kwargs selects the zstd filter at level 1."""
    kwargs = _compression.h5_dataset_kwargs()
    assert kwargs["compression"] == _compression._H5_ZSTD_FILTER_ID
    assert tuple(kwargs["compression_opts"]) == (1,)


def test_zarr_dataset_kwargs_options():
    """Verify zarr_dataset_kwargs correctly builds compressor dict."""
    kwargs = _compression.zarr_dataset_kwargs()
    assert "zstd" in type(kwargs["compressor"]).__name__.lower()


def test_numeric_only_compression_invalid_backend():
    """Verify that numeric_only_compression raises ValueError for an unsupported format."""
    with (
        pytest.raises(ValueError, match="store_kind must be 'h5' or 'zarr'"),
        _compression.numeric_only_compression("invalid_format"),
    ):
        pass


def test_is_string_like_detection():
    """Verify _is_string_like correctly detects string, unicode, and object dtypes."""
    assert _compression._is_string_like(np.dtype("O"), None)
    assert _compression._is_string_like(np.dtype("U10"), None)
    assert _compression._is_string_like(np.dtype("S10"), None)
    assert not _compression._is_string_like(np.dtype("int32"), None)
    assert not _compression._is_string_like(np.dtype("float64"), None)


def test_numeric_only_compression_skips_strings(tmp_path):
    """Verify that string/object dataset writes do not receive numeric compression filters."""
    path = tmp_path / "test_strings.h5"
    kwargs = _compression.h5_dataset_kwargs()

    with (
        _compression.numeric_only_compression("h5"),
        h5py.File(path, "w") as f,
    ):
        # Numeric dataset gets compressed
        f.create_dataset("numbers", data=np.array([1, 2, 3], dtype=np.int32), **kwargs)
        # String array should have compression removed automatically
        f.create_dataset("strings", data=np.array(["a", "b", "c"], dtype=object), **kwargs)

    with h5py.File(path, "r") as f:
        # Numeric dataset has the zstd filter
        assert str(_compression._H5_ZSTD_FILTER_ID) in f["numbers"]._filters
        # String dataset has no compression filters
        assert dict(f["strings"]._filters) == {}


def _written_chunks(tmp_path, data, **extra):
    path = tmp_path / "chunks.h5"
    with _compression.numeric_only_compression("h5"), h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, **_compression.h5_dataset_kwargs(), **extra)
    with h5py.File(path, "r") as f:
        return f["a"].chunks


def test_large_1d_arrays_get_1mib_chunks(tmp_path):
    """Compressed 1-D arrays larger than one chunk get ~1 MiB chunks for the parallel reader."""
    for dtype in (np.uint8, np.int64):
        per_chunk = (1 << 20) // np.dtype(dtype).itemsize
        assert _written_chunks(tmp_path, np.zeros(3 * per_chunk + 7, dtype=dtype)) == (per_chunk,)


def test_small_explicit_and_2d_chunks_untouched(tmp_path):
    """Arrays that fit in one chunk, explicit chunks, and 2-D arrays keep h5py's choice."""
    small = np.zeros(1000, dtype=np.int64)
    big = np.zeros(1 << 20, dtype=np.int64)
    assert _written_chunks(tmp_path, small) != ((1 << 20) // 8,)
    assert _written_chunks(tmp_path, big, chunks=(4096,)) == (4096,)
    assert len(_written_chunks(tmp_path, np.zeros((1 << 18, 4)))) == 2
