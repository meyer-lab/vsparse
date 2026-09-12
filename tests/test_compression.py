"""Tests for compression helper options, codec dispatch, and string column exclusion."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from vsparse import _compression


def test_h5_dataset_kwargs_options():
    """Verify h5_dataset_kwargs correctly builds filter parameters."""
    blosc_kwargs = _compression.h5_dataset_kwargs()
    assert "compression" in blosc_kwargs
    assert "compression_opts" in blosc_kwargs


def test_zarr_dataset_kwargs_options():
    """Verify zarr_dataset_kwargs correctly builds compressor dict."""
    blosc_kwargs = _compression.zarr_dataset_kwargs()
    assert "compressor" in blosc_kwargs or "codecs" in blosc_kwargs


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
        # Numeric dataset has blosc2 filter
        assert any(int(fid) >= 32000 for fid in f["numbers"]._filters)
        # String dataset has no compression filters
        assert dict(f["strings"]._filters) == {}
