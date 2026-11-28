"""Tests for IVCS v2 static canonical Huffman codec, selective pread, and v1 fallback."""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from vsparse import VCSCAnnData, VCSCArray, VCSRArray, _compression, _huffman, _ivcsc, load_packed


def test_package_merge_kraft_and_edge_cases():
    # Empty
    lengths = _huffman._package_merge_lengths(np.zeros(256, dtype=np.int64))
    assert np.all(lengths == 0)

    # Single active symbol -> length 1 (not 0)
    counts = np.zeros(256, dtype=np.int64)
    counts[42] = 100
    lengths = _huffman._package_merge_lengths(counts)
    assert lengths[42] == 1
    assert np.sum(lengths) == 1

    # Steep Fibonacci-like weights that would exceed length 13 without Package-Merge
    counts = np.zeros(256, dtype=np.int64)
    w = 1
    for i in range(40):
        counts[i] = w
        w = min(w * 2, 1 << 50)
    lengths = _huffman._package_merge_lengths(counts)
    active = lengths[lengths > 0]
    assert active.max() <= _huffman.MAX_CODE_LEN
    kraft = sum(1 << (_huffman.MAX_CODE_LEN - int(L)) for L in active)
    assert kraft == (1 << _huffman.MAX_CODE_LEN)


def test_pack_unpack_empty_and_single_symbol():
    # Empty
    major_ptr = np.zeros(5, dtype=np.int64)
    value_ptr = np.zeros(1, dtype=np.int64)
    indices = np.empty(0, dtype=np.uint16)
    packed, lengths, bit_offsets = _huffman.pack_indices(major_ptr, value_ptr, indices)
    assert packed.shape == (0,)
    assert np.all(bit_offsets == 0)
    out = _huffman.unpack_indices(
        major_ptr, value_ptr, packed, lengths, bit_offsets, np.dtype(np.uint16)
    )
    assert out.shape == (0,)

    # Single symbol: all gaps are 0 (every row has [0, 1, 2])
    dense = np.ones((10, 3), dtype=np.float32)
    v = VCSRArray.from_scipy(sp.csr_array(dense))
    packed, lengths, bit_offsets = _huffman.pack_indices(v.major_ptr, v.value_ptr, v.indices)
    assert lengths[0] == 1
    out = _huffman.unpack_indices(
        v.major_ptr, v.value_ptr, packed, lengths, bit_offsets, v.indices.dtype
    )
    np.testing.assert_array_equal(out, v.indices)


def test_pack_unpack_large_escaped_gaps():
    # Gaps >> 255 requiring multi-byte LEB128 escapes
    dense = sp.lil_array((10, 100_000), dtype=np.float32)
    dense[0, 500] = 1.0
    dense[0, 70_000] = 1.0
    dense[3, 99_999] = 2.0
    dense[9, 0] = 1.0
    dense[9, 255] = 1.0
    dense[9, 511] = 1.0
    v = VCSRArray.from_scipy(dense.tocsr())
    packed, lengths, bit_offsets = _huffman.pack_indices(v.major_ptr, v.value_ptr, v.indices)
    out = _huffman.unpack_indices(
        v.major_ptr, v.value_ptr, packed, lengths, bit_offsets, v.indices.dtype
    )
    np.testing.assert_array_equal(out, v.indices)


def test_pack_unpack_parallel_path():
    rng = np.random.default_rng(7)
    n_rows, n_cols = 1200, 1000
    mat = sp.random_array((n_rows, n_cols), density=0.05, format="csr", random_state=rng)
    mat.data = rng.integers(1, 10, size=mat.nnz).astype(np.float32)
    v = VCSRArray.from_scipy(mat)

    packed, lengths, bit_offsets = _huffman.pack_indices(v.major_ptr, v.value_ptr, v.indices)
    out = _huffman.unpack_indices(
        v.major_ptr, v.value_ptr, packed, lengths, bit_offsets, v.indices.dtype
    )
    rebuilt = VCSRArray(v.shape, v.major_ptr, v.values, v.value_ptr, out)
    np.testing.assert_array_equal(rebuilt.toarray(), v.toarray())


@settings(max_examples=30, deadline=None)
@given(
    n_rows=st.integers(1, 40),
    n_cols=st.integers(1, 400),
    density=st.floats(0.0, 0.4),
    seed=st.integers(0, 10_000),
)
def test_huffman_roundtrip_hypothesis(n_rows, n_cols, density, seed):
    rng = np.random.default_rng(seed)
    mat = sp.random_array((n_rows, n_cols), density=density, format="csr", random_state=rng)
    if mat.nnz:
        mat.data = rng.integers(1, 8, size=mat.nnz).astype(np.float32)
    v = VCSRArray.from_scipy(mat)
    packed, lengths, bit_offsets = _huffman.pack_indices(v.major_ptr, v.value_ptr, v.indices)
    out = _huffman.unpack_indices(
        v.major_ptr, v.value_ptr, packed, lengths, bit_offsets, v.indices.dtype
    )
    rebuilt = VCSRArray(v.shape, v.major_ptr, v.values, v.value_ptr, out)
    np.testing.assert_array_equal(rebuilt.toarray(), v.toarray())


def test_selective_pread_reads_only_requested_bytes(tmp_path, monkeypatch):
    """Selecting a small subset of cells only preads a tiny fraction of packed_indices (#102)."""
    rng = np.random.default_rng(42)
    n_obs, n_vars = 2000, 1500
    mat = sp.random_array((n_obs, n_vars), density=0.1, format="csr", random_state=rng)
    mat.data = rng.integers(1, 15, size=mat.nnz).astype(np.float32)
    va = VCSCAnnData(
        X=VCSRArray.from_scipy(mat),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]),
    )
    path = tmp_path / "big.h5ad"
    va.write_h5ad(path)

    with h5py.File(path, "r") as f:
        ds = f["X"]["packed_indices"]
        assert dict(ds._filters) == {}
        assert ds.chunks is None
        total_packed_bytes = int(ds.shape[0])
        assert total_packed_bytes > 200_000

    bytes_read = 0
    real_pread_spans = _huffman._pread_spans

    def counting_pread_spans(filename, file_offset, span_starts, span_ends, out):
        nonlocal bytes_read
        bytes_read += int(np.sum(span_ends - span_starts))
        return real_pread_spans(filename, file_offset, span_starts, span_ends, out)

    monkeypatch.setattr(_huffman, "_pread_spans", counting_pread_spans)
    sel = np.array([10, 500, 1500, 1990], dtype=np.int64)
    sub = load_packed(path, obs_idx=sel)
    assert va.X is not None and sub.X is not None
    np.testing.assert_array_equal(sub.X.toarray(), va.X[sel].toarray())
    assert 0 < bytes_read < total_packed_bytes * 0.05


def test_v2_zarr_and_core_h5_fallback(tmp_path):
    rng = np.random.default_rng(3)
    mat = sp.random_array((60, 400), density=0.05, format="csr", random_state=rng)
    mat.data = rng.integers(1, 10, size=mat.nnz).astype(np.float32)
    va = VCSCAnnData(
        X=VCSRArray.from_scipy(mat),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(400)]),
    )
    # Zarr roundtrip + subset
    zpath = tmp_path / "v2.zarr"
    va.write_zarr(zpath)
    back_z = VCSCAnnData.read_zarr(zpath)
    assert va.X is not None and back_z.X is not None
    np.testing.assert_array_equal(back_z.X.toarray(), va.X.toarray())

    # In-memory HDF5 (core driver) subset fallback
    hpath = tmp_path / "v2.h5ad"
    va.write_h5ad(hpath)
    with h5py.File(hpath, "r", driver="core", backing_store=False) as f:
        from vsparse._io import read_vcs_subset

        sub = read_vcs_subset(f["X"], np.array([0, 5, 59]), np.array([1, 100, 399]))
    np.testing.assert_array_equal(
        sub.toarray(), va.X[np.array([0, 5, 59]), np.array([1, 100, 399])].toarray()
    )


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray], ids=["ivcsr_v1", "ivcsc_v1"])
def test_v1_backward_compatibility(tmp_path, cls):
    """Existing v1 (0.1.0, LEB128 + zstd-1) files still load fully and selectively."""
    rng = np.random.default_rng(9)
    mat = sp.random_array((50, 300), density=0.06, format="csr", random_state=rng)
    mat.data = rng.integers(1, 10, size=mat.nnz).astype(np.float32)
    arr = cls.from_scipy(mat)

    path = tmp_path / "v1.h5ad"
    kwargs = _compression.h5_dataset_kwargs()
    with _compression.numeric_only_compression("h5"), h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.attrs["shape"] = arr.shape
        g.attrs["indices_dtype"] = arr.indices.dtype.name
        g.attrs["encoding-type"] = "ivcsc" if cls is VCSCArray else "ivcsr"
        g.attrs["encoding-version"] = "0.1.0"
        for name in ("major_ptr", "values", "value_ptr"):
            ad.io.write_elem(g, name, getattr(arr, name), dataset_kwargs=kwargs)
        packed = _ivcsc.pack_indices(arr.value_ptr, arr.indices)
        ad.io.write_elem(g, "packed_indices", packed, dataset_kwargs=kwargs)
        ad.io.write_elem(f, "obs", pd.DataFrame(index=[f"c{i}" for i in range(50)]))
        ad.io.write_elem(f, "var", pd.DataFrame(index=[f"g{j}" for j in range(300)]))
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

    full = VCSCAnnData.read_h5ad(path)
    assert full.X is not None
    np.testing.assert_array_equal(full.X.toarray(), arr.toarray())

    sub = load_packed(path, obs_idx=np.array([2, 10, 49]), var_idx=np.array([0, 50, 299]))
    assert sub.X is not None
    np.testing.assert_array_equal(
        sub.X.toarray(), arr[np.array([2, 10, 49]), np.array([0, 50, 299])].toarray()
    )
