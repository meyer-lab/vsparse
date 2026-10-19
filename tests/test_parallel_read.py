"""Tests for the parallel zstd chunk reader, its byte-range form, and their fallbacks.

See ``vsparse._compression``: :func:`read_range` is the primitive,
:func:`read_array` is it over a whole dataset.
"""

from __future__ import annotations

import subprocess
import sys

import h5py
import hdf5plugin
import numcodecs
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vsparse import VCSRArray, VCSCAnnData, _compression

ZSTD = _compression.h5_dataset_kwargs()


@pytest.fixture
def spy(monkeypatch):
    """Record whether read_array took the parallel path."""
    calls = []
    real = _compression._read_chunks_parallel

    def wrapped(*args):
        calls.append(args[0].name)
        return real(*args)

    monkeypatch.setattr(_compression, "_read_chunks_parallel", wrapped)
    return calls


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64, np.float32])
@pytest.mark.parametrize("n", [0, 1, 999, 1000, 4321])
def test_parallel_read_matches(tmp_path, spy, dtype, n):
    """Chunked zstd datasets decode in parallel, including a partial final chunk."""
    data = (np.random.default_rng(0).integers(0, 200, n)).astype(dtype)
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, chunks=(1000,) if n else None, maxshape=(None,), **ZSTD)
    with h5py.File(path, "r") as f:
        out = _compression.read_array(f["a"])
    assert out.dtype == data.dtype
    np.testing.assert_array_equal(out, data)
    assert spy == ["/a"]


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (0, 4321),  # the whole dataset
        (0, 1),  # one element, first chunk
        (999, 1001),  # across a chunk boundary
        (1000, 2000),  # exactly one whole chunk
        (1500, 4321),  # partial first chunk through the partial last one
        (4320, 4321),  # last element
        (2000, 2000),  # empty, mid-stream
        (0, 0),  # empty, at the start
    ],
)
def test_read_range_matches_slicing(tmp_path, spy, start, stop):
    """A byte range decodes only the chunks that cover it, and matches h5py's slice."""
    data = np.arange(4321, dtype=np.int64)
    path = tmp_path / "r.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, chunks=(1000,), **ZSTD)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            _compression.read_range(f["a"], start, stop), data[start:stop]
        )
    assert spy == ["/a"]


def test_read_range_reuses_a_cached_chunk_table(tmp_path, spy):
    """A caller streaming in windows builds the chunk table once and passes it back."""
    data = np.arange(5000, dtype=np.int32)
    path = tmp_path / "cached.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, chunks=(1000,), **ZSTD)
    with h5py.File(path, "r") as f:
        table = _compression.chunk_table(f["a"])
        assert table is not None
        for lo in range(0, 5000, 700):
            np.testing.assert_array_equal(
                _compression.read_range(f["a"], lo, lo + 700, table=table),
                data[lo : lo + 700],
            )
    assert spy == ["/a"] * 8


def test_chunk_table_is_none_for_other_layouts(tmp_path):
    """``chunk_table`` is the eligibility test as well as the table."""
    data = np.arange(1234, dtype=np.int64)
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data)
    with h5py.File(path, "r") as f:
        assert _compression.chunk_table(f["a"]) is None
        np.testing.assert_array_equal(_compression.read_range(f["a"], 10, 20), data[10:20])


def test_userblock_offsets(tmp_path, spy):
    """Chunk offsets from chunk_iter already include any userblock."""
    data = np.arange(50_000, dtype=np.int64)
    path = tmp_path / "ub.h5"
    with h5py.File(path, "w", userblock_size=512) as f:
        f.create_dataset("a", data=data, chunks=(4096,), **ZSTD)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(_compression.read_array(f["a"]), data)
    assert spy == ["/a"]


def test_chunk_stored_without_filter(tmp_path, spy):
    """A chunk HDF5 stored raw (filter-mask bit set) is copied, not decoded."""
    data = np.arange(3000, dtype=np.int32)
    path = tmp_path / "mask.h5"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("a", shape=(3000,), dtype=np.int32, chunks=(1000,), **ZSTD)
        ds[:] = data
        ds.id.write_direct_chunk((1000,), data[1000:2000].tobytes(), filter_mask=1)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(_compression.read_array(f["a"]), data)
    assert spy == ["/a"]


def test_unwritten_chunk_falls_back(tmp_path, spy):
    """A never-written chunk reads as the fill value via the ordinary path."""
    path = tmp_path / "sparse.h5"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "a", shape=(3000,), dtype=np.int64, chunks=(1000,), fillvalue=7, **ZSTD
        )
        ds[:1000] = 1
    with h5py.File(path, "r") as f:
        out = _compression.read_array(f["a"])
    np.testing.assert_array_equal(out, np.r_[np.ones(1000), np.full(2000, 7)])
    assert spy == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # contiguous, uncompressed
        {"chunks": (100,)},  # chunked, uncompressed
        dict(hdf5plugin.Blosc2(cname="lz4")),  # the previous default
        {"chunks": (100,), "shuffle": True, **ZSTD},  # zstd plus another filter
    ],
    ids=["contiguous", "chunked", "blosc2", "zstd+shuffle"],
)
def test_other_layouts_fall_back(tmp_path, spy, kwargs):
    data = np.arange(1234, dtype=np.int64)
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, **kwargs)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(_compression.read_array(f["a"]), data)
    assert spy == []


def test_writable_or_in_memory_file_falls_back(tmp_path, spy):
    data = np.arange(5000, dtype=np.int64)
    path = tmp_path / "rw.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, chunks=(1000,), **ZSTD)
        np.testing.assert_array_equal(_compression.read_array(f["a"]), data)
    with h5py.File(path, "r", driver="core") as f:
        np.testing.assert_array_equal(_compression.read_array(f["a"]), data)
    assert spy == []


def test_zarr_array_reads(tmp_path):
    zarr = pytest.importorskip("zarr")
    data = np.arange(5000, dtype=np.int64)
    arr = zarr.create_array(tmp_path / "z.zarr", shape=data.shape, dtype=data.dtype, chunks=(1000,))
    arr[:] = data
    np.testing.assert_array_equal(_compression.read_array(arr), data)


def _vcsr_anndata(n_obs=400, n_vars=60) -> VCSCAnnData:
    rng = np.random.default_rng(1)
    m = sp.random(n_obs, n_vars, density=0.2, format="csr", random_state=rng)
    m.data = rng.integers(1, 20, m.nnz).astype(np.float32)
    return VCSCAnnData(
        X=VCSRArray.from_scipy(m),
        obs=pd.DataFrame(
            {"depth": np.asarray(m.sum(axis=1)).ravel()}, index=[f"c{i}" for i in range(n_obs)]
        ),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]),
    )


def test_write_h5ad_reads_in_parallel(tmp_path, spy):
    """Every IVCSR array written by write_h5ad takes the parallel read path."""
    va = _vcsr_anndata()
    path = tmp_path / "x.h5ad"
    va.write_h5ad(path)
    back = VCSCAnnData.read_h5ad(path)
    assert back.X is not None and va.X is not None
    np.testing.assert_array_equal(back.X.toarray(), va.X.toarray())
    assert {"/X/major_ptr", "/X/values", "/X/value_ptr", "/X/packed_indices"} <= set(spy)


def test_blosc2_file_still_reads(tmp_path, spy):
    """Files written with the earlier Blosc2+LZ4 default keep reading."""
    va = _vcsr_anndata()
    path = tmp_path / "legacy.h5ad"
    va.write_h5ad(path, dataset_kwargs=dict(hdf5plugin.Blosc2(cname="lz4")))
    with h5py.File(path, "r") as f:
        assert "32026" in f["X"]["packed_indices"]._filters
    back = VCSCAnnData.read_h5ad(path)
    assert back.X is not None and va.X is not None
    np.testing.assert_array_equal(back.X.toarray(), va.X.toarray())
    np.testing.assert_array_equal(back.obs["depth"], va.obs["depth"])
    assert spy == []


def test_read_h5ad_in_fresh_process(tmp_path):
    """read_h5ad registers the HDF5 filters itself; compressed obs columns read before X does."""
    path = tmp_path / "fresh.h5ad"
    _vcsr_anndata().write_h5ad(path)
    code = f"import vsparse; print(vsparse.VCSCAnnData.read_h5ad({str(path)!r}).shape)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "(400, 60)"


def test_numcodecs_decodes_hdf5_zstd_chunks(tmp_path):
    """The parallel path relies on the HDF5 zstd filter writing plain zstd frames."""
    data = np.arange(1000, dtype=np.int64)
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=data, chunks=(1000,), **ZSTD)
    with h5py.File(path, "r") as f:
        _mask, raw = f["a"].id.read_direct_chunk((0,))
    np.testing.assert_array_equal(np.frombuffer(numcodecs.Zstd().decode(raw), np.int64), data)
