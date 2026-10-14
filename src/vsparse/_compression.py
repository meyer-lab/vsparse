"""On-disk compression: zstd-1 on write, parallel chunk decoding on read.

zstd at level 1 is the default for every numeric array written. On the
IVCSR ``packed_indices`` stream -- delta+varint bytes that are mostly
single-byte values with a skewed distribution -- byte-oriented LZ codecs
find nothing to match (Blosc2+LZ4, the previous default, stored real files
at 1.00x), while zstd's entropy coding still gets ~1.15-1.5x. Higher zstd
levels don't help, since there are no long-range repeats to find. It
requires the ``hdf5plugin`` filter to be registered for HDF5; zarr ships zstd
support via ``numcodecs``.

HDF5 runs a dataset's filter pipeline one chunk at a time on a single thread,
which makes decompression, not I/O, the bottleneck on read.
:func:`read_array` sidesteps that for zstd-filtered 1-D datasets: it looks up
every chunk's file offset in one pass (``chunk_iter``), reads the compressed
chunks itself, and decodes them on a thread pool (``numcodecs.Zstd`` releases
the GIL). Anything else -- older Blosc2-compressed files, uncompressed or
contiguous datasets, in-memory/non-POSIX drivers -- falls back to an ordinary
h5py read, so those files keep loading exactly as before.

Compression is only ever applied to *numeric* arrays. anndata applies
whatever ``dataset_kwargs`` it's given uniformly to every array it writes,
including the small variable-length string arrays nested inside dataframes
and categoricals (category labels, ``obs``/``var`` string columns, the
``_index``). At least some HDF5 filter-plugin builds (seen in this
environment: h5py 3.16 / HDF5 2.0.0 / hdf5plugin's Blosc2) segfault
(``SIGFPE``) when a plugin filter is applied to a variable-length-string
dataset. :func:`numeric_only_compression` patches the relevant
``create_dataset``/``create_array`` calls for the duration of a write so
string/object arrays always land uncompressed, regardless of what
``dataset_kwargs`` was passed in -- callers don't have to know about this.

Strings are left uncompressed rather than given a different codec: gzip and
lzf are safe on variable-length strings here, but the size problem is the
per-row string itself, not its compression. Encoding low-cardinality
columns as categoricals turns the per-row data numeric, which the numeric
path then compresses -- see
:meth:`vsparse.VCSCAnnData.write_h5ad`'s ``convert_strings_to_categoricals``.
"""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import numba
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["h5_dataset_kwargs", "numeric_only_compression", "read_array", "zarr_dataset_kwargs"]

_H5_COMPRESSION_KEYS = ("compression", "compression_opts")
_ZARR_V2_COMPRESSION_KEYS = ("compressor",)
_ZARR_V3_COMPRESSION_KEYS = ("compressors", "compressor")

_ZSTD_LEVEL = 1
# Registered HDF5 filter id for zstd (https://github.com/HDFGroup/hdf5_plugins).
_H5_ZSTD_FILTER_ID = 32015
# Chunk size for large compressed 1-D arrays -- see _chunk_large_1d.
_H5_CHUNK_BYTES = 1 << 20


def h5_dataset_kwargs() -> dict[str, Any]:
    """``h5py.Group.create_dataset`` kwargs for zstd-1 compression."""
    import hdf5plugin

    return dict(hdf5plugin.Zstd(clevel=_ZSTD_LEVEL))


def _is_zarr_v2() -> bool:
    try:
        import zarr

        return getattr(zarr, "__version__", "").startswith("2.")
    except Exception:
        return False


def zarr_dataset_kwargs() -> dict[str, Any]:
    """``dataset_kwargs`` for zstd-1 compression, understood by anndata's zarr writers."""
    if _is_zarr_v2():
        import numcodecs

        return {"compressor": numcodecs.Zstd(level=_ZSTD_LEVEL)}

    import zarr.codecs

    return {"compressor": zarr.codecs.ZstdCodec(level=_ZSTD_LEVEL)}


def _is_string_like(dtype: Any, data: Any) -> bool:
    """True if a dataset write looks like a variable-length/string/object array."""
    for candidate in (dtype, getattr(data, "dtype", None)):
        if candidate is None:
            continue
        try:
            import h5py

            if h5py.check_string_dtype(candidate) is not None:
                return True
        except Exception:
            # h5py's check only understands its own/numpy dtypes -- anything
            # else (e.g. zarr's own dtype objects) falls through to the
            # numpy-dtype and class-name checks below.
            pass
        try:
            if np.dtype(candidate).kind in ("O", "U", "S"):
                return True
        except TypeError:
            # Not numpy-dtype-like, e.g. zarr v3's VariableLengthUTF8() dtype object.
            if any(tag in type(candidate).__name__ for tag in ("UTF8", "String")):
                return True
    return False


def _chunk_large_1d(kwargs: dict[str, Any], data: Any) -> None:
    """Give a compressed 1-D array larger than one chunk ~1 MiB chunks, unless chunks were chosen.

    h5py's automatic chunking picks ~64-330 KiB chunks for the VCS arrays,
    and :func:`read_array` pays a fixed, GIL-bound cost per chunk -- so
    fewer, larger chunks read several times faster, at the same zstd ratio.
    """
    if "compression" not in kwargs or kwargs.get("chunks") is not None:
        return
    shape, dtype = getattr(data, "shape", None), getattr(data, "dtype", None)
    if shape is None or dtype is None or len(shape) != 1:
        return
    per_chunk = max(1, _H5_CHUNK_BYTES // np.dtype(dtype).itemsize)
    if shape[0] > per_chunk:
        kwargs["chunks"] = (per_chunk,)


@contextlib.contextmanager
def _numeric_only_h5() -> Iterator[None]:
    import h5py

    original = h5py.Group.create_dataset

    def patched(self: h5py.Group, name: str, *args: Any, **kwargs: Any) -> h5py.Dataset:
        data = kwargs.get("data", args[0] if args else None)
        if _is_string_like(kwargs.get("dtype"), data):
            for key in _H5_COMPRESSION_KEYS:
                kwargs.pop(key, None)
        else:
            _chunk_large_1d(kwargs, data)
        return original(self, name, *args, **kwargs)

    h5py.Group.create_dataset = patched
    try:
        yield
    finally:
        h5py.Group.create_dataset = original


@contextlib.contextmanager
def _numeric_only_zarr() -> Iterator[None]:
    import zarr

    if _is_zarr_v2():
        zarr_group = cast(Any, zarr.Group)
        original_ds = getattr(zarr_group, "create_dataset", None)
        if original_ds is not None:
            keys = _ZARR_V2_COMPRESSION_KEYS

            def patched_ds(self: Any, name: str, **kwargs: Any) -> Any:
                if _is_string_like(kwargs.get("dtype"), kwargs.get("data")):
                    for key in keys:
                        kwargs[key] = None
                return original_ds(self, name, **kwargs)

            zarr_group.create_dataset = patched_ds
            try:
                yield
            finally:
                zarr_group.create_dataset = original_ds
            return

    original = zarr.Group.create_array
    keys = _ZARR_V3_COMPRESSION_KEYS

    def patched(self: zarr.Group, name: str, **kwargs: Any) -> Any:
        if _is_string_like(kwargs.get("dtype"), kwargs.get("data")):
            for key in keys:
                if key in kwargs:
                    kwargs[key] = None if key == "compressor" else ()
        return original(self, name, **kwargs)

    zarr.Group.create_array = patched
    try:
        yield
    finally:
        zarr.Group.create_array = original


def numeric_only_compression(store_kind: str) -> contextlib.AbstractContextManager[None]:
    """Scoped patch making compression apply to numeric arrays only.

    ``store_kind`` is ``"h5"`` or ``"zarr"``. See the module docstring for why
    this is needed. For HDF5 it also gives large compressed 1-D arrays ~1 MiB
    chunks (see :func:`_chunk_large_1d`). Use as
    ``with numeric_only_compression("h5"): ...`` around the write.
    """
    if store_kind == "h5":
        return _numeric_only_h5()
    if store_kind == "zarr":
        return _numeric_only_zarr()
    raise ValueError(f"store_kind must be 'h5' or 'zarr', got {store_kind!r}")


# -- parallel zstd read ---------------------------------------------------------


def _is_parallel_zstd(ds: Any) -> bool:
    """True if ``ds`` is a chunked, zstd-only, 1-D dataset in a plain read-only HDF5 file."""
    import h5py

    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or ds.chunks is None:
        return False
    # The chunks are read with os.pread on the file itself, which needs a
    # POSIX file on disk that nothing is still writing to.
    if not hasattr(os, "pread") or ds.file.mode != "r" or ds.file.driver != "sec2":
        return False
    plist = ds.id.get_create_plist()
    return plist.get_nfilters() == 1 and plist.get_filter(0)[0] == _H5_ZSTD_FILTER_ID


def _chunk_table(ds: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-chunk ``(file offset, stored size, filter mask)``, in order.

    ``None`` if any chunk was never written (HDF5 reads those as the fill
    value, which only the ordinary read path reproduces) or this h5py/HDF5
    build lacks ``chunk_iter``.
    """
    chunk_len = ds.chunks[0]
    n_chunks = -(-ds.shape[0] // chunk_len)
    offsets = np.full(n_chunks, -1, dtype=np.int64)
    sizes = np.zeros(n_chunks, dtype=np.int64)
    masks = np.zeros(n_chunks, dtype=np.int64)

    def record(info: Any) -> None:
        i = info.chunk_offset[0] // chunk_len
        offsets[i], sizes[i], masks[i] = info.byte_offset, info.size, info.filter_mask

    try:
        ds.id.chunk_iter(record)
    except (AttributeError, NotImplementedError):
        return None
    return None if (offsets < 0).any() else (offsets, sizes, masks)


def _read_chunks_parallel(
    ds: Any, offsets: np.ndarray, sizes: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    import numcodecs

    out = np.empty(ds.shape, dtype=ds.dtype)
    flat = out.view(np.uint8)
    chunk_bytes = ds.chunks[0] * ds.dtype.itemsize
    codec = numcodecs.Zstd()
    fd = os.open(ds.file.filename, os.O_RDONLY)

    def load(i: int) -> None:
        raw = os.pread(fd, int(sizes[i]), int(offsets[i]))
        dst = flat[i * chunk_bytes : (i + 1) * chunk_bytes]
        if masks[i] & 1:  # filter-mask bit 0: HDF5 skipped zstd, stored the chunk as-is
            dst[:] = np.frombuffer(raw, dtype=np.uint8, count=dst.size)
        elif dst.size == chunk_bytes:
            codec.decode(raw, out=dst)
        else:  # the last chunk decodes to a full chunk; only its leading part is data
            dst[:] = np.frombuffer(codec.decode(raw), dtype=np.uint8, count=dst.size)

    # One contiguous run of chunks per thread, rather than one task per
    # chunk, keeps the per-chunk (GIL-holding) Python overhead minimal.
    n_workers = max(1, min(numba.get_num_threads(), offsets.shape[0]))
    bounds = np.linspace(0, offsets.shape[0], n_workers + 1).astype(np.int64)

    def load_run(t: int) -> None:
        for i in range(bounds[t], bounds[t + 1]):
            load(i)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(load_run, range(n_workers)))
    finally:
        os.close(fd)
    return out


def read_array(ds: Any) -> np.ndarray:
    """Read a whole dataset into memory, decoding zstd chunks in parallel where possible.

    ``ds`` is an h5py dataset or zarr array. A zstd-compressed, chunked, 1-D
    h5py dataset (what :func:`h5_dataset_kwargs` writes for every VCS array)
    is decoded chunk-by-chunk on a thread pool sized by
    ``numba.get_num_threads()``; everything else, including Blosc2-compressed
    files written by earlier versions, is read the ordinary way. See the
    module docstring.
    """
    table = _chunk_table(ds) if _is_parallel_zstd(ds) else None
    if table is None:
        import hdf5plugin  # noqa: F401  -- registers the zstd/Blosc2 HDF5 filters

        return np.asarray(ds[...])
    return _read_chunks_parallel(ds, *table)
