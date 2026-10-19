"""Read a packed IVCSR ``X`` a slab of bytes at a time, instead of all of it.

:func:`vsparse.load_packed`'s ``obs_idx``/``var_idx`` decode only the slices you
ask for, but they still hold the whole packed stream while doing it (see
:func:`vsparse._ivcsc.unpack_selected`). That is the remaining half of
https://github.com/meyer-lab/vsparse/issues/67: 33 GB resident for the HEK293T
Xaira screen, 22 GB for Parse-10M, before the output is counted.

This module reads the stream in windows instead. What makes that possible is
the layout (see :mod:`vsparse._ivcsc`):

- ``major_ptr[r]:major_ptr[r + 1]`` are slice ``r``'s value groups, and groups
  are written in slice order, so every slice owns one contiguous byte run;
- every value group restarts its delta at ``-1``, so a run of whole slices
  decodes in isolation given the byte it starts at;
- so a window cut **at a slice boundary** needs only ``(next slice, next byte)``
  carried across it -- two integers, no decoder state.

It is a sequential read, not a random one: ``value_ptr`` counts elements rather
than bytes, so nothing in the file says where slice ``r`` begins and the stream
is still walked front to back. Compression is not in the way --
:func:`vsparse._compression.read_range` fetches whole zstd chunks through the
same ``chunk_table``/``pread``/``numcodecs`` path a full read uses, and never
seeks into a compressed chunk. The only overhead is re-decoding at most one
boundary chunk per window.

``major_ptr``, ``value_ptr`` and ``values`` are read whole: ~12 bytes per value
group against 4 per nonzero, which on that screen is 4.1 GB rather than the
116 GB a decoded index array needs. That is this module's floor, and it scales
with value groups, not with ``nnz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vsparse import _compression, _ivcsc
from vsparse._anndata_class import _FIELD_KEYS, VCSCAnnData, subset_fields
from vsparse._base import VCSCArray, VCSRArray

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

__all__ = ["iter_blocks", "iter_packed", "open_packed", "rows_per_window"]

IVCS_ENCODINGS = ("ivcsr", "ivcsc")

#: Elements a window aims to cover. The packed bytes are ~1.1x this and a
#: decoded block is 4-8x it, so this sets the working set of any read the caller
#: didn't size: ~18 MB of bytes, ~67 MB decoded. Larger windows buy little --
#: the per-window cost is one parallel byte scan and a few Python calls -- and
#: pay for it in residency.
_WINDOW_ELEMS = 1 << 24
#: Bytes added on top of a sized read, so rounding never costs a whole retry.
_READ_SLACK = 1 << 12
#: Headroom on a sized read, for a window whose varints run wider than the
#: file's average.
_SIZE_SLACK = 1.05


@dataclass(frozen=True, slots=True)
class PackedX:
    """An IVCSR/IVCSC group with everything but ``packed_indices`` in memory."""

    #: ``(n_rows, n_cols)``, as stored.
    shape: tuple[int, int]
    #: ``"csr"`` or ``"csc"`` -- which axis the major one is.
    format: str
    #: Width the stored minor indices decode to.
    indices_dtype: np.dtype
    major_ptr: np.ndarray
    values: np.ndarray
    value_ptr: np.ndarray
    #: ``value_ptr[major_ptr]``: the exact per-slice nnz indptr, free in
    #: ``O(n_major)``, which sizes every read and output before a byte is read.
    elem_ptr: np.ndarray
    #: The undecoded ``packed_indices`` dataset.
    packed: Any
    #: Its :func:`vsparse._compression.chunk_table`, built once for all windows.
    table: Any
    #: Packed bytes per stored nonzero over the whole file, which sizes a read:
    #: 1.0 if every delta fits one varint, up to :func:`_max_varint_bytes`. It is
    #: known for free and a window sized from it lands first try.
    bytes_per_elem: float

    @property
    def n_major(self) -> int:
        return self.major_ptr.shape[0] - 1

    @property
    def n_minor(self) -> int:
        return self.shape[1] if self.format == "csr" else self.shape[0]

    def major_shape(self, n_major: int) -> tuple[int, int]:
        """The shape of a block holding ``n_major`` major slices."""
        return (self.n_minor, n_major) if self.format == "csc" else (n_major, self.n_minor)


def open_packed(g: Any) -> PackedX:
    """Read everything but ``packed_indices`` out of an IVCSR/IVCSC group."""
    encoding = g.attrs.get("encoding-type")
    if encoding not in IVCS_ENCODINGS:
        raise ValueError(f"{g.name} is stored {encoding!r}, not one of {IVCS_ENCODINGS}")
    major_ptr = _compression.read_array(g["major_ptr"])
    value_ptr = _compression.read_array(g["value_ptr"])
    packed = g["packed_indices"]
    return PackedX(
        shape=(int(g.attrs["shape"][0]), int(g.attrs["shape"][1])),
        format="csr" if encoding == "ivcsr" else "csc",
        indices_dtype=np.dtype(g.attrs["indices_dtype"]),
        major_ptr=major_ptr,
        values=_compression.read_array(g["values"]),
        value_ptr=value_ptr,
        elem_ptr=value_ptr[major_ptr],
        packed=packed,
        table=_compression.chunk_table(packed),
        bytes_per_elem=packed.shape[0] / max(1, int(value_ptr[-1])),
    )


def _max_varint_bytes(n_minor: int) -> int:
    """Widest varint the encoder can emit, since every delta is ``< n_minor``."""
    return -(-max(1, int(n_minor - 1).bit_length()) // 7)


class _Cursor:
    """Forward-only cursor over ``packed_indices``, carrying only ``(slice, byte)``."""

    __slots__ = ("_px", "byte", "row")

    def __init__(self, px: PackedX) -> None:
        self._px = px
        self.row = 0
        self.byte = 0

    def read_rows(self, row_hi: int) -> np.ndarray:
        """Exactly the bytes of major slices ``[self.row, row_hi)``, and advance past them."""
        px = self._px
        n_elem = int(px.elem_ptr[row_hi] - px.elem_ptr[self.row])
        buf, real_starts, cum = self._read_at_least(n_elem)

        # Where slice row_hi starts. The window's own span table already says
        # how many varints precede each span, so this is a bounded walk inside
        # one span (~a window/n_spans of bytes), not another pass over the
        # window -- the counting pass is the expensive part and we have it.
        span = int(np.searchsorted(cum, n_elem, side="left"))
        if int(cum[span]) == n_elem:
            end = int(real_starts[span])
        else:
            end = int(
                _ivcsc._advance_n_varints(buf, real_starts[span - 1], n_elem - int(cum[span - 1]))
            )
        self.row = row_hi
        self.byte += end
        return buf[:end]

    def _read_at_least(self, n_elem: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A window holding at least ``n_elem`` whole varints, with its span table.

        A short read costs a re-read *and* a re-count of the window, which is the
        expensive half, so the first guess matters: it comes from the file's own
        bytes-per-nonzero rather than from assuming single-byte varints. A retry
        then rescales by what this window actually measured, which converges in
        one step instead of creeping up by the shortfall.
        """
        px = self._px
        hard = max(int(n_elem * _max_varint_bytes(px.n_minor)), 1)
        want = min(hard, max(int(n_elem * px.bytes_per_elem * _SIZE_SLACK) + _READ_SLACK, 1))
        while True:
            buf = self._read(want)
            real_starts, cum = _ivcsc._varint_chunks(buf, _ivcsc._num_chunks(buf.shape[0]))
            have = int(cum[-1])
            if have >= n_elem:
                return buf, real_starts, cum
            if want >= hard or self.byte + want >= px.packed.shape[0]:
                raise ValueError(
                    f"packed_indices holds {have} of the {n_elem} elements major slices "
                    f"starting at {self.row} should: the stream is truncated or corrupt"
                )
            scaled = int(want * _SIZE_SLACK * n_elem / max(have, 1)) + _READ_SLACK
            want = min(hard, max(scaled, want + 1))
            del buf  # don't hold the short read while taking a longer one

    def _read(self, n_bytes: int) -> np.ndarray:
        px = self._px
        raw = _compression.read_range(px.packed, self.byte, self.byte + n_bytes, table=px.table)
        # A window out of the middle of the stream usually ends mid-varint, and
        # everything downstream of here assumes a true boundary.
        return raw[: int(_ivcsc.trim_to_boundary(raw))]


def rows_per_window(px: PackedX) -> int:
    """A major-slice count whose decoded block is roughly one window of elements."""
    per_row = max(1, int(px.elem_ptr[-1]) // max(1, px.n_major))
    return max(1, _WINDOW_ELEMS // per_row)


def iter_blocks(px: PackedX, *, chunk_size: int) -> Iterator[tuple[int, int, Any]]:
    """``(row_lo, row_hi, array)`` for consecutive blocks of ``chunk_size`` major slices.

    The array layer of :func:`iter_packed`, without the AnnData fields -- what a
    whole-file reduction that only needs the data wants, since reading ``obs``
    whole can cost more than the data does.

    A block is a contiguous run of slices, so its pointer arrays are slices of
    the parent's and only the indices are decoded; nothing here needs
    :func:`~vsparse._ivcsc.unpack_selected`'s selective machinery.
    """
    cursor = _Cursor(px)
    array_type = VCSRArray if px.format == "csr" else VCSCArray
    for row_lo in range(0, px.n_major, chunk_size):
        row_hi = min(row_lo + chunk_size, px.n_major)
        buf = cursor.read_rows(row_hi)
        g0, g1 = int(px.major_ptr[row_lo]), int(px.major_ptr[row_hi])
        value_local = px.value_ptr[g0 : g1 + 1] - int(px.elem_ptr[row_lo])
        yield (
            row_lo,
            row_hi,
            array_type(
                px.major_shape(row_hi - row_lo),
                px.major_ptr[row_lo : row_hi + 1] - g0,
                px.values[g0:g1],
                value_local,
                _ivcsc.unpack_indices(value_local, buf, px.indices_dtype),
            ),
        )


def iter_packed(
    path: str | PathLike[str], *, chunk_size: int = 50_000, x_key: str = "X"
) -> Iterator[VCSCAnnData]:
    """Yield consecutive row blocks of an IVCSR-backed ``.h5ad``, one block resident at a time.

    Block ``k`` is exactly ``load_packed(path)[lo:hi]``: ``obs`` and the other
    per-cell fields are windowed to the block, ``var`` and the per-gene fields
    come through whole, and stored normalization statistics are marked stale
    (see :func:`vsparse._anndata_class.subset_fields`).

    This is what a whole-file reduction wants and ``obs_idx=`` cannot express.
    Per-group means over every cell, say, are one pass here; as selective reads
    they are one pass *per group*, each running the whole stream again, since
    nothing in the file indexes a slice's bytes.

    Peak memory is about ``chunk_size * nnz_per_row * indices.itemsize`` for the
    block plus the packed window, on top of the pointer arrays this module always
    holds (see the module docstring). Blocks deliberately carry no statistics
    sidecar: ``row_sums``/``row_nnz`` would window correctly but
    ``col_sums``/``col_nnz``/``col_depth_sums`` are whole-matrix quantities, and
    a block's are not the file's.

    Normalizing block by block is exact for a recipe whose scales are per-cell --
    ``cp10k_log1p`` is ``log1p(x * 1e4 / row_sums[i])``, so a block's own
    statistics give the whole file's answer. It is **not** exact for ``parafac2``
    or ``pearson``, whose ``median(row_sums)`` target and ``col_depth_sums`` gene
    scale are whole-population quantities; take those from the file's sidecar
    (:func:`vsparse._stats.read_stats`) instead of per block.

    Parameters
    ----------
    path
        An ``.h5ad`` whose ``X`` (or the group named ``x_key``) was stored IVCSR.
    chunk_size
        Rows per block. The last block is whatever is left.
    x_key
        Top-level group holding the IVCSR array.

    Raises
    ------
    NotImplementedError
        If the array is stored IVCSC. There the major axis is genes, so a row
        block is a *minor*-axis selection, which the packed layout can't serve
        without decoding all of it; rewrite the file with
        ``VCSCAnnData.from_anndata(..., format="csr")``.
    """
    import anndata as ad
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the zstd/Blosc2 HDF5 filters

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    with h5py.File(Path(path), "r") as f:
        px = open_packed(f[x_key])
        if px.format != "csr":
            raise NotImplementedError(
                f"{x_key} is stored IVCSC (gene-major), so a row block is a minor-axis "
                "selection and can't be read without decoding the whole array; rewrite it "
                'with VCSCAnnData.from_anndata(..., format="csr") to stream by cell'
            )
        fields = {k: ad.io.read_elem(f[k]) for k in _FIELD_KEYS if k in f}
        for row_lo, row_hi, block in iter_blocks(px, chunk_size=chunk_size):
            yield VCSCAnnData(X=block, **subset_fields(fields, slice(row_lo, row_hi), slice(None)))
