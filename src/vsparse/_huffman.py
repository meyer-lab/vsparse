"""Static canonical Huffman codec for IVCSC/IVCSR index gaps (format v2).

Replaces the v1 ``LEB128 + zstd-1`` encoding of ``packed_indices`` with a
per-file static canonical Huffman code on index gaps (256 symbols: ``0..254``
directly, ``255`` as an escape followed by LEB128 of ``gap - 255``, max code
length 13) plus a per-major-slice ``int64`` bit-offset table.

Because every major slice has an explicit bit offset and ``packed_indices`` is
stored uncompressed on disk:

- Full decodes run in a single parallel pass directly from bits to indices
  (no zstd pass, no varint boundary realignment).
- Selective reads (:func:`read_selected_bytes` + :func:`unpack_selected`) only
  read the byte ranges covering the requested major slices from disk, instead
  of loading the entire packed stream into memory.
- GPU streaming can transfer raw Huffman bytes directly to VRAM and decode one
  thread per major slice without a GPU zstd decompressor.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numba
import numpy as np

from vsparse._indexutils import smallest_index_dtype

__all__ = [
    "ESCAPE",
    "MAX_CODE_LEN",
    "N_SYMBOLS",
    "PAD_BYTES",
    "build_tables",
    "pack_indices",
    "read_selected_bytes",
    "unpack_indices",
    "unpack_selected",
]

MAX_CODE_LEN = 13
TABLE_SIZE = 1 << MAX_CODE_LEN
N_SYMBOLS = 256
ESCAPE = 255
# Trailing zero bytes appended to every packed buffer so the 4-byte bit-buffer
# refill never reads past the end of the array.
PAD_BYTES = 8

# Merge nearby byte spans separated by at most this gap when reading a subset
# from disk, trading tiny over-reads for far fewer syscalls.
_COALESCE_GAP_BYTES = 64 << 10  # 64 KiB
_MIN_PARALLEL_MAJOR = 256


# -- canonical Huffman table construction -------------------------------------


def _package_merge_lengths(counts: np.ndarray) -> np.ndarray:
    """Optimal length-limited Huffman code lengths (<= ``MAX_CODE_LEN``) via Package-Merge."""
    lengths = np.zeros(N_SYMBOLS, dtype=np.uint8)
    active = [(int(counts[s]), [s]) for s in range(N_SYMBOLS) if counts[s] > 0]
    n_active = len(active)
    if n_active == 0:
        return lengths
    if n_active == 1:
        lengths[active[0][1][0]] = 1
        return lengths

    items = sorted(active, key=lambda x: x[0])
    prev: list[tuple[int, list[int]]] = []
    for _ in range(MAX_CODE_LEN):
        packages = [
            (prev[i][0] + prev[i + 1][0], prev[i][1] + prev[i + 1][1])
            for i in range(0, len(prev) - 1, 2)
        ]
        prev = sorted(items + packages, key=lambda x: x[0])

    for _, syms in prev[: 2 * n_active - 2]:
        for s in syms:
            lengths[s] += 1
    return lengths


def _reverse_bits(code: int, length: int) -> int:
    rev = 0
    for _ in range(length):
        rev = (rev << 1) | (code & 1)
        code >>= 1
    return rev


def build_tables(code_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LSB-first canonical encode/decode tables from ``code_lengths`` (uint8[256]).

    Returns ``(enc_codes, enc_lens, dec_table)`` where:

    - ``enc_codes`` (``uint16[256]``): bit-reversed canonical code per symbol.
    - ``enc_lens`` (``uint8[256]``): code length in bits (``0`` if unused).
    - ``dec_table`` (``uint16[8192]``): ``(length << 8) | symbol`` indexed by
      the next ``MAX_CODE_LEN`` LSB-first bits.
    """
    enc_lens = np.ascontiguousarray(code_lengths, dtype=np.uint8)
    enc_codes = np.zeros(N_SYMBOLS, dtype=np.uint16)
    dec_table = np.zeros(TABLE_SIZE, dtype=np.uint16)

    order = sorted(
        ((int(enc_lens[s]), s) for s in range(N_SYMBOLS) if enc_lens[s] > 0),
    )
    if not order:
        return enc_codes, enc_lens, dec_table

    code = 0
    prev_len = 0
    for length, sym in order:
        code <<= length - prev_len
        rev = _reverse_bits(code, length)
        enc_codes[sym] = rev
        entry = np.uint16((length << 8) | sym)
        dec_table[rev :: (1 << length)] = entry
        code += 1
        prev_len = length

    return enc_codes, enc_lens, dec_table


# -- encoding kernels ---------------------------------------------------------


@numba.njit(cache=True)
def _leb128_bits(v: np.uint64) -> np.int64:
    bits = np.int64(8)
    while v >= np.uint64(0x80):
        v >>= np.uint64(7)
        bits += 8
    return bits


@numba.njit(cache=True, parallel=True)
def _sort_and_histogram(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    indices: np.ndarray,
    chunk_bounds: np.ndarray,
    sorted_out: np.ndarray,
    hist_per_chunk: np.ndarray,
) -> None:
    """Sort each value group ascending into ``sorted_out`` and histogram its gap symbols."""
    n_chunks = chunk_bounds.shape[0] - 1
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        hist = hist_per_chunk[t]
        for i in range(chunk_bounds[t], chunk_bounds[t + 1]):
            for g in range(major_ptr[i], major_ptr[i + 1]):
                start = value_ptr[g]
                end = value_ptr[g + 1]
                if end <= start:
                    continue
                seg = np.sort(indices[start:end])
                prev = np.int64(-1)
                for k in range(seg.shape[0]):
                    cur = np.int64(seg[k])
                    sorted_out[start + k] = seg[k]
                    gap = cur - prev - 1
                    prev = cur
                    sym = gap if gap < ESCAPE else ESCAPE
                    hist[sym] += 1


@numba.njit(cache=True, parallel=True)
def _slice_bit_lengths(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    sorted_indices: np.ndarray,
    enc_lens: np.ndarray,
    chunk_bounds: np.ndarray,
    out_bits: np.ndarray,
) -> None:
    """Compute exact encoded bit length for every major slice."""
    n_chunks = chunk_bounds.shape[0] - 1
    esc_len = np.int64(enc_lens[ESCAPE])
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        for i in range(chunk_bounds[t], chunk_bounds[t + 1]):
            total = np.int64(0)
            for g in range(major_ptr[i], major_ptr[i + 1]):
                prev = np.int64(-1)
                for k in range(value_ptr[g], value_ptr[g + 1]):
                    cur = np.int64(sorted_indices[k])
                    gap = cur - prev - 1
                    prev = cur
                    if gap < ESCAPE:
                        total += np.int64(enc_lens[gap])
                    else:
                        total += esc_len + _leb128_bits(np.uint64(gap - ESCAPE))
            out_bits[i] = total


@numba.njit(cache=True)
def _flush_bytes(buf: np.ndarray, byte_pos: np.int64, bit_buf: np.uint64, nbits: np.int64):
    while nbits >= 8:
        buf[byte_pos] = np.uint8(bit_buf & np.uint64(0xFF))
        byte_pos += 1
        bit_buf >>= np.uint64(8)
        nbits -= 8
    return byte_pos, bit_buf, nbits


@numba.njit(cache=True, parallel=True)
def _encode_chunks(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    sorted_indices: np.ndarray,
    enc_codes: np.ndarray,
    enc_lens: np.ndarray,
    chunk_bounds: np.ndarray,
    chunk_byte_starts: np.ndarray,
    out_buf: np.ndarray,
) -> None:
    """Encode each chunk of major slices starting at its byte-aligned offset."""
    n_chunks = chunk_bounds.shape[0] - 1
    esc_code = np.uint64(enc_codes[ESCAPE])
    esc_len = np.int64(enc_lens[ESCAPE])
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        byte_pos = chunk_byte_starts[t]
        bit_buf = np.uint64(0)
        nbits = np.int64(0)
        for i in range(chunk_bounds[t], chunk_bounds[t + 1]):
            for g in range(major_ptr[i], major_ptr[i + 1]):
                prev = np.int64(-1)
                for k in range(value_ptr[g], value_ptr[g + 1]):
                    cur = np.int64(sorted_indices[k])
                    gap = cur - prev - 1
                    prev = cur
                    if gap < ESCAPE:
                        bit_buf |= np.uint64(enc_codes[gap]) << np.uint64(nbits)
                        nbits += np.int64(enc_lens[gap])
                        byte_pos, bit_buf, nbits = _flush_bytes(out_buf, byte_pos, bit_buf, nbits)
                    else:
                        bit_buf |= esc_code << np.uint64(nbits)
                        nbits += esc_len
                        byte_pos, bit_buf, nbits = _flush_bytes(out_buf, byte_pos, bit_buf, nbits)
                        v = np.uint64(gap - ESCAPE)
                        while v >= np.uint64(0x80):
                            b = (v & np.uint64(0x7F)) | np.uint64(0x80)
                            bit_buf |= b << np.uint64(nbits)
                            nbits += 8
                            v >>= np.uint64(7)
                            byte_pos, bit_buf, nbits = _flush_bytes(
                                out_buf, byte_pos, bit_buf, nbits
                            )
                        bit_buf |= v << np.uint64(nbits)
                        nbits += 8
                        byte_pos, bit_buf, nbits = _flush_bytes(out_buf, byte_pos, bit_buf, nbits)
        if nbits > 0:
            out_buf[byte_pos] = np.uint8(bit_buf & np.uint64(0xFF))


def _nnz_chunk_bounds(major_ptr: np.ndarray, value_ptr: np.ndarray, n_chunks: int) -> np.ndarray:
    """Split ``[0, n_major)`` into ``n_chunks`` roughly equal-nnz contiguous slice ranges."""
    n_major = major_ptr.shape[0] - 1
    if n_chunks <= 1 or n_major == 0:
        return np.array([0, n_major], dtype=np.int64)
    slice_end_nnz = value_ptr[major_ptr[1:]]
    total = int(slice_end_nnz[-1])
    if total == 0:
        return np.linspace(0, n_major, n_chunks + 1).astype(np.int64)
    targets = np.linspace(0, total, n_chunks + 1).astype(np.int64)
    bounds = np.searchsorted(slice_end_nnz, targets, side="right").astype(np.int64)
    bounds[0] = 0
    bounds[-1] = n_major
    return bounds


def _compute_aligned_bit_offsets(
    slice_bits: np.ndarray, chunk_bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Prefix-sum ``slice_bits`` into ``bit_offsets``, byte-aligning each chunk boundary."""
    n_major = slice_bits.shape[0]
    n_chunks = chunk_bounds.shape[0] - 1
    bit_offsets = np.empty(n_major + 1, dtype=np.int64)
    chunk_byte_starts = np.empty(n_chunks, dtype=np.int64)
    cur_bit = np.int64(0)
    for t in range(n_chunks):
        s, e = int(chunk_bounds[t]), int(chunk_bounds[t + 1])
        cur_bit = (cur_bit + 7) & ~np.int64(7)
        chunk_byte_starts[t] = cur_bit >> 3
        if e > s:
            bit_offsets[s] = cur_bit
            local_cum = np.cumsum(slice_bits[s:e])
            bit_offsets[s + 1 : e] = cur_bit + local_cum[:-1]
            cur_bit += int(local_cum[-1])
    bit_offsets[n_major] = (cur_bit + 7) & ~np.int64(7)
    return bit_offsets, chunk_byte_starts


def pack_indices(
    major_ptr: np.ndarray, value_ptr: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static-Huffman-encode ``indices`` gaps with per-major-slice bit offsets.

    Returns ``(packed, code_lengths, bit_offsets)``:

    - ``packed``: ``uint8`` bitstream (with ``PAD_BYTES`` trailing zero bytes).
    - ``code_lengths``: ``uint8[256]`` canonical Huffman code lengths.
    - ``bit_offsets``: ``int64[n_major + 1]`` bit start of each major slice.
    """
    major_ptr = np.ascontiguousarray(major_ptr, dtype=np.int64)
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices)
    n_major = major_ptr.shape[0] - 1
    nnz = indices.shape[0]

    if nnz == 0:
        return (
            np.empty(0, dtype=np.uint8),
            np.zeros(N_SYMBOLS, dtype=np.uint8),
            np.zeros(n_major + 1, dtype=np.int64),
        )

    n_chunks = 1 if n_major < _MIN_PARALLEL_MAJOR else max(1, min(numba.get_num_threads(), n_major))
    chunk_bounds = _nnz_chunk_bounds(major_ptr, value_ptr, n_chunks)
    sorted_indices = np.empty_like(indices)
    hist_per_chunk = np.zeros((n_chunks, N_SYMBOLS), dtype=np.int64)
    _sort_and_histogram(major_ptr, value_ptr, indices, chunk_bounds, sorted_indices, hist_per_chunk)

    code_lengths = _package_merge_lengths(hist_per_chunk.sum(axis=0))
    enc_codes, enc_lens, _ = build_tables(code_lengths)

    slice_bits = np.empty(n_major, dtype=np.int64)
    _slice_bit_lengths(major_ptr, value_ptr, sorted_indices, enc_lens, chunk_bounds, slice_bits)
    bit_offsets, chunk_byte_starts = _compute_aligned_bit_offsets(slice_bits, chunk_bounds)

    total_bytes = int(bit_offsets[-1] >> 3) + PAD_BYTES
    packed = np.zeros(total_bytes, dtype=np.uint8)
    _encode_chunks(
        major_ptr,
        value_ptr,
        sorted_indices,
        enc_codes,
        enc_lens,
        chunk_bounds,
        chunk_byte_starts,
        packed,
    )
    return packed, code_lengths, bit_offsets


# -- decoding kernels ---------------------------------------------------------


@numba.njit(cache=True)
def _refill(buf: np.ndarray, byte_pos: np.int64, bit_buf: np.uint64, nbits: np.int64):
    b0 = np.uint64(buf[byte_pos])
    b1 = np.uint64(buf[byte_pos + 1])
    b2 = np.uint64(buf[byte_pos + 2])
    b3 = np.uint64(buf[byte_pos + 3])
    bit_buf |= (
        b0 | (b1 << np.uint64(8)) | (b2 << np.uint64(16)) | (b3 << np.uint64(24))
    ) << np.uint64(nbits)
    return byte_pos + 4, bit_buf, nbits + 32


@numba.njit(cache=True)
def _decode_gap(
    buf: np.ndarray,
    byte_pos: np.int64,
    bit_buf: np.uint64,
    nbits: np.int64,
    dec_table: np.ndarray,
):
    if nbits < MAX_CODE_LEN:
        byte_pos, bit_buf, nbits = _refill(buf, byte_pos, bit_buf, nbits)
    entry = dec_table[bit_buf & np.uint64(TABLE_SIZE - 1)]
    sym = np.int64(entry & np.uint16(0xFF))
    length = np.int64(entry >> np.uint16(8))
    bit_buf >>= np.uint64(length)
    nbits -= length
    if sym < ESCAPE:
        return sym, byte_pos, bit_buf, nbits

    val = np.uint64(0)
    shift = np.uint64(0)
    while True:
        if nbits < 8:
            byte_pos, bit_buf, nbits = _refill(buf, byte_pos, bit_buf, nbits)
        b = bit_buf & np.uint64(0xFF)
        bit_buf >>= np.uint64(8)
        nbits -= 8
        val |= (b & np.uint64(0x7F)) << shift
        if (b & np.uint64(0x80)) == 0:
            break
        shift += np.uint64(7)
    return np.int64(ESCAPE) + np.int64(val), byte_pos, bit_buf, nbits


@numba.njit(cache=True)
def _init_bit_reader(buf: np.ndarray, bit_offset: np.int64):
    byte_pos = bit_offset >> 3
    skip = bit_offset & 7
    bit_buf = np.uint64(buf[byte_pos]) >> np.uint64(skip)
    return byte_pos + 1, bit_buf, np.int64(8 - skip)


@numba.njit(cache=True, parallel=True)
def _decode_chunks(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    buf: np.ndarray,
    bit_offsets: np.ndarray,
    dec_table: np.ndarray,
    chunk_bounds: np.ndarray,
    out: np.ndarray,
) -> None:
    n_chunks = chunk_bounds.shape[0] - 1
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        for i in range(chunk_bounds[t], chunk_bounds[t + 1]):
            g_start = major_ptr[i]
            g_end = major_ptr[i + 1]
            if g_start == g_end:
                continue
            byte_pos, bit_buf, nbits = _init_bit_reader(buf, bit_offsets[i])
            for g in range(g_start, g_end):
                prev = np.int64(-1)
                for k in range(value_ptr[g], value_ptr[g + 1]):
                    gap, byte_pos, bit_buf, nbits = _decode_gap(
                        buf, byte_pos, bit_buf, nbits, dec_table
                    )
                    prev = prev + 1 + gap
                    out[k] = prev


def unpack_indices(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    packed: np.ndarray,
    code_lengths: np.ndarray,
    bit_offsets: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Invert :func:`pack_indices`, returning ``indices`` of the requested dtype."""
    major_ptr = np.ascontiguousarray(major_ptr, dtype=np.int64)
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    bit_offsets = np.ascontiguousarray(bit_offsets, dtype=np.int64)
    out = np.empty(int(value_ptr[-1]), dtype=dtype)
    if out.shape[0] == 0:
        return out

    buf = np.ascontiguousarray(packed, dtype=np.uint8)
    _, _, dec_table = build_tables(code_lengths)
    n_major = major_ptr.shape[0] - 1
    n_chunks = 1 if n_major < _MIN_PARALLEL_MAJOR else max(1, min(numba.get_num_threads(), n_major))
    chunk_bounds = _nnz_chunk_bounds(major_ptr, value_ptr, n_chunks)
    _decode_chunks(major_ptr, value_ptr, buf, bit_offsets, dec_table, chunk_bounds, out)
    return out


# -- selective read & decode (#102) -------------------------------------------


def _coalesce_spans(
    bit_offsets: np.ndarray, sel: np.ndarray, packed_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coalesced byte spans covering ``sel`` and each selected slice's local bit offset.

    Returns ``(span_starts, span_ends, local_bit_offsets)`` where ``local_bit_offsets``
    has length ``len(sel)`` and indexes into the concatenation of ``packed[s:e]``
    for each ``(s, e)`` span.
    """
    n_sel = sel.shape[0]
    local_bit_offsets = np.zeros(n_sel, dtype=np.int64)
    if n_sel == 0 or packed_len == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, local_bit_offsets

    b0 = bit_offsets[sel]
    b1 = bit_offsets[sel + 1]
    nonempty = b1 > b0
    if not bool(nonempty.any()):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, local_bit_offsets

    starts = b0[nonempty] >> 3
    ends = np.minimum(((b1[nonempty] + 7) >> 3) + PAD_BYTES, packed_len)

    # Merge spans whose gap is <= _COALESCE_GAP_BYTES (inputs are sorted because sel is sorted).
    new_span = np.empty(starts.shape[0], dtype=bool)
    new_span[0] = True
    if starts.shape[0] > 1:
        # Track running max end in case PAD_BYTES causes overlaps.
        running_end = np.maximum.accumulate(ends[:-1])
        new_span[1:] = starts[1:] > running_end + _COALESCE_GAP_BYTES

    span_idx = np.flatnonzero(new_span)
    span_starts = starts[span_idx]
    span_ends = np.empty(span_idx.shape[0], dtype=np.int64)
    for m in range(span_idx.shape[0]):
        lo = int(span_idx[m])
        hi = int(span_idx[m + 1]) if m + 1 < span_idx.shape[0] else starts.shape[0]
        span_ends[m] = int(ends[lo:hi].max())

    span_lengths = span_ends - span_starts
    local_byte_base = np.zeros(span_starts.shape[0], dtype=np.int64)
    if span_starts.shape[0] > 1:
        np.cumsum(span_lengths[:-1], out=local_byte_base[1:])

    group_of = np.cumsum(new_span) - 1
    nonempty_local_bits = (b0[nonempty] - span_starts[group_of] * 8) + local_byte_base[group_of] * 8
    local_bit_offsets[nonempty] = nonempty_local_bits
    return span_starts, span_ends, local_bit_offsets


def _h5_contiguous_offset(ds: Any) -> int | None:
    """File byte offset of a plain contiguous, uncompressed, read-only POSIX HDF5 dataset."""
    try:
        import h5py
    except ImportError:
        return None
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or ds.chunks is not None:
        return None
    if not hasattr(os, "pread") or ds.file.mode != "r" or ds.file.driver != "sec2":
        return None
    if ds.id.get_create_plist().get_nfilters() != 0:
        return None
    offset = ds.id.get_offset()
    return None if offset is None else int(offset)


def _pread_spans(
    filename: str,
    file_offset: int,
    span_starts: np.ndarray,
    span_ends: np.ndarray,
    out: np.ndarray,
) -> None:
    n_spans = span_starts.shape[0]
    lengths = span_ends - span_starts
    dest_offsets = np.zeros(n_spans, dtype=np.int64)
    if n_spans > 1:
        np.cumsum(lengths[:-1], out=dest_offsets[1:])

    fd = os.open(filename, os.O_RDONLY)
    n_workers = max(1, min(numba.get_num_threads(), n_spans))
    bounds = np.linspace(0, n_spans, n_workers + 1).astype(np.int64)

    def read_run(t: int) -> None:
        for m in range(bounds[t], bounds[t + 1]):
            nbytes = int(lengths[m])
            raw = os.pread(fd, nbytes, file_offset + int(span_starts[m]))
            dst = int(dest_offsets[m])
            out[dst : dst + nbytes] = np.frombuffer(raw, dtype=np.uint8, count=nbytes)

    try:
        if n_workers == 1:
            read_run(0)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(read_run, range(n_workers)))
    finally:
        os.close(fd)


def read_selected_bytes(
    ds: Any, bit_offsets: np.ndarray, sel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Read only the byte spans of ``ds`` needed to decode major slices ``sel``.

    Returns ``(local_buf, local_bit_offsets)`` where ``local_buf`` holds the
    coalesced byte spans (with ``PAD_BYTES`` zero padding) and
    ``local_bit_offsets[i]`` is slice ``sel[i]``'s bit offset in ``local_buf``.
    """
    packed_len = int(ds.shape[0])
    span_starts, span_ends, local_bit_offsets = _coalesce_spans(bit_offsets, sel, packed_len)
    if span_starts.shape[0] == 0:
        return np.zeros(PAD_BYTES, dtype=np.uint8), local_bit_offsets

    total_bytes = int(np.sum(span_ends - span_starts))
    local_buf = np.zeros(total_bytes + PAD_BYTES, dtype=np.uint8)

    file_offset = _h5_contiguous_offset(ds)
    if file_offset is not None:
        _pread_spans(ds.file.filename, file_offset, span_starts, span_ends, local_buf)
    else:
        pos = 0
        for s, e in zip(span_starts, span_ends, strict=True):
            n = int(e - s)
            local_buf[pos : pos + n] = np.asarray(ds[int(s) : int(e)], dtype=np.uint8)
            pos += n
    return local_buf, local_bit_offsets


@numba.njit(cache=True, parallel=True)
def _selected_groups(major_ptr, value_ptr, sel, group_offset, groups, sizes):
    for i in numba.prange(sel.shape[0]):  # ty: ignore[not-iterable]
        out_g = group_offset[i]
        for g in range(major_ptr[sel[i]], major_ptr[sel[i] + 1]):
            groups[out_g] = g
            sizes[out_g] = value_ptr[g + 1] - value_ptr[g]
            out_g += 1


@numba.njit(cache=True, parallel=True)
def _count_kept(
    major_ptr, value_ptr, buf, sel, local_bit_offsets, dec_table, group_offset, remap, out_counts
):
    for i in numba.prange(sel.shape[0]):  # ty: ignore[not-iterable]
        g_start = major_ptr[sel[i]]
        g_end = major_ptr[sel[i] + 1]
        if g_start == g_end:
            continue
        byte_pos, bit_buf, nbits = _init_bit_reader(buf, local_bit_offsets[i])
        out_g = group_offset[i]
        for g in range(g_start, g_end):
            prev = np.int64(-1)
            kept = np.int64(0)
            for _ in range(value_ptr[g], value_ptr[g + 1]):
                gap, byte_pos, bit_buf, nbits = _decode_gap(
                    buf, byte_pos, bit_buf, nbits, dec_table
                )
                prev = prev + 1 + gap
                if remap[prev] >= 0:
                    kept += 1
            out_counts[out_g] = kept
            out_g += 1


@numba.njit(cache=True, parallel=True)
def _fill_selected(
    major_ptr, value_ptr, buf, sel, local_bit_offsets, dec_table, group_offset, remap, out_ptr, out
):
    for i in numba.prange(sel.shape[0]):  # ty: ignore[not-iterable]
        g_start = major_ptr[sel[i]]
        g_end = major_ptr[sel[i] + 1]
        if g_start == g_end:
            continue
        byte_pos, bit_buf, nbits = _init_bit_reader(buf, local_bit_offsets[i])
        out_g = group_offset[i]
        for g in range(g_start, g_end):
            prev = np.int64(-1)
            k = out_ptr[out_g]
            for _ in range(value_ptr[g], value_ptr[g + 1]):
                gap, byte_pos, bit_buf, nbits = _decode_gap(
                    buf, byte_pos, bit_buf, nbits, dec_table
                )
                prev = prev + 1 + gap
                new = remap[prev]
                if new >= 0:
                    out[k] = new
                    k += 1
            out_g += 1


def unpack_selected(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    local_buf: np.ndarray,
    local_bit_offsets: np.ndarray,
    code_lengths: np.ndarray,
    sel: np.ndarray,
    remap: np.ndarray | None,
    n_minor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode selected major slices ``sel`` from ``local_buf`` and apply optional ``remap``."""
    major_ptr = np.ascontiguousarray(major_ptr, dtype=np.int64)
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    buf = np.ascontiguousarray(local_buf, dtype=np.uint8)
    local_bit_offsets = np.ascontiguousarray(local_bit_offsets, dtype=np.int64)
    sel = np.ascontiguousarray(sel, dtype=np.int64)
    _, _, dec_table = build_tables(code_lengths)

    group_offset = np.zeros(sel.shape[0] + 1, dtype=np.int64)
    np.cumsum(major_ptr[sel + 1] - major_ptr[sel], out=group_offset[1:])
    groups = np.empty(int(group_offset[-1]), dtype=np.int64)
    counts = np.empty(groups.shape[0], dtype=np.int64)
    _selected_groups(major_ptr, value_ptr, sel, group_offset, groups, counts)

    filtering = remap is not None
    if remap is None:
        remap = np.arange(n_minor, dtype=np.int64)
    else:
        _count_kept(
            major_ptr,
            value_ptr,
            buf,
            sel,
            local_bit_offsets,
            dec_table,
            group_offset,
            remap,
            counts,
        )

    out_ptr = np.zeros(groups.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=out_ptr[1:])
    indices = np.empty(int(out_ptr[-1]), dtype=smallest_index_dtype(n_minor))
    _fill_selected(
        major_ptr,
        value_ptr,
        buf,
        sel,
        local_bit_offsets,
        dec_table,
        group_offset,
        remap,
        out_ptr,
        indices,
    )
    if not filtering:
        return group_offset, groups, out_ptr, indices

    nonempty = counts > 0
    nonempty_before = np.zeros(groups.shape[0] + 1, dtype=np.int64)
    np.cumsum(nonempty, out=nonempty_before[1:])
    new_value_ptr = np.append(out_ptr[:-1][nonempty], out_ptr[-1])
    return nonempty_before[group_offset], groups[nonempty], new_value_ptr, indices
