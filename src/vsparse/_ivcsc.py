"""Byte-packed delta encoding of VCS indices, for compact on-disk storage.

This backs the "IVCSC"/"IVCSR" file format (named after `IVSparse's IVCSC
<https://github.com/Seth-Wolfgang/IVSparse>`_, which inspired it): the same
VCSC/VCSR layout, but with the ``indices`` array replaced by a variable-
length-integer (LEB128), delta-encoded byte stream, which is typically much
smaller on disk than a plain int32/int64 array.

Indices within each (major-slice, value) group are unordered with respect to
the original layout -- decompression re-sorts them per major slice regardless
(see :mod:`vsparse._construct`) -- so packing is free to sort each group
ascending before delta-encoding it. That keeps every delta non-negative and
usually small, which is what makes the byte packing worth doing.

This module only implements the byte codec; :mod:`vsparse._io` uses it to decode
a packed group straight back into a plain VCSCArray/VCSRArray with an
ordinary ``indices`` array on read, so none of the compute paths in
:mod:`vsparse._ops` need to know this encoding exists.

Decoding is the expensive direction (see :func:`unpack_indices`): a serial
byte-at-a-time varint walk over the whole packed stream. For large arrays,
``unpack_indices`` instead uses a parallel decode that exploits two facts:

- LEB128 is self-synchronizing at the byte level -- a byte with the
  continuation bit clear is *always* the last byte of some varint, and the
  byte after it *always* starts the next one, no matter where in the stream
  you started reading. So a handful of arbitrary byte offsets can each be
  cheaply walked forward to a true varint boundary with no global context.
- ``value_ptr`` (which group each index belongs to) is plain, uncompressed,
  and already fully known -- so once a chunk's byte boundary is realigned to
  *some* varint boundary, snapping it onto the nearest *group* boundary is a
  short bounded scan, not a rescan from byte 0.

That turns what looks like an inherently serial stream into: a cheap fully
parallel counting pass, a tiny (thread-count-sized) serial prefix sum, a
cheap bounded per-boundary snap, then a fully parallel real decode. See
:func:`_unpack_parallel`.
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = ["pack_indices", "unpack_indices"]

# Below this many packed bytes, decode serially -- the fixed overhead of the
# counting pass plus a second (parallel) pass isn't worth it.
_PARALLEL_MIN_BYTES = 1 << 20  # 1 MiB
# Lower bound on how small a per-thread chunk is allowed to get, so we don't
# over-split modestly-sized arrays into slivers.
_MIN_CHUNK_BYTES = 1 << 16  # 64 KiB


@numba.njit(cache=True)
def _pack(value_ptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n_groups = value_ptr.shape[0] - 1
    nnz = indices.shape[0]
    # Worst case (index deltas needing the full 10-byte varint width for a
    # 64-bit value) plus one byte per empty group.
    buf = np.empty(nnz * 10 + n_groups, dtype=np.uint8)
    pos = 0
    for g in range(n_groups):
        start, end = value_ptr[g], value_ptr[g + 1]
        seg = np.sort(indices[start:end])
        prev = np.int64(-1)
        for k in range(seg.shape[0]):
            cur = np.int64(seg[k])
            v = np.uint64(cur - prev - 1)
            prev = cur
            while v >= 0x80:
                buf[pos] = np.uint8((v & 0x7F) | 0x80)
                pos += 1
                v >>= np.uint64(7)
            buf[pos] = np.uint8(v)
            pos += 1
    return buf[:pos].copy()


@numba.njit(cache=True)
def _unpack(value_ptr: np.ndarray, buf: np.ndarray, out: np.ndarray) -> None:
    n_groups = value_ptr.shape[0] - 1
    pos = 0
    for g in range(n_groups):
        start, end = value_ptr[g], value_ptr[g + 1]
        prev = np.int64(-1)
        for k in range(start, end):
            shift = np.uint64(0)
            result = np.uint64(0)
            while True:
                b = buf[pos]
                pos += 1
                result |= np.uint64(b & 0x7F) << shift
                if b & 0x80 == 0:
                    break
                shift += np.uint64(7)
            prev = prev + 1 + np.int64(result)
            out[k] = prev


def pack_indices(value_ptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Delta+varint encode ``indices``, grouped by ``value_ptr``, into ``uint8`` bytes."""
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices)
    return _pack(value_ptr, indices)


# -- parallel decode ----------------------------------------------------------
#
# See the module docstring for the overall strategy. Three passes:
#
#   1. _realign_forward / _count_all: for K guessed byte splits, walk forward
#      to a true varint boundary (self-synchronizing, no context needed), then
#      count (not decode) the varints in each resulting span. Fully parallel.
#   2. A serial prefix sum over just the K counts, then _advance_n_varints
#      snaps each boundary from "some varint" onto "the start of a value_ptr
#      group" with a short bounded scan (at most one group's worth of bytes).
#   3. _decode_chunks: the real decode, now with exact (start_group, end_group,
#      start_byte) per chunk, so each chunk is fully independent. Parallel.


@numba.njit(cache=True)
def _realign_forward(buf: np.ndarray, guess: np.int64) -> np.int64:
    """Smallest byte offset >= guess that starts a fresh varint."""
    if guess == 0:
        return np.int64(0)
    pos = guess
    n = buf.shape[0]
    while pos < n and (buf[pos - 1] & 0x80) != 0:
        pos += 1
    return pos


@numba.njit(cache=True)
def _count_varints_range(buf: np.ndarray, start: np.int64, end: np.int64) -> np.int64:
    """Count complete varints in buf[start:end); start/end must be true boundaries."""
    pos = start
    count = np.int64(0)
    while pos < end:
        while buf[pos] & 0x80:
            pos += 1
        pos += 1
        count += 1
    return count


@numba.njit(cache=True)
def _advance_n_varints(buf: np.ndarray, start_byte: np.int64, n: np.int64) -> np.int64:
    """Byte offset n complete varints after start_byte."""
    pos = start_byte
    for _ in range(n):
        while buf[pos] & 0x80:
            pos += 1
        pos += 1
    return pos


@numba.njit(cache=True, parallel=True)
def _count_all(buf: np.ndarray, real_starts: np.ndarray, out_counts: np.ndarray) -> None:
    k = real_starts.shape[0] - 1
    for t in numba.prange(k):  # ty: ignore[not-iterable]
        out_counts[t] = _count_varints_range(buf, real_starts[t], real_starts[t + 1])


@numba.njit(cache=True, parallel=True)
def _decode_chunks(
    value_ptr: np.ndarray,
    buf: np.ndarray,
    out: np.ndarray,
    chunk_group: np.ndarray,
    chunk_byte: np.ndarray,
) -> None:
    k = chunk_group.shape[0] - 1
    for t in numba.prange(k):  # ty: ignore[not-iterable]
        pos = chunk_byte[t]
        for g in range(chunk_group[t], chunk_group[t + 1]):
            start, end = value_ptr[g], value_ptr[g + 1]
            prev = np.int64(-1)
            for kk in range(start, end):
                shift = np.uint64(0)
                result = np.uint64(0)
                while True:
                    b = buf[pos]
                    pos += 1
                    result |= np.uint64(b & 0x7F) << shift
                    if b & 0x80 == 0:
                        break
                    shift += np.uint64(7)
                prev = prev + 1 + np.int64(result)
                out[kk] = prev


def _group_chunk_boundaries(
    value_ptr: np.ndarray, buf: np.ndarray, n_chunks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``buf`` into ``n_chunks`` pieces landing on exact group boundaries.

    Returns ``(chunk_group, chunk_byte)``, each of length ``n_chunks + 1``:
    chunk ``t`` owns groups ``[chunk_group[t], chunk_group[t + 1])``, whose
    packed bytes start at ``chunk_byte[t]``. Used by :func:`_unpack_parallel`
    to parallelize decoding safely.
    """
    n_bytes = buf.shape[0]
    n_groups = value_ptr.shape[0] - 1
    nnz = int(value_ptr[-1])

    guesses = np.linspace(0, n_bytes, n_chunks + 1).astype(np.int64)
    real_starts = np.empty(n_chunks + 1, dtype=np.int64)
    real_starts[0] = 0
    real_starts[n_chunks] = n_bytes
    for t in range(1, n_chunks):
        real_starts[t] = _realign_forward(buf, guesses[t])

    counts = np.empty(n_chunks, dtype=np.int64)
    _count_all(buf, real_starts, counts)
    cum = np.empty(n_chunks + 1, dtype=np.int64)
    cum[0] = 0
    cum[1:] = np.cumsum(counts)
    assert cum[n_chunks] == nnz

    chunk_group = np.empty(n_chunks + 1, dtype=np.int64)
    chunk_byte = np.empty(n_chunks + 1, dtype=np.int64)
    chunk_group[0], chunk_byte[0] = 0, 0
    chunk_group[n_chunks], chunk_byte[n_chunks] = n_groups, n_bytes
    for t in range(1, n_chunks):
        target = cum[t]
        g = int(np.searchsorted(value_ptr, target, side="right") - 1)
        if value_ptr[g] == target:
            chunk_group[t] = g
            chunk_byte[t] = real_starts[t]
        else:
            extra = np.int64(value_ptr[g + 1] - target)
            chunk_group[t] = g + 1
            chunk_byte[t] = _advance_n_varints(buf, real_starts[t], extra)

    return chunk_group, chunk_byte


def _unpack_parallel(
    value_ptr: np.ndarray, buf: np.ndarray, out: np.ndarray, n_chunks: int
) -> None:
    chunk_group, chunk_byte = _group_chunk_boundaries(value_ptr, buf, n_chunks)
    _decode_chunks(value_ptr, buf, out, chunk_group, chunk_byte)


def _num_chunks(n_bytes: int) -> int:
    if n_bytes < _PARALLEL_MIN_BYTES:
        return 1
    return max(1, min(numba.get_num_threads(), n_bytes // _MIN_CHUNK_BYTES))


def unpack_indices(value_ptr: np.ndarray, packed: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Invert :func:`pack_indices`, producing an ``indices`` array of the given dtype.

    Decodes in parallel across ``packed``-sized chunks once the packed array
    is large enough to be worth it (see :func:`_unpack_parallel`); small
    arrays fall back to a plain serial decode.
    """
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    out = np.empty(int(value_ptr[-1]), dtype=dtype)
    n_chunks = _num_chunks(packed.shape[0])
    if out.shape[0] == 0 or n_chunks <= 1:
        _unpack(value_ptr, packed, out)
    else:
        _unpack_parallel(value_ptr, packed, out, n_chunks)
    return out
