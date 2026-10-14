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

from vsparse._indexutils import smallest_index_dtype

__all__ = ["pack_indices", "unpack_indices", "unpack_selected"]

# Below this many packed bytes, decode serially -- the fixed overhead of the
# counting pass plus a second (parallel) pass isn't worth it.
_PARALLEL_MIN_BYTES = 1 << 20  # 1 MiB
# Lower bound on how small a per-thread chunk is allowed to get, so we don't
# over-split modestly-sized arrays into slivers.
_MIN_CHUNK_BYTES = 1 << 16  # 64 KiB
# Bytes :func:`_locate_ranks` counts at a time when skipping ahead.
_LOCATE_BLOCK = 256


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
def _decode_varint(buf: np.ndarray, pos: np.int64) -> tuple[np.uint64, np.int64]:
    """Decode one varint starting at ``pos``; returns ``(value, next_pos)``."""
    shift = np.uint64(0)
    result = np.uint64(0)
    while True:
        b = buf[pos]
        pos += 1
        result |= np.uint64(b & 0x7F) << shift
        if b & 0x80 == 0:
            break
        shift += np.uint64(7)
    return result, pos


@numba.njit(cache=True)
def _unpack(value_ptr: np.ndarray, buf: np.ndarray, out: np.ndarray) -> None:
    n_groups = value_ptr.shape[0] - 1
    pos = np.int64(0)
    for g in range(n_groups):
        start, end = value_ptr[g], value_ptr[g + 1]
        prev = np.int64(-1)
        for k in range(start, end):
            result, pos = _decode_varint(buf, pos)
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
    """Count varints ending in buf[start:end) -- i.e. bytes with the continuation bit clear.

    Between true boundaries that's the number of complete varints. Counted
    branch-free, one byte at a time, so the loop vectorizes.
    """
    count = np.int64(0)
    for pos in range(start, end):
        count += 1 - (buf[pos] >> 7)
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
                result, pos = _decode_varint(buf, pos)
                prev = prev + 1 + np.int64(result)
                out[kk] = prev


def _varint_chunks(buf: np.ndarray, n_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``buf`` into ``n_chunks`` spans starting on varint boundaries.

    Returns ``(real_starts, cum)``, each of length ``n_chunks + 1``: span ``t``
    covers bytes ``[real_starts[t], real_starts[t + 1])``, and its first varint
    is varint number ``cum[t]`` of the whole stream.
    """
    n_bytes = buf.shape[0]
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
    return real_starts, cum


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
    real_starts, cum = _varint_chunks(buf, n_chunks)
    assert cum[n_chunks] == int(value_ptr[-1])

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


# -- selective decode -----------------------------------------------------------
#
# Decoding only some major slices (rows of an IVCSR array) still needs every
# byte of the stream in memory, since nothing records where a slice's bytes
# start. But it only has to *count* varints up to each selected slice, not
# decode them: :func:`_varint_chunks` splits the stream on varint boundaries
# with a parallel count, each span then walks forward to the selected slices
# that start inside it (:func:`_locate_ranks`), and only those slices are
# decoded (:func:`_count_kept`/:func:`_fill_selected`), in parallel. A minor-axis
# selection is applied during that same decode by renumbering each index
# through a lookup table, dropping the ones mapped to ``-1``.


@numba.njit(cache=True, parallel=True)
def _locate_ranks(
    buf: np.ndarray,
    real_starts: np.ndarray,
    cum: np.ndarray,
    targets: np.ndarray,
    span_ptr: np.ndarray,
    out: np.ndarray,
) -> None:
    """Byte offset of varint number ``targets[i]``, for sorted ``targets``.

    Span ``t`` (from :func:`_varint_chunks`) owns targets
    ``[span_ptr[t], span_ptr[t + 1])`` and walks forward to each in turn, so
    the whole pass touches each byte at most once.
    """
    for t in numba.prange(real_starts.shape[0] - 1):  # ty: ignore[not-iterable]
        pos = real_starts[t]
        end = real_starts[t + 1]
        rank = cum[t]
        for i in range(span_ptr[t], span_ptr[t + 1]):
            need = targets[i] - rank
            # Skip whole blocks the target isn't in, counting with the
            # vectorized kernel. That can stop partway through a varint;
            # _advance_n_varints then finishes it as the first of `need`.
            while need > 0 and pos + _LOCATE_BLOCK <= end:
                ended = _count_varints_range(buf, pos, pos + _LOCATE_BLOCK)
                if ended >= need:
                    break
                need -= ended
                pos += _LOCATE_BLOCK
            pos = _advance_n_varints(buf, pos, need)
            rank = targets[i]
            out[i] = pos


@numba.njit(cache=True, parallel=True)
def _selected_groups(major_ptr, value_ptr, sel, group_offset, groups, sizes):
    """Every stored group of the selected slices, in order, and how many indices each holds."""
    for i in numba.prange(sel.shape[0]):  # ty: ignore[not-iterable]
        out_g = group_offset[i]
        for g in range(major_ptr[sel[i]], major_ptr[sel[i] + 1]):
            groups[out_g] = g
            sizes[out_g] = value_ptr[g + 1] - value_ptr[g]
            out_g += 1


@numba.njit(cache=True, parallel=True)
def _count_kept(major_ptr, value_ptr, buf, sel, starts, group_offset, remap, out_counts):
    """How many indices of each selected group ``remap`` keeps (maps to ``>= 0``)."""
    for i in numba.prange(sel.shape[0]):  # ty: ignore[not-iterable]
        pos = starts[i]
        out_g = group_offset[i]
        for g in range(major_ptr[sel[i]], major_ptr[sel[i] + 1]):
            prev = np.int64(-1)
            kept = 0
            for _ in range(value_ptr[g], value_ptr[g + 1]):
                result, pos = _decode_varint(buf, pos)
                prev = prev + 1 + np.int64(result)
                if remap[prev] >= 0:
                    kept += 1
            out_counts[out_g] = kept
            out_g += 1


@numba.njit(cache=True, parallel=True)
def _fill_selected(major_ptr, value_ptr, buf, sel, starts, group_offset, remap, out_ptr, out):
    """Decode the selected groups, writing each kept index's ``remap`` from ``out_ptr``."""
    for i in numba.prange(sel.shape[0]):  # ty: ignore[not-iterable]
        pos = starts[i]
        out_g = group_offset[i]
        for g in range(major_ptr[sel[i]], major_ptr[sel[i] + 1]):
            prev = np.int64(-1)
            k = out_ptr[out_g]
            for _ in range(value_ptr[g], value_ptr[g + 1]):
                result, pos = _decode_varint(buf, pos)
                prev = prev + 1 + np.int64(result)
                new = remap[prev]
                if new >= 0:
                    out[k] = new
                    k += 1
            out_g += 1


def _slice_starts(
    major_ptr: np.ndarray, value_ptr: np.ndarray, buf: np.ndarray, sel: np.ndarray
) -> np.ndarray:
    """Byte offset in ``buf`` where each selected major slice's indices start."""
    targets = value_ptr[major_ptr[sel]]
    n_chunks = _num_chunks(buf.shape[0])
    real_starts, cum = _varint_chunks(buf, n_chunks)
    # A target at the very end of the stream (an empty last slice) belongs to
    # the last span, which walks it up to the final byte.
    span_of = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, n_chunks - 1)
    span_ptr = np.searchsorted(span_of, np.arange(n_chunks + 1)).astype(np.int64)
    starts = np.empty(sel.shape[0], dtype=np.int64)
    _locate_ranks(buf, real_starts, cum, targets, span_ptr, starts)
    return starts


def unpack_selected(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    packed: np.ndarray,
    sel: np.ndarray,
    remap: np.ndarray | None,
    n_minor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode only major slices ``sel``, and only the minor indices ``remap`` keeps.

    Parameters
    ----------
    major_ptr, value_ptr, packed
        The stored IVCS arrays.
    sel
        Sorted, unique major indices to decode.
    remap
        ``None`` to keep every minor index, or an ``int64`` array giving each
        old minor index its new one, ``-1`` to drop it.
    n_minor
        Length of the minor axis afterwards.

    Returns
    -------
    ``(major_ptr, groups, value_ptr, indices)`` for the selection, where
    ``groups`` are the stored groups it keeps (so ``values[groups]`` are its
    values). Groups left empty by ``remap`` are dropped, as indexing an
    in-memory array would.
    """
    major_ptr = np.ascontiguousarray(major_ptr, dtype=np.int64)
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    buf = np.ascontiguousarray(packed, dtype=np.uint8)
    sel = np.ascontiguousarray(sel, dtype=np.int64)

    # Every group of the selected slices, in order: group_offset is the
    # selection's major_ptr before any group is emptied by `remap`.
    group_offset = np.zeros(sel.shape[0] + 1, dtype=np.int64)
    np.cumsum(major_ptr[sel + 1] - major_ptr[sel], out=group_offset[1:])
    groups = np.empty(int(group_offset[-1]), dtype=np.int64)
    counts = np.empty(groups.shape[0], dtype=np.int64)
    _selected_groups(major_ptr, value_ptr, sel, group_offset, groups, counts)

    starts = _slice_starts(major_ptr, value_ptr, buf, sel)
    filtering = remap is not None
    if remap is None:
        remap = np.arange(n_minor, dtype=np.int64)
    else:
        _count_kept(major_ptr, value_ptr, buf, sel, starts, group_offset, remap, counts)

    out_ptr = np.zeros(groups.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=out_ptr[1:])
    indices = np.empty(int(out_ptr[-1]), dtype=smallest_index_dtype(n_minor))
    _fill_selected(major_ptr, value_ptr, buf, sel, starts, group_offset, remap, out_ptr, indices)
    if not filtering:
        return group_offset, groups, out_ptr, indices

    nonempty = counts > 0
    nonempty_before = np.zeros(groups.shape[0] + 1, dtype=np.int64)
    np.cumsum(nonempty, out=nonempty_before[1:])
    new_value_ptr = np.append(out_ptr[:-1][nonempty], out_ptr[-1])
    return nonempty_before[group_offset], groups[nonempty], new_value_ptr, indices
