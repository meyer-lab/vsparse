"""Numba-accelerated construction and decompression of the VCS layout.

The value-compressed sparse (VCS) layout groups the nonzero entries of each
major-axis slice (columns for VCSC, rows for VCSR) by their *value* rather
than storing one value per nonzero. For data with heavy value repetition
(e.g. integer count matrices) this makes the value array much smaller than
the number of nonzeros.

Arrays (major axis of size ``n_major``, ``nnz`` total stored nonzeros,
``n_unique`` total unique (major, value) groups):

``major_ptr``
    ``int64[n_major + 1]``. ``major_ptr[i]:major_ptr[i + 1]`` indexes into
    ``values``/``value_ptr`` for the unique values of major-slice ``i``.
``values``
    ``dtype[n_unique]``. The unique nonzero values, grouped by major slice.
``value_ptr``
    ``int64[n_unique + 1]``. ``value_ptr[k]:value_ptr[k + 1]`` indexes into
    ``indices`` for the minor indices sharing ``values[k]``.
``indices``
    ``int32/int64[nnz]``. Minor-axis indices (rows for VCSC, columns for
    VCSR), grouped by the unique value they belong to.
"""

from __future__ import annotations

import numba
import numpy as np

from vsparse._indexutils import smallest_index_dtype

__all__ = ["compress", "decompress", "transpose_major"]


@numba.njit(cache=True)
def _compress(major_ptr_in, minor_indices, data, n_major):
    nnz = minor_indices.shape[0]
    out_indices = np.empty(nnz, dtype=minor_indices.dtype)
    out_values = np.empty(nnz, dtype=data.dtype)
    out_value_ptr = np.empty(nnz + 1, dtype=np.int64)
    out_major_ptr = np.empty(n_major + 1, dtype=np.int64)

    value_count = 0
    pos = 0
    out_major_ptr[0] = 0

    for j in range(n_major):
        start, end = major_ptr_in[j], major_ptr_in[j + 1]
        seg_len = end - start
        if seg_len == 0:
            out_major_ptr[j + 1] = value_count
            continue

        seg_vals = data[start:end]
        seg_minor = minor_indices[start:end]
        order = np.argsort(seg_vals)
        sorted_vals = seg_vals[order]
        sorted_minor = seg_minor[order]

        out_indices[pos : pos + seg_len] = sorted_minor

        out_values[value_count] = sorted_vals[0]
        out_value_ptr[value_count] = pos
        for k in range(1, seg_len):
            if sorted_vals[k] != sorted_vals[k - 1]:
                value_count += 1
                out_values[value_count] = sorted_vals[k]
                out_value_ptr[value_count] = pos + k
        value_count += 1
        pos += seg_len
        out_major_ptr[j + 1] = value_count

    out_value_ptr[value_count] = pos
    return (
        out_major_ptr,
        out_values[:value_count].copy(),
        out_value_ptr[: value_count + 1].copy(),
        out_indices,
    )


@numba.njit(cache=True)
def _decompress(major_ptr, values, value_ptr, indices, n_major, nnz):
    out_indices = np.empty(nnz, dtype=indices.dtype)
    out_data = np.empty(nnz, dtype=values.dtype)
    out_major_ptr = np.empty(n_major + 1, dtype=np.int64)

    pos = 0
    out_major_ptr[0] = 0
    for j in range(n_major):
        u_start, u_end = major_ptr[j], major_ptr[j + 1]
        col_start = pos
        for u in range(u_start, u_end):
            v = values[u]
            i_start, i_end = value_ptr[u], value_ptr[u + 1]
            cnt = i_end - i_start
            out_indices[pos : pos + cnt] = indices[i_start:i_end]
            out_data[pos : pos + cnt] = v
            pos += cnt
        order = np.argsort(out_indices[col_start:pos])
        out_indices[col_start:pos] = out_indices[col_start:pos][order]
        out_data[col_start:pos] = out_data[col_start:pos][order]
        out_major_ptr[j + 1] = pos

    return out_major_ptr, out_indices, out_data


def compress(
    major_ptr: np.ndarray,
    minor_indices: np.ndarray,
    data: np.ndarray,
    n_major: int,
    n_minor: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the VCS layout from a standard compressed-sparse layout.

    Parameters mirror scipy's ``indptr``/``indices``/``data`` for either a
    CSC (major axis = columns) or CSR (major axis = rows) matrix.

    ``n_minor`` picks the stored ``indices`` dtype; left at ``None`` the
    input's dtype is preserved.
    """
    major_ptr = np.ascontiguousarray(major_ptr, dtype=np.int64)
    idx_dtype = None if n_minor is None else smallest_index_dtype(n_minor)
    if idx_dtype is not None and idx_dtype.itemsize > minor_indices.dtype.itemsize:
        idx_dtype = None  # never widen a caller's already-narrower indices
    minor_indices = np.ascontiguousarray(minor_indices, dtype=idx_dtype)
    data = np.ascontiguousarray(data)
    return _compress(major_ptr, minor_indices, data, n_major)


def decompress(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    indices: np.ndarray,
    n_major: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert :func:`compress`, producing standard ``indptr``/``indices``/``data``."""
    nnz = indices.shape[0]
    return _decompress(major_ptr, values, value_ptr, indices, n_major, nnz)


def transpose_major(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    indices: np.ndarray,
    n_minor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-group the VCS layout by the *other* axis (VCSC indices <-> VCSR indices).

    This is a storage-layout conversion, not a mathematical transpose: shape
    is unchanged, only which axis is deduplicated-by-value changes. Every
    stored ``(old_major, value, old_minor)`` triple becomes a
    ``(new_major=old_minor, value, new_minor=old_major)`` triple in the
    output.

    Unlike going through :func:`decompress` (expand to plain nonzeros,
    sorting *within each major slice* to restore minor-index order) followed
    by re-:func:`compress` (sorting *within each major slice* again to
    re-dedupe), this does the whole regrouping as a single global
    ``np.lexsort`` by ``(new_major, value)`` -- one vectorized C sort over
    all ``nnz`` entries, rather than one Python-level call into a per-slice
    sort for every one of the (potentially many thousands of) major slices
    in each direction. That per-slice-call overhead, not the sorting itself,
    is what makes the decompress-then-recompress route slow at scale.
    """
    n_major = major_ptr.shape[0] - 1
    nnz = indices.shape[0]

    if nnz == 0:
        return (
            np.zeros(n_minor + 1, dtype=np.int64),
            values[:0].copy(),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=indices.dtype),
        )

    group_sizes = np.diff(value_ptr)  # nnz-count per unique (major, value) group
    value_of_entry = np.repeat(values, group_sizes)
    major_of_group = np.repeat(np.arange(n_major, dtype=np.int64), np.diff(major_ptr))
    old_major_of_entry = np.repeat(major_of_group, group_sizes)  # -> new minor index
    new_major_of_entry = np.asarray(indices)  # old minor index IS the new major

    order = np.lexsort((value_of_entry, new_major_of_entry))
    sorted_major = new_major_of_entry[order]
    sorted_value = value_of_entry[order]
    sorted_minor = old_major_of_entry[order]

    change = np.empty(nnz, dtype=bool)
    change[0] = True
    change[1:] = (sorted_major[1:] != sorted_major[:-1]) | (sorted_value[1:] != sorted_value[:-1])
    group_starts = np.flatnonzero(change)

    values_out = sorted_value[group_starts]
    value_ptr_out = np.concatenate([group_starts, [nnz]]).astype(np.int64)
    major_ptr_out = np.searchsorted(sorted_major[group_starts], np.arange(n_minor + 1)).astype(
        np.int64
    )

    # The output's minor axis is the input's major axis, so that's the bound.
    indices_out = sorted_minor.astype(smallest_index_dtype(n_major), copy=False)

    return major_ptr_out, values_out, value_ptr_out, indices_out
