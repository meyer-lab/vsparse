"""Numba-accelerated numerical kernels on the VCS layout."""

from __future__ import annotations

import numba
import numpy as np

__all__ = [
    "accumulator_threads",
    "major_matmat",
    "major_matvec",
    "minor_counts",
    "minor_extrema",
    "minor_matmat",
    "minor_matvec",
    "minor_select_counts",
    "minor_select_fill",
    "minor_sums",
]

# Thread-local accumulators for the minor-axis scatters below cost
# ``nthreads * n_minor * bytes_per_element``. The thread count is capped to
# keep that block under this budget.
_ACCUMULATOR_BUDGET_BYTES = 64 << 20  # 64 MiB


def accumulator_threads(n_minor: int, bytes_per_element: int = 8) -> int:
    """Threads to run a minor-axis scatter with, capped by accumulator size."""
    if n_minor <= 0:
        return 1
    affordable = max(1, _ACCUMULATOR_BUDGET_BYTES // (n_minor * max(1, bytes_per_element)))
    return int(min(numba.get_num_threads(), affordable))


@numba.njit(cache=True)
def _major_matvec(major_ptr, values, value_ptr, indices, x, n_major, n_minor):
    """y = A @ x where A's major axis (columns for VCSC) has length n_major.

    x has length n_major, output y has length n_minor.
    """
    y = np.zeros(n_minor, dtype=values.dtype)
    for j in range(n_major):
        xj = x[j]
        if xj == 0:
            continue
        for u in range(major_ptr[j], major_ptr[j + 1]):
            val = values[u] * xj
            for k in range(value_ptr[u], value_ptr[u + 1]):
                y[indices[k]] += val
    return y


@numba.njit(cache=True, parallel=True)
def _minor_matvec(major_ptr, values, value_ptr, indices, x, n_major):
    """y = x @ A, i.e. contract over the minor axis. x has length n_minor.

    Output y has length n_major. Safe to parallelize over major slices since
    each output element is written by exactly one iteration.
    """
    y = np.zeros(n_major, dtype=values.dtype)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        s = 0.0
        for u in range(major_ptr[j], major_ptr[j + 1]):
            acc = 0.0
            for k in range(value_ptr[u], value_ptr[u + 1]):
                acc += x[indices[k]]
            s += values[u] * acc
        y[j] = s
    return y


@numba.njit(cache=True)
def _major_matmat(major_ptr, values, value_ptr, indices, b, n_major, n_minor):
    """Y = A @ B where A's major axis has length n_major, B is (n_major, k)."""
    k = b.shape[1]
    y = np.zeros((n_minor, k), dtype=values.dtype)
    for j in range(n_major):
        for u in range(major_ptr[j], major_ptr[j + 1]):
            val = values[u]
            for idx in range(value_ptr[u], value_ptr[u + 1]):
                row = indices[idx]
                for c in range(k):
                    y[row, c] += val * b[j, c]
    return y


@numba.njit(cache=True, parallel=True)
def _minor_matmat(major_ptr, values, value_ptr, indices, b, n_major):
    """Y = B @ A, i.e. B is (k, n_minor), contracting over the minor axis."""
    k = b.shape[0]
    y = np.zeros((k, n_major), dtype=values.dtype)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        for u in range(major_ptr[j], major_ptr[j + 1]):
            val = values[u]
            for idx in range(value_ptr[u], value_ptr[u + 1]):
                col = indices[idx]
                for c in range(k):
                    y[c, j] += val * b[c, col]
    return y


@numba.njit(cache=True, parallel=True)
def minor_sums(values, value_ptr, indices, n_minor, nthreads):
    n_groups = values.shape[0]
    chunk = (n_groups + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_minor), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_groups, start + chunk)
        local = partial[t]
        for u in range(start, end):
            v = np.float64(values[u])
            for k in range(value_ptr[u], value_ptr[u + 1]):
                local[indices[k]] += v
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _minor_extrema_kernel(values, value_ptr, indices, n_minor, nthreads, initial, is_max):
    # The count comes along for free and tells the caller which minor indices
    # have an implicit zero to fold in.
    n_groups = values.shape[0]
    chunk = (n_groups + nthreads - 1) // nthreads
    part_val = np.full((nthreads, n_minor), initial, dtype=values.dtype)
    part_cnt = np.zeros((nthreads, n_minor), dtype=np.int64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_groups, start + chunk)
        local_val = part_val[t]
        local_cnt = part_cnt[t]
        for u in range(start, end):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                idx = indices[k]
                local_cnt[idx] += 1
                if is_max:
                    if v > local_val[idx]:
                        local_val[idx] = v
                else:
                    if v < local_val[idx]:
                        local_val[idx] = v
    return part_val, part_cnt


def minor_extrema(values, value_ptr, indices, n_minor, initial, is_max):
    """Extremum over the stored values, and the stored count, per minor index."""
    nthreads = accumulator_threads(n_minor, values.dtype.itemsize + 8)
    part_val, part_cnt = _minor_extrema_kernel(
        values, value_ptr, indices, n_minor, nthreads, initial, is_max
    )
    extrema = part_val.max(axis=0) if is_max else part_val.min(axis=0)
    return extrema, part_cnt.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def minor_counts(value_ptr, indices, n_minor, nthreads):
    n_groups = value_ptr.shape[0] - 1
    chunk = (n_groups + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_minor), dtype=np.int64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_groups, start + chunk)
        local = partial[t]
        for u in range(start, end):
            for k in range(value_ptr[u], value_ptr[u + 1]):
                local[indices[k]] += 1
    return partial.sum(axis=0)


def major_matvec(major_ptr, values, value_ptr, indices, x, n_major, n_minor):
    return _major_matvec(major_ptr, values, value_ptr, indices, np.asarray(x), n_major, n_minor)


def minor_matvec(major_ptr, values, value_ptr, indices, x, n_major):
    return _minor_matvec(major_ptr, values, value_ptr, indices, np.asarray(x), n_major)


def major_matmat(major_ptr, values, value_ptr, indices, b, n_major, n_minor):
    b = np.ascontiguousarray(b)
    return _major_matmat(major_ptr, values, value_ptr, indices, b, n_major, n_minor)


def minor_matmat(major_ptr, values, value_ptr, indices, b, n_major):
    b = np.ascontiguousarray(b)
    return _minor_matmat(major_ptr, values, value_ptr, indices, b, n_major)


# -- minor-axis selection ----------------------------------------------------


@numba.njit(cache=True, parallel=True)
def _minor_select_counts(value_ptr, indices, fanout, out_counts):
    # A selection may name one index several times, so each stored element
    # contributes ``fanout`` output entries rather than one.
    n_slots = value_ptr.shape[0] - 1
    for u in numba.prange(n_slots):  # ty: ignore[not-iterable]
        c = 0
        for k in range(value_ptr[u], value_ptr[u + 1]):
            c += fanout[indices[k]]
        out_counts[u] = c


@numba.njit(cache=True, parallel=True)
def _minor_select_fill(
    value_ptr, indices, offsets, positions, kept_slots, new_value_ptr, out_indices
):
    for s in numba.prange(kept_slots.shape[0]):  # ty: ignore[not-iterable]
        u = kept_slots[s]
        pos = new_value_ptr[s]  # each slot fills a disjoint range
        for k in range(value_ptr[u], value_ptr[u + 1]):
            o = indices[k]
            for j in range(offsets[o], offsets[o + 1]):  # every destination of o
                out_indices[pos] = positions[j]
                pos += 1


def minor_select_counts(value_ptr, indices, fanout):
    """Output element count per stored (major, value) slot."""
    counts = np.empty(value_ptr.shape[0] - 1, dtype=np.int64)
    _minor_select_counts(value_ptr, indices, fanout, counts)
    return counts


def minor_select_fill(
    value_ptr, indices, offsets, positions, kept_slots, new_value_ptr, out_indices
):
    """Write the remapped minor indices for the surviving slots, in place."""
    _minor_select_fill(
        value_ptr, indices, offsets, positions, kept_slots, new_value_ptr, out_indices
    )
