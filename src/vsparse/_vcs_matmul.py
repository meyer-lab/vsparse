"""Parallel numba kernels for ``VCSCArrayNormalized``/``VCSRArrayNormalized`` @ dense.

For row-scale ``r = 1/a``, gene-scale ``gs = 1/b``, transform ``g``, center
``c = col_mean``, and post-scale ``s = col_post_scale`` (see
:mod:`vsparse._norm_common` for the recipes these come from),

    A_norm = broadcast_rows(-s * c) + Delta,   Delta[i, j] = s[j] * g(raw[i, j] / r[i] / gs[j])

with ``Delta`` exactly zero off the structural nonzeros (since every recipe's
``g`` has ``g(0) == 0``), so ``A_norm @ B = ones(n_rows) (x) (-(s * c) @ B) +
Delta @ B`` and ``B @ A_norm = (B.sum(axis=1)) (x) (-(s * c)) + B @ Delta``,
with only the genuinely ``O(nnz)`` ``Delta @ B`` / ``B @ Delta`` needing a
kernel.

Both ``self @ B`` and ``B @ self`` need a kernel that's *major-aligned*:
parallelizing safely over the major axis requires the major axis to be
exactly the output's disjoint axis (:class:`~vsparse.VCSRArray` for
``self @ B``, :class:`~vsparse.VCSCArray` for ``B @ self`` -- see
:func:`_vcsr_matmul_delta`/:func:`_vcsc_rmatmul_delta`). The other direction
on a given array (:class:`~vsparse.VCSCArray` for ``self @ B``,
:class:`~vsparse.VCSRArray` for ``B @ self``) doesn't have that alignment:
parallelizing naively over major slices there would have different threads
scatter-add into the *same* output row/column, a data race.

That misaligned direction is handled the same way :mod:`vsparse._ops`
already handles the minor-axis reductions (``minor_sums``/``minor_counts``/
``minor_extrema``): each thread gets its own private, full-output-sized
accumulator and walks a disjoint contiguous range of major slices, scattering
into that private copy; the per-thread copies are summed once at the end
(:func:`_vcsc_matmul_delta_minor`/:func:`_vcsr_rmatmul_delta_minor`). No
second copy of the sparse array's structure is ever built, and no chunk of it
is ever regrouped into the other VCS format -- the kernel walks the array's
own ``indices`` exactly as stored, so the cost of a call is ``O(nnz * width)``
work plus one fixed-size accumulator allocation, never a sequence of
allocate/free cycles that scales with array size.

The one thing this can't avoid is that the accumulator block itself costs
``nthreads * out_major_dim * width * 8`` bytes.
:func:`~vsparse._ops.accumulator_threads` (shared with the reduction
kernels) caps ``nthreads`` to keep that block under a fixed budget -- for a
huge output axis (e.g. millions of cells) that caps down to a single thread,
trading parallelism for a hard memory bound. That is the right trade at
that scale: a small, constant footprint and a correct answer, rather than
either an unbounded thread count or a chunked-transpose fallback whose total
allocation churn (thousands of variably-sized chunk buffers, over many
power iterations and many ranks/trials in a BiCV sweep) is what previously
fragmented the allocator and drove RSS up to ~140 GB on a shared machine
(see the project issue this replaced).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from vsparse._norm_common import _g
from vsparse._ops import accumulator_threads

if TYPE_CHECKING:
    from vsparse._vcs_norm import _VCSNormalizedBase

#: The major-aligned kernels cut the major axis into this many nonzero-balanced chunks per
#: thread. More than one lets a thread that finishes early take up the remaining work only if
#: the schedule is dynamic, which Numba's is not; the extra chunks cost nothing, and they keep
#: each chunk small enough that the rounding of a cut to a slice boundary cannot unbalance it.
_CHUNKS_PER_THREAD = 4


@numba.njit(cache=True)
def _nnz_balanced_bounds(major_ptr, value_ptr, nchunks):
    """Cut the major slices into ``nchunks`` contiguous ranges of about equal nonzeros.

    Returns ``bounds`` of length ``nchunks + 1`` with ``bounds[0] == 0`` and
    ``bounds[-1] == n_major``; chunk ``c`` is slices ``bounds[c]:bounds[c + 1]``
    (possibly empty). Real count matrices have heavy-tailed slice lengths (28% of
    Lupus's nonzeros sit in 1% of its genes, and the longest gene is in every cell),
    and the long slices cluster along the axis, so an equal-count split leaves the
    thread that owns them with many times the mean work. One serial pass over the
    ``n_major + 1`` pointers; the cut falls at the first slice end that reaches each
    target, so no chunk is more than one slice over its share.
    """
    n_major = major_ptr.shape[0] - 1
    bounds = np.empty(nchunks + 1, dtype=np.int64)
    bounds[0] = 0
    base = value_ptr[major_ptr[0]]
    total = value_ptr[major_ptr[n_major]] - base
    c = 1
    for j in range(n_major):
        done = value_ptr[major_ptr[j + 1]] - base
        while c < nchunks and done * nchunks >= c * total:
            bounds[c] = j + 1
            c += 1
    while c <= nchunks:
        bounds[c] = n_major
        c += 1
    return bounds


__all__ = ["dense_at_normalized", "normalized_at_dense"]


# -- major-aligned: VCSR, self @ B (out rows == major slices) ---------------


@numba.njit(cache=True, parallel=True)
def _vcsr_matmul_delta(
    major_ptr,
    values,
    value_ptr,
    indices,
    row_scale,
    gene_scale,
    col_post_scale,
    g_code,
    hoist,
    bounds,
    B,
    out,
):
    """``Delta @ B`` for a VCSR array, parallel over nonzero-balanced chunks of rows.

    ``hoist`` is true when ``gene_scale`` is all ones (the recipe has no per-gene
    scaling): then ``g(v / rs / gs)`` is the same for every index of a value group, and
    is evaluated once per group instead of once per stored entry.
    """
    k = B.shape[1]
    for t in numba.prange(bounds.shape[0] - 1):  # ty: ignore[not-iterable]
        for i in range(bounds[t], bounds[t + 1]):
            rs = row_scale[i]
            orow = out[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                gv = _g(v / rs, g_code) if hoist else 0.0
                for kk in range(value_ptr[u], value_ptr[u + 1]):
                    col = indices[kk]
                    gs = gene_scale[col]
                    if gs > 0.0:
                        g = gv if hoist else _g(v / rs / gs, g_code)
                        delta = col_post_scale[col] * g
                        brow = B[col]
                        for c in range(k):
                            orow[c] += delta * brow[c]


# -- major-aligned: VCSC, B @ self (out cols == major slices) ---------------
#
# ``B`` arrives as (p, n_rows) and the natural output is (p, n_cols) -- but
# with ``out``/``B`` in that shape, the innermost loop over ``c`` (0..p) hits
# out[c, j] and B[c, row], both strided by the *other* array's full width
# (n_cols/n_rows elements between consecutive c), not the unit stride a
# C-contiguous inner loop needs. Every one of the up to nnz*p inner-loop
# steps is a separate cache line, so this is a per-nonzero cost, not a
# one-off. Transposing both to (n_rows, p)/(n_cols, p) first makes the
# inner loop over ``c`` walk one contiguous row of each -- the transposes
# themselves are O(p * (n_rows + n_cols)), negligible next to the O(nnz * p)
# kernel they speed up.


@numba.njit(cache=True, parallel=True)
def _vcsc_rmatmul_delta(
    major_ptr,
    values,
    value_ptr,
    indices,
    row_scale,
    gene_scale,
    col_post_scale,
    g_code,
    bounds,
    Bt,
    out_t,
):
    """``B @ Delta`` for a VCSC array, parallel over nonzero-balanced chunks of columns.

    ``g`` cannot be hoisted out of the group here: the argument ``v / row_scale[row]``
    changes with the row of every index.
    """
    p = Bt.shape[1]
    for t in numba.prange(bounds.shape[0] - 1):  # ty: ignore[not-iterable]
        for j in range(bounds[t], bounds[t + 1]):
            gs = gene_scale[j]
            if gs == 0.0:
                continue
            s = col_post_scale[j]
            acc = out_t[j]
            for u in range(major_ptr[j], major_ptr[j + 1]):
                v = values[u]
                for kk in range(value_ptr[u], value_ptr[u + 1]):
                    row = indices[kk]
                    delta = s * _g(v / row_scale[row] / gs, g_code)
                    brow = Bt[row]
                    for c in range(p):
                        acc[c] += delta * brow[c]


# -- misaligned direction: per-thread private accumulators, no regrouping ----


@numba.njit(cache=True, parallel=True)
def _vcsc_matmul_delta_minor(
    major_ptr,
    values,
    value_ptr,
    indices,
    row_scale,
    gene_scale,
    col_post_scale,
    g_code,
    bounds,
    B,
    n_rows,
):
    """``Delta @ B`` for a VCSC array (major=columns), walked directly (no VCSR regroup).

    ``B`` is ``(n_cols, k)``; returns ``(n_rows, k)``. Parallel-safe because
    each thread scatters into its own private ``(n_rows, k)`` accumulator
    while owning a disjoint, contiguous range of columns; ``bounds`` has one
    range per thread, balanced by nonzeros.
    """
    nthreads = bounds.shape[0] - 1
    k = B.shape[1]
    partial = np.zeros((nthreads, n_rows, k), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        local = partial[t]
        for j in range(bounds[t], bounds[t + 1]):
            gs = gene_scale[j]
            if gs == 0.0:
                continue
            s = col_post_scale[j]
            brow = B[j]
            for u in range(major_ptr[j], major_ptr[j + 1]):
                v = values[u]
                for kk in range(value_ptr[u], value_ptr[u + 1]):
                    row = indices[kk]
                    delta = s * _g(v / row_scale[row] / gs, g_code)
                    acc = local[row]
                    for c in range(k):
                        acc[c] += delta * brow[c]
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _vcsr_rmatmul_delta_minor(
    major_ptr,
    values,
    value_ptr,
    indices,
    row_scale,
    gene_scale,
    col_post_scale,
    g_code,
    hoist,
    bounds,
    Bt,
    n_cols,
):
    """``B @ Delta`` for a VCSR array (major=rows), walked directly (no VCSC regroup).

    ``Bt`` is ``(n_rows, p)`` (``B`` transposed); returns ``(n_cols, p)``.
    Parallel-safe because each thread scatters into its own private
    ``(n_cols, p)`` accumulator while owning a disjoint, contiguous range of
    rows; ``bounds`` has one range per thread, balanced by nonzeros. ``hoist``
    is as for :func:`_vcsr_matmul_delta`.
    """
    nthreads = bounds.shape[0] - 1
    p = Bt.shape[1]
    partial = np.zeros((nthreads, n_cols, p), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        local = partial[t]
        for i in range(bounds[t], bounds[t + 1]):
            rs = row_scale[i]
            brow = Bt[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                gv = _g(v / rs, g_code) if hoist else 0.0
                for kk in range(value_ptr[u], value_ptr[u + 1]):
                    col = indices[kk]
                    gs = gene_scale[col]
                    if gs > 0.0:
                        g = gv if hoist else _g(v / rs / gs, g_code)
                        delta = col_post_scale[col] * g
                        acc = local[col]
                        for c in range(p):
                            acc[c] += delta * brow[c]
    return partial.sum(axis=0)


# -- public entry points: dense correction + sparse delta -------------------


def _prep_dense(other: Any, expect_rows: int) -> tuple[np.ndarray, bool]:
    B = np.asarray(other, dtype=np.float64)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return np.ascontiguousarray(B), squeeze


def _bounds(arr: Any, nchunks: int) -> np.ndarray:
    """Nonzero-balanced chunk bounds over the major slices of ``arr`` (see :func:`_nnz_balanced_bounds`)."""
    return _nnz_balanced_bounds(arr.major_ptr, arr.value_ptr, nchunks)


def normalized_at_dense(nview: _VCSNormalizedBase, other: Any) -> np.ndarray:
    """``nview @ other`` -- normalized-view-on-the-left sparse-dense product."""
    arr = nview._arr
    n_cols = arr.shape[1]
    B, squeeze = _prep_dense(other, n_cols)
    out = np.zeros((arr.shape[0], B.shape[1]), dtype=np.float64)

    g_code = nview.recipe.g_code

    if arr._format == "csr":
        # self @ B is major-aligned for VCSR.
        _vcsr_matmul_delta(
            arr.major_ptr,
            arr.values,
            arr.value_ptr,
            arr.indices,
            nview.row_scale,
            nview.gene_scale,
            nview.col_post_scale,
            g_code,
            not nview.recipe.gene_scale,
            _bounds(arr, _CHUNKS_PER_THREAD * numba.get_num_threads()),
            B,
            out,
        )
    else:
        # arr is VCSC: self @ B is the misaligned direction for this format.
        nthreads = accumulator_threads(arr.shape[0], bytes_per_element=8 * B.shape[1])
        out += _vcsc_matmul_delta_minor(
            arr.major_ptr,
            arr.values,
            arr.value_ptr,
            arr.indices,
            nview.row_scale,
            nview.gene_scale,
            nview.col_post_scale,
            g_code,
            _bounds(arr, nthreads),
            B,
            arr.shape[0],
        )

    offset = nview.col_mean * nview.col_post_scale
    baseline = (-offset) @ B  # (k,): every row's implicit-zero contribution
    out += baseline[None, :]
    return out[:, 0] if squeeze else out


def dense_at_normalized(nview: _VCSNormalizedBase, other: Any) -> np.ndarray:
    """``other @ nview`` -- normalized-view-on-the-right dense-sparse product."""
    arr = nview._arr
    n_rows = arr.shape[0]
    other_arr = np.asarray(other, dtype=np.float64)
    squeeze = other_arr.ndim == 1
    B2 = other_arr.reshape(1, -1) if squeeze else other_arr
    if B2.ndim != 2 or B2.shape[1] != n_rows:
        raise ValueError(f"shape mismatch: expected last dimension {n_rows}, got {B2.shape}")
    B2 = np.ascontiguousarray(B2)
    p = B2.shape[0]
    Bt = np.ascontiguousarray(B2.T)  # (n_rows, p) -- see the note above the kernel
    out_t = np.zeros((arr.shape[1], p), dtype=np.float64)

    g_code = nview.recipe.g_code

    if arr._format == "csc":
        # B @ self is major-aligned for VCSC.
        _vcsc_rmatmul_delta(
            arr.major_ptr,
            arr.values,
            arr.value_ptr,
            arr.indices,
            nview.row_scale,
            nview.gene_scale,
            nview.col_post_scale,
            g_code,
            _bounds(arr, _CHUNKS_PER_THREAD * numba.get_num_threads()),
            Bt,
            out_t,
        )
    else:
        # arr is VCSR: B @ self is the misaligned direction for this format.
        nthreads = accumulator_threads(arr.shape[1], bytes_per_element=8 * p)
        out_t += _vcsr_rmatmul_delta_minor(
            arr.major_ptr,
            arr.values,
            arr.value_ptr,
            arr.indices,
            nview.row_scale,
            nview.gene_scale,
            nview.col_post_scale,
            g_code,
            not nview.recipe.gene_scale,
            _bounds(arr, nthreads),
            Bt,
            arr.shape[1],
        )

    out = np.ascontiguousarray(out_t.T)
    offset = nview.col_mean * nview.col_post_scale
    baseline = B2.sum(axis=1)[:, None] * (-offset)[None, :]  # (m, n_cols)
    out += baseline
    return out[0, :] if squeeze else out
