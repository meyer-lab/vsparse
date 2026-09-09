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
:class:`~vsparse.VCSRArray` for ``B @ self``) doesn't have that alignment --
rather than run a scatter kernel there (thread-local output-shaped
accumulators, reduced across threads: cache-unfriendly, and memory-hungry
enough for wide ``B`` to need throttling), the storage is regrouped into the
other VCS format via :meth:`~vsparse._base._VCSBase._transpose_major`, a
chunk of major slices at a time so the extra memory is one chunk's worth
rather than a second copy of the array.

``row_scale``/``gene_scale``/``col_mean``/``col_post_scale`` are per-row/
per-column statistics, so a chunk just takes the slice of them its own axis
covers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from vsparse._norm_common import _g

if TYPE_CHECKING:
    from vsparse._vcs_norm import _VCSNormalizedBase

__all__ = ["dense_at_normalized", "normalized_at_dense"]


# -- major-aligned: VCSR, self @ B (out rows == major slices) ---------------


@numba.njit(cache=True, parallel=True)
def _vcsr_matmul_delta(
    major_ptr, values, value_ptr, indices, row_scale, gene_scale, col_post_scale, g_code, B, out
):
    n_major = major_ptr.shape[0] - 1
    k = B.shape[1]
    for i in numba.prange(n_major):  # ty: ignore[not-iterable]
        rs = row_scale[i]
        for u in range(major_ptr[i], major_ptr[i + 1]):
            v = values[u]
            for kk in range(value_ptr[u], value_ptr[u + 1]):
                col = indices[kk]
                gs = gene_scale[col]
                if gs > 0.0:
                    delta = col_post_scale[col] * _g(v / rs / gs, g_code)
                    for c in range(k):
                        out[i, c] += delta * B[col, c]


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
    major_ptr, values, value_ptr, indices, row_scale, gene_scale, col_post_scale, g_code, Bt, out_t
):
    n_major = major_ptr.shape[0] - 1
    p = Bt.shape[1]
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
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


# ``Delta @ B`` splits over the contracted axis and ``B @ Delta`` over rows,
# so a contiguous range of major slices can be regrouped on its own,
# accumulated into the shared output, and dropped.

_CHUNK_BUDGET_BYTES = 128 << 20  # 128 MiB of transient regrouping per chunk

# transpose_major sorts globally over the chunk's nonzeros, so its peak is
# several nnz-sized temporaries. Deliberately generous, since underestimating
# means the budget doesn't hold.
_TRANSPOSE_BYTES_PER_NNZ = 64


def _chunk_bounds(arr, budget_bytes: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` major-slice ranges, each within the byte budget."""
    n_major = arr.n_major
    if n_major == 0:
        return []
    max_nnz = max(1, budget_bytes // _TRANSPOSE_BYTES_PER_NNZ)
    if arr.nnz <= max_nnz:
        return [(0, n_major)]

    # nnz of major slices [0, j) -- value_ptr indexed by the group boundary.
    cumulative = arr.value_ptr[arr.major_ptr]
    bounds = []
    start = 0
    while start < n_major:
        # Furthest stop whose chunk stays under budget; always advance by >= 1.
        stop = int(np.searchsorted(cumulative, cumulative[start] + max_nnz, side="right")) - 1
        stop = min(max(stop, start + 1), n_major)
        bounds.append((start, stop))
        start = stop
    return bounds


def _aligned_source(nview: _VCSNormalizedBase, needed_format: str):
    """``(array, None)`` to run one major-aligned pass, or ``(None, chunk bounds)``."""
    arr = nview._arr
    if arr._format == needed_format:
        return arr, None
    if nview._dual_arr is not None:
        return nview._dual_arr, None
    bounds = _chunk_bounds(arr, _CHUNK_BUDGET_BYTES)
    if len(bounds) <= 1:
        # Regrouping the whole array already fits the per-call budget, so
        # keeping it costs no extra peak memory and saves every later call.
        return _build_dual(nview), None
    return None, bounds


def _build_dual(nview: _VCSNormalizedBase):
    """Build and cache the opposite-format copy of ``nview``'s array, once."""
    if nview._dual_arr is None:
        nview._dual_arr = nview._arr._transpose_major()
    return nview._dual_arr


# -- public entry points: dense correction + sparse delta -------------------


def _prep_dense(other: Any, expect_rows: int) -> tuple[np.ndarray, bool]:
    B = np.asarray(other, dtype=np.float64)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return np.ascontiguousarray(B), squeeze


def normalized_at_dense(nview: _VCSNormalizedBase, other: Any) -> np.ndarray:
    """``nview @ other`` -- normalized-view-on-the-left sparse-dense product."""
    arr = nview._arr
    n_cols = arr.shape[1]
    B, squeeze = _prep_dense(other, n_cols)
    out = np.zeros((arr.shape[0], B.shape[1]), dtype=np.float64)

    g_code = nview.recipe.g_code

    # self @ B is major-aligned for VCSR; a VCSC array needs regrouping.
    src, bounds = _aligned_source(nview, "csr")
    if src is not None:
        _vcsr_matmul_delta(
            src.major_ptr,
            src.values,
            src.value_ptr,
            src.indices,
            nview.row_scale,
            nview.gene_scale,
            nview.col_post_scale,
            g_code,
            B,
            out,
        )
    else:
        # Column chunk at a time: Delta @ B == sum over chunks of
        # Delta[:, chunk] @ B[chunk, :], so each chunk's contribution
        # accumulates into the same output and is then discarded.
        for start, stop in bounds:
            chunk = arr._major_range(start, stop)._transpose_major()
            _vcsr_matmul_delta(
                chunk.major_ptr,
                chunk.values,
                chunk.value_ptr,
                chunk.indices,
                nview.row_scale,
                nview.gene_scale[start:stop],
                nview.col_post_scale[start:stop],
                g_code,
                np.ascontiguousarray(B[start:stop]),
                out,
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

    # B @ self is major-aligned for VCSC; a VCSR array needs regrouping.
    src, bounds = _aligned_source(nview, "csc")
    if src is not None:
        _vcsc_rmatmul_delta(
            src.major_ptr,
            src.values,
            src.value_ptr,
            src.indices,
            nview.row_scale,
            nview.gene_scale,
            nview.col_post_scale,
            g_code,
            Bt,
            out_t,
        )
    else:
        # Row chunk at a time, accumulating into the same output.
        for start, stop in bounds:
            chunk = arr._major_range(start, stop)._transpose_major()
            _vcsc_rmatmul_delta(
                chunk.major_ptr,
                chunk.values,
                chunk.value_ptr,
                chunk.indices,
                nview.row_scale[start:stop],
                nview.gene_scale,
                nview.col_post_scale,
                g_code,
                np.ascontiguousarray(Bt[start:stop]),
                out_t,
            )

    out = np.ascontiguousarray(out_t.T)
    offset = nview.col_mean * nview.col_post_scale
    baseline = B2.sum(axis=1)[:, None] * (-offset)[None, :]  # (m, n_cols)
    out += baseline
    return out[0, :] if squeeze else out
