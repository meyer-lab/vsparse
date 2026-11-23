"""Grouped (per-condition) reductions on :class:`~vsparse.VCSCArray`/:class:`~vsparse.VCSRArray`.

``group_sum``/``group_nnz``/``group_mean`` collapse one axis of a raw (not
normalized) array into ``n_groups`` bins given a group label per row (or
column) -- the per-sample/per-condition sums, nonzero counts and means that
multi-condition methods such as PARAFAC2 work with.

Which kernel runs depends on whether the grouped axis is the major axis
(the value-compressed one: columns for VCSC, rows for VCSR):

* **Major axis, per-group totals** (``total=True``): one ``bincount`` over
  the per-major-slice totals, which come from the value groups alone --
  ``indices`` is never read, exactly like :meth:`sum`'s major-axis path.
* **Major axis, full** (``total=False``): each major slice is added into its
  group's output row. With at least as many groups as threads, slices are
  ordered by group (a stable ``argsort``; already-grouped data, as sorted by
  condition, then walks memory in order) and each thread owns whole groups,
  so no output is shared and no scratch memory is needed. With fewer groups
  than threads, each thread instead accumulates into a private
  ``(n_groups, n_minor)`` block, capped by the same budget as the minor-axis
  scatters in :mod:`vsparse._ops`.
* **Minor axis**: every major slice owns its own output row, so the scatter
  of ``indices`` through the group labels parallelizes over slices without
  any scratch memory. Per-group totals are a ``bincount`` over the minor
  sums.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from vsparse import _ops

if TYPE_CHECKING:
    from vsparse._base import _VCSBase

__all__ = ["group_counts", "group_reduce", "validate_groups"]


# -- kernels -------------------------------------------------------------------
#
# ``count`` selects nonzero counting (every stored element contributes 1)
# instead of summing values, so ``group_nnz`` shares these kernels without
# allocating an all-ones copy of ``values``.


@numba.njit(cache=True, inline="always")
def _add_major_slice(row, j, major_ptr, values, value_ptr, indices, count):
    """Add major slice ``j`` into the dense minor-axis vector ``row``."""
    for u in range(major_ptr[j], major_ptr[j + 1]):
        v = 1.0 if count else np.float64(values[u])
        for k in range(value_ptr[u], value_ptr[u + 1]):
            row[indices[k]] += v


@numba.njit(cache=True, parallel=True)
def _major_by_group_owned(
    major_ptr, values, value_ptr, indices, order, group_ptr, n_groups, n_minor, count
):
    """Per-group minor-axis totals; each thread owns whole groups (no shared writes)."""
    out = np.zeros((n_groups, n_minor), dtype=np.float64)
    for g in numba.prange(n_groups):  # ty: ignore[not-iterable]
        row = out[g]
        for p in range(group_ptr[g], group_ptr[g + 1]):
            _add_major_slice(row, order[p], major_ptr, values, value_ptr, indices, count)
    return out


@numba.njit(cache=True, parallel=True)
def _major_by_group_partial(
    major_ptr, values, value_ptr, indices, groups, n_groups, n_minor, nthreads, count
):
    """Per-group minor-axis totals via one private ``(n_groups, n_minor)`` block per thread."""
    n_major = major_ptr.shape[0] - 1
    span = (n_major + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_groups, n_minor), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        local = partial[t]
        for j in range(t * span, min(n_major, (t + 1) * span)):
            _add_major_slice(local[groups[j]], j, major_ptr, values, value_ptr, indices, count)
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _minor_by_group(major_ptr, values, value_ptr, indices, groups, n_groups, count):
    """Per-major-slice totals of each minor-axis group; each slice owns its output row."""
    n_major = major_ptr.shape[0] - 1
    out = np.zeros((n_major, n_groups), dtype=np.float64)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        row = out[j]
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = 1.0 if count else np.float64(values[u])
            for k in range(value_ptr[u], value_ptr[u + 1]):
                row[groups[indices[k]]] += v
    return out


# -- dispatch ------------------------------------------------------------------


def validate_groups(
    arr: _VCSBase, group_idxs: Any, n_groups: Any, axis: Any
) -> tuple[np.ndarray, int, int]:
    """Check and normalize ``group_sum``-style arguments.

    Returns ``(groups, n_groups, axis)`` with ``groups`` as an ``int64`` array.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")
    n_groups = int(n_groups)
    if n_groups < 0:
        raise ValueError(f"n_groups must be non-negative, got {n_groups}")
    groups = np.asarray(group_idxs)
    if groups.ndim != 1 or groups.shape[0] != arr.shape[axis]:
        raise ValueError(
            f"group_idxs must be 1-D with length {arr.shape[axis]} (shape[{axis}]), "
            f"got shape {groups.shape}"
        )
    if groups.size and groups.dtype.kind not in "iu":
        raise TypeError(f"group_idxs must be integers, got dtype {groups.dtype}")
    groups = groups.astype(np.int64, copy=False)
    if not _ops.indices_in_bounds(groups, n_groups):
        raise ValueError(f"group_idxs must lie in [0, n_groups) = [0, {n_groups})")
    return groups, n_groups, axis


def group_counts(groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of rows (or columns) labeled with each group."""
    return np.bincount(groups, minlength=n_groups).astype(np.int64)


def _major_full(arr: _VCSBase, groups: np.ndarray, n_groups: int, count: bool) -> np.ndarray:
    """``(n_groups, n_minor)`` per-group totals for groups along the major axis."""
    n_minor = arr.n_minor
    if n_groups == 0 or n_minor == 0:
        return np.zeros((n_groups, n_minor), dtype=np.float64)
    args = (arr.major_ptr, arr.values, arr.value_ptr, arr.indices)
    if n_groups >= numba.get_num_threads():
        order = np.argsort(groups, kind="stable")
        group_ptr = np.zeros(n_groups + 1, dtype=np.int64)
        np.cumsum(group_counts(groups, n_groups), out=group_ptr[1:])
        return _major_by_group_owned(*args, order, group_ptr, n_groups, n_minor, count)
    nthreads = _ops.accumulator_threads(n_groups * n_minor)
    return _major_by_group_partial(*args, groups, n_groups, n_minor, nthreads, count)


def _major_totals(arr: _VCSBase, groups: np.ndarray, n_groups: int, count: bool) -> np.ndarray:
    """``(n_groups,)`` per-group totals for groups along the major axis -- skips ``indices``."""
    per_major = arr._major_nnz() if count else arr._major_sums()
    return np.bincount(groups, weights=per_major, minlength=n_groups)


def _minor_totals(arr: _VCSBase, groups: np.ndarray, n_groups: int, count: bool) -> np.ndarray:
    """``(n_groups,)`` per-group totals for groups along the minor axis."""
    per_minor = arr._minor_nnz() if count else arr._minor_sums()
    return np.bincount(groups, weights=per_minor, minlength=n_groups)


def group_reduce(
    arr: _VCSBase, groups: np.ndarray, n_groups: int, axis: int, *, total: bool, count: bool
) -> np.ndarray:
    """Grouped sum (``count=False``) or stored-element count (``count=True``) of ``arr``.

    ``groups``/``n_groups``/``axis`` must already have passed
    :func:`validate_groups`. Returns ``float64``; callers cast counts.
    """
    major_axis = 0 if arr._format == "csr" else 1
    on_major = axis == major_axis
    if total:
        reducer = _major_totals if on_major else _minor_totals
        return reducer(arr, groups, n_groups, count)
    if on_major:
        out = _major_full(arr, groups, n_groups, count)  # (n_groups, n_minor)
    else:
        out = _minor_by_group(
            arr.major_ptr, arr.values, arr.value_ptr, arr.indices, groups, n_groups, count
        )  # (n_major, n_groups)
    # Both kernels are laid out along (major, minor); flip to (rows, cols) for VCSC.
    out = out if arr._format == "csr" else out.T
    return np.ascontiguousarray(out)
