"""Per-row/per-column summary statistics carried by a VCS array -- the "stats sidecar".

Every :class:`~vsparse.VCSCArray`/:class:`~vsparse.VCSRArray` can carry a
:class:`VCSStats` (see :attr:`~vsparse.VCSCArray.stats`): the per-cell and
per-gene totals QC filtering and normalization keep needing, held at
``O(n_rows + n_cols)`` instead of being re-derived from the ``O(nnz)`` data
each time (https://github.com/meyer-lab/vsparse/issues/26).

The fields are chosen so that the per-cell ``a`` and per-gene ``b`` of every
normalization recipe (``y = (g(x * a * b) - c) * s``, see
:mod:`vsparse._norm_common`) follow from them without touching the data:

- ``a = target / row_sums`` (or ``1``), where ``target`` is a constant or
  ``median(row_sums)``;
- ``b = 1 / sum_i(x[i, j] * a[i])``, which is ``1 / col_sums`` when ``a = 1``
  and ``1 / (target * col_depth_sums)`` otherwise, since ``target`` factors
  out of the sum.

``c``/``s`` are moments of ``g(x * a * b)``, a nonlinear function of the
recipe, so there's no recipe-independent summary of them; they still take
one pass over the data per recipe.

The statistics are computed lazily, on first access, and persisted alongside
the array by :meth:`~vsparse.VCSCAnnData.write_h5ad`/:meth:`~vsparse.VCSCAnnData.write_zarr`
as a ``stats`` subgroup of the array's own group (:func:`write_stats`). On read
(:func:`read_stats`) a missing, unrecognized or malformed sidecar is ignored
rather than trusted: the array then simply recomputes its statistics on first
use, so files written before the sidecar existed load exactly as before.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from vsparse import _ops
from vsparse._compression import read_array

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["VCSStats", "compute_stats", "read_stats", "write_stats"]

STATS_KEY = "stats"
STATS_ENCODING_TYPE = "vsparse-stats"
STATS_ENCODING_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True, eq=False)
class VCSStats:
    """Per-row/per-column statistics of one VCS array -- see :mod:`vsparse._stats`.

    The arrays are read-only, since a :class:`VCSStats` is shared between an
    array and its copies.
    """

    #: Per-row sum of stored values (scanpy's ``obs["total_counts"]``).
    row_sums: np.ndarray
    #: Per-row count of stored values that are nonzero (``obs["n_genes_by_counts"]``).
    row_nnz: np.ndarray
    #: Per-column sum of stored values (``var["total_counts"]``).
    col_sums: np.ndarray
    #: Per-column count of stored values that are nonzero (``var["n_cells_by_counts"]``).
    col_nnz: np.ndarray
    #: Per-column sum of depth-normalized values, ``sum_i(x[i, j] / row_sums[i])``
    #: (rows summing to zero divide by ``1`` instead).
    col_depth_sums: np.ndarray

    def __post_init__(self) -> None:
        for f in fields(self):
            dtype = np.int64 if f.name.endswith("_nnz") else np.float64
            arr = np.array(getattr(self, f.name), dtype=dtype)
            arr.flags.writeable = False
            object.__setattr__(self, f.name, arr)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.row_sums.shape[0]), int(self.col_sums.shape[0]))

    def _is_consistent(self) -> bool:
        n_rows, n_cols = self.shape
        return (
            self.row_nnz.shape == (n_rows,)
            and self.col_nnz.shape == (n_cols,)
            and self.col_depth_sums.shape == (n_cols,)
        )


def depth_divisors(row_sums: np.ndarray) -> np.ndarray:
    """``row_sums``, with rows summing to zero mapped to ``1`` so they can be divided by."""
    return np.where(row_sums != 0.0, row_sums, 1.0)


def _nonzero_counts(arr: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-major and per-minor counts of stored values that are nonzero."""
    nonzero = arr.values != 0
    nz_group_sizes = np.diff(arr.value_ptr) * nonzero
    group_of_major = np.repeat(np.arange(arr.n_major, dtype=np.int64), np.diff(arr.major_ptr))
    major = np.bincount(group_of_major, weights=nz_group_sizes, minlength=arr.n_major)
    minor = _ops.minor_sums(
        nonzero.astype(np.float64),
        arr.value_ptr,
        arr.indices,
        arr.n_minor,
        _ops.accumulator_threads(arr.n_minor),
    )
    return major.astype(np.int64), np.rint(minor).astype(np.int64)


@numba.njit(cache=True, inline="always")
def _is_sorted(seg: np.ndarray) -> bool:
    for k in range(1, seg.shape[0]):  # noqa: SIM110 -- numba has no all(genexpr)
        if seg[k] < seg[k - 1]:
            return False
    return True


@numba.njit(cache=True, parallel=True)
def _depth_sums_major_is_col(major_ptr, values, value_ptr, indices, inv_depth):
    """``inv_depth @ arr`` for a VCSC ``arr``, summing each value group in row order.

    A plain gather (``_ops.minor_matvec``) would add a group's rows up in
    whatever order they're stored, and the IVCSC on-disk format re-sorts them
    (:mod:`vsparse._ivcsc`), so a sidecar computed before writing could miss
    one recomputed after reading by an ulp. Summing in sorted order makes the
    result independent of that order; groups are usually sorted already.
    """
    n_major = major_ptr.shape[0] - 1
    out = np.zeros(n_major, dtype=np.float64)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        total = 0.0
        for u in range(major_ptr[j], major_ptr[j + 1]):
            seg = indices[value_ptr[u] : value_ptr[u + 1]]
            if not _is_sorted(seg):
                seg = np.sort(seg)
            acc = 0.0
            for r in seg:
                acc += inv_depth[r]
            total += np.float64(values[u]) * acc
        out[j] = total
    return out


def _col_depth_sums(arr: Any, row_sums: np.ndarray) -> np.ndarray:
    inv_depth = 1.0 / depth_divisors(row_sums)
    if arr._format == "csc":
        return _depth_sums_major_is_col(
            arr.major_ptr, arr.values, arr.value_ptr, arr.indices, inv_depth
        )
    # Major=rows scatters each row once into each of its columns, in row
    # order, so storage order within a value group doesn't matter here.
    return inv_depth @ arr


def compute_stats(arr: Any) -> VCSStats:
    """Compute ``arr``'s :class:`VCSStats` from its data, in ``O(nnz)``."""
    row_sums = np.asarray(arr.sum(axis=1), dtype=np.float64)
    col_sums = np.asarray(arr.sum(axis=0), dtype=np.float64)
    major_nnz, minor_nnz = _nonzero_counts(arr)
    row_nnz, col_nnz = (minor_nnz, major_nnz) if arr._format == "csc" else (major_nnz, minor_nnz)
    col_depth_sums = _col_depth_sums(arr, row_sums)
    return VCSStats(row_sums, row_nnz, col_sums, col_nnz, col_depth_sums)


def write_stats(
    g: Any,
    stats: VCSStats,
    *,
    write_elem: Callable[..., None],
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    """Write ``stats`` as the ``stats`` subgroup of an array's group ``g``."""
    sg = g.require_group(STATS_KEY)
    for f in fields(stats):
        write_elem(sg, f.name, getattr(stats, f.name), dataset_kwargs=dataset_kwargs)
    sg.attrs["encoding-type"] = STATS_ENCODING_TYPE
    sg.attrs["encoding-version"] = STATS_ENCODING_VERSION


def read_stats(g: Any, shape: tuple[int, int]) -> VCSStats | None:
    """The sidecar stored under an array's group ``g``, or ``None`` if there isn't a usable one.

    ``None`` means the caller should recompute: the sidecar is absent (a file
    written before it existed), of an encoding this version doesn't know, or
    missing a field. One whose lengths don't match ``shape`` is also
    discarded, with a warning, since that can only mean it's corrupt.
    """
    if STATS_KEY not in g:
        return None
    sg = g[STATS_KEY]
    if (
        sg.attrs.get("encoding-type") != STATS_ENCODING_TYPE
        or sg.attrs.get("encoding-version") != STATS_ENCODING_VERSION
    ):
        return None
    names = [f.name for f in fields(VCSStats)]
    if not all(name in sg for name in names):
        return None
    stats = VCSStats(**{name: read_array(sg[name]) for name in names})
    if not (stats.shape == tuple(shape) and stats._is_consistent()):
        warnings.warn(
            f"ignoring a stats sidecar of shape {stats.shape} stored with an array "
            f"of shape {tuple(shape)}; it will be recomputed",
            stacklevel=2,
        )
        return None
    return stats
