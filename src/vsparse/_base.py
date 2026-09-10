"""Value-Compressed Sparse Column/Row array types.

VCSC/VCSR are compressed-sparse layouts, inspired by IVSparse's VCSC
(https://github.com/Seth-Wolfgang/IVSparse), that additionally deduplicate
the stored values within each major-axis slice (columns for VCSC, rows for
VCSR). Instead of one stored value per nonzero, each unique value in a
slice is stored once alongside the list of minor-axis indices that share
it. This is a strict win in memory whenever values repeat heavily within a
slice, as is common for integer count matrices (e.g. single-cell RNA-seq
counts).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from vsparse import _construct, _ops
from vsparse._indexutils import is_full_slice as _is_full_slice
from vsparse._indexutils import normalize_major_idx as _normalize_major_idx
from vsparse._indexutils import smallest_index_dtype as _smallest_index_dtype
from vsparse._norm_common import DEFAULT_RECIPE as _DEFAULT_RECIPE
from vsparse._norm_common import Recipe as _Recipe
from vsparse._norm_common import _NormCache

__all__ = ["VCSCArray", "VCSRArray"]


class _VCSBase:
    """Shared implementation for :class:`VCSCArray` and :class:`VCSRArray`.

    Subclasses fix ``_format`` to ``"csc"`` or ``"csr"``, which determines
    whether the major axis (the axis whose slices are value-compressed) is
    columns or rows.
    """

    _format: str

    # Tell numpy to defer binary operators (e.g. ndarray @ VCSCArray) to our
    # __rmatmul__/__rmul__ instead of trying to broadcast us as an ndarray.
    __array_ufunc__ = None

    __slots__ = ("_norm_cache", "indices", "major_ptr", "shape", "value_ptr", "values")

    def __init__(
        self,
        shape: tuple[int, int],
        major_ptr: np.ndarray,
        values: np.ndarray,
        value_ptr: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        n_major, n_minor = self._swap(shape)
        major_ptr = np.asarray(major_ptr, dtype=np.int64)
        value_ptr = np.asarray(value_ptr, dtype=np.int64)
        values = np.asarray(values)
        indices = np.asarray(indices)

        if major_ptr.shape[0] != n_major + 1:
            raise ValueError(f"major_ptr has length {major_ptr.shape[0]}, expected {n_major + 1}")
        if value_ptr.shape[0] != values.shape[0] + 1:
            raise ValueError("value_ptr must have length len(values) + 1")
        if major_ptr[-1] != values.shape[0]:
            raise ValueError("major_ptr[-1] must equal len(values)")
        if value_ptr[-1] != indices.shape[0]:
            raise ValueError("value_ptr[-1] must equal len(indices)")
        if indices.shape[0] and (indices.min() < 0 or indices.max() >= n_minor):
            raise ValueError("indices out of bounds for the given shape")

        # Narrow after the bounds check above, so an out-of-range index is
        # rejected rather than silently truncated by the cast.
        idx_dtype = _smallest_index_dtype(n_minor)
        if idx_dtype.itemsize < indices.dtype.itemsize:
            indices = indices.astype(idx_dtype, copy=False)

        self.shape = (int(shape[0]), int(shape[1]))
        self.major_ptr = major_ptr
        self.values = values
        self.value_ptr = value_ptr
        self.indices = indices
        self._norm_cache = _NormCache()

    # -- axis bookkeeping ------------------------------------------------

    @classmethod
    def _swap(cls, shape: tuple[int, int]) -> tuple[int, int]:
        """Return ``(n_major, n_minor)`` for the given ``(n_rows, n_cols)``."""
        n_rows, n_cols = shape
        return (n_cols, n_rows) if cls._format == "csc" else (n_rows, n_cols)

    @property
    def n_major(self) -> int:
        return self._swap(self.shape)[0]

    @property
    def n_minor(self) -> int:
        return self._swap(self.shape)[1]

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def n_unique(self) -> int:
        """Number of stored (major-slice, unique-value) entries."""
        return int(self.values.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cls = type(self).__name__
        return (
            f"<{cls} shape={self.shape} dtype={self.dtype} nnz={self.nnz} n_unique={self.n_unique}>"
        )

    def copy(self):
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            self.values.copy(),
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    # -- construction / conversion ---------------------------------------

    @classmethod
    def from_scipy(cls, mat) -> _VCSBase:
        """Build from any scipy sparse array/matrix (converted internally)."""
        mat = mat.tocsc() if cls._format == "csc" else mat.tocsr()
        n_major, n_minor = cls._swap(mat.shape)
        major_ptr, values, value_ptr, indices = _construct.compress(
            mat.indptr, mat.indices, mat.data, n_major, n_minor
        )
        return cls(mat.shape, major_ptr, values, value_ptr, indices)

    def to_scipy(self):
        """Decompress to the equivalent scipy ``csc_array``/``csr_array``."""
        n_major, _n_minor = self._swap(self.shape)
        major_ptr, indices, data = _construct.decompress(
            self.major_ptr, self.values, self.value_ptr, self.indices, n_major
        )
        cls = sp.csc_array if self._format == "csc" else sp.csr_array
        return cls((data, indices, major_ptr), shape=self.shape)

    def to_csc(self) -> sp.csc_array:
        return self.to_scipy().tocsc()

    def to_csr(self) -> sp.csr_array:
        return self.to_scipy().tocsr()

    def toarray(self) -> np.ndarray:
        return self.to_scipy().toarray()

    def astype(self, dtype: Any, copy: bool = True) -> _VCSBase:
        """Cast the stored values to ``dtype``. Structural zeros stay zero implicitly."""
        dtype = np.dtype(dtype)
        if not copy and dtype == self.dtype:
            return self
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            self.values.astype(dtype),
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    # -- structural ops ----------------------------------------------------

    @property
    def T(self) -> _VCSBase:
        """Transpose. Free: shares buffers and swaps the VCSC/VCSR dual class."""
        other_cls = VCSRArray if self._format == "csc" else VCSCArray
        return other_cls(
            (self.shape[1], self.shape[0]),
            self.major_ptr,
            self.values,
            self.value_ptr,
            self.indices,
        )

    def transpose(self) -> _VCSBase:
        return self.T

    def _transpose_major(self) -> _VCSBase:
        """Convert to the other VCS format (VCSC<->VCSR) of the *same* shape.

        Unlike :attr:`T` (free: reinterprets the same buffers as the
        transposed matrix), this physically re-groups the stored values by
        the other axis -- see :func:`vsparse._construct.transpose_major`. Used
        to give a major-aligned (parallel-safe, no scatter) kernel something
        to run against, in the direction the array wasn't built for.
        """
        other_cls = VCSRArray if self._format == "csc" else VCSCArray
        major_ptr, values, value_ptr, indices = _construct.transpose_major(
            self.major_ptr, self.values, self.value_ptr, self.indices, self.n_minor
        )
        return other_cls(self.shape, major_ptr, values, value_ptr, indices)

    def normalized(self, view: str | _Recipe = _DEFAULT_RECIPE, *, recalculate: bool = True) -> Any:
        """A normalized *view* of this array -- see :mod:`vsparse._vcs_norm`/:mod:`vsparse._norm_common`.

        Parameters
        ----------
        view
            Which normalization recipe to apply: the name of one of
            :data:`vsparse._norm_common.RECIPES` (``"raw"``, ``"cp10k_log1p"``,
            ``"parafac2"`` (the default), ``"scanpy"``, ``"pearson"``), or a
            :class:`~vsparse._norm_common.Recipe` built by the caller.
        recalculate
            If ``True`` (the default), (re)compute the recipe's statistics
            fresh from this array. If ``False``, reuse a previously computed
            view for ``view`` on this same array, if one exists (from an
            earlier ``.normalized(view, ...)`` call, with any value of
            ``recalculate``) -- this is how switching between recipes avoids
            recomputing each one every time. If no such view has been
            computed yet, one is still computed (there is nothing to reuse).
            The cache holds at most
            :data:`~vsparse._norm_common.NORM_CACHE_MAXSIZE` recipes, and holds
            each view weakly -- see :class:`~vsparse._norm_common._NormCache`.
        """
        from vsparse._norm_common import resolve_recipe
        from vsparse._vcs_norm import VCSCArrayNormalized, VCSRArrayNormalized

        recipe = resolve_recipe(view)
        cache = self._norm_cache
        if not recalculate:
            cached = cache.get(recipe, self)
            if cached is not None:
                return cached

        cls = VCSCArrayNormalized if self._format == "csc" else VCSRArrayNormalized
        result = cls(self, recipe)
        cache.put(recipe, result)
        return result

    def log1p(self) -> _VCSBase:
        """Elementwise ``log1p``. Structural zeros stay zero implicitly."""
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            np.log1p(self.values),
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    # -- reductions -----------------------------------------------------------

    def _major_sums(self) -> np.ndarray:
        """Per-major-slice totals -- cheap, doesn't touch ``indices``."""
        group_sizes = np.diff(self.value_ptr)
        weighted = self.values.astype(np.float64) * group_sizes
        group_of_major = np.repeat(np.arange(self.n_major, dtype=np.int64), np.diff(self.major_ptr))
        return np.bincount(group_of_major, weights=weighted, minlength=self.n_major)

    def _minor_sums(self) -> np.ndarray:
        """A parallel scatter-add over every nonzero to get per-minor-index totals."""
        return _ops.minor_sums(
            self.values,
            self.value_ptr,
            self.indices,
            self.n_minor,
            _ops.accumulator_threads(self.n_minor),
        )

    def sum(self, axis: int | None = None) -> np.ndarray | float:
        """Sum of (structural) values along ``axis`` (0=rows, 1=columns), or overall if ``None``."""
        if axis is None:
            group_sizes = np.diff(self.value_ptr)
            return float(np.sum(self.values.astype(np.float64) * group_sizes))
        if axis not in (0, 1):
            raise ValueError(f"axis must be None, 0, or 1, got {axis!r}")
        # axis=0 reduces over rows (column totals); axis=1 reduces over
        # columns (row totals). Major-slice totals are column totals for
        # VCSC (major=columns) and row totals for VCSR (major=rows).
        major_axis = 0 if self._format == "csc" else 1
        return self._major_sums() if axis == major_axis else self._minor_sums()

    def _major_nnz(self) -> np.ndarray:
        """Per-major-slice stored-element counts -- doesn't touch ``indices``."""
        return (self.value_ptr[self.major_ptr[1:]] - self.value_ptr[self.major_ptr[:-1]]).astype(
            np.int64
        )

    def _minor_nnz(self) -> np.ndarray:
        """A parallel scatter over every nonzero to get per-minor-index counts."""
        return _ops.minor_counts(
            self.value_ptr,
            self.indices,
            self.n_minor,
            _ops.accumulator_threads(self.n_minor),
        )

    def getnnz(self, axis: int | None = None) -> np.ndarray | int:
        """Count of stored elements along ``axis``, or overall if ``None``."""
        if axis is None:
            return self.nnz
        if axis not in (0, 1):
            raise ValueError(f"axis must be None, 0, or 1, got {axis!r}")
        major_axis = 0 if self._format == "csc" else 1
        return self._major_nnz() if axis == major_axis else self._minor_nnz()

    def count_nonzero(self) -> int:
        """Count of stored elements that are actually nonzero (unlike :attr:`nnz`/``getnnz``)."""
        group_sizes = np.diff(self.value_ptr)
        return int(np.sum(group_sizes[self.values != 0]))

    def mean(self, axis: int | None = None) -> np.ndarray | float:
        """Mean of (structural + implicit-zero) values along ``axis``, or overall if ``None``."""
        if axis is None:
            total = self.shape[0] * self.shape[1]
            return self.sum() / total if total else float("nan")
        if axis not in (0, 1):
            raise ValueError(f"axis must be None, 0, or 1, got {axis!r}")
        denom = self.shape[0] if axis == 0 else self.shape[1]
        return self.sum(axis=axis) / denom if denom else np.full(0, float("nan"))

    def _reduce_initial(self, kind: str) -> Any:
        """Identity element for a max/min reduction over ``self.values.dtype``."""
        dt = self.values.dtype
        if np.issubdtype(dt, np.integer):
            return np.iinfo(dt).min if kind == "max" else np.iinfo(dt).max
        return -np.inf if kind == "max" else np.inf

    def _major_reduce(self, ufunc: np.ufunc, initial: Any) -> np.ndarray:
        """Per-major-slice max/min, accounting for implicit zeros in sparse slices."""
        out = np.full(self.n_major, initial, dtype=self.values.dtype)
        group_of_major = np.repeat(np.arange(self.n_major, dtype=np.int64), np.diff(self.major_ptr))
        ufunc.at(out, group_of_major, self.values)
        nnz_per_major = self.value_ptr[self.major_ptr[1:]] - self.value_ptr[self.major_ptr[:-1]]
        not_dense = nnz_per_major < self.n_minor
        out[not_dense] = ufunc(out[not_dense], 0)
        return out

    def _minor_reduce(self, ufunc: np.ufunc, initial: Any) -> np.ndarray:
        """Per-minor-index max/min, accounting for implicit zeros in sparse slices."""
        out, counts = _ops.minor_extrema(
            self.values, self.value_ptr, self.indices, self.n_minor, initial, ufunc is np.maximum
        )
        # A minor index missing from some major slice has a structural zero
        # competing in the reduction.
        not_dense = counts < self.n_major
        out[not_dense] = ufunc(out[not_dense], 0)
        return out

    def _reduce(self, ufunc: np.ufunc, kind: str, axis: int | None) -> np.ndarray | Any:
        initial = self._reduce_initial(kind)
        if axis is None:
            m = ufunc.reduce(self.values, initial=initial) if self.values.size else initial
            if self.nnz < self.shape[0] * self.shape[1]:
                m = ufunc(m, 0)
            return m.item() if hasattr(m, "item") else m
        if axis not in (0, 1):
            raise ValueError(f"axis must be None, 0, or 1, got {axis!r}")
        major_axis = 0 if self._format == "csc" else 1
        return (
            self._major_reduce(ufunc, initial)
            if axis == major_axis
            else self._minor_reduce(ufunc, initial)
        )

    def max(self, axis: int | None = None) -> np.ndarray | Any:
        """Maximum value (including implicit zeros) along ``axis``, or overall if ``None``."""
        return self._reduce(np.maximum, "max", axis)

    def min(self, axis: int | None = None) -> np.ndarray | Any:
        """Minimum value (including implicit zeros) along ``axis``, or overall if ``None``."""
        return self._reduce(np.minimum, "min", axis)

    # -- elementwise arithmetic ----------------------------------------------

    def _elementwise(self, other: Any, op_name: str) -> _VCSBase | np.ndarray:
        """Fall back to scipy to compute an elementwise binary op, re-wrapping a sparse result."""
        other_arg = other.to_scipy() if isinstance(other, _VCSBase) else other
        self_scipy = self.to_scipy()
        if op_name == "multiply":
            result = self_scipy.multiply(other_arg)
        else:
            result = getattr(self_scipy, op_name)(other_arg)
        if result is NotImplemented:
            return NotImplemented
        if sp.issparse(result):
            return type(self).from_scipy(result)
        return np.asarray(result)

    def __add__(self, other):
        if np.isscalar(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError("adding a nonzero scalar to a sparse array is not supported")
        return self._elementwise(other, "__add__")

    __radd__ = __add__

    def __sub__(self, other):
        if np.isscalar(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError(
                "subtracting a nonzero scalar from a sparse array is not supported"
            )
        return self._elementwise(other, "__sub__")

    def __rsub__(self, other):
        if np.isscalar(other):
            if other == 0:
                return -self
            raise NotImplementedError(
                "subtracting a sparse array from a nonzero scalar is not supported"
            )
        other_arg = other.to_scipy() if isinstance(other, _VCSBase) else other
        result = other_arg - self.to_scipy()
        if sp.issparse(result):
            return type(self).from_scipy(result)
        return np.asarray(result)

    def multiply(self, other) -> _VCSBase | np.ndarray:
        """Elementwise multiplication (matches scipy's sparse-array ``.multiply``)."""
        return self * other

    # -- scalar arithmetic --------------------------------------------------

    def _empty_like(self) -> _VCSBase:
        n_major, _ = self._swap(self.shape)
        return type(self)(
            self.shape,
            np.zeros(n_major + 1, dtype=np.int64),
            np.empty(0, dtype=self.values.dtype),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=self.indices.dtype),
        )

    def __mul__(self, other):
        if np.isscalar(other):
            if other == 0:
                return self._empty_like()
            return type(self)(
                self.shape,
                self.major_ptr.copy(),
                self.values * other,
                self.value_ptr.copy(),
                self.indices.copy(),
            )
        return self._elementwise(other, "multiply")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if np.isscalar(other):
            return type(self)(
                self.shape,
                self.major_ptr.copy(),
                self.values / other,
                self.value_ptr.copy(),
                self.indices.copy(),
            )
        return self._elementwise(other, "__truediv__")

    def __neg__(self):
        return self * -1

    # -- matrix products -----------------------------------------------------

    def _dot_right(self, x: np.ndarray) -> np.ndarray:
        """Compute ``self @ x`` for 1-D ``x`` of length ``n_cols``."""
        n_major, n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.major_matvec(*args, x, n_major, n_minor)
        return _ops.minor_matvec(*args, x, n_major)

    def _dot_left(self, x: np.ndarray) -> np.ndarray:
        """Compute ``x @ self`` for 1-D ``x`` of length ``n_rows``."""
        n_major, _n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.minor_matvec(*args, x, n_major)
        return _ops.major_matvec(*args, x, n_major, self.shape[1])

    def _dot_right_mat(self, b: np.ndarray) -> np.ndarray:
        n_major, n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.major_matmat(*args, b, n_major, n_minor)
        return _ops.minor_matmat(*args, b.T, n_major).T

    def _dot_left_mat(self, b: np.ndarray) -> np.ndarray:
        n_major, n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.minor_matmat(*args, b, n_major)
        return _ops.major_matmat(*args, b.T, n_major, n_minor).T

    def __matmul__(self, other):
        other_arr = np.asarray(other)
        if other_arr.ndim == 1:
            if other_arr.shape[0] != self.shape[1]:
                raise ValueError(f"shapes {self.shape} and {other_arr.shape} not aligned")
            return self._dot_right(other_arr)
        if other_arr.ndim == 2:
            if other_arr.shape[0] != self.shape[1]:
                raise ValueError(f"shapes {self.shape} and {other_arr.shape} not aligned")
            return self._dot_right_mat(other_arr)
        return NotImplemented

    def __rmatmul__(self, other):
        other_arr = np.asarray(other)
        if other_arr.ndim == 1:
            if other_arr.shape[0] != self.shape[0]:
                raise ValueError(f"shapes {other_arr.shape} and {self.shape} not aligned")
            return self._dot_left(other_arr)
        if other_arr.ndim == 2:
            if other_arr.shape[1] != self.shape[0]:
                raise ValueError(f"shapes {other_arr.shape} and {self.shape} not aligned")
            return self._dot_left_mat(other_arr)
        return NotImplemented

    # -- indexing -------------------------------------------------------------

    def _select_major(self, key: Any) -> _VCSBase:
        idx = _normalize_major_idx(key, self.n_major)
        starts = self.major_ptr[idx]
        ends = self.major_ptr[idx + 1]
        counts = ends - starts
        new_major_ptr = np.zeros(idx.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=new_major_ptr[1:])

        value_slots = (
            np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])
            if idx.shape[0]
            else np.empty(0, dtype=np.int64)
        )
        new_values = self.values[value_slots]

        v_starts = self.value_ptr[value_slots]
        v_ends = self.value_ptr[value_slots + 1]
        idx_counts = v_ends - v_starts
        new_value_ptr = np.zeros(value_slots.shape[0] + 1, dtype=np.int64)
        np.cumsum(idx_counts, out=new_value_ptr[1:])
        new_indices = (
            np.concatenate([self.indices[s:e] for s, e in zip(v_starts, v_ends, strict=True)])
            if value_slots.shape[0]
            else np.empty(0, dtype=self.indices.dtype)
        )

        n_minor = self.n_minor
        new_shape = (n_minor, idx.shape[0]) if self._format == "csc" else (idx.shape[0], n_minor)
        return type(self)(new_shape, new_major_ptr, new_values, new_value_ptr, new_indices)

    def _major_range(self, start: int, stop: int) -> _VCSBase:
        """The contiguous major-slice range ``[start, stop)``, without copying values.

        ``values``/``indices`` come back as views; only the two pointer
        arrays are rebuilt, rebased to the new start.
        """
        u0, u1 = int(self.major_ptr[start]), int(self.major_ptr[stop])
        k0, k1 = int(self.value_ptr[u0]), int(self.value_ptr[u1])
        n_sel = stop - start
        new_shape = (self.n_minor, n_sel) if self._format == "csc" else (n_sel, self.n_minor)
        return type(self)(
            new_shape,
            self.major_ptr[start : stop + 1] - u0,
            self.values[u0:u1],
            self.value_ptr[u0 : u1 + 1] - k0,
            self.indices[k0:k1],
        )

    def _select_minor(self, key: Any) -> _VCSBase:
        """Select along the minor axis (rows for VCSC, columns for VCSR).

        A key may name the same index more than once, so this is a fan-out
        rather than a remap and one stored element can appear in several
        output positions.
        """
        idx = _normalize_major_idx(key, self.n_minor)
        n_minor_new = idx.shape[0]

        # Output positions grouped by the original index they came from.
        fanout = np.bincount(idx, minlength=self.n_minor)
        offsets = np.zeros(self.n_minor + 1, dtype=np.int64)
        np.cumsum(fanout, out=offsets[1:])
        positions = np.argsort(idx, kind="stable").astype(np.int64, copy=False)

        slot_counts = _ops.minor_select_counts(self.value_ptr, self.indices, fanout)
        kept_slots = np.flatnonzero(slot_counts)

        new_values = self.values[kept_slots]
        new_value_ptr = np.zeros(kept_slots.shape[0] + 1, dtype=np.int64)
        np.cumsum(slot_counts[kept_slots], out=new_value_ptr[1:])

        # Keep the parent's index dtype, as every other structural op does.
        new_indices = np.empty(int(new_value_ptr[-1]), dtype=self.indices.dtype)
        _ops.minor_select_fill(
            self.value_ptr,
            self.indices,
            offsets,
            positions,
            kept_slots,
            new_value_ptr,
            new_indices,
        )

        # Slots keep their original order, so each major slice owns a
        # contiguous run of them.
        major_of_slot = np.searchsorted(self.major_ptr, kept_slots, side="right") - 1
        major_counts = np.bincount(major_of_slot, minlength=self.n_major)
        new_major_ptr = np.zeros(self.n_major + 1, dtype=np.int64)
        np.cumsum(major_counts, out=new_major_ptr[1:])

        new_shape = (
            (n_minor_new, self.n_major) if self._format == "csc" else (self.n_major, n_minor_new)
        )
        return type(self)(new_shape, new_major_ptr, new_values, new_value_ptr, new_indices)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("VCSC/VCSR arrays are 2-D")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        # A bare int on *both* axes must collapse to a scalar, which a 2-D
        # VCSC/VCSR array can't represent -- only that case needs to convert.
        if isinstance(row_key, int | np.integer) and isinstance(col_key, int | np.integer):
            return self.to_scipy()[row_key, col_key]

        major_key, minor_key = (col_key, row_key) if self._format == "csc" else (row_key, col_key)
        if _is_full_slice(major_key) and _is_full_slice(minor_key):
            return self.copy()

        result = self
        if not _is_full_slice(major_key):
            result = result._select_major(major_key)
        if not _is_full_slice(minor_key):
            result = result._select_minor(minor_key)
        return result


class VCSCArray(_VCSBase):
    """Value-Compressed Sparse Column array. Values are deduplicated per column."""

    __slots__ = ()
    _format = "csc"


class VCSRArray(_VCSBase):
    """Value-Compressed Sparse Row array. Values are deduplicated per row."""

    __slots__ = ()
    _format = "csr"
