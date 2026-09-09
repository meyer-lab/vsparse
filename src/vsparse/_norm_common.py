"""Shared statistics/materialization logic for normalized VCS views.

:mod:`vsparse._vcs_norm` (:class:`~vsparse.VCSCArrayNormalized`/:class:`~vsparse.
VCSRArrayNormalized`) wraps a raw VCS array and behaves like a
read-depth-normalized, transformed matrix -- without ever materializing it.
The transform follows the declarative recipe proposed in
https://github.com/meyer-lab/vsparse/issues/40#issuecomment-5546322485::

    y[i, j] = (g(x[i, j] * a[i] * b[j]) - c[j]) * s[j]

where ``a`` is a per-cell scale (depth normalization), ``b`` a per-gene
scale, ``g`` a monotone scalar function with ``g(0) == 0``, ``c`` a per-gene
center, and ``s`` a per-gene post-scale. :data:`RECIPES` collects several
common instantiations of this shape (``"raw"``, ``"cp10k_log1p"``,
``"parafac2"``, ``"scanpy"``, ``"pearson"``).

Three properties of this shape keep it representable as a view rather than a
materialized ``n_rows * n_cols`` array:

1. **Separability** -- ``a`` depends only on the cell index, ``b``/``c``/``s``
   only on the gene index, so the whole state of a normalization is
   ``O(n_cells + n_genes)``, never ``O(n_cells * n_genes)``.
2. **Entrywise independence** -- ``y[i, j]`` depends only on ``x[i, j]``, so
   the transform fuses into any kernel already visiting a nonzero.
3. **Rank-1 dense part** -- because ``g(0) = 0``, every structural (implicit)
   zero maps to exactly ``-c[j] * s[j]``, so the full dense matrix is a rank-1
   "baseline" plus a sparse "delta" that is zero off the stored nonzeros.

Statistics (``a``, ``b``, ``c``, ``s``) are computed once, at construction,
over the whole wrapped array, and cached on the view -- see
:meth:`~NormalizedViewBase.select`/:meth:`_VCSBase.normalized` for how a
subset or an alternate recipe is (re)computed. Computing them still requires
touching every nonzero (once for ``b``, once more for ``c``/``s``, since
those need ``b`` first), done with parallel numba kernels below, specialized
per storage format:

- major=columns (VCSC): both passes collapse into one, fully parallel
  over columns with no cross-thread writes -- each column's own elements
  carry everything needed to compute both its ``b`` and its ``c``/``s``
  (:func:`_column_stats_major_is_col`, :func:`_column_gstats_major_is_col`).
- major=rows (VCSR): each pass is a scatter-add across columns from
  many rows, so it's parallelized row-chunked with thread-local partial
  column arrays, reduced by summing across threads
  (:func:`_scaled_col_sums_vcs`, :func:`_gstats_col_sums_vcs`).

These kernels only need ``major_ptr``/``values``/``value_ptr``/``indices``
arrays, from the plain (:mod:`vsparse._base`) array types. The matmul kernels
in :mod:`vsparse._vcs_matmul` walk that same already-materialized ``indices``
array directly. :class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized`
each supply their own ``__matmul__``/``__rmatmul__`` wired to those kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

__all__ = ["DEFAULT_RECIPE", "RECIPES", "NormalizedViewBase", "Recipe", "resolve_recipe"]


# -- recipes ------------------------------------------------------------------

G_IDENTITY = 0
G_LOG1P = 1
G_LOG1P_1000X = 2
G_SQRT = 3


@dataclass(frozen=True, slots=True)
class Recipe:
    """One instantiation of ``y = (g(x * a * b) - c) * s``."""

    name: str
    #: Target used by ``a``: ``None`` -> ``a = 1`` for every cell (no depth
    #: normalization); ``"median"`` -> ``a = median(depth) / depth``;
    #: a float -> ``a = target / depth``.
    depth_target: float | str | None
    #: Whether ``b`` is ``1 / sum_i(x[i, j] * a[i])`` (``True``) or ``1`` (``False``).
    gene_scale: bool
    g_code: int
    center: bool
    post_scale: bool


RECIPES: dict[str, Recipe] = {
    "raw": Recipe("raw", None, False, G_IDENTITY, False, False),
    "cp10k_log1p": Recipe("cp10k_log1p", 1e4, False, G_LOG1P, False, False),
    "parafac2": Recipe("parafac2", "median", True, G_LOG1P_1000X, True, False),
    "scanpy": Recipe("scanpy", 1e4, False, G_LOG1P, True, True),
    "pearson": Recipe("pearson", 1.0, True, G_SQRT, True, True),
}
DEFAULT_RECIPE = "parafac2"


def resolve_recipe(view: str | Recipe) -> Recipe:
    if isinstance(view, Recipe):
        return view
    try:
        return RECIPES[view]
    except KeyError:
        raise ValueError(
            f"unknown normalization view {view!r}; choose from {sorted(RECIPES)}"
        ) from None


# -- elementwise transform ----------------------------------------------------


@numba.njit(cache=True, inline="always")
def _g(x: float, g_code: int) -> float:
    if g_code == G_LOG1P:
        return np.log1p(x)
    if g_code == G_LOG1P_1000X:
        return np.log10(1.0 + 1000.0 * x)
    if g_code == G_SQRT:
        return np.sqrt(x) if x > 0.0 else 0.0
    return x


def _g_np(x: np.ndarray, g_code: int) -> np.ndarray:
    """Vectorized (non-numba) counterpart of :func:`_g`, for the ``__getitem__`` path."""
    if g_code == G_LOG1P:
        return np.log1p(x)
    if g_code == G_LOG1P_1000X:
        return np.log10(1.0 + 1000.0 * x)
    if g_code == G_SQRT:
        return np.sqrt(np.clip(x, 0.0, None))
    return x


# -- statistics: major=columns -- fused, no cross-thread writes -------------


@numba.njit(cache=True, parallel=True)
def _column_stats_major_is_col(major_ptr, values, value_ptr, indices, row_scale):
    """``gsum[j] = sum_i values[i, j] / row_scale[i]`` -- the raw material for ``b``."""
    n_major = major_ptr.shape[0] - 1
    gsum = np.zeros(n_major, dtype=np.float64)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = 0.0
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                gs += v / row_scale[indices[k]]
        gsum[j] = gs
    return gsum


@numba.njit(cache=True, parallel=True)
def _column_gstats_major_is_col(
    major_ptr, values, value_ptr, indices, row_scale, gene_scale, g_code
):
    """Per-column sum/sum-of-squares of ``g(x / row_scale / gene_scale)`` over stored entries."""
    n_major = major_ptr.shape[0] - 1
    col_sum = np.zeros(n_major, dtype=np.float64)
    col_sumsq = np.zeros(n_major, dtype=np.float64)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = gene_scale[j]
        if gs <= 0.0:
            continue
        s0 = 0.0
        s1 = 0.0
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                scaled = v / row_scale[indices[k]] / gs
                gy = _g(scaled, g_code)
                s0 += gy
                s1 += gy * gy
        col_sum[j] = s0
        col_sumsq[j] = s1
    return col_sum, col_sumsq


# -- statistics: major=rows -- scatter-add passes ----------------------------


@numba.njit(cache=True, parallel=True)
def _scaled_col_sums_vcs(major_ptr, values, value_ptr, indices, row_scale, n_cols, nthreads):
    n_major = major_ptr.shape[0] - 1
    chunk = (n_major + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_cols), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_major, start + chunk)
        local = partial[t]
        for i in range(start, end):
            rs = row_scale[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                for k in range(value_ptr[u], value_ptr[u + 1]):
                    local[indices[k]] += v / rs
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _gstats_col_sums_vcs(
    major_ptr, values, value_ptr, indices, row_scale, gene_scale, g_code, n_cols, nthreads
):
    n_major = major_ptr.shape[0] - 1
    chunk = (n_major + nthreads - 1) // nthreads
    partial_sum = np.zeros((nthreads, n_cols), dtype=np.float64)
    partial_sumsq = np.zeros((nthreads, n_cols), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_major, start + chunk)
        local_sum = partial_sum[t]
        local_sumsq = partial_sumsq[t]
        for i in range(start, end):
            rs = row_scale[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                for k in range(value_ptr[u], value_ptr[u + 1]):
                    c = indices[k]
                    gs = gene_scale[c]
                    if gs > 0.0:
                        scaled = v / rs / gs
                        gy = _g(scaled, g_code)
                        local_sum[c] += gy
                        local_sumsq[c] += gy * gy
    return partial_sum.sum(axis=0), partial_sumsq.sum(axis=0)


# -- full materialization -----------------------------------------------------
#
# Both kernels start from an ``out`` already filled with ``-col_offset``
# (``col_post_scale * col_mean`` -- the value every implicit structural zero
# takes) and only overwrite the entries that are actually stored -- each
# parallelized over the major axis, which owns disjoint rows (major=rows) or
# columns (major=columns) of ``out``, so there's no cross-thread write.


@numba.njit(cache=True, parallel=True)
def _fill_normalized_major_is_col(
    major_ptr,
    values,
    value_ptr,
    indices,
    row_scale,
    gene_scale,
    col_mean,
    col_post_scale,
    g_code,
    out,
):
    n_major = major_ptr.shape[0] - 1
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = gene_scale[j]
        if gs <= 0.0:
            continue
        cm = col_mean[j]
        s = col_post_scale[j]
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                r = indices[k]
                scaled = v / row_scale[r] / gs
                out[r, j] = (_g(scaled, g_code) - cm) * s


@numba.njit(cache=True, parallel=True)
def _fill_normalized_major_is_row(
    major_ptr,
    values,
    value_ptr,
    indices,
    row_scale,
    gene_scale,
    col_mean,
    col_post_scale,
    g_code,
    out,
):
    n_major = major_ptr.shape[0] - 1
    for i in numba.prange(n_major):  # ty: ignore[not-iterable]
        rs = row_scale[i]
        for u in range(major_ptr[i], major_ptr[i + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                c = indices[k]
                gs = gene_scale[c]
                if gs <= 0.0:
                    continue
                scaled = v / rs / gs
                out[i, c] = (_g(scaled, g_code) - col_mean[c]) * col_post_scale[c]


def _prep_key(key: Any) -> Any:
    """Turn a bare int into a length-1 list, so fancy indexing never drops that axis."""
    if isinstance(key, int | np.integer):
        return [int(key)]
    return key


def _compute_row_scale(arr: Any, recipe: Recipe) -> np.ndarray:
    n_rows, _n_cols = arr.shape
    if recipe.depth_target is None:
        return np.ones(n_rows, dtype=np.float64)

    row_totals = np.asarray(arr.sum(axis=1), dtype=np.float64)
    if recipe.depth_target == "median":
        target = float(np.median(row_totals)) if row_totals.shape[0] else 0.0
    else:
        target = float(recipe.depth_target)
    if target <= 0.0:
        return np.ones(n_rows, dtype=np.float64)
    row_scale = row_totals / target
    row_scale[row_scale == 0.0] = 1.0  # rows with no counts: avoid div-by-zero (unused otherwise)
    return row_scale


class NormalizedViewBase:
    """Shared implementation for the normalized VCSC/VCSR views.

    Subclasses fix ``_format`` (``"csc"``/``"csr"``) and supply
    ``__matmul__``/``__rmatmul__`` wired to :mod:`vsparse._vcs_matmul`.

    Internally, ``row_scale``/``gene_scale`` hold ``1 / a``/``1 / b`` (the
    reciprocals of the recipe's ``a``/``b``) -- that's the form the numba
    kernels above divide by. ``a``/``b`` (as named in the recipe) are exposed
    via the :attr:`a`/:attr:`b` properties.
    """

    _format: str

    __array_ufunc__ = None

    __slots__ = ("_arr", "col_mean", "col_post_scale", "gene_scale", "recipe", "row_scale", "stale")

    def __init__(
        self, arr: Any, recipe: str | Recipe = DEFAULT_RECIPE, *, stale: bool = False
    ) -> None:
        if arr._format != self._format:
            raise ValueError(
                f"{type(self).__name__} wraps a {self._format!r}-format array, "
                f"got {type(arr).__name__}"
            )
        self._arr = arr
        self.recipe = resolve_recipe(recipe)
        self.stale = stale

        n_rows, n_cols = arr.shape
        row_scale = _compute_row_scale(arr, self.recipe)
        self.row_scale = row_scale

        indices = arr.indices  # decode once; shared by both statistics passes below
        if self.recipe.gene_scale:
            if self._format == "csc":
                gsum = _column_stats_major_is_col(
                    arr.major_ptr, arr.values, arr.value_ptr, indices, row_scale
                )
            else:
                nthreads = numba.get_num_threads()
                gsum = _scaled_col_sums_vcs(
                    arr.major_ptr, arr.values, arr.value_ptr, indices, row_scale, n_cols, nthreads
                )
            gene_scale = gsum
        else:
            gene_scale = np.ones(n_cols, dtype=np.float64)
        self.gene_scale = gene_scale

        if self.recipe.center or self.recipe.post_scale:
            if self._format == "csc":
                col_sum, col_sumsq = _column_gstats_major_is_col(
                    arr.major_ptr,
                    arr.values,
                    arr.value_ptr,
                    indices,
                    row_scale,
                    gene_scale,
                    self.recipe.g_code,
                )
            else:
                nthreads = numba.get_num_threads()
                col_sum, col_sumsq = _gstats_col_sums_vcs(
                    arr.major_ptr,
                    arr.values,
                    arr.value_ptr,
                    indices,
                    row_scale,
                    gene_scale,
                    self.recipe.g_code,
                    n_cols,
                    nthreads,
                )
            mean = col_sum / n_rows if n_rows > 0 else np.zeros(n_cols, dtype=np.float64)
            variance = np.clip(
                col_sumsq / n_rows - mean**2 if n_rows > 0 else np.zeros(n_cols), 0.0, None
            )
            std = np.sqrt(variance)
            col_mean = mean if self.recipe.center else np.zeros(n_cols, dtype=np.float64)
            if self.recipe.post_scale:
                with np.errstate(divide="ignore", invalid="ignore"):
                    col_post_scale = np.where(std > 0.0, 1.0 / std, 1.0)
            else:
                col_post_scale = np.ones(n_cols, dtype=np.float64)
        else:
            col_mean = np.zeros(n_cols, dtype=np.float64)
            col_post_scale = np.ones(n_cols, dtype=np.float64)
        self.col_mean = col_mean
        self.col_post_scale = col_post_scale

    @classmethod
    def from_stats(
        cls,
        arr: Any,
        recipe: str | Recipe,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        s: np.ndarray,
        *,
        stale: bool = True,
    ) -> NormalizedViewBase:
        """Build directly from precomputed ``a``/``b``/``c``/``s``, skipping the ``O(nnz)`` passes.

        Used to carry a previously-computed normalization (e.g. from
        ``.obs``/``.varm``) onto a (possibly different) array without
        recalculating -- see ``recalculate=False`` on
        :meth:`vsparse._base._VCSBase.normalized`. Defaults to marking the
        result :attr:`stale`, since the caller is asserting these statistics
        rather than deriving them from ``arr`` itself.
        """
        if arr._format != cls._format:
            raise ValueError(
                f"{cls.__name__} wraps a {cls._format!r}-format array, got {type(arr).__name__}"
            )
        self = object.__new__(cls)
        self._arr = arr
        self.recipe = resolve_recipe(recipe)
        self.stale = stale
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        self.row_scale = np.where(a > 0.0, 1.0 / a, 0.0)
        self.gene_scale = np.where(b > 0.0, 1.0 / b, 0.0)
        self.col_mean = np.asarray(c, dtype=np.float64)
        self.col_post_scale = np.asarray(s, dtype=np.float64)
        # Subclasses with the ``_dual_arr`` slot (VCS views) need it initialized
        # too, since ``__init__`` (which normally does) is bypassed here.
        self._dual_arr = None
        return self

    # -- recipe-facing statistics (a/b/c/s, as named in the issue) -------------

    @property
    def a(self) -> np.ndarray:
        """Per-cell scale -- ``1 / row_scale``."""
        return np.where(self.row_scale > 0.0, 1.0 / self.row_scale, 0.0)

    @property
    def b(self) -> np.ndarray:
        """Per-gene scale -- ``1 / gene_scale``."""
        return np.where(self.gene_scale > 0.0, 1.0 / self.gene_scale, 0.0)

    @property
    def c(self) -> np.ndarray:
        """Per-gene center."""
        return self.col_mean

    @property
    def s(self) -> np.ndarray:
        """Per-gene post-scale."""
        return self.col_post_scale

    @property
    def shape(self) -> tuple[int, int]:
        return self._arr.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        stale = " stale=True" if self.stale else ""
        return f"<{type(self).__name__} shape={self.shape} dtype={self.dtype} recipe={self.recipe.name!r}{stale}>"

    # -- materialization ------------------------------------------------------

    def toarray(self) -> np.ndarray:
        """The full normalized matrix, densely."""
        n_rows, n_cols = self.shape
        baseline = -self.col_mean * self.col_post_scale
        out = np.broadcast_to(baseline, (n_rows, n_cols)).copy()
        arr = self._arr
        if self._format == "csc":
            _fill_normalized_major_is_col(
                arr.major_ptr,
                arr.values,
                arr.value_ptr,
                arr.indices,
                self.row_scale,
                self.gene_scale,
                self.col_mean,
                self.col_post_scale,
                self.recipe.g_code,
                out,
            )
        else:
            _fill_normalized_major_is_row(
                arr.major_ptr,
                arr.values,
                arr.value_ptr,
                arr.indices,
                self.row_scale,
                self.gene_scale,
                self.col_mean,
                self.col_post_scale,
                self.recipe.g_code,
                out,
            )
        return out

    # -- selection ---------------------------------------------------------------

    def select(self, rows: Any = slice(None), cols: Any = slice(None)) -> Any:
        """A normalized view of the selected sub-array, with statistics recomputed for it.

        Returns a view, not a dense array, so it still composes with
        ``@``/:meth:`toarray`.
        """
        # A column selection re-derives read depth from only the selected
        # columns, which is rarely what a caller wants; select genes first
        # and normalize after if it matters.
        sub: Any = self._arr[_prep_key(rows), _prep_key(cols)]
        if not isinstance(sub, type(self._arr)):
            sub = type(self._arr).from_scipy(sub)
        return type(self)(sub, self.recipe)

    # -- on-the-fly elementwise access ------------------------------------------

    def __getitem__(self, key: Any) -> np.ndarray:
        """``toarray()[key]``, computed on the fly without materializing the full matrix.

        Applies this view's statistics, so it is not the same as normalizing
        the selected sub-matrix; use :meth:`select` for that.
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"{type(self).__name__} arrays are 2-D")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        row_key = _prep_key(row_key)
        col_key = _prep_key(col_key)

        dense_raw = self._arr[row_key, col_key].toarray().astype(np.float64)
        rs = np.asarray(self.row_scale)[row_key].reshape(-1, 1)
        gs = np.asarray(self.gene_scale)[col_key].reshape(1, -1)
        cm = np.asarray(self.col_mean)[col_key].reshape(1, -1)
        s = np.asarray(self.col_post_scale)[col_key].reshape(1, -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(gs > 0.0, dense_raw / rs / gs, 0.0)
        return (_g_np(scaled, self.recipe.g_code) - cm) * s

    # -- explicitly-unsupported operations ------------------------------------

    def _unsupported(self, op: str) -> Any:
        raise RuntimeError(
            f"{op} is not supported on {type(self).__name__}. Call .toarray() first if you need it."
        )

    def __add__(self, other: Any) -> Any:
        return self._unsupported("addition")

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        return self._unsupported("subtraction")

    def __mul__(self, other: Any) -> Any:
        return self._unsupported("multiplication")

    __rmul__ = __mul__
