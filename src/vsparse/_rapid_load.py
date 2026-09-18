"""Fast filter/normalize loader for IVCSR-stored ``.h5ad`` files.

Mirrors the preprocessing in `parafac2.normalize.prepare_dataset
<https://github.com/meyer-lab/parafac2/blob/main/parafac2/normalize.py>`_
(minimum-count cell filtering, minimum-expression gene filtering, per-cell
read-depth normalization, and a log10 transform) but is written directly
against the on-disk VCSR/IVCSR layout instead of going through
:meth:`~vsparse.VCSCAnnData.read_h5ad` + generic ``AnnData`` indexing.

Why this needs its own code path
---------------------------------
A VCSR major (row/cell) slice stores each *unique* value once, alongside the
list of minor-axis (gene) indices that share it -- ``values``/``value_ptr``
are indexed per unique-value group, not per nonzero. Two consequences drive
the design here:

- **Cell filtering is (almost) free.** A cell's total count is
  ``sum(value * group_size for each unique-value group in that row)``, a sum
  over ``n_unique`` (~46x fewer entries than ``nnz`` on the bundled example
  dataset) that touches only ``major_ptr``/``values``/``value_ptr``. It needs
  none of the (compressed, delta+varint-packed) ``indices`` array, so the
  cell mask can be computed without ever decoding it.
- **Gene filtering and normalization must touch every retained nonzero.** A
  gene's total count is a scatter-add keyed by minor-axis index, so there is
  no shortcut analogous to the cell case. Without an ``obs_filter``, every
  index is decoded. With an ``obs_filter``, the packed stream must still be
  scanned because row byte offsets are not stored, but indices and values
  are materialized only for selected rows. Once decoded, the per-cell scale
  factor differs per row while the per-gene scale factor differs per column,
  so after both are applied a row's values are no longer mostly-repeated --
  the whole point of VCSC/VCSR's deduplication is gone. Normalization is
  therefore done on a plain CSR array, built once, right after the two masks
  are known -- via a fused numba pass rather than scipy's generic per-step
  (row sum, row scale, column sum, column scale, multiply, add, log10)
  elementwise pipeline, each step of which is another single-threaded
  full-``nnz`` pass with its own temporary.

The one thing this implementation does *not* do is what
:func:`parafac2.normalize.prepare_dataset` calls "indexing for subsetting the
data": ``X[cell_mask, gene_mask]``. That line re-derives a filtered matrix via
scipy's generic fancy indexing (row selection, then a column selection that
internally goes through a CSC conversion). Here, row/gene filtering,
compaction, and column-index remapping happen in one fused parallel pass
(:func:`_count_kept`/:func:`_fill_kept`) directly on the decoded arrays, so
the filtered CSR is built once instead of assembled and then re-sliced.
Gene means (``X.var["means"]``) aren't computed either -- nothing here needs
them, and computing them is a simple ``X.mean(axis=0)`` for callers that do.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numba
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from vsparse import _ivcsc
from vsparse._anndata_class import VCSCAnnData
from vsparse._indexutils import smallest_index_dtype
from vsparse._ops import accumulator_threads

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["load_and_normalize", "load_packed"]


# -- group-level (no-decode) cell totals -------------------------------------


def _cell_totals(major_ptr: np.ndarray, values: np.ndarray, value_ptr: np.ndarray) -> np.ndarray:
    """Per-row total counts, computed without touching ``indices``.

    Each unique-value group of size ``value_ptr[k + 1] - value_ptr[k]``
    contributes ``values[k] * group_size`` to its row's total -- so this
    only costs ``O(n_unique)``, not ``O(nnz)``.
    """
    n_major = major_ptr.shape[0] - 1
    group_sizes = np.diff(value_ptr)
    weighted = values.astype(np.float64) * group_sizes
    row_of_group = np.repeat(np.arange(n_major, dtype=np.int64), np.diff(major_ptr))
    return np.bincount(row_of_group, weights=weighted, minlength=n_major)


# -- parallel value expansion (values/value_ptr -> per-nonzero data) --------


@numba.njit(cache=True, parallel=True)
def _expand_values(values: np.ndarray, value_ptr: np.ndarray, out: np.ndarray) -> None:
    n_unique = values.shape[0]
    for k in numba.prange(n_unique):  # ty: ignore[not-iterable]
        v = values[k]
        for p in range(value_ptr[k], value_ptr[k + 1]):
            out[p] = v


def _build_data(values: np.ndarray, value_ptr: np.ndarray, nnz: int) -> np.ndarray:
    out = np.empty(nnz, dtype=values.dtype)
    _expand_values(values, value_ptr, out)
    return out


# -- selective packed decode (obs-filtered IVCSR rows only) -----------------


@numba.njit(cache=True)
def _decode_selected_rows(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    packed: np.ndarray,
    row_mask: np.ndarray,
    out_indices: np.ndarray,
    out_data: np.ndarray,
) -> None:
    """Decode only selected IVCSR rows while scanning past excluded rows."""
    pos = 0
    out_pos = 0
    n_rows = major_ptr.shape[0] - 1

    for r in range(n_rows):
        keep = row_mask[r]
        for g in range(major_ptr[r], major_ptr[r + 1]):
            if keep:
                prev = np.int64(-1)
                value = values[g]
                for _ in range(value_ptr[g], value_ptr[g + 1]):
                    result, pos = _ivcsc._decode_varint(packed, pos)
                    prev = prev + 1 + np.int64(result)
                    out_indices[out_pos] = prev
                    out_data[out_pos] = value
                    out_pos += 1
            else:
                for _ in range(value_ptr[g], value_ptr[g + 1]):
                    while packed[pos] & 0x80:
                        pos += 1
                    pos += 1


def _build_selected_rows(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    packed: np.ndarray,
    indices_dtype: np.dtype,
    row_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_indptr = value_ptr[major_ptr]
    row_nnz = np.diff(full_indptr)[row_mask]
    nnz = int(row_nnz.sum())
    ptr_dtype = smallest_index_dtype(nnz)

    indptr = np.zeros(row_nnz.shape[0] + 1, dtype=ptr_dtype)
    np.cumsum(row_nnz, out=indptr[1:])
    indices = np.empty(nnz, dtype=indices_dtype)
    data = np.empty(nnz, dtype=values.dtype)
    _decode_selected_rows(major_ptr, values, value_ptr, packed, row_mask, indices, data)
    return indptr, indices, data


# -- parallel weighted bincount (raw per-gene totals, for gene_mask) --------


@numba.njit(cache=True, parallel=True)
def _weighted_bincount(
    indices: np.ndarray, data: np.ndarray, n_bins: int, nthreads: int
) -> np.ndarray:
    n = indices.shape[0]
    chunk = (n + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_bins), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n, start + chunk)
        local = partial[t]
        for k in range(start, end):
            local[indices[k]] += data[k]
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _gene_detection_counts(
    indices: np.ndarray, data: np.ndarray, n_genes: int, nthreads: int
) -> np.ndarray:
    """Number of cells with positive expression for each gene."""
    n = indices.shape[0]
    chunk = (n + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_genes), dtype=np.int64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n, start + chunk)
        local = partial[t]
        for k in range(start, end):
            if data[k] > 0:
                local[indices[k]] += 1
    return partial.sum(axis=0)


# -- fused row+column filter/compaction --------------------------------------


@numba.njit(cache=True, parallel=True)
def _count_kept(
    row_indptr: np.ndarray,
    indices: np.ndarray,
    gene_mask: np.ndarray,
    kept_rows: np.ndarray,
    out_counts: np.ndarray,
) -> None:
    for j in numba.prange(kept_rows.shape[0]):  # ty: ignore[not-iterable]
        r = kept_rows[j]
        c = 0
        for k in range(row_indptr[r], row_indptr[r + 1]):
            if gene_mask[indices[k]]:
                c += 1
        out_counts[j] = c


@numba.njit(cache=True, parallel=True)
def _fill_kept(
    row_indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    gene_remap: np.ndarray,
    kept_rows: np.ndarray,
    new_indptr: np.ndarray,
    out_indices: np.ndarray,
    out_data: np.ndarray,
) -> None:
    for j in numba.prange(kept_rows.shape[0]):  # ty: ignore[not-iterable]
        r = kept_rows[j]
        pos = new_indptr[j]
        for k in range(row_indptr[r], row_indptr[r + 1]):
            g = gene_remap[indices[k]]
            if g >= 0:
                out_indices[pos] = g
                out_data[pos] = data[k]
                pos += 1


def _filter_and_compact(
    row_indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    cell_mask: np.ndarray,
    gene_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    kept_rows = np.nonzero(cell_mask)[0]
    n_kept_genes = int(gene_mask.sum())
    gene_remap = (np.cumsum(gene_mask) - 1).astype(smallest_index_dtype(n_kept_genes))
    gene_remap[~gene_mask] = -1

    # ``new_indptr`` is indexed by nonzero count and needs int64 once nnz
    # passes INT32_MAX; ``out_indices`` holds gene indices and never does.
    ptr_dtype = smallest_index_dtype(int(indices.shape[0]))
    col_dtype = smallest_index_dtype(n_kept_genes)

    counts = np.empty(kept_rows.shape[0], dtype=ptr_dtype)
    _count_kept(row_indptr, indices, gene_mask, kept_rows, counts)
    new_indptr = np.zeros(kept_rows.shape[0] + 1, dtype=ptr_dtype)
    np.cumsum(counts, out=new_indptr[1:])

    nnz_filtered = int(new_indptr[-1])
    out_indices = np.empty(nnz_filtered, dtype=col_dtype)
    out_data = np.empty(nnz_filtered, dtype=np.float32)
    _fill_kept(row_indptr, indices, data, gene_remap, kept_rows, new_indptr, out_indices, out_data)

    return new_indptr, out_indices, out_data, kept_rows, n_kept_genes


# -- fused row-scale / gene-scale / log10 transform --------------------------
#
# scipy's route to the same result -- X.sum(axis=1), a np.repeat of the
# per-row scale over nnz, an in-place divide, X.sum(axis=0), a fancy-index
# gather of the per-column scale over nnz, another in-place divide, then
# three more elementwise passes (*1000, +1, log10) -- is ~9 passes over the
# (still nnz-scale, filtering rarely drops much on real data) array plus two
# extra full-nnz temporaries (the repeat and the gather). Every one of those
# passes and temporaries is single-threaded generic numpy code. Fusing the
# whole tail into three parallel passes over the already-filtered CSR (row
# sums -> per-column sums of the row-scaled data -> the final transform,
# each row-parallel with no nnz-sized temporary beyond the output) removes
# both the redundant passes and the memory pressure from those temporaries.


@numba.njit(cache=True, parallel=True)
def _row_sums(indptr: np.ndarray, data: np.ndarray, out: np.ndarray) -> None:
    for r in numba.prange(out.shape[0]):  # ty: ignore[not-iterable]
        s = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            s += data[k]
        out[r] = s


@numba.njit(cache=True, parallel=True)
def _scaled_col_sums(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_scale: np.ndarray,
    n_bins: int,
    nthreads: int,
) -> np.ndarray:
    n_rows = indptr.shape[0] - 1
    chunk = (n_rows + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_bins), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_rows, start + chunk)
        local = partial[t]
        for r in range(start, end):
            sc = row_scale[r]
            for k in range(indptr[r], indptr[r + 1]):
                local[indices[k]] += data[k] / sc
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _fused_normalize_transform(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_scale: np.ndarray,
    gene_sums: np.ndarray,
) -> None:
    """Overwrites ``data`` in place -- each output only depends on its own input."""
    n_rows = indptr.shape[0] - 1
    for r in numba.prange(n_rows):  # ty: ignore[not-iterable]
        sc = row_scale[r]
        for k in range(indptr[r], indptr[r + 1]):
            v = (data[k] / sc) / gene_sums[indices[k]]
            data[k] = np.log10(np.float32(1.0) + np.float32(1000.0) * v)


def _normalize_and_transform(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_genes: int
) -> np.ndarray:
    """Depth normalization + log10(1000x + 1), fused, overwriting ``data`` in place."""
    n_rows = indptr.shape[0] - 1
    nthreads = numba.get_num_threads()

    counts_per_cell = np.empty(n_rows, dtype=np.float64)
    _row_sums(indptr, data, counts_per_cell)
    counts_per_cell /= np.median(counts_per_cell)

    gene_sums = _scaled_col_sums(indptr, indices, data, counts_per_cell, n_genes, nthreads)

    _fused_normalize_transform(indptr, indices, data, counts_per_cell, gene_sums)
    return data


# -- top-level entry point ---------------------------------------------------


def _compute_gene_mask(
    indices: np.ndarray,
    data: np.ndarray,
    n_genes: int,
    denom: int,
    gene_threshold: float,
    min_cells: int | None,
) -> np.ndarray:
    """Genes with total raw counts above ``gene_threshold * denom``, and detected in >= ``min_cells``."""
    gene_totals_raw = _weighted_bincount(indices, data, n_genes, accumulator_threads(n_genes))
    gene_mask = gene_totals_raw > (gene_threshold * denom)
    if min_cells is not None:
        gene_detection_counts = _gene_detection_counts(
            indices, data, n_genes, accumulator_threads(n_genes)
        )
        gene_mask &= gene_detection_counts >= min_cells
    return gene_mask


def _rows_without_obs_filter(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    packed: np.ndarray,
    indices_dtype: np.dtype,
    cell_totals: np.ndarray,
    min_cell_counts: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode every row; ``metadata_cell_mask`` is just ``cell_mask`` since none were dropped upfront."""
    cell_mask = cell_totals > min_cell_counts
    indices = _ivcsc.unpack_indices(value_ptr, packed, indices_dtype)
    data = _build_data(values, value_ptr, indices.shape[0])
    row_indptr = value_ptr[major_ptr]
    return cell_mask, row_indptr, indices, data, cell_mask


def _rows_with_obs_filter(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    packed: np.ndarray,
    indices_dtype: np.dtype,
    cell_totals: np.ndarray,
    obs: Any,
    obs_filter: Callable[[pd.DataFrame], object],
    min_cell_counts: float,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate ``obs_filter``, then decode only the rows it selects."""
    if not isinstance(obs, pd.DataFrame):
        raise ValueError("obs_filter requires an obs table in the h5ad file")
    if not callable(obs_filter):
        raise TypeError("obs_filter must be callable or None")

    obs_mask = np.asarray(obs_filter(obs))
    if obs_mask.ndim != 1 or obs_mask.shape[0] != n_cells:
        raise ValueError(f"obs_filter must return a one-dimensional mask of length {n_cells}")
    if obs_mask.dtype != np.bool_:
        raise ValueError("obs_filter must return a boolean mask")
    if not np.any(obs_mask):
        raise ValueError("obs_filter selected no cells")
    obs_mask = np.ascontiguousarray(obs_mask)

    selected_rows = np.nonzero(obs_mask)[0]
    cell_mask = cell_totals[obs_mask] > min_cell_counts
    row_indptr, indices, data = _build_selected_rows(
        major_ptr, values, value_ptr, packed, indices_dtype, obs_mask
    )

    metadata_cell_mask = np.zeros(n_cells, dtype=np.bool_)
    metadata_cell_mask[selected_rows[cell_mask]] = True
    return cell_mask, row_indptr, indices, data, metadata_cell_mask, selected_rows.shape[0]


_FIELD_KEYS = ("obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns")


def _validate_min_cells(min_cells: int | None) -> None:
    if min_cells is None:
        return
    if isinstance(min_cells, bool) or not isinstance(min_cells, int):
        raise TypeError("min_cells must be an integer or None")
    if min_cells < 0:
        raise ValueError("min_cells must be non-negative")


def _read_ivcsr_group(
    f: Any, x_key: str
) -> tuple[
    tuple[int, int], np.dtype, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    """Read the packed IVCSR array at ``f[x_key]`` plus the surrounding obs/var/etc. fields."""
    g = f[x_key]
    shape = (int(g.attrs["shape"][0]), int(g.attrs["shape"][1]))
    indices_dtype = np.dtype(g.attrs["indices_dtype"])
    major_ptr = g["major_ptr"][...]
    values = g["values"][...]
    value_ptr = g["value_ptr"][...]
    packed = g["packed_indices"][...]

    kwargs = {k: ad.io.read_elem(f[k]) for k in _FIELD_KEYS if k in f}
    if "raw" in f:
        kwargs["raw"] = ad.io.read_elem(f["raw"])
    return shape, indices_dtype, major_ptr, values, value_ptr, packed, kwargs


def load_and_normalize(
    path: str | PathLike[str],
    *,
    min_cell_counts: float = 10.0,
    gene_threshold: float = 0.0,
    min_cells: int | None = None,
    obs_filter: Callable[[pd.DataFrame], object] | None = None,
    x_key: str = "X",
) -> ad.AnnData:
    """Load, filter, and depth-normalize a VCSR/IVCSR-backed ``.h5ad`` file.

    Reproduces ``parafac2.normalize.prepare_dataset``: cells with total
    counts <= ``min_cell_counts`` and genes with total counts <=
    ``gene_threshold * n_cells`` are dropped. When ``min_cells`` is given,
    genes expressed in fewer than ``min_cells`` cells are also dropped. Gene
    filters are measured on the raw counts after any ``obs_filter``.
    The remaining matrix is row-normalized to the median per-cell depth, then
    column-normalized by gene sum, then transformed as ``log10(1000x + 1)``.
    Surrounding metadata (``obs``, ``var``, ``obsm``, etc.) is sliced to
    match the retained cells and genes.

    Parameters
    ----------
    path
        Path to an ``.h5ad`` file whose ``X`` (or ``layers[x_key]``) was
        written with ``format="ivcsc"``/``"ivcsr"`` (see
        :meth:`~vsparse.VCSCAnnData.write_h5ad`).
    min_cell_counts
        Cells with total raw counts <= this are dropped.
    gene_threshold
        Minimum threshold fraction for gene inclusion, as in
        ``parafac2.normalize.prepare_dataset``: genes with total raw counts
        <= ``gene_threshold * n_cells`` are dropped.
    min_cells
        Optional gene filter. Genes expressed in fewer than this
        many cells are dropped. Expression is defined as a raw count > 0.
    obs_filter
        Optional callable receiving ``obs`` and returning a one-dimensional
        boolean mask. When provided, rows are subset before cell filtering,
        gene filtering, and normalization, so gene totals and
        ``gene_threshold`` are computed using only the selected cells. The
        packed IVCSR stream is still read in full, but indices and values are
        materialized only for selected rows.
    x_key
        Top-level h5ad group holding the IVCSR array (``"X"`` by default).

    Returns
    -------
    ad.AnnData
        Filtered, depth-normalized AnnData object with ``X`` as a CSR array
        and sliced metadata.

    Examples
    --------
    Select cells using multiple ``obs`` columns and multiple accepted values::

        load_and_normalize(
            path,
            obs_filter=lambda obs: (
                obs["condition"].isin(["control", "vehicle"])
                & (obs["timepoint"] == "T3")
            ),
        )
    """
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the Blosc2 HDF5 filter

    _validate_min_cells(min_cells)

    with h5py.File(Path(path), "r") as f:
        shape, indices_dtype, major_ptr, values, value_ptr, packed, kwargs = _read_ivcsr_group(
            f, x_key
        )

    n_cells, n_genes = shape

    # Cell mask: no decode needed.
    cell_totals = _cell_totals(major_ptr, values, value_ptr)

    if obs_filter is None:
        cell_mask, row_indptr, indices, data, metadata_cell_mask = _rows_without_obs_filter(
            major_ptr, values, value_ptr, packed, indices_dtype, cell_totals, min_cell_counts
        )
        gene_denom = n_cells
    else:
        cell_mask, row_indptr, indices, data, metadata_cell_mask, gene_denom = _rows_with_obs_filter(
            major_ptr,
            values,
            value_ptr,
            packed,
            indices_dtype,
            cell_totals,
            kwargs.get("obs"),
            obs_filter,
            min_cell_counts,
            n_cells,
        )
    del packed

    gene_mask = _compute_gene_mask(indices, data, n_genes, gene_denom, gene_threshold, min_cells)

    new_indptr, out_indices, out_data, kept_rows, n_kept_genes = _filter_and_compact(
        row_indptr, indices, data, cell_mask, gene_mask
    )
    del indices, data, row_indptr

    normalized = _normalize_and_transform(new_indptr, out_indices, out_data, n_kept_genes)
    X = csr_array((normalized, out_indices, new_indptr), shape=(kept_rows.shape[0], n_kept_genes))

    if "obs" in kwargs and "var" in kwargs:
        adata = ad.AnnData(**kwargs)  # ty: ignore[invalid-argument-type]
    else:
        adata = ad.AnnData(shape=shape, **kwargs)  # ty: ignore[invalid-argument-type]
    adata = adata[metadata_cell_mask, gene_mask].copy()
    adata.X = X

    return adata


def load_packed(path: str | PathLike[str], *, x_key: str = "X") -> VCSCAnnData:
    """Load an IVCSR/IVCSC-backed ``.h5ad`` file, decoding ``X`` immediately.

    Unlike :func:`load_and_normalize`, this does no filtering or
    normalization -- ``X`` comes back as an ordinary VCSCArray/VCSRArray with
    plain ``indices``, decoded from the packed bytes read from disk.

    Parameters
    ----------
    path
        Path to an ``.h5ad`` file whose ``X`` (or a top-level group named
        ``x_key``) was written with ``format="ivcsc"``/``"ivcsr"`` (see
        :meth:`~vsparse.VCSCAnnData.write_h5ad`).
    x_key
        Top-level h5ad group holding the IVCSR/IVCSC array (``"X"`` by default).
    """
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the Blosc2 HDF5 filter

    with h5py.File(Path(path), "r") as f:
        X = ad.io.read_elem(f[x_key])
        kwargs = {k: ad.io.read_elem(f[k]) for k in _FIELD_KEYS if k in f}

    return VCSCAnnData(X=X, **kwargs)  # ty: ignore[invalid-argument-type]
