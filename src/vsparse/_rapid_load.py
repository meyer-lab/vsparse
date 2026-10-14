"""Loader for IVCSR/IVCSC-stored ``.h5ad`` files.

:func:`load_packed` reads the packed ``X`` and decodes it immediately into an
ordinary :class:`~vsparse.VCSCArray`/:class:`~vsparse.VCSRArray` inside a
:class:`~vsparse.VCSCAnnData`. Filtering and normalization are separate steps
(subsetting, then the ``.normalized()`` views), not part of loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad

from vsparse._anndata_class import VCSCAnnData
from vsparse._compression import read_array
from vsparse._indexutils import smallest_index_dtype
from vsparse._ops import accumulator_threads

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["load_packed"]

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
    major_ptr = read_array(g["major_ptr"])
    values = read_array(g["values"])
    value_ptr = read_array(g["value_ptr"])
    packed = read_array(g["packed_indices"])

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
        stored as IVCSC/IVCSR (see
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
    import hdf5plugin  # noqa: F401  -- registers the zstd/Blosc2 HDF5 filters

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
        cell_mask, row_indptr, indices, data, metadata_cell_mask, gene_denom = (
            _rows_with_obs_filter(
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
        adata = ad.AnnData(**kwargs)
    else:
        adata = ad.AnnData(shape=shape, **kwargs)
    adata = adata[metadata_cell_mask, gene_mask].copy()
    adata.X = X

    return adata


def load_packed(path: str | PathLike[str], *, x_key: str = "X") -> VCSCAnnData:
    """Load an IVCSR/IVCSC-backed ``.h5ad`` file, decoding ``X`` immediately.

    ``X`` comes back as an ordinary VCSCArray/VCSRArray with plain
    ``indices``, decoded from the packed bytes read from disk. Subset and
    normalize afterwards with the usual indexing and ``.normalized()`` views.

    Parameters
    ----------
    path
        Path to an ``.h5ad`` file whose ``X`` (or a top-level group named
        ``x_key``) was stored as IVCSC/IVCSR (see
        :meth:`~vsparse.VCSCAnnData.write_h5ad`).
    x_key
        Top-level h5ad group holding the IVCSR/IVCSC array (``"X"`` by default).
    """
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the zstd/Blosc2 HDF5 filters

    with h5py.File(Path(path), "r") as f:
        X = ad.io.read_elem(f[x_key])
        kwargs = {k: ad.io.read_elem(f[k]) for k in _FIELD_KEYS if k in f}

    return VCSCAnnData(X=X, **kwargs)  # ty: ignore[invalid-argument-type]
