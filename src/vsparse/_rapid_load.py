"""Loader for IVCSR/IVCSC-stored ``.h5ad`` files.

:func:`load_packed` reads the packed ``X`` and decodes it immediately into an
ordinary :class:`~vsparse.VCSCArray`/:class:`~vsparse.VCSRArray` inside a
:class:`~vsparse.VCSCAnnData`, optionally keeping only some cells and genes
as it decodes (``obs_idx``/``var_idx``). Normalization is a separate step (the
``.normalized()`` views), not part of loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np

from vsparse import _io
from vsparse._anndata_class import VCSCAnnData
from vsparse._indexutils import normalize_major_idx

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["load_packed"]

_FIELD_KEYS = ("obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns")


def _positions(key: Any, n: int, name: str) -> np.ndarray | None:
    """``key`` (a boolean mask, integer positions, or a slice) as int positions on an axis of ``n``."""
    if key is None:
        return None
    if isinstance(key, slice):
        return np.arange(*key.indices(n), dtype=np.int64)
    arr = np.asarray(key)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    _check_selection(arr, n, name)
    return normalize_major_idx(arr, n).astype(np.int64, copy=False)


def _check_selection(arr: np.ndarray, n: int, name: str) -> None:
    if arr.ndim != 1 or not (arr.dtype == bool or np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(
            f"{name} must be a 1-D boolean mask, an array of integer positions, "
            f"or a slice; got {arr.dtype} with shape {arr.shape}"
        )
    if arr.dtype == bool and arr.shape[0] != n:
        raise IndexError(f"{name} is a boolean mask of length {arr.shape[0]}, expected {n}")


def load_packed(
    path: str | PathLike[str],
    *,
    x_key: str = "X",
    obs_idx: Any = None,
    var_idx: Any = None,
) -> VCSCAnnData:
    """Load an IVCSR/IVCSC-backed ``.h5ad`` file, decoding ``X`` immediately.

    ``X`` comes back as an ordinary VCSCArray/VCSRArray with plain
    ``indices``, decoded from the packed bytes read from disk. Normalize
    afterwards with the ``.normalized()`` views.

    ``load_packed(path, obs_idx=o, var_idx=v)`` equals ``load_packed(path)[o, v]``,
    but decodes only the selected cells (for IVCSR; genes for IVCSC) and drops
    unselected genes (cells) during that same decode, so the full matrix is
    never built. Every packed byte is still read from disk.

    Parameters
    ----------
    path
        Path to an ``.h5ad`` file whose ``X`` (or a top-level group named
        ``x_key``) was stored as IVCSC/IVCSR (see
        :meth:`~vsparse.VCSCAnnData.write_h5ad`).
    x_key
        Top-level h5ad group holding the IVCSR/IVCSC array (``"X"`` by default).
    obs_idx, var_idx
        Cells/genes to keep: a boolean mask over the axis, an array of integer
        positions (negative ones count from the end; unsorted or repeated
        positions give that order), or a slice. ``None`` (the default) keeps
        all. A mask built from metadata, e.g. ``obs["tissue"] == "blood"``,
        works as is; to select by label, convert first with
        ``obs.index.get_indexer(labels)``.
    """
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the zstd/Blosc2 HDF5 filters

    with h5py.File(Path(path), "r") as f:
        kwargs = {k: ad.io.read_elem(f[k]) for k in _FIELD_KEYS if k in f}
        if obs_idx is None and var_idx is None:
            X = ad.io.read_elem(f[x_key])
            return VCSCAnnData(X=X, **kwargs)  # ty: ignore[invalid-argument-type]
        return _load_subset(f[x_key], kwargs, obs_idx, var_idx)


def _load_subset(x_elem: Any, kwargs: dict[str, Any], obs_idx: Any, var_idx: Any) -> VCSCAnnData:
    """:func:`load_packed` with a selection: ``X`` subset as it's decoded, the rest after reading."""
    n_obs, n_vars = _io._read_shape(x_elem)
    rows = _positions(obs_idx, n_obs, "obs_idx")
    cols = _positions(var_idx, n_vars, "var_idx")
    X = _io.read_vcs_subset(x_elem, rows, cols)
    meta = VCSCAnnData(**kwargs)
    fields = meta._subset_fields(
        slice(None) if rows is None else rows, slice(None) if cols is None else cols
    )
    return VCSCAnnData(X=X, **fields)
