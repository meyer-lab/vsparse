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

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["load_packed"]

_FIELD_KEYS = ("obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns")


def load_packed(path: str | PathLike[str], *, x_key: str = "X") -> VCSCAnnData:
    """Load an IVCSR/IVCSC-backed ``.h5ad`` file, decoding ``X`` immediately.

    ``X`` comes back as an ordinary VCSCArray/VCSRArray with plain
    ``indices``, decoded from the packed bytes read from disk. Subset and
    normalize afterwards with the usual indexing and ``.normalized()`` views.

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
