from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSRArray
from vsparse._rapid_load import load_packed


def _write_ivcsr(tmp_path, dense: np.ndarray, name="data.ivcsr.h5ad"):
    adata = ad.AnnData(X=sp.csr_array(dense))
    vad = VCSCAnnData.from_anndata(adata, format="csr")
    path = tmp_path / name
    vad.write_h5ad(path)
    return path


def test_load_packed_decodes_x(tmp_path, rng):
    """X is decoded eagerly into a plain VCSRArray."""
    dense = rng.integers(0, 8, size=(20, 12)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.4] = 0.0
    path = _write_ivcsr(tmp_path, dense)

    result = load_packed(path)

    assert isinstance(result, VCSCAnnData)
    assert isinstance(result.X, VCSRArray)
    assert result.shape == dense.shape
    np.testing.assert_allclose(result.X.toarray(), dense)


def test_load_packed_metadata_preserved(tmp_path, rng):
    import pandas as pd

    dense = rng.integers(0, 5, size=(10, 6)).astype(np.float64)
    adata = ad.AnnData(X=sp.csr_array(dense))
    adata.obs["grp"] = [str(i % 2) for i in range(10)]
    adata.var["gene"] = [f"g{i}" for i in range(6)]
    vad = VCSCAnnData.from_anndata(adata, format="csr")
    path = tmp_path / "meta.ivcsr.h5ad"
    vad.write_h5ad(path)

    result = load_packed(path)
    pd.testing.assert_index_equal(result.obs.index, adata.obs.index)
    assert list(result.obs["grp"]) == list(adata.obs["grp"])
    assert list(result.var["gene"]) == list(adata.var["gene"])
