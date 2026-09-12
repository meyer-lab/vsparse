from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import vsparse
from vsparse import VCSCArray, VCSRArray


@pytest.fixture
def adata(dense) -> ad.AnnData:
    obj = ad.AnnData(X=sp.csr_array(dense))
    obj.raw = obj
    obj.layers["dense_layer"] = sp.csc_array(dense)
    return obj


def test_from_anndata_x_csc(adata, dense):
    v = vsparse.from_anndata(adata, format="csc")
    assert isinstance(v, VCSCArray)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_x_csr(adata, dense):
    v = vsparse.from_anndata(adata, format="csr")
    assert isinstance(v, VCSRArray)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_raw(adata, dense):
    v = vsparse.from_anndata(adata, use_raw=True)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_layer(adata, dense):
    v = vsparse.from_anndata(adata, layer="dense_layer")
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_rejects_layer_and_raw(adata):
    with pytest.raises(ValueError):
        vsparse.from_anndata(adata, layer="dense_layer", use_raw=True)


def test_to_layer_roundtrip(adata, dense):
    v = vsparse.from_anndata(adata)
    vsparse.to_layer(adata, v, "vcsc_out")
    np.testing.assert_allclose(adata.layers["vcsc_out"].toarray(), dense)


def test_from_anndata_dense_x(dense):
    obj = ad.AnnData(X=dense)
    v = vsparse.from_anndata(obj)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_raw_none_raises(dense):
    """Verify that requesting use_raw=True on an AnnData with no .raw raises ValueError."""
    obj = ad.AnnData(X=sp.csr_array(dense))
    with pytest.raises(ValueError, match=r"adata\.raw is None"):
        vsparse.from_anndata(obj, use_raw=True)


def test_from_anndata_invalid_format_raises(adata):
    """Verify that passing an invalid format string raises ValueError."""
    with pytest.raises(ValueError, match="format must be 'csc' or 'csr'"):
        vsparse.from_anndata(adata, format="invalid_format")


def test_to_layer_shape_mismatch_raises(adata, dense):
    """Verify that storing an array with mismatched dimensions to layer raises ValueError."""
    v = vsparse.from_anndata(adata)
    # Create an incompatible shape AnnData
    mismatched_adata = ad.AnnData(X=np.zeros((dense.shape[0] + 1, dense.shape[1])))
    with pytest.raises(ValueError, match="shape mismatch"):
        vsparse.to_layer(mismatched_adata, v, key="layer_key")
