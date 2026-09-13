from __future__ import annotations

from typing import cast

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSCArray, VCSRArray


@pytest.fixture
def base_adata(dense) -> ad.AnnData:
    obj = ad.AnnData(X=sp.csr_array(dense))
    obj.raw = obj
    obj.obs["grp"] = [str(i % 2) for i in range(dense.shape[0])]
    obj.var["gene"] = [f"g{i}" for i in range(dense.shape[1])]
    return obj


def test_from_anndata_and_back(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata)
    assert isinstance(va.X, VCSCArray)
    assert isinstance(va.raw_X, VCSCArray)
    assert va.shape == dense.shape

    back = va.to_anndata()
    assert type(back) is ad.AnnData
    np.testing.assert_allclose(sp.csr_array(back.X).toarray(), dense)
    assert back.raw is not None
    np.testing.assert_allclose(sp.csr_array(back.raw.X).toarray(), dense)
    assert list(back.obs["grp"]) == list(base_adata.obs["grp"])
    assert list(back.var["gene"]) == list(base_adata.var["gene"])


def test_from_anndata_csr_format(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata, format="csr")
    assert isinstance(va.X, VCSRArray)
    np.testing.assert_allclose(va.X.to_scipy().toarray(), dense)


def test_from_anndata_no_raw(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    assert va.raw_X is None


def test_constructor_rejects_non_vcs_x(dense):
    with pytest.raises(TypeError):
        VCSCAnnData(X=dense)


def test_constructor_rejects_non_vcs_raw_x(dense):
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    with pytest.raises(TypeError):
        VCSCAnnData(X=v, raw_X=dense)


def test_constructor_rejects_raw_kwarg(dense):
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    with pytest.raises(TypeError):
        VCSCAnnData(X=v, raw="not supported")


def test_x_setter_validates_shape(dense):
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    va = VCSCAnnData(X=v)
    other_shape = (dense.shape[0] + 1, dense.shape[1])
    bad = VCSCArray.from_scipy(sp.csc_array(np.zeros(other_shape)))
    with pytest.raises(ValueError):
        va.X = bad


def test_h5ad_roundtrip(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "test.h5ad"
    va.write_h5ad(path)

    read_back = VCSCAnnData.read_h5ad(path)
    assert isinstance(read_back.X, VCSCArray)
    assert isinstance(read_back.raw_X, VCSCArray)
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)
    np.testing.assert_allclose(read_back.raw_X.to_scipy().toarray(), dense)
    assert list(read_back.obs["grp"]) == list(base_adata.obs["grp"])
    assert list(read_back.var["gene"]) == list(base_adata.var["gene"])


def test_h5ad_roundtrip_no_raw(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    path = tmp_path / "test_noraw.h5ad"
    va.write_h5ad(path)

    read_back = VCSCAnnData.read_h5ad(path)
    assert read_back.raw_X is None
    assert read_back.X is not None
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)


def test_zarr_roundtrip(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "test.zarr"
    va.write_zarr(path)

    read_back = VCSCAnnData.read_zarr(path)
    assert isinstance(read_back.X, VCSCArray)
    assert isinstance(read_back.raw_X, VCSCArray)
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)
    np.testing.assert_allclose(read_back.raw_X.to_scipy().toarray(), dense)


def test_h5ad_ivcsc_roundtrip(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "test_ivcsc.h5ad"
    va.write_h5ad(path, format="ivcsc")

    import h5py

    with h5py.File(path, "r") as f:
        assert f["X"].attrs["encoding-type"] == "ivcsc"
        assert "packed_indices" in f["X"]
        assert "indices" not in f["X"]

    read_back = VCSCAnnData.read_h5ad(path)
    assert isinstance(read_back.X, VCSCArray)
    assert isinstance(read_back.raw_X, VCSCArray)
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)
    np.testing.assert_allclose(read_back.raw_X.to_scipy().toarray(), dense)


def test_zarr_ivcsc_roundtrip(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata, format="csr")
    path = tmp_path / "test_ivcsr.zarr"
    va.write_zarr(path, format="ivcsc")

    read_back = VCSCAnnData.read_zarr(path)
    assert isinstance(read_back.X, VCSRArray)
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)


def test_write_h5ad_rejects_bad_format(base_adata, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    with pytest.raises(ValueError):
        va.write_h5ad(tmp_path / "bad.h5ad", format="bogus")


def test_write_h5ad_default_compression(base_adata, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "compressed.h5ad"
    va.write_h5ad(path)

    import h5py

    with h5py.File(path, "r") as f:
        # Blosc2 is registered as a dynamically-loaded HDF5 filter, so h5py
        # doesn't recognize it via `compression` -- check the raw filter id.
        assert any(int(fid) >= 32000 for fid in f["X"]["values"]._filters)


def test_write_h5ad_dataset_kwargs_override(base_adata, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "uncompressed.h5ad"
    va.write_h5ad(path, dataset_kwargs={})

    import h5py

    with h5py.File(path, "r") as f:
        assert dict(f["X"]["values"]._filters) == {}


def test_uns_embedding_roundtrips_via_registry(base_adata, dense, tmp_path):
    """VCSCArray registered as an IO codec also works nested inside uns."""
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    base_adata.uns["vcsc_extra"] = v
    path = tmp_path / "uns_test.h5ad"

    import h5py

    with h5py.File(path, "w") as f:
        ad.io.write_elem(f, "uns", dict(base_adata.uns))

    with h5py.File(path, "r") as f:
        read_back = cast(dict, ad.io.read_elem(f["uns"]))

    assert isinstance(read_back["vcsc_extra"], VCSCArray)
    np.testing.assert_allclose(read_back["vcsc_extra"].to_scipy().toarray(), dense)


def test_x_setter_validation(base_adata, dense):
    """Verify that assigning a valid X matrix updates the X property and shape mismatch raises ValueError."""
    va = VCSCAnnData.from_anndata(base_adata)
    mismatched_v = VCSCArray.from_scipy(
        sp.csc_array(np.zeros((dense.shape[0] + 1, dense.shape[1])))
    )
    with pytest.raises(ValueError, match="does not match adata shape"):
        va.X = mismatched_v

    new_x = VCSCArray.from_scipy(sp.csc_array(dense))
    va.X = new_x
    assert va.X is new_x
    va.X = None
    assert va.X is None


def test_raw_x_setter_validation(base_adata, dense):
    """Verify type checking and assignment behavior on the raw_X property."""
    va = VCSCAnnData.from_anndata(base_adata)
    with pytest.raises(TypeError, match="raw_X must be a VCSCArray or VCSRArray"):
        va.raw_X = "invalid_string"  # type: ignore[assignment]

    new_v = VCSRArray.from_scipy(sp.csr_array(dense))
    va.raw_X = new_v
    assert va.raw_X is new_v


def test_getitem_slices_obs_and_var(base_adata, dense):
    if dense.shape[0] < 3 or dense.shape[1] < 3:
        pytest.skip("shape too small")
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    sub = va[1:3, [0, 2]]

    assert isinstance(sub, VCSCAnnData)
    assert isinstance(sub.X, VCSCArray)
    assert sub.shape == (2, 2)
    np.testing.assert_allclose(sub.X.toarray(), dense[1:3][:, [0, 2]])
    assert list(sub.obs["grp"]) == list(base_adata.obs["grp"].iloc[1:3])
    assert list(sub.var["gene"]) == [base_adata.var["gene"].iloc[0], base_adata.var["gene"].iloc[2]]


def test_getitem_boolean_mask(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    obs_mask = np.zeros(dense.shape[0], dtype=bool)
    obs_mask[::2] = True
    sub = va[obs_mask, :]
    assert isinstance(sub.X, VCSCArray)
    np.testing.assert_allclose(sub.X.toarray(), dense[obs_mask])


def test_getitem_subsets_raw_x_and_returns_new_object(base_adata, dense):
    if dense.shape[0] < 2:
        pytest.skip("shape too small")
    va = VCSCAnnData.from_anndata(base_adata)
    sub = va[0:2, :]
    assert isinstance(sub.raw_X, VCSCArray)
    np.testing.assert_allclose(sub.raw_X.toarray(), dense[0:2])
    assert sub is not va


def test_getitem_single_int_row(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    sub = va[0, :]
    assert sub.shape == (1, dense.shape[1])
    assert isinstance(sub.X, VCSCArray)
    np.testing.assert_allclose(sub.X.toarray(), dense[0:1])


# -- copy() -- the inherited anndata.AnnData.copy() only knows how to copy
# the standard private _X attribute, which this class never sets (X/raw_X
# live in _vcs_X/_vcs_raw_X instead), so it silently drops them.


def test_copy_preserves_x_and_raw_x(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata)
    c = va.copy()
    assert type(c) is VCSCAnnData
    assert isinstance(c.X, VCSCArray)
    assert isinstance(c.raw_X, VCSCArray)
    np.testing.assert_allclose(c.X.toarray(), dense)
    np.testing.assert_allclose(c.raw_X.toarray(), dense)


def test_copy_is_independent_of_the_original(base_adata, dense):
    if dense.shape[0] == 0 or dense.shape[1] == 0:
        pytest.skip("shape too small")
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    c = va.copy()
    assert c.X is not va.X
    assert c.obs is not va.obs

    c.obs["grp"] = "mutated"
    assert list(va.obs["grp"]) != list(c.obs["grp"])


def test_copy_preserves_obs_var_uns(base_adata, dense):
    base_adata.uns["note"] = {"k": "v"}
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    c = va.copy()
    assert list(c.obs["grp"]) == list(va.obs["grp"])
    assert list(c.var["gene"]) == list(va.var["gene"])
    assert c.uns["note"] == {"k": "v"}


def test_copy_preserves_a_normalized_x_subclass(base_adata, dense):
    """A subclass overriding the `X` getter (as BAL-Pf2's own
    lazy-normalized-view AnnData does) must round-trip through copy()."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")

    class _NormalizedView(VCSCAnnData):
        @property
        def X(self):
            return self.normalized("parafac2")

    nv = _NormalizedView.from_anndata(base_adata, include_raw=False)
    c = nv.copy()
    assert type(c) is _NormalizedView
    c_x, nv_x = c.X, nv.X
    assert c_x is not None
    assert nv_x is not None
    np.testing.assert_allclose(c_x.toarray(), nv_x.toarray())


def test_copy_of_a_slice_round_trips_x(base_adata, dense):
    """The exact pattern a caller like `parafac2`'s BiCV split uses:
    `adata[obs_mask][:, var_mask].copy()`."""
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small")
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    sub = va[0:2, 0:2]
    c = sub.copy()
    assert c.X is not None
    np.testing.assert_allclose(c.X.toarray(), dense[0:2, 0:2])


# -- to_memory() -- same underlying problem as copy(): the inherited
# anndata.AnnData.to_memory() doesn't know about _vcs_X/_vcs_raw_X either.


def test_to_memory_preserves_x_and_raw_x(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata)
    m = va.to_memory()
    assert type(m) is VCSCAnnData
    assert isinstance(m.X, VCSCArray)
    assert isinstance(m.raw_X, VCSCArray)
    np.testing.assert_allclose(m.X.toarray(), dense)
    np.testing.assert_allclose(m.raw_X.toarray(), dense)


def test_to_memory_preserves_a_normalized_x_subclass(base_adata, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")

    class _NormalizedView(VCSCAnnData):
        @property
        def X(self):
            return self.normalized("parafac2")

    nv = _NormalizedView.from_anndata(base_adata, include_raw=False)
    m = nv.to_memory()
    assert type(m) is _NormalizedView
    m_x, nv_x = m.X, nv.X
    assert m_x is not None
    assert nv_x is not None
    np.testing.assert_allclose(m_x.toarray(), nv_x.toarray())
