from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSRArray


def _adata(n_cells: int = 400, n_genes: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "cell_type": [f"type_{i}" for i in rng.integers(0, 5, n_cells)],
            "sample_id": [f"SAMPLE_{i:03d}" for i in rng.integers(0, 12, n_cells)],
            "total_counts": rng.normal(1000, 50, n_cells),
        },
        index=[f"cell_{i:05d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {
            "gene_symbol": [f"GENE{i:04d}" for i in range(n_genes)],  # unique per gene
            "chromosome": [f"chr{i}" for i in rng.integers(1, 5, n_genes)],  # repeats
        },
        index=[f"ENSG{i:05d}" for i in range(n_genes)],
    )
    X = sp.random_array((n_cells, n_genes), density=0.2, format="csr", random_state=0)
    X.data = np.round(X.data * 8 + 1)
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.mark.parametrize("fmt", ["vcsc", "ivcsc"])
def test_string_columns_are_written_as_categoricals(tmp_path, fmt):
    va = VCSCAnnData.from_anndata(_adata(), format="csr")
    assert not isinstance(va.obs["cell_type"].dtype, pd.CategoricalDtype)

    path = tmp_path / f"data.{fmt}.h5ad"
    va.write_h5ad(path, format=fmt)
    back = VCSCAnnData.read_h5ad(path)

    assert isinstance(back.obs["cell_type"].dtype, pd.CategoricalDtype)
    assert isinstance(back.obs["sample_id"].dtype, pd.CategoricalDtype)
    assert isinstance(back.var["chromosome"].dtype, pd.CategoricalDtype)
    # gene_symbol is unique per gene: nothing to gain, so it's left alone.
    assert not isinstance(back.var["gene_symbol"].dtype, pd.CategoricalDtype)


def test_values_survive_the_conversion(tmp_path):
    original = _adata()
    va = VCSCAnnData.from_anndata(original, format="csr")
    path = tmp_path / "data.h5ad"
    va.write_h5ad(path)
    back = VCSCAnnData.read_h5ad(path)

    for col in ("cell_type", "sample_id"):
        pd.testing.assert_series_equal(
            back.obs[col].astype(str), original.obs[col].astype(str), check_names=False
        )
    np.testing.assert_allclose(back.obs["total_counts"], original.obs["total_counts"])
    np.testing.assert_array_equal(back.obs_names, original.obs_names)
    np.testing.assert_array_equal(back.var_names, original.var_names)
    assert isinstance(back.X, VCSRArray)
    np.testing.assert_allclose(back.X.toarray(), sp.csr_array(original.X).toarray())


def test_conversion_can_be_turned_off(tmp_path):
    va = VCSCAnnData.from_anndata(_adata(), format="csr")
    path = tmp_path / "raw_strings.h5ad"
    va.write_h5ad(path, convert_strings_to_categoricals=False)
    back = VCSCAnnData.read_h5ad(path)

    assert not isinstance(back.obs["cell_type"].dtype, pd.CategoricalDtype)
    assert not isinstance(va.obs["cell_type"].dtype, pd.CategoricalDtype)  # not mutated


def test_categorical_encoding_shrinks_the_file(tmp_path):
    """Categorical codes compress where per-row strings cannot."""
    va_plain = VCSCAnnData.from_anndata(_adata(n_cells=4000), format="csr")
    va_cat = VCSCAnnData.from_anndata(_adata(n_cells=4000), format="csr")

    plain = tmp_path / "plain.h5ad"
    cat = tmp_path / "cat.h5ad"
    va_plain.write_h5ad(plain, convert_strings_to_categoricals=False)
    va_cat.write_h5ad(cat, convert_strings_to_categoricals=True)

    assert cat.stat().st_size < plain.stat().st_size


def test_high_cardinality_columns_are_left_alone(tmp_path):
    """A column with a distinct value per row is left alone."""
    adata = _adata(n_cells=100)
    adata.obs["barcode"] = [f"barcode_{i}" for i in range(adata.n_obs)]
    va = VCSCAnnData.from_anndata(adata, format="csr")

    path = tmp_path / "unique.h5ad"
    va.write_h5ad(path)
    back = VCSCAnnData.read_h5ad(path)

    assert not isinstance(back.obs["barcode"].dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(back.obs["barcode"], adata.obs["barcode"])


def test_zarr_write_converts_too(tmp_path):
    va = VCSCAnnData.from_anndata(_adata(), format="csr")
    store = tmp_path / "data.zarr"
    va.write_zarr(store)
    back = VCSCAnnData.read_zarr(store)

    assert isinstance(back.obs["cell_type"].dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        back.obs["cell_type"].astype(str), va.obs["cell_type"].astype(str)
    )
