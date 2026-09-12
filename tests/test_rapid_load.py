from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSRArray
from vsparse._rapid_load import load_and_normalize, load_packed


def _reference_prepare(
    dense: np.ndarray,
    min_cell_counts: float,
    gene_threshold: float,
    min_cells: int | None = None,
):
    """Direct translation of parafac2.normalize.prepare_dataset's math, on a plain array."""
    X = sp.csr_array(dense)
    cell_mask = np.ravel(X.sum(axis=1)) > min_cell_counts
    gene_mask = np.ravel(X.sum(axis=0)) > (gene_threshold * X.shape[0])
    if min_cells is not None:
        gene_mask &= np.ravel((X > 0).sum(axis=0)) >= min_cells
    Xf = sp.csr_array(X[cell_mask][:, gene_mask])
    Xf.data = Xf.data.astype(np.float32)

    counts_per_cell = np.ravel(Xf.sum(axis=1)).astype(np.float32)
    counts_per_cell /= np.median(counts_per_cell)
    Xf.data /= np.repeat(counts_per_cell, np.diff(Xf.indptr))

    gene_sums = np.ravel(Xf.sum(axis=0)).astype(np.float32)
    Xf.data /= gene_sums[Xf.indices]

    Xf.data *= np.float32(1000.0)
    Xf.data += np.float32(1.0)
    np.log10(Xf.data, out=Xf.data)

    return Xf, np.nonzero(cell_mask)[0], np.nonzero(gene_mask)[0]


def _write_ivcsr(tmp_path, dense: np.ndarray, name="data.ivcsr.h5ad"):
    adata = ad.AnnData(X=sp.csr_array(dense))
    vad = VCSCAnnData.from_anndata(adata, format="csr")
    path = tmp_path / name
    vad.write_h5ad(path, format="ivcsc")
    return path


@pytest.mark.parametrize(
    ("min_cell_counts", "gene_threshold"),
    [(-1.0, 0.0), (10.0, 0.05), (3.0, 0.2)],
)
def test_matches_reference_normalization(tmp_path, rng, min_cell_counts, gene_threshold):
    dense = rng.integers(0, 8, size=(120, 40)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.5] = 0.0
    path = _write_ivcsr(tmp_path, dense)

    result = load_and_normalize(
        path, min_cell_counts=min_cell_counts, gene_threshold=gene_threshold
    )
    ref_X, _, _ = _reference_prepare(dense, min_cell_counts, gene_threshold)

    assert isinstance(result, ad.AnnData)
    assert result.shape == ref_X.shape
    assert isinstance(result.X, sp.csr_array)
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


def test_no_filtering_keeps_everything(tmp_path, rng):
    dense = rng.integers(0, 5, size=(30, 10)).astype(np.float64)
    path = _write_ivcsr(tmp_path, dense)

    result = load_and_normalize(path, min_cell_counts=-1.0, gene_threshold=0.0)

    assert isinstance(result, ad.AnnData)
    assert result.shape == dense.shape
    assert result.n_obs == dense.shape[0]
    assert result.n_vars == dense.shape[1]


def test_min_cells_filters_genes_by_detection_count(tmp_path):
    dense = np.array(
        [
            [5, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    path = _write_ivcsr(tmp_path, dense)

    result = load_and_normalize(
        path,
        min_cell_counts=-1.0,
        gene_threshold=0.0,
        min_cells=3,
    )
    ref_X, _, ref_genes = _reference_prepare(dense, -1.0, 0.0, min_cells=3)

    assert list(result.var_names) == [str(i) for i in ref_genes]
    assert list(result.var_names) == ["2", "3"]
    assert isinstance(result.X, sp.csr_array)
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


def test_min_cells_uses_obs_filtered_cohort(tmp_path):
    import pandas as pd

    dense = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.float64,
    )
    obs = pd.DataFrame(
        {"condition": ["control", "treated"] * 3},
        index=[f"cell_{i}" for i in range(dense.shape[0])],
    )
    adata = ad.AnnData(X=sp.csr_array(dense), obs=obs)
    path = tmp_path / "min_cells_obs_filter.ivcsr.h5ad"
    VCSCAnnData.from_anndata(adata, format="csr").write_h5ad(path, format="ivcsc")

    obs_mask = np.asarray(obs["condition"] == "control")
    result = load_and_normalize(
        path,
        min_cell_counts=-1.0,
        gene_threshold=0.0,
        min_cells=2,
        obs_filter=lambda obs: obs["condition"] == "control",
    )
    ref_X, _, ref_genes = _reference_prepare(dense[obs_mask], -1.0, 0.0, min_cells=2)

    assert list(result.var_names) == [str(i) for i in ref_genes]
    assert list(result.var_names) == ["1", "2"]
    assert isinstance(result.X, sp.csr_array)
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("min_cells", "error", "match"),
    [
        (-1, ValueError, "non-negative"),
        (1.5, TypeError, "integer or None"),
        (True, TypeError, "integer or None"),
    ],
)
def test_min_cells_validation(tmp_path, rng, min_cells, error, match):
    dense = rng.integers(0, 5, size=(10, 4)).astype(np.float64)
    path = _write_ivcsr(tmp_path, dense)

    with pytest.raises(error, match=match):
        load_and_normalize(path, min_cells=min_cells)


def test_obs_filter_matches_subset_reference(tmp_path):
    import pandas as pd

    dense = np.array(
        [
            [10, 0, 2],
            [0, 100, 0],
            [10, 0, 2],
            [0, 100, 0],
            [10, 0, 2],
            [0, 100, 0],
        ],
        dtype=np.float64,
    )
    obs = pd.DataFrame(
        {"condition": ["control", "treated"] * 3},
        index=[f"cell_{i}" for i in range(dense.shape[0])],
    )
    adata = ad.AnnData(X=sp.csr_array(dense), obs=obs)
    path = tmp_path / "obs_filter.ivcsr.h5ad"
    VCSCAnnData.from_anndata(adata, format="csr").write_h5ad(path, format="ivcsc")

    obs_mask = np.asarray(obs["condition"] == "control")
    result = load_and_normalize(
        path,
        min_cell_counts=-1.0,
        gene_threshold=1.0,
        obs_filter=lambda obs: obs["condition"] == "control",
    )
    ref_X, _, ref_genes = _reference_prepare(dense[obs_mask], -1.0, 1.0)

    assert list(result.obs_names) == ["cell_0", "cell_2", "cell_4"]
    assert list(result.var_names) == [str(i) for i in ref_genes]
    assert isinstance(result.X, sp.csr_array)
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


def test_obs_filter_applied_before_cell_filtering_and_slices_metadata(tmp_path, rng):
    import pandas as pd

    n_cells, n_genes = 12, 6
    dense = rng.integers(0, 8, size=(n_cells, n_genes)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.4] = 0.0
    dense[3] = 0.0

    obs = pd.DataFrame(
        {
            "condition": ["control", "treated", "control"] * 4,
            "batch": np.arange(n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    obsm = {"embedding": rng.random((n_cells, 2))}
    obsp = {"distances": rng.random((n_cells, n_cells))}
    adata = ad.AnnData(X=sp.csr_array(dense), obs=obs, var=var, obsm=obsm, obsp=obsp)
    path = tmp_path / "obs_filter_metadata.ivcsr.h5ad"
    VCSCAnnData.from_anndata(adata, format="csr").write_h5ad(path, format="ivcsc")

    obs_mask = np.asarray(obs["condition"] == "control")
    selected_rows = np.nonzero(obs_mask)[0]
    ref_X, ref_cells, ref_genes = _reference_prepare(dense[obs_mask], 5.0, 0.2)
    retained_rows = selected_rows[ref_cells]

    result = load_and_normalize(
        path,
        min_cell_counts=5.0,
        gene_threshold=0.2,
        obs_filter=lambda obs: obs["condition"] == "control",
    )

    assert list(result.obs_names) == [f"cell_{i}" for i in retained_rows]
    assert list(result.var_names) == [f"gene_{j}" for j in ref_genes]
    assert list(result.obs["batch"]) == list(retained_rows)
    np.testing.assert_allclose(
        np.asarray(result.obsm["embedding"]), np.asarray(obsm["embedding"])[retained_rows]
    )
    np.testing.assert_allclose(
        np.asarray(result.obsp["distances"]),
        np.asarray(obsp["distances"])[retained_rows][:, retained_rows],
    )
    assert isinstance(result.X, sp.csr_array)
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("obs_filter", "match"),
    [
        (lambda obs: np.ones(len(obs) - 1, dtype=bool), "mask of length"),
        (lambda obs: np.ones(len(obs), dtype=np.int8), "boolean mask"),
        (lambda obs: np.zeros(len(obs), dtype=bool), "selected no cells"),
    ],
)
def test_obs_filter_validation(tmp_path, rng, obs_filter, match):
    dense = rng.integers(0, 5, size=(10, 4)).astype(np.float64)
    path = _write_ivcsr(tmp_path, dense)

    with pytest.raises(ValueError, match=match):
        load_and_normalize(path, obs_filter=obs_filter)


def test_obs_filter_must_be_callable(tmp_path, rng):
    dense = rng.integers(0, 5, size=(10, 4)).astype(np.float64)
    path = _write_ivcsr(tmp_path, dense)

    with pytest.raises(TypeError, match="must be callable"):
        load_and_normalize(path, obs_filter=True)  # ty: ignore[invalid-argument-type]


def test_obs_filter_requires_obs(tmp_path):
    import h5py

    from vsparse import _io

    dense = np.array([[1.0, 0.0], [0.0, 1.0]])
    path = tmp_path / "no_obs.ivcsr.h5ad"
    with h5py.File(path, "w") as f:
        _io.write_ivcs_elem(f, "X", VCSRArray.from_scipy(sp.csr_array(dense)))

    with pytest.raises(ValueError, match="requires an obs table"):
        load_and_normalize(path, obs_filter=lambda obs: np.ones(len(obs), dtype=bool))


def test_metadata_sliced_correctly(tmp_path, rng):
    import pandas as pd

    n_cells, n_genes = 60, 20
    dense = rng.integers(0, 8, size=(n_cells, n_genes)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.5] = 0.0

    obs = pd.DataFrame(
        {"cell_type": [f"type_{i % 3}" for i in range(n_cells)], "batch": range(n_cells)},
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"SYM_{j}" for j in range(n_genes)]},
        index=pd.Index([f"gene_{j}" for j in range(n_genes)]),
    )
    obsm = {"pca": rng.random((n_cells, 5))}
    varm = {"loadings": rng.random((n_genes, 3))}
    obsp = {"distances": rng.random((n_cells, n_cells))}
    varp = {"correlations": rng.random((n_genes, n_genes))}
    uns = {"dataset_info": "test_metadata", "version": 1}

    adata = ad.AnnData(
        X=sp.csr_array(dense),
        obs=obs,
        var=var,
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        varp=varp,
        uns=uns,
    )

    path = tmp_path / "metadata_test.ivcsr.h5ad"
    vad = VCSCAnnData.from_anndata(adata, format="csr")
    vad.write_h5ad(path, format="ivcsc")

    min_cell_counts = 10.0
    gene_threshold = 0.05
    ref_X, ref_cells, ref_genes = _reference_prepare(dense, min_cell_counts, gene_threshold)

    result = load_and_normalize(
        path, min_cell_counts=min_cell_counts, gene_threshold=gene_threshold
    )

    assert isinstance(result, ad.AnnData)
    assert result.shape == ref_X.shape

    # Check obs/var indices and contents
    assert list(result.obs_names) == [f"cell_{i}" for i in ref_cells]
    assert list(result.var_names) == [f"gene_{j}" for j in ref_genes]
    assert list(result.obs["cell_type"]) == [f"type_{i % 3}" for i in ref_cells]
    assert list(result.var["gene_symbol"]) == [f"SYM_{j}" for j in ref_genes]

    # Check obsm/varm
    np.testing.assert_allclose(np.asarray(result.obsm["pca"]), np.asarray(obsm["pca"])[ref_cells])
    np.testing.assert_allclose(
        np.asarray(result.varm["loadings"]), np.asarray(varm["loadings"])[ref_genes]
    )

    # Check obsp/varp
    np.testing.assert_allclose(
        np.asarray(result.obsp["distances"]), np.asarray(obsp["distances"])[ref_cells][:, ref_cells]
    )
    np.testing.assert_allclose(
        np.asarray(result.varp["correlations"]),
        np.asarray(varp["correlations"])[ref_genes][:, ref_genes],
    )

    # Check uns
    assert result.uns == uns

    # Check X
    assert isinstance(result.X, sp.csr_array)
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


def test_rapid_load_preserves_raw(tmp_path, rng):
    """Verify that load_and_normalize restores adata.raw when present in the h5ad file."""
    import h5py

    from vsparse import VCSRArray, _io

    dense = rng.integers(0, 8, size=(40, 20)).astype(np.float64)
    adata = ad.AnnData(X=sp.csr_array(dense))
    adata.raw = adata

    path = tmp_path / "raw_test.ivcsr.h5ad"
    with h5py.File(path, "w") as f:
        _io.write_ivcs_elem(f, "X", VCSRArray.from_scipy(sp.csr_array(dense)))
        ad.io.write_elem(f, "obs", adata.obs)
        ad.io.write_elem(f, "var", adata.var)
        ad.io.write_elem(f, "raw", {"X": sp.csr_array(dense), "var": adata.var})

    result = load_and_normalize(path, min_cell_counts=-1.0, gene_threshold=0.0)
    assert result.raw is not None


def test_rapid_load_custom_x_key(tmp_path, rng):
    """Verify load_and_normalize reads from a custom top-level group key."""
    import h5py

    from vsparse import VCSRArray, _io

    dense = rng.integers(0, 8, size=(30, 15)).astype(np.float64)
    vcsr = VCSRArray.from_scipy(sp.csr_array(dense))

    path = tmp_path / "custom_key.h5ad"
    with h5py.File(path, "w") as f:
        _io.write_ivcs_elem(f, "custom_matrix", vcsr)

    result = load_and_normalize(
        path, x_key="custom_matrix", min_cell_counts=-1.0, gene_threshold=0.0
    )
    assert isinstance(result, ad.AnnData)
    assert result.shape == dense.shape


def test_load_packed_decodes_x(tmp_path, rng):
    """load_packed is the alternative route: no filtering/normalization, X decoded eagerly."""
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
    vad.write_h5ad(path, format="ivcsc")

    result = load_packed(path)
    pd.testing.assert_index_equal(result.obs.index, adata.obs.index)
    assert list(result.obs["grp"]) == list(adata.obs["grp"])
    assert list(result.var["gene"]) == list(adata.var["gene"])
