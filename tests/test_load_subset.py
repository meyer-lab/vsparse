"""``load_packed(obs_idx=..., var_idx=...)``: subsetting while unpacking IVCSR/IVCSC."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSCArray, VCSRArray, _ivcsc, load_packed

N_OBS, N_VARS = 120, 3000  # wide, so many index deltas need multi-byte varints


def _counts(rng: np.random.Generator) -> sp.csr_array:
    mat = sp.random_array((N_OBS, N_VARS), density=0.02, format="lil", random_state=0)
    mat[:7] = 0  # empty rows, at both ends and in the middle
    mat[-5:] = 0
    mat[50:53] = 0
    mat[:, 100:300] = 0  # and empty columns
    out = sp.csr_array(mat)
    out.data = rng.integers(1, 40, size=out.data.shape[0]).astype(np.float64)
    return out


@pytest.fixture(params=[VCSRArray, VCSCArray], ids=["ivcsr", "ivcsc"])
def packed_path(request, tmp_path):
    rng = np.random.default_rng(0)
    adata = VCSCAnnData(
        X=request.param.from_scipy(_counts(rng)),
        obs=pd.DataFrame(
            {"group": rng.integers(0, 4, N_OBS)}, index=[f"c{i}" for i in range(N_OBS)]
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
        obsm={"pca": rng.normal(size=(N_OBS, 3))},
        varm={"loadings": rng.normal(size=(N_VARS, 2))},
        obsp={"dist": sp.random_array((N_OBS, N_OBS), density=0.1, format="csr")},
        uns={"note": "kept"},
    )
    path = tmp_path / "packed.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture(params=[False, True], ids=["serial", "many_chunks"])
def chunking(request, monkeypatch):
    """Exercise both a single-span scan and many spans (normally only past 1 MiB)."""
    if request.param:
        monkeypatch.setattr(_ivcsc, "_num_chunks", lambda n_bytes: max(1, min(13, n_bytes)))


def _dense(mat: Any) -> np.ndarray:
    return mat.toarray()


def _assert_same_x(left, right) -> None:
    assert left.shape == right.shape
    assert type(left) is type(right)
    for name in ("major_ptr", "values", "value_ptr", "indices"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


def _assert_matches_indexing(path, obs_idx, var_idx) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # repeated positions -> duplicate names
        subset = load_packed(path, obs_idx=obs_idx, var_idx=var_idx)
        expected = load_packed(path)[
            slice(None) if obs_idx is None else obs_idx,
            slice(None) if var_idx is None else var_idx,
        ]
    _assert_same_x(subset.X, expected.X)
    pd.testing.assert_frame_equal(subset.obs, expected.obs)
    pd.testing.assert_frame_equal(subset.var, expected.var)
    np.testing.assert_array_equal(subset.obsm["pca"], expected.obsm["pca"])
    np.testing.assert_array_equal(subset.varm["loadings"], expected.varm["loadings"])
    np.testing.assert_array_equal(_dense(subset.obsp["dist"]), _dense(expected.obsp["dist"]))
    assert subset.uns["note"] == "kept"


_RNG = np.random.default_rng(1)
SELECTIONS = {
    "obs_mask": (_RNG.random(N_OBS) < 0.2, None),
    "var_mask": (None, _RNG.random(N_VARS) < 0.3),
    "both_masks": (_RNG.random(N_OBS) < 0.5, _RNG.random(N_VARS) < 0.1),
    "sorted_positions": (np.array([0, 6, 7, 40, N_OBS - 1]), np.arange(0, N_VARS, 7)),
    "unsorted_repeated_negative": (np.array([9, 3, 3, -1, 60]), np.array([2999, 5, 5, -3, 0])),
    "slices": (slice(5, 100, 3), slice(None, None, -2)),
    "empty_rows_only": (np.array([0, 1, 50, 51, N_OBS - 1]), None),
    "nothing": (np.array([], dtype=np.int64), np.zeros(N_VARS, dtype=bool)),
    "everything": (np.ones(N_OBS, dtype=bool), np.arange(N_VARS)),
    "list_of_ints": ([4, 8, 15, 16, 23, 42], None),
}


@pytest.mark.parametrize("selection", SELECTIONS.values(), ids=SELECTIONS.keys())
def test_subset_load_matches_indexing_a_full_load(packed_path, chunking, selection):
    _assert_matches_indexing(packed_path, *selection)


def test_metadata_masks_work_directly(packed_path):
    full = load_packed(packed_path)
    mask = full.obs["group"] == 2  # a pandas Series
    _assert_matches_indexing(packed_path, mask, None)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"obs_idx": np.array(["c1", "c2"])}, TypeError),
        ({"var_idx": np.array([0.5, 1.0])}, TypeError),
        ({"obs_idx": np.ones((2, 2), dtype=bool)}, TypeError),
        ({"obs_idx": np.ones(N_OBS - 1, dtype=bool)}, IndexError),
        ({"var_idx": np.array([N_VARS])}, IndexError),
        ({"obs_idx": np.array([-N_OBS - 1])}, IndexError),
    ],
)
def test_invalid_selections_are_rejected(packed_path, kwargs, error):
    with pytest.raises(error):
        load_packed(packed_path, **kwargs)


@pytest.mark.parametrize("seed", range(5))
def test_unpack_selected_matches_a_full_decode(chunking, seed):
    rng = np.random.default_rng(seed)
    arr = VCSRArray.from_scipy(_counts(rng))
    packed = _ivcsc.pack_indices(arr.value_ptr, arr.indices)
    full = VCSRArray(
        arr.shape,
        arr.major_ptr,
        arr.values,
        arr.value_ptr,
        _ivcsc.unpack_indices(arr.value_ptr, packed, arr.indices.dtype),
    )
    sel = np.flatnonzero(rng.random(N_OBS) < 0.3)
    cols = np.flatnonzero(rng.random(N_VARS) < 0.4)
    remap = np.full(N_VARS, -1, dtype=np.int64)
    remap[cols] = np.arange(cols.shape[0])

    major_ptr, groups, value_ptr, indices = _ivcsc.unpack_selected(
        arr.major_ptr, arr.value_ptr, packed, sel, remap, cols.shape[0]
    )
    values = arr.values[groups]  # noqa: PD011 -- a VCSRArray, not pandas
    got = VCSRArray((sel.shape[0], cols.shape[0]), major_ptr, values, value_ptr, indices)
    _assert_same_x(got, full[sel][:, cols])
