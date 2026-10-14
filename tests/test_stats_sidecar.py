"""The per-row/per-column statistics sidecar (``arr.stats``, :mod:`vsparse._stats`)."""

from __future__ import annotations

import warnings
from typing import Any

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given

from _hypothesis_strategies import dense_matrices, slow_first_call
from vsparse import RECIPES, VCSCAnnData, VCSCArray, VCSRArray, load_packed
from vsparse import _stats as stats_mod

CLASSES = [VCSCArray, VCSRArray]
_FIELDS = ("row_sums", "row_nnz", "col_sums", "col_nnz", "col_depth_sums")


def _counts(seed: int = 0, shape: tuple[int, int] = (40, 25)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dense = rng.poisson(1.0, size=shape).astype(np.float64)
    dense[rng.random(shape) < 0.5] = 0.0
    return dense


def _arrays(adata: VCSCAnnData) -> tuple[Any, Any]:
    """``adata``'s ``X``/``raw_X``, typed loosely: both are set in every test here."""
    return adata.X, adata.raw_X


def _assert_stats_equal(left: Any, right: Any) -> None:
    for name in _FIELDS:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


def _assert_views_equal(left, right) -> None:
    for name in ("row_scale", "gene_scale", "col_mean", "col_post_scale"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


@slow_first_call
@given(dense=dense_matrices())
@pytest.mark.parametrize("cls", CLASSES)
def test_stats_match_dense(cls, dense):
    stats = cls.from_scipy(sp.csr_array(dense)).stats
    row_sums = dense.sum(axis=1)
    divisors = np.where(row_sums != 0, row_sums, 1.0)
    np.testing.assert_allclose(stats.row_sums, row_sums)
    np.testing.assert_allclose(stats.col_sums, dense.sum(axis=0))
    np.testing.assert_array_equal(stats.row_nnz, (dense != 0).sum(axis=1))
    np.testing.assert_array_equal(stats.col_nnz, (dense != 0).sum(axis=0))
    np.testing.assert_allclose(stats.col_depth_sums, (dense / divisors[:, None]).sum(axis=0))
    assert stats.shape == dense.shape


@pytest.mark.parametrize("cls", CLASSES)
def test_explicit_zeros_are_not_counted(cls):
    mat = sp.csr_array((np.array([0.0, 2.0]), np.array([0, 1]), np.array([0, 2, 2])), shape=(2, 2))
    stats = cls.from_scipy(mat).stats
    np.testing.assert_array_equal(stats.row_nnz, [1, 0])
    np.testing.assert_array_equal(stats.col_nnz, [0, 1])


@pytest.mark.parametrize("cls", CLASSES)
def test_stats_are_cached_read_only_and_shared_by_copies(cls):
    arr = cls.from_scipy(sp.csr_array(_counts()))
    assert arr._stats is None
    stats = arr.stats
    assert arr.stats is stats
    assert arr.copy()._stats is stats
    assert arr[:, :]._stats is stats
    with pytest.raises(ValueError, match="read-only"):
        stats.row_sums[0] = 1.0
    # A subset is different data, so it gets its own.
    assert arr[:5]._stats is None


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_normalization_scales_come_from_the_sidecar(cls, recipe, monkeypatch):
    arr = cls.from_scipy(sp.csr_array(_counts()))
    expected = arr.normalized(recipe)

    # Restored statistics are used as-is: no recomputation, identical results.
    restored = cls.from_scipy(sp.csr_array(_counts()))
    restored._stats = arr.stats
    monkeypatch.setattr(stats_mod, "compute_stats", lambda _arr: pytest.fail("stats recomputed"))
    _assert_views_equal(restored.normalized(recipe), expected)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("fmt", ["h5ad", "zarr"])
def test_write_read_round_trips_the_sidecar(cls, fmt, tmp_path):
    adata = VCSCAnnData(
        X=cls.from_scipy(sp.csr_array(_counts(0))),
        raw_X=cls.from_scipy(sp.csr_array(_counts(1))),
        # Nested arrays go through the registered (unpacked) VCSC/VCSR codec.
        uns={"counts": cls.from_scipy(sp.csr_array(_counts(2)))},
    )
    x, raw_x = _arrays(adata)
    assert x._stats is None  # never computed before writing ...
    path = tmp_path / f"out.{fmt}"
    getattr(adata, f"write_{fmt}")(path)
    back = getattr(VCSCAnnData, f"read_{fmt}")(path)
    back_x, back_raw_x = _arrays(back)

    # ... but written anyway, and read back exactly.
    for name, original, read in (
        ("X", x, back_x),
        ("raw_X", raw_x, back_raw_x),
        ("uns", adata.uns["counts"], back.uns["counts"]),
    ):
        assert read._stats is not None, name
        _assert_stats_equal(read._stats, stats_mod.compute_stats(read))
        _assert_stats_equal(read._stats, original.stats)

    for recipe in sorted(RECIPES):
        _assert_views_equal(back_x.normalized(recipe), x.normalized(recipe))


def test_load_packed_attaches_the_sidecar(tmp_path):
    adata = VCSCAnnData(X=VCSRArray.from_scipy(sp.csr_array(_counts())))
    path = tmp_path / "out.h5ad"
    adata.write_h5ad(path)
    _assert_stats_equal(_arrays(load_packed(path))[0]._stats, _arrays(adata)[0].stats)


def _write_then_edit(tmp_path, edit) -> Any:
    """Write a file, apply ``edit`` to its ``X`` group, and read back ``X``."""
    adata = VCSCAnnData(X=VCSCArray.from_scipy(sp.csr_array(_counts())))
    path = tmp_path / "out.h5ad"
    adata.write_h5ad(path)
    with h5py.File(path, "r+") as f:
        edit(f["X"])
    return _arrays(VCSCAnnData.read_h5ad(path))[0]


def _drop_group(g):
    del g["stats"]


def _bump_version(g):
    g["stats"].attrs["encoding-version"] = "99.0.0"


def _drop_field(g):
    del g["stats/col_depth_sums"]


@pytest.mark.parametrize("edit", [_drop_group, _bump_version, _drop_field])
def test_unusable_sidecar_is_ignored_and_recomputed(tmp_path, edit):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back_x = _write_then_edit(tmp_path, edit)
    assert back_x._stats is None
    expected = VCSCArray.from_scipy(sp.csr_array(_counts()))
    _assert_stats_equal(back_x.stats, expected.stats)
    _assert_views_equal(back_x.normalized(), expected.normalized())


def test_mismatched_sidecar_warns_and_is_recomputed(tmp_path):
    def truncate(g):
        del g["stats/row_sums"]
        g["stats"].create_dataset("row_sums", data=np.ones(3))

    with pytest.warns(UserWarning, match="stats sidecar"):
        back_x = _write_then_edit(tmp_path, truncate)
    assert back_x._stats is None
    assert back_x.stats.shape == back_x.shape


@pytest.mark.parametrize("cls", CLASSES)
def test_stats_do_not_depend_on_order_within_value_groups(cls):
    # IVCS storage re-sorts each value group's indices, so the sidecar
    # written before a round trip has to match what's recomputed after it.
    arr = cls.from_scipy(sp.csr_array(_counts(shape=(300, 40))))
    rng = np.random.default_rng(0)
    shuffled = arr.indices.copy()
    for u in range(arr.n_unique):
        seg = shuffled[arr.value_ptr[u] : arr.value_ptr[u + 1]]
        seg[:] = rng.permutation(seg)
    other = cls(arr.shape, arr.major_ptr, arr.values, arr.value_ptr, shuffled)
    _assert_stats_equal(other.stats, arr.stats)
