"""Tests for group_sum/group_nnz/group_mean on VCSCArray/VCSRArray (#81)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import dense_matrices, slow_first_call
from hypothesis import given
from hypothesis import strategies as st

from vsparse import VCSCArray, VCSRArray, _grouped


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _reference(dense, groups, n_groups, axis, reducer):
    """Dense per-group reduction: ``reducer`` over each group's rows/columns along ``axis``."""
    parts = []
    for g in range(n_groups):
        sel = np.take(dense, np.flatnonzero(groups == g), axis=axis)
        parts.append(reducer(sel, axis=axis))
    if not parts:
        other = dense.shape[1 - axis]
        return np.zeros((0, other)) if axis == 0 else np.zeros((other, 0))
    return np.stack(parts, axis=axis)


def _check_all(v, dense, groups, n_groups, axis):
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        exp_mean = _reference(dense, groups, n_groups, axis, np.mean)
        exp_total_mean = np.array(
            [np.take(dense, np.flatnonzero(groups == g), axis=axis).mean() for g in range(n_groups)]
        )
    exp_sum = _reference(dense, groups, n_groups, axis, np.sum)
    exp_nnz = _reference(dense != 0, groups, n_groups, axis, np.sum)

    np.testing.assert_allclose(v.group_sum(groups, n_groups, axis=axis), exp_sum)
    np.testing.assert_array_equal(v.group_nnz(groups, n_groups, axis=axis), exp_nnz)
    np.testing.assert_allclose(v.group_mean(groups, n_groups, axis=axis), exp_mean)

    np.testing.assert_allclose(
        v.group_sum(groups, n_groups, axis=axis, total=True), exp_sum.sum(axis=1 - axis)
    )
    np.testing.assert_array_equal(
        v.group_nnz(groups, n_groups, axis=axis, total=True), exp_nnz.sum(axis=1 - axis)
    )
    np.testing.assert_allclose(
        v.group_mean(groups, n_groups, axis=axis, total=True), exp_total_mean.reshape(-1)
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_grouped_matches_dense(vcls, axis, dense, data):
    n_groups = data.draw(st.integers(0 if dense.shape[axis] == 0 else 1, 6), label="n_groups")
    groups = np.array(
        data.draw(
            st.lists(
                st.integers(0, max(n_groups - 1, 0)),
                min_size=dense.shape[axis],
                max_size=dense.shape[axis],
            ),
            label="groups",
        ),
        dtype=np.int64,
    )
    _check_all(vcls.from_scipy(sp.csr_array(dense)), dense, groups, n_groups, axis)


@pytest.mark.parametrize("axis", [0, 1])
def test_major_full_both_kernels(dense, vcls, axis, rng, monkeypatch):
    """The group-owned and per-thread-partial kernels must agree with the dense reference."""
    v = vcls.from_scipy(sp.csr_array(dense))
    n_groups = 3
    groups = rng.integers(0, n_groups, size=dense.shape[axis])
    expected = _reference(dense, groups, n_groups, axis, np.sum)
    # Few groups vs. many threads -> partial kernel; patched thread count -> owned kernel.
    for fake_threads in (n_groups + 1, 1):
        monkeypatch.setattr(_grouped.numba, "get_num_threads", lambda n=fake_threads: n)
        np.testing.assert_allclose(v.group_sum(groups, n_groups, axis=axis), expected)


def test_single_group_matches_sum(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    for axis in (0, 1):
        ones = np.zeros(dense.shape[axis], dtype=np.int64)
        np.testing.assert_allclose(v.group_sum(ones, 1, axis=axis).reshape(-1), v.sum(axis=axis))
        np.testing.assert_allclose(v.group_mean(ones, 1, axis=axis).reshape(-1), v.mean(axis=axis))
        np.testing.assert_array_equal(
            v.group_nnz(ones, 1, axis=axis).reshape(-1), v.getnnz(axis=axis)
        )
        assert v.group_sum(ones, 1, axis=axis, total=True)[0] == pytest.approx(v.sum())


def test_sorted_and_shuffled_groups_agree(vcls, rng):
    """Contiguous-by-condition data and the same labels shuffled give identical results."""
    dense = rng.poisson(1.0, size=(60, 9)).astype(np.float64)
    groups = np.repeat(np.arange(5), 12)
    perm = rng.permutation(60)
    v_sorted = vcls.from_scipy(sp.csr_array(dense))
    v_shuffled = vcls.from_scipy(sp.csr_array(dense[perm]))
    np.testing.assert_allclose(v_sorted.group_sum(groups, 5), v_shuffled.group_sum(groups[perm], 5))


def test_empty_group_mean_is_nan(vcls):
    dense = np.array([[1.0, 0.0], [2.0, 3.0]])
    v = vcls.from_scipy(sp.csr_array(dense))
    means = v.group_mean([0, 0], 2)
    np.testing.assert_allclose(means[0], [1.5, 1.5])
    assert np.isnan(means[1]).all()
    np.testing.assert_array_equal(v.group_sum([0, 0], 2)[1], [0.0, 0.0])
    assert np.isnan(v.group_mean([0, 0], 2, total=True)[1])


def test_return_dtypes(vcls):
    v = vcls.from_scipy(sp.csr_array(np.array([[1, 0], [2, 3]], dtype=np.int32)))
    assert v.group_sum([0, 1], 2).dtype == np.float64
    assert v.group_nnz([0, 1], 2).dtype == np.int64
    assert v.group_nnz([0, 1], 2, total=True).dtype == np.int64


@pytest.mark.parametrize(
    ("group_idxs", "n_groups", "axis", "exc"),
    [
        ([0, 1], 2, 2, ValueError),  # bad axis
        ([0, 1], -1, 0, ValueError),  # negative n_groups
        ([0, 1, 0], 2, 0, ValueError),  # wrong length
        ([[0, 1]], 2, 0, ValueError),  # not 1-D
        ([0, 2], 2, 0, ValueError),  # label out of range
        ([0, -1], 2, 0, ValueError),  # negative label
        ([0.0, 1.0], 2, 0, TypeError),  # non-integer labels
    ],
)
def test_invalid_arguments(vcls, group_idxs, n_groups, axis, exc):
    v = vcls.from_scipy(sp.csr_array(np.eye(2)))
    for method in (v.group_sum, v.group_nnz, v.group_mean):
        with pytest.raises(exc):
            method(group_idxs, n_groups, axis=axis)


def test_major_totals_skip_indices(vcls):
    """Per-group totals along the major axis come from value groups alone."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
    v = vcls.from_scipy(sp.csr_array(dense))
    major_axis = 0 if vcls is VCSRArray else 1
    groups = np.array([1, 0, 1])
    expected = _reference(dense, groups, 2, major_axis, np.sum).sum(axis=1 - major_axis)
    # Corrupt indices: a path that read them would scatter into the wrong place.
    v.indices = np.zeros_like(v.indices)
    np.testing.assert_allclose(v.group_sum(groups, 2, axis=major_axis, total=True), expected)
