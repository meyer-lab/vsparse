"""Property-based tests for VCSCArrayNormalized/VCSRArrayNormalized: the
default normalized() view, every RECIPES entry, and select(), against a plain
numpy reference implementation of the same math.

Caching/staleness/identity contracts (recalculate=, weak-cache eviction, the
select()-vs-getitem() qualitative divergence on realistic data) aren't numeric
properties of arbitrary input and stay as designed-example tests in
test_vcs_norm.py, test_vcs_norm_recipes.py, and test_norm_selection.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _hypothesis_strategies import axis_key, dense_matrices, slow_first_call
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from vsparse import RECIPES, VCSCArray, VCSRArray


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _reference(dense: np.ndarray) -> np.ndarray:
    """Read-depth normalize, log-transform, and mean-center a dense matrix directly."""
    row_totals = dense.sum(axis=1)
    median = np.median(row_totals) if row_totals.shape[0] else 0.0
    if median > 0.0:
        row_scale = row_totals / median
        row_scale[row_scale == 0.0] = 1.0
    else:
        # A non-positive median row total means depth normalization is skipped
        # entirely (see _norm_common._compute_row_scale's `target <= 0.0` guard).
        row_scale = np.ones_like(row_totals)
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


def _recipe_reference(dense: np.ndarray, recipe: str) -> np.ndarray:
    """A plain-numpy version of ``y = (g(x * a * b) - c) * s`` for each recipe."""
    depth = dense.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if recipe == "raw":
            a = np.ones_like(depth)
        elif recipe in ("cp10k_log1p", "scanpy"):
            a = np.where(depth > 0, 1e4 / depth, 1.0)
        elif recipe == "parafac2":
            median = np.median(depth)
            a = np.where(depth > 0, median / depth, 1.0) if median > 0 else np.ones_like(depth)
        elif recipe == "pearson":
            a = np.where(depth > 0, 1.0 / depth, 1.0)
        else:
            raise ValueError(recipe)

    scaled = dense * a[:, None]
    if recipe in ("parafac2", "pearson"):
        gsum = scaled.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(gsum > 0, 1.0 / gsum, 0.0)
    else:
        b = np.ones(dense.shape[1])

    x = scaled * b[None, :]
    if recipe in ("cp10k_log1p", "scanpy"):
        g = np.log1p(x)
    elif recipe == "parafac2":
        g = np.log10(1.0 + 1000.0 * x)
    elif recipe == "pearson":
        g = np.sqrt(np.clip(x, 0.0, None))
    else:
        g = x

    c = g.mean(axis=0) if recipe in ("parafac2", "scanpy", "pearson") else np.zeros(dense.shape[1])

    if recipe in ("scanpy", "pearson"):
        std = g.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(std > 0, 1.0 / std, 1.0)
    else:
        s = np.ones(dense.shape[1])

    return (g - c) * s


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_default_normalized_matches_reference(vcls, dense):
    assume(dense.sum() > 0)  # median row total must be nonzero
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    np.testing.assert_allclose(nv.toarray(), _reference(dense), atol=1e-8)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_default_normalized_matmul_matches_reference(vcls, dense, data):
    assume(dense.sum() > 0)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    p = data.draw(st.integers(0, 4))
    b = data.draw(arrays(dtype=np.float64, shape=(dense.shape[1], p), elements=st.floats(-10, 10)))
    np.testing.assert_allclose(nv @ b, ref @ b, atol=1e-6)

    q = data.draw(st.integers(0, 4))
    c = data.draw(arrays(dtype=np.float64, shape=(q, dense.shape[0]), elements=st.floats(-10, 10)))
    np.testing.assert_allclose(c @ nv, c @ ref, atol=1e-6)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@pytest.mark.parametrize("recipe", sorted(RECIPES))
@slow_first_call
@given(dense=dense_matrices())
def test_recipe_matches_reference(vcls, recipe, dense):
    assume(dense.sum() > 0)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized(recipe)
    assert nv.recipe.name == recipe
    np.testing.assert_allclose(nv.toarray(), _recipe_reference(dense, recipe), atol=1e-5)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices())
def test_select_with_no_args_is_the_whole_view(vcls, dense):
    assume(dense.sum() > 0)
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()
    np.testing.assert_allclose(nv.select().toarray(), nv.toarray(), atol=1e-8)


@pytest.mark.parametrize("vcls", [VCSCArray, VCSRArray])
@slow_first_call
@given(dense=dense_matrices(), data=st.data())
def test_select_matches_reference_on_the_selection(vcls, dense, data):
    """select(rows, cols) recomputes statistics for that sub-block, so it must
    agree with normalizing the dense sub-block directly."""
    assume(dense.sum() > 0)
    rows = axis_key(data, dense.shape[0])
    cols = axis_key(data, dense.shape[1])
    sub_dense = dense[rows, :][:, cols]
    assume(sub_dense.sum() > 0)

    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()
    got = nv.select(rows, cols).toarray()
    np.testing.assert_allclose(got, _reference(sub_dense), atol=1e-6)
