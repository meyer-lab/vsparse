"""Tests for normalization recipes (issue #40): multiple views, caching, and staleness."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vsparse import (
    RECIPES,
    Recipe,
    VCSCAnnData,
    VCSCArray,
    VCSCArrayNormalized,
    VCSRArray,
    VCSRArrayNormalized,
)
from vsparse._norm_common import NORM_CACHE_MAXSIZE


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _norm_cls(vcls):
    return VCSCArrayNormalized if vcls is VCSCArray else VCSRArrayNormalized


def _reference(dense: np.ndarray, recipe: str) -> np.ndarray:
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

    if recipe in ("parafac2", "scanpy", "pearson"):
        c = g.mean(axis=0)
    else:
        c = np.zeros(dense.shape[1])

    if recipe in ("scanpy", "pearson"):
        std = g.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(std > 0, 1.0 / std, 1.0)
    else:
        s = np.ones(dense.shape[1])

    return (g - c) * s


# -- per-recipe numerical correctness -----------------------------------------


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_toarray_matches_reference_for_every_recipe(dense, vcls, recipe):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized(recipe)
    assert isinstance(nv, _norm_cls(vcls))
    assert nv.recipe.name == recipe
    np.testing.assert_allclose(nv.toarray(), _reference(dense, recipe), atol=1e-6)


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_matmul_matches_reference_for_every_recipe(dense, vcls, recipe):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized(recipe)
    ref = _reference(dense, recipe)

    rng = np.random.default_rng(11)
    B = rng.normal(size=(dense.shape[1], 3))
    np.testing.assert_allclose(nv @ B, ref @ B, atol=1e-5)

    Bl = rng.normal(size=(3, dense.shape[0]))
    np.testing.assert_allclose(Bl @ nv, Bl @ ref, atol=1e-5)


def test_unknown_view_raises(vcls, dense):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    with pytest.raises(ValueError, match="unknown normalization view"):
        v.normalized("not-a-recipe")


# -- a/b/c/s exposed as named in the issue ------------------------------------


def test_abcs_properties_are_named_per_issue_40(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized("scanpy")
    assert nv.a.shape == (dense.shape[0],)
    assert nv.b.shape == (dense.shape[1],)
    assert nv.c.shape == (dense.shape[1],)
    assert nv.s.shape == (dense.shape[1],)
    # scanpy: b == 1 everywhere (no per-gene scale)
    np.testing.assert_allclose(nv.b, 1.0)


# -- .normalized(view, recalculate=...) caching -------------------------------


def test_recalculate_false_reuses_cached_view(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv1 = v.normalized("parafac2")
    nv2 = v.normalized("parafac2", recalculate=False)
    assert nv1 is nv2


def test_recalculate_true_always_recomputes(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv1 = v.normalized("parafac2")
    nv2 = v.normalized("parafac2", recalculate=True)
    assert nv1 is not nv2
    np.testing.assert_allclose(nv1.toarray(), nv2.toarray())


def test_switching_views_without_recalculate_uses_independent_caches(vcls, dense):
    """Each recipe gets its own cache slot; switching doesn't disturb the others."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    parafac2 = v.normalized("parafac2")
    raw = v.normalized("raw", recalculate=False)  # not cached yet -> computed once
    assert not np.allclose(parafac2.toarray(), raw.toarray())

    # Switching back doesn't recompute -- same object, same values as before.
    back = v.normalized("parafac2", recalculate=False)
    assert back is parafac2
    back_raw = v.normalized("raw", recalculate=False)
    assert back_raw is raw


def test_indexing_the_raw_array_starts_with_an_empty_cache(vcls, dense):
    """A freshly indexed array is a new object with nothing to reuse."""
    if dense.sum() == 0 or dense.shape[0] < 2:
        pytest.skip("shape too small or all-zero")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    v.normalized("parafac2")
    sub = v[0:1, :]
    assert len(sub._norm_cache) == 0


# -- select() carries the recipe forward --------------------------------------


def test_select_keeps_the_same_recipe(vcls, dense):
    if dense.sum() == 0 or dense.shape[0] < 2:
        pytest.skip("shape too small or all-zero")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized("scanpy")
    sub = nv.select(np.arange(min(2, dense.shape[0])))
    assert sub.recipe.name == "scanpy"
    np.testing.assert_allclose(
        sub.toarray(), _reference(dense[: sub.shape[0]], "scanpy"), atol=1e-6
    )


# -- VCSCAnnData: obs/varm/uns storage and staleness --------------------------


def _small_adata(rng: np.random.Generator, n_obs=24, n_vars=10) -> VCSCAnnData:
    dense = rng.poisson(1.5, size=(n_obs, n_vars)).astype(np.float64)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return VCSCAnnData(X=VCSCArray.from_scipy(sp.csc_array(dense)), obs=obs, var=var)


def test_anndata_normalized_records_obs_varm_uns():
    rng = np.random.default_rng(0)
    adata = _small_adata(rng)
    nv = adata.normalized("parafac2")

    np.testing.assert_allclose(adata.obs["vsparse_a"].to_numpy(), nv.a)
    np.testing.assert_allclose(np.asarray(adata.varm["vsparse_b"]).reshape(-1), nv.b)
    np.testing.assert_allclose(np.asarray(adata.varm["vsparse_c"]).reshape(-1), nv.c)
    np.testing.assert_allclose(np.asarray(adata.varm["vsparse_s"]).reshape(-1), nv.s)
    assert adata.uns["vsparse"] == {"recipe": "parafac2", "stale": False}


def test_anndata_normalized_recalculate_false_hits_in_memory_cache():
    rng = np.random.default_rng(1)
    adata = _small_adata(rng)
    nv1 = adata.normalized("parafac2")
    nv2 = adata.normalized("parafac2", recalculate=False)
    assert nv1 is nv2


def test_anndata_indexing_marks_normalization_stale():
    rng = np.random.default_rng(2)
    adata = _small_adata(rng)
    nv = adata.normalized("parafac2")

    sub = adata[0:8, :]
    assert sub.uns["vsparse"] == {"recipe": "parafac2", "stale": True}
    # Parent is untouched by the child's bookkeeping.
    assert adata.uns["vsparse"]["stale"] is False

    # recalculate=False: reuse the carried-over (stale) per-cell/per-gene
    # values without recomputing -- exactly the parent's own values, windowed.
    reused = sub.normalized("parafac2", recalculate=False)
    assert reused.stale is True
    np.testing.assert_allclose(reused.a, nv.a[0:8])
    np.testing.assert_allclose(reused.b, nv.b)

    # recalculate=True: fresh statistics for just this subset, no longer stale.
    fresh = sub.normalized("parafac2", recalculate=True)
    assert fresh.stale is False
    assert sub.uns["vsparse"]["stale"] is False
    assert not np.allclose(fresh.b, nv.b)  # different population -> different gene scale


def test_anndata_normalized_requires_x():
    adata = VCSCAnnData(obs=pd.DataFrame(index=["a"]), var=pd.DataFrame(index=["g"]))
    with pytest.raises(ValueError, match="requires X"):
        adata.normalized()


# -- caller-built Recipe objects ----------------------------------------------


def _custom_recipe() -> Recipe:
    """A recipe that is *not* in RECIPES: cp10k + log1p, centered and variance-scaled."""
    return Recipe("custom_cp10k_scaled", 1e4, False, RECIPES["scanpy"].g_code, True, True)


def test_custom_recipe_works_on_the_array(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized(_custom_recipe())
    assert nv.recipe.name == "custom_cp10k_scaled"
    # Same (a, b, g, c, s) as "scanpy", so it must agree with that reference.
    np.testing.assert_allclose(nv.toarray(), _reference(dense, "scanpy"), atol=1e-6)


def test_custom_recipe_works_on_the_anndata():
    """Regression: VCSCAnnData.normalized() used to hand ``recipe.name`` back to
    the array, which re-resolved it through RECIPES and raised for anything the
    caller built themselves."""
    rng = np.random.default_rng(0)
    adata = _small_adata(rng)
    recipe = _custom_recipe()
    nv = adata.normalized(recipe)

    assert nv.recipe is recipe
    assert adata.uns["vsparse"]["recipe"] == "custom_cp10k_scaled"
    np.testing.assert_allclose(adata.obs["vsparse_a"].to_numpy(), nv.a)
    np.testing.assert_allclose(
        nv.toarray(), _reference(np.asarray(adata.X.toarray()), "scanpy"), atol=1e-6
    )


def test_custom_recipe_round_trips_through_recalculate_false():
    rng = np.random.default_rng(0)
    adata = _small_adata(rng)
    recipe = _custom_recipe()
    ref = adata.normalized(recipe).toarray()
    adata._vcs_norm_cache.clear()  # force the obs/varm/uns path, not the in-memory one
    again = adata.normalized(recipe, recalculate=False)
    np.testing.assert_allclose(again.toarray(), ref)


def test_two_custom_recipes_sharing_a_name_do_not_collide(vcls, dense):
    """The cache keys on the Recipe itself, so a shared ``name`` is not a shared slot."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    centered = Recipe("dup", 1e4, False, RECIPES["scanpy"].g_code, True, False)
    plain = Recipe("dup", 1e4, False, RECIPES["scanpy"].g_code, False, False)
    nv_centered = v.normalized(centered)
    nv_plain = v.normalized(plain, recalculate=False)
    assert nv_plain is not nv_centered
    np.testing.assert_allclose(nv_plain.c, 0.0)
    assert v.normalized(centered, recalculate=False) is nv_centered


def test_recipe_with_an_unknown_g_code_is_rejected(vcls, dense):
    """``_g`` falls through to the identity for an unrecognized code -- fail loudly instead."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    with pytest.raises(ValueError, match="unknown g_code"):
        v.normalized(Recipe("bogus", None, False, 99, False, False))


# -- cache eviction -----------------------------------------------------------


def test_cache_does_not_pin_a_dropped_view(vcls, dense):
    """The cache holds views weakly, so it can't keep an O(nnz) _dual_arr alive."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    ref = weakref.ref(v.normalized("parafac2"))
    gc.collect()
    assert ref() is None, "cached view outlived the caller's last reference"
    assert len(v._norm_cache) == 1, "the statistics themselves should still be cached"


def test_cache_rebuilds_an_evicted_view_from_retained_statistics(vcls, dense):
    """A collected view is rebuilt exactly, without redoing the O(nnz) passes."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    first = v.normalized("scanpy")
    expected = first.toarray()
    row_scale, gene_scale = first.row_scale, first.gene_scale
    del first
    gc.collect()

    again = v.normalized("scanpy", recalculate=False)
    # Bit-identical, not merely close: the internal arrays are carried over
    # rather than round-tripped through the a/b reciprocals.
    np.testing.assert_array_equal(again.row_scale, row_scale)
    np.testing.assert_array_equal(again.gene_scale, gene_scale)
    np.testing.assert_array_equal(again.toarray(), expected)


def test_cache_is_bounded_and_evicts_least_recently_used(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    names = sorted(RECIPES)
    assert len(names) > NORM_CACHE_MAXSIZE, "test needs more recipes than the cache holds"

    held = [v.normalized(n) for n in names]
    assert len(v._norm_cache) == NORM_CACHE_MAXSIZE
    # The first recipe touched is the one dropped.
    assert RECIPES[names[0]] not in v._norm_cache
    assert RECIPES[names[-1]] in v._norm_cache
    assert len(held) == len(names)


def test_cache_hit_is_identity_stable_while_the_caller_holds_the_view(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized("parafac2")
    assert v.normalized("parafac2", recalculate=False) is nv


def test_anndata_cache_does_not_pin_a_dropped_view():
    rng = np.random.default_rng(0)
    adata = _small_adata(rng)
    ref = weakref.ref(adata.normalized("scanpy"))
    gc.collect()
    assert ref() is None
    # Still reusable -- from obs/varm/uns if not from the retained statistics.
    assert adata.normalized("scanpy", recalculate=False) is not None
