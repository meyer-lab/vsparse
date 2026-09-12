"""Tests for select() vs __getitem__() on normalized views, using a designed
two-population dataset (differing depth and marker genes) where the two
diverge substantially -- not a property that holds for arbitrary small random
matrices. General select()-matches-reference and select()-with-no-args
properties are covered in test_property_normalization.py instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSCArrayNormalized, VCSRArray, VCSRArrayNormalized


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _norm_cls(vcls):
    return VCSCArrayNormalized if vcls is VCSCArray else VCSRArrayNormalized


def _reference(dense: np.ndarray) -> np.ndarray:
    """Read-depth normalize, log-transform, and mean-center a dense matrix directly."""
    row_totals = dense.sum(axis=1)
    row_scale = row_totals / np.median(row_totals)
    row_scale[row_scale == 0.0] = 1.0
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


def _mixed_population(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two cell types differing in both sequencing depth and marker genes."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 240, 80
    dense = np.empty((n_cells, n_genes))
    dense[: n_cells // 2] = rng.poisson(0.6, size=(n_cells // 2, n_genes))
    dense[n_cells // 2 :] = rng.poisson(4.0, size=(n_cells // 2, n_genes))
    dense[n_cells // 2 :, :16] *= 6  # markers expressed only in the second type

    mask = np.zeros(n_cells, dtype=bool)
    mask[n_cells // 2 :] = True
    return dense, mask


def _relative_frobenius(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# -- select(): statistics recomputed for the selection -----------------------


def test_select_matches_normalizing_the_selection_directly(vcls):
    """select() gives the same answer as normalizing those cells on their own."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    got = nv.select(mask).toarray()
    want = _reference(dense[mask])

    assert _relative_frobenius(got, want) < 1e-12
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_select_equals_selecting_on_the_raw_array_first(vcls):
    """select() matches selecting on the raw array and normalizing after."""
    dense, mask = _mixed_population()
    arr = vcls.from_scipy(_scipy_for(vcls, dense))

    raw_sub = arr[mask, :]
    assert isinstance(raw_sub, vcls)

    np.testing.assert_allclose(
        arr.normalized().select(mask).toarray(),
        raw_sub.normalized().toarray(),
        atol=1e-12,
    )


def test_select_returns_a_view_that_still_composes(vcls):
    """The result is a view, so it still composes with @ and toarray()."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    sub = nv.select(mask)
    assert isinstance(sub, _norm_cls(vcls))
    assert sub.shape == (int(mask.sum()), dense.shape[1])

    rng = np.random.default_rng(3)
    B = rng.normal(size=(dense.shape[1], 4))
    np.testing.assert_allclose(sub @ B, _reference(dense[mask]) @ B, atol=1e-8)


# -- __getitem__: a window that keeps the parent's statistics ----------------


def test_getitem_is_a_window_into_the_parent_matrix(vcls):
    """Indexing gives the same values as the fully materialized view."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    full = nv.toarray()
    np.testing.assert_allclose(nv[mask, :], full[mask, :], atol=1e-10)
    np.testing.assert_allclose(nv[0:4, 0:4], full[0:4, 0:4], atol=1e-10)


def test_getitem_and_select_disagree_substantially_on_a_real_selection(vcls):
    """Windowing and renormalizing diverge by tens of percent on a real selection."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    want = _reference(dense[mask])
    windowed = np.asarray(nv[mask, :])
    renormalized = nv.select(mask).toarray()

    assert _relative_frobenius(renormalized, want) < 1e-12
    assert _relative_frobenius(windowed, want) > 0.1
