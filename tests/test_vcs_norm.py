"""Tests for VCSCArrayNormalized/VCSRArrayNormalized: normalized VCSC/VCSR views.

Numeric correctness of the default view against a dense reference (toarray,
matmul) is covered by property-based tests in test_property_normalization.py.
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
    row_scale[row_scale == 0.0] = 1.0  # rows with no counts: avoid div-by-zero (unused otherwise)
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


def test_getitem_matches_reference_block(dense, vcls):
    if dense.sum() == 0 or dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small or all-zero")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    sub = nv[0:2, 0:2]
    np.testing.assert_allclose(sub, ref[0:2, 0:2], atol=1e-8)


def test_getitem_single_row_and_col(dense, vcls):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    np.testing.assert_allclose(nv[0, :], ref[0:1, :], atol=1e-8)
    np.testing.assert_allclose(nv[:, 0], ref[:, 0:1], atol=1e-8)


def test_getitem_boolean_mask(dense, vcls):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    mask = np.zeros(dense.shape[0], dtype=bool)
    mask[::2] = True
    np.testing.assert_allclose(nv[mask, :], ref[mask, :], atol=1e-8)


def test_shape_and_dtype(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    assert nv.shape == dense.shape
    assert nv.dtype == np.float64


def test_wrong_format_raises(vcls):
    other = VCSRArray if vcls is VCSCArray else VCSCArray
    dense = np.eye(3)
    other_arr = other.from_scipy(_scipy_for(other, dense))
    with pytest.raises(ValueError, match="format"):
        _norm_cls(vcls)(other_arr)


@pytest.mark.parametrize(
    "op",
    [
        lambda nv: nv + nv,
        lambda nv: nv * 2,
    ],
)
def test_unsupported_operations_raise_runtime_error(dense, vcls, op):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    with pytest.raises(RuntimeError, match="not supported"):
        op(nv)


# -- matmul: error paths (numeric correctness is property-tested) ------------


def test_matmul_bad_shape_raises(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    with pytest.raises(ValueError, match="shape mismatch"):
        nv @ np.ones(dense.shape[1] + 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        np.ones(dense.shape[0] + 1) @ nv


def test_all_zero_matrix_does_not_crash(vcls):
    dense = np.zeros((5, 4))
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    np.testing.assert_allclose(nv.toarray(), np.zeros((5, 4)))


def test_small_array_dual_is_cached_lazily(dense, vcls):
    """An array whose whole regrouping fits one chunk is transposed once and kept."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    assert nv._dual_arr is None  # not built at construction

    rng = np.random.default_rng(12)
    B = rng.normal(size=(dense.shape[1], 3))
    Bl = rng.normal(size=(3, dense.shape[0]))
    nv @ B  # major-aligned for VCSR self@B; misaligned for VCSC
    Bl @ nv  # major-aligned for VCSC B@self; misaligned for VCSR

    dual_after_matmul = nv._dual_arr
    assert dual_after_matmul is not None
    assert dual_after_matmul._format != v._format

    nv @ B
    Bl @ nv
    assert nv._dual_arr is dual_after_matmul  # reused, not rebuilt


def test_transpose_major_roundtrip(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    dual = v._transpose_major()
    assert dual.shape == v.shape
    assert dual._format != v._format
    np.testing.assert_allclose(dual.toarray(), dense)
