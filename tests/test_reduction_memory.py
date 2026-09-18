"""Minor-axis reductions: correctness, and ceilings on what they allocate.

The ``pytest-memray`` ceilings here assert a structural property -- a reduction
producing an ``n_minor``-sized result must not allocate anything that grows
with ``nnz``. They rely on ``pinned_threads`` to fix the thread count the
accumulators are sized by, and on module-scoped fixtures for the inputs, since
a ``limit_memory`` mark measures the test body alone.
"""

from __future__ import annotations

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray
from vsparse._ops import _ACCUMULATOR_BUDGET_BYTES, accumulator_threads


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _minor_axis(vcls):
    return 1 if vcls is VCSCArray else 0


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum_matches_dense(dense, vcls, axis):
    """Totals along each axis, and overall."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(v.sum(axis=axis), dense.sum(axis=axis))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_extrema_match_dense(dense, vcls, axis, kind):
    """Extrema have to fold in the implicit zeros the layout never stores."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(getattr(v, kind)(axis=axis), getattr(dense, kind)(axis=axis))


@pytest.mark.parametrize("axis", [0, 1])
def test_getnnz_matches_dense(dense, vcls, axis):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_array_equal(v.getnnz(axis=axis), (dense != 0).sum(axis=axis))


def test_reductions_on_an_all_zero_array(vcls):
    """Every entry is an implicit zero, so the stored values are empty."""
    v = vcls.from_scipy(_scipy_for(vcls, np.zeros((4, 3))))
    np.testing.assert_allclose(v.max(axis=0), np.zeros(3))
    np.testing.assert_allclose(v.min(axis=0), np.zeros(3))
    np.testing.assert_allclose(v.sum(axis=0), np.zeros(3))
    np.testing.assert_array_equal(v.getnnz(axis=0), np.zeros(3, dtype=np.int64))


def test_integer_values_use_integer_sentinels(vcls):
    """A float sentinel would make an integer max come back wrong or upcast."""
    dense = np.array([[7, 0, 2], [0, 3, 9]], dtype=np.int32)
    v = vcls.from_scipy(_scipy_for(vcls, dense.astype(np.float64)))
    np.testing.assert_array_equal(v.max(axis=0), dense.max(axis=0))
    np.testing.assert_array_equal(v.min(axis=0), dense.min(axis=0))


def test_sums_accumulate_in_float64(vcls):
    """Accumulating in the stored dtype would overflow or lose precision."""
    v = vcls.from_scipy(_scipy_for(vcls, np.array([[1.0, 0.0, 2.0], [3.0, 4.0, 0.0]])))
    assert v._minor_sums().dtype == np.float64


@pytest.mark.parametrize("n_minor", [0, 1_000, 10**8])
@pytest.mark.parametrize("bytes_per_element", [8, 16])
def test_accumulator_block_stays_within_budget(n_minor, bytes_per_element):
    """Thread-local accumulators must not become the new unbounded allocation."""
    nthreads = accumulator_threads(n_minor, bytes_per_element)
    assert 1 <= nthreads <= numba.get_num_threads()
    if nthreads > 1:
        assert nthreads * n_minor * bytes_per_element <= _ACCUMULATOR_BUDGET_BYTES


@pytest.fixture(scope="module")
def reduction_array():
    """A 2_000 x 500 array with every reduction's JIT warmed, built outside the body."""
    rng = np.random.default_rng(0)
    dense = rng.integers(1, 5, size=(2_000, 500)).astype(np.float64)
    v = VCSRArray.from_scipy(sp.csr_array(dense))
    for warm in (v.sum, v.max, v.getnnz):
        warm(axis=0)
    return v


# 1e6 nonzeros here: 8 MB of values, 4 MB of indices. The accumulator block is
# `MEMORY_TEST_THREADS * 500 * 8` = 16 KB, doubled for the extrema kernels,
# which carry two. The ceilings sit ~2x over that and orders of magnitude under
# nnz-scale.
#
# Marks go on each `pytest.param`: pytest-memray reads them at collection, so
# one applied from the test body is never seen and asserts nothing.
@pytest.mark.parametrize(
    ("label", "call"),
    [
        pytest.param("sum", lambda v: v.sum(axis=0), marks=pytest.mark.limit_memory("64 KB")),
        pytest.param("max", lambda v: v.max(axis=0), marks=pytest.mark.limit_memory("96 KB")),
        pytest.param("getnnz", lambda v: v.getnnz(axis=0), marks=pytest.mark.limit_memory("64 KB")),
    ],
)
def test_minor_axis_reductions_allocate_nothing_nnz_sized(
    pinned_threads, reduction_array, label, call
):
    """An n_minor-sized result must not cost nnz-sized scratch."""
    out = call(reduction_array)
    assert out.shape == (500,)


@pytest.fixture(scope="module")
def selection_array():
    """A 2_000 x 500 array with the selection kernels' JIT warmed."""
    rng = np.random.default_rng(0)
    dense = rng.integers(1, 5, size=(2_000, 500)).astype(np.float64)
    v = VCSRArray.from_scipy(sp.csr_array(dense))
    v[:, np.arange(0, 500, 2)]
    return v


# A selection's result does grow with what was selected: half of 1e6 nonzeros
# is ~2.3 MB of output, which is the answer rather than scratch. The ceiling
# sits above that and well below output-plus-an-nnz-sized-temporary.
@pytest.mark.limit_memory("4 MB")
def test_minor_axis_selection_allocates_no_nnz_sized_scratch(pinned_threads, selection_array):
    """A selection may pay for its output, but not for a copy of the input."""
    out = selection_array[:, np.arange(0, 500, 2)]
    assert out.shape == (2_000, 250)
