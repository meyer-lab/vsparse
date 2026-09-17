"""Misaligned-direction normalized-view matmul: native minor-axis kernel.

Replaces the old chunked-transpose/dual-copy approach (see
``vsparse._vcs_matmul``): the misaligned direction now walks the array's own
storage directly with per-thread private accumulators, never building a
second copy of the sparse structure or regrouping any part of it into the
other VCS format.
"""

from __future__ import annotations


import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray
from vsparse._ops import accumulator_threads


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _reference(dense: np.ndarray) -> np.ndarray:
    row_totals = dense.sum(axis=1)
    row_scale = row_totals / np.median(row_totals)
    row_scale[row_scale == 0.0] = 1.0
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


def test_misaligned_matmul_matches_reference(dense, vcls):
    """Both matmul directions match a dense reference, whichever direction is misaligned."""
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    rng = np.random.default_rng(7)
    B = rng.normal(size=(dense.shape[1], 3))
    Bl = rng.normal(size=(2, dense.shape[0]))

    np.testing.assert_allclose(nv @ B, ref @ B, atol=1e-7)
    np.testing.assert_allclose(Bl @ nv, Bl @ ref, atol=1e-7)
    np.testing.assert_allclose(nv @ B[:, 0], ref @ B[:, 0], atol=1e-7)
    np.testing.assert_allclose(Bl[0] @ nv, Bl[0] @ ref, atol=1e-7)


def test_matmul_result_is_accumulator_thread_count_invariant(monkeypatch, vcls, rng):
    """Same answer regardless of how many private accumulators the kernel splits work across."""
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(0, 5, size=(40, 24)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    ref = _reference(dense)
    B = rng.normal(size=(dense.shape[1], 3))
    Bl = rng.normal(size=(2, dense.shape[0]))
    nv = v.normalized()

    for forced_threads in (1, 2, 8):
        monkeypatch.setattr(
            vcs_matmul, "accumulator_threads", lambda *a, _n=forced_threads, **k: _n
        )
        np.testing.assert_allclose(nv @ B, ref @ B, atol=1e-7)
        np.testing.assert_allclose(Bl @ nv, Bl @ ref, atol=1e-7)


def test_no_second_copy_of_the_array_is_built(vcls, rng):
    """The normalized view carries nothing beyond the original array and O(n_rows+n_cols) stats."""
    dense = rng.integers(1, 5, size=(30, 20)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()

    B = rng.normal(size=(dense.shape[1], 2))
    Bl = rng.normal(size=(2, dense.shape[0]))
    nv @ B
    Bl @ nv

    assert not hasattr(nv, "_dual_arr")
    # The wrapped array's own buffers are untouched/unreplaced by any matmul call.
    assert nv._arr is v


@pytest.fixture(scope="module")
def misaligned_setup():
    """A 1_200 x 400 VCSC view and a width-2 operand, JIT already warmed.

    In a fixture so that neither the array nor the one-off compilation counts
    against the ceiling below -- `limit_memory` measures the test body only.
    """
    rng = np.random.default_rng(0)
    dense = rng.integers(1, 5, size=(1_200, 400)).astype(np.float64)
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    B = rng.normal(size=(dense.shape[1], 2))
    v.normalized() @ B  # warm up the JIT
    return v, B, dense


# 1_200 x 400 with no zeros is 480_000 nonzeros: 3.8 MB of values and 1.9 MB of
# indices. The accumulator block is `MEMORY_TEST_THREADS * 1_200 * 2 * 8` =
# 77 KB, plus the 19 KB output. A ceiling of 256 KB is ~2x that and ~15x under
# the values array, so it fails if this path ever goes back to building a
# second copy of the array and passes on any runner.
@pytest.mark.limit_memory("256 KB")
def test_misaligned_matmul_peak_is_bounded_by_the_accumulator_budget(
    pinned_threads, misaligned_setup
):
    """Peak memory tracks the (fixed) accumulator budget, not the size of the array."""
    v, B, _ = misaligned_setup
    out = v.normalized() @ B
    assert out.shape == (1_200, 2)


def test_misaligned_matmul_result_is_still_correct(misaligned_setup):
    """The bounded-memory path above must also produce the right numbers."""
    v, B, dense = misaligned_setup
    np.testing.assert_allclose(v.normalized() @ B, _reference(dense) @ B, atol=1e-7)


def test_accumulator_threads_degrades_to_one_for_a_huge_output_axis():
    """A huge output axis (e.g. millions of cells) must not blow the accumulator budget."""
    huge_axis = 2_000_000
    wide_b = 200  # e.g. rank + oversampling in a randomized SVD
    assert accumulator_threads(huge_axis, bytes_per_element=8 * wide_b) == 1


# `limit_leaks` fails when any single call stack still holds memory once the
# body returns, which is the shape of a cache that grows with use rather than
# of a big one-off allocation -- the bug d286cf3 fixed, where the normalization
# cache pinned O(nnz) duals. `limit_memory` cannot see it: each pass on its own
# stays under any sane ceiling, and only the accumulation across passes is
# wrong. It traces native stacks for every allocation and so is markedly
# slower than the ceilings above, which is why there is one of these and not
# one per operation.
@pytest.mark.limit_leaks("128 KB")
def test_repeated_matmul_on_one_view_retains_nothing(pinned_threads, misaligned_setup):
    """Iterating on a view must not accumulate: every pass frees what it took."""
    v, B, _ = misaligned_setup
    nv = v.normalized()
    for _ in range(8):
        nv @ B
