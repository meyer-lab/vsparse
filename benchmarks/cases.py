from __future__ import annotations

import gc
from collections.abc import Callable

import numpy as np

from benchmarks.harness import (
    best_cpu_time,
    best_gpu_time,
    best_time,
    integer_counts_csr,
    ratio_vs_scipy,
)

FAST: dict[str, Callable[[], dict[str, float]]] = {}
SLOW: dict[str, Callable[[], dict[str, float]]] = {}
CUDA: dict[str, Callable[[], dict[str, float]]] = {}


def fast(fn):
    FAST[fn.__name__] = fn
    return fn


def slow(fn):
    SLOW[fn.__name__] = fn
    return fn


def cuda(fn):
    CUDA[fn.__name__] = fn
    return fn


# -- layout size -------------------------------------------------------------


@fast
def layout_bytes_per_nonzero() -> dict[str, float]:
    """Bytes the layout holds per stored nonzero, and the same for ``indices`` alone."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(20_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    stored = v.major_ptr.nbytes + v.values.nbytes + v.value_ptr.nbytes + v.indices.nbytes
    scipy_bytes = mat.indptr.nbytes + mat.indices.nbytes + mat.data.nbytes
    return {
        "bytes_per_nonzero": stored / v.nnz,
        "indices_bytes_per_nonzero": v.indices.nbytes / v.nnz,
        "vs_scipy_ratio": stored / scipy_bytes,
    }


# -- throughput, relative to scipy -------------------------------------------


@fast
def matvec_vs_scipy() -> dict[str, float]:
    """Aligned-direction matrix-vector product, against scipy's CSR."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(20_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    x = rng.normal(size=mat.shape[1])
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ x, lambda: mat @ x)}


@fast
def matmat_vs_scipy() -> dict[str, float]:
    """Aligned-direction matrix-matrix product, against scipy's CSR."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(20_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(mat.shape[1], 8))
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ B, lambda: mat @ B)}


# -- normalized views (issue #40 recipes): view-op vs materialize-then-op ---
#
# For every recipe, the view-based matmul/matvec should be faster than fully
# materializing the (dense, implicit-zero-filling) normalized matrix and
# multiplying that -- the whole point of a *view*.
# ``time_ratio_view_over_materialize`` < 1 is the expectation for every case
# below. The matching memory claim is asserted as a ceiling in
# ``tests/test_minor_axis_matmul.py`` rather than recorded here.


def _normalized_bench(recipe: str, *, vector: bool) -> Callable[[], dict[str, float]]:
    def bench() -> dict[str, float]:
        from vsparse import VCSRArray

        mat = integer_counts_csr(20_000, 2_000, density=0.05)
        v = VCSRArray.from_scipy(mat)
        nv = v.normalized(recipe)
        rng = np.random.default_rng(0)
        B = rng.normal(size=mat.shape[1]) if vector else rng.normal(size=(mat.shape[1], 8))

        def via_view() -> np.ndarray:
            return nv @ B

        def via_materialize() -> np.ndarray:
            return nv.toarray() @ B

        return {
            "time_ratio_view_over_materialize": best_time(via_view) / best_time(via_materialize),
        }

    bench.__name__ = f"normalized_{recipe}_{'matvec' if vector else 'matmat'}_vs_materialize"
    return bench


def _normalized_rbench(recipe: str, *, vector: bool) -> Callable[[], dict[str, float]]:
    def bench() -> dict[str, float]:
        from vsparse import VCSCArray

        mat = integer_counts_csr(20_000, 2_000, density=0.05)
        v = VCSCArray.from_scipy(mat)
        nv = v.normalized(recipe)
        rng = np.random.default_rng(0)
        B = rng.normal(size=mat.shape[0]) if vector else rng.normal(size=(8, mat.shape[0]))

        def via_view() -> np.ndarray:
            return B @ nv

        def via_materialize() -> np.ndarray:
            return B @ nv.toarray()

        return {
            "time_ratio_view_over_materialize": best_time(via_view) / best_time(via_materialize),
        }

    bench.__name__ = f"normalized_{recipe}_{'rmatvec' if vector else 'rmatmat'}_vs_materialize"
    return bench


def _register_normalized_benchmarks() -> None:
    from vsparse import RECIPES

    for recipe in sorted(RECIPES):
        for vector in (False, True):
            fast(_normalized_bench(recipe, vector=vector))
            fast(_normalized_rbench(recipe, vector=vector))


_register_normalized_benchmarks()


# -- normalized views vs the *sparse* baseline -------------------------------
#
# The cases above compare against `nv.toarray() @ B`, a dense materialization
# nobody would actually perform -- it makes the view look good for a reason
# that has nothing to do with the kernels. The honest baseline is the one the
# prototype benchmarks used: build the sparse delta once, multiply that with
# scipy, and add the same rank-1 correction the view adds. Both sides then do
# identical arithmetic and the ratio isolates the kernel.
#
# CPU, not just wall, is the point here. Our kernels are `parallel=True` and
# scipy's are single-threaded, so wall time hides a per-nonzero cost that a
# shared workstation still pays -- and every recipe but "raw" applies a
# transcendental to every stored nonzero.


def _sparse_delta(nv, mat):
    """``Delta`` as scipy CSR: what ``to_csr()`` on the view would return.

    ``Delta[i, j] = s[j] * g(x[i, j] / row_scale[i] / gene_scale[j])`` on the
    stored nonzeros and exactly zero off them, so
    ``A_norm = Delta + 1 (x) (-c * s)``.
    """
    import scipy.sparse as sp

    from vsparse._norm_common import _g_np

    coo = mat.tocoo()
    gs = nv.gene_scale[coo.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(gs > 0.0, coo.data / nv.row_scale[coo.row] / gs, 0.0)
    data = nv.col_post_scale[coo.col] * _g_np(scaled, nv.recipe.g_code)
    return sp.csr_array((data, (coo.row, coo.col)), shape=mat.shape)


def _normalized_vs_sparse(recipe: str, *, vector: bool) -> Callable[[], dict[str, float]]:
    def bench() -> dict[str, float]:
        from vsparse import VCSRArray

        mat = integer_counts_csr(20_000, 2_000, density=0.05)
        v = VCSRArray.from_scipy(mat)
        nv = v.normalized(recipe)
        rng = np.random.default_rng(0)
        B = rng.normal(size=mat.shape[1]) if vector else rng.normal(size=(mat.shape[1], 8))

        delta = _sparse_delta(nv, mat)
        offset = -(nv.col_mean * nv.col_post_scale)

        def via_view() -> np.ndarray:
            return nv @ B

        def via_sparse() -> np.ndarray:
            return delta @ B + (offset @ B)

        np.testing.assert_allclose(via_view(), via_sparse(), rtol=1e-9, atol=1e-9)

        # Context, and core-count dependent: the kernel is `parallel=True`
        # against a single-threaded scipy. The gated, machine-portable CPU
        # comparison lives in `normalized_cpu_vs_sparse_1t` below, which is
        # also the only case here that pays for CPU timing -- doing it in
        # every case doubled the suite's runtime for a number nothing gates.
        return {
            "wall_ratio_view_over_sparse": best_time(via_view) / best_time(via_sparse),
        }

    bench.__name__ = f"normalized_{recipe}_{'matvec' if vector else 'matmat'}_vs_sparse"
    return bench


def _register_sparse_baseline_benchmarks() -> None:
    from vsparse import RECIPES

    for recipe in sorted(RECIPES):
        for vector in (False, True):
            fast(_normalized_vs_sparse(recipe, vector=vector))


_register_sparse_baseline_benchmarks()


@fast
def normalized_cpu_vs_sparse_1t() -> dict[str, float]:
    """Per-nonzero CPU cost of each recipe's kernel, against the same math in scipy.

    Runs with numba pinned to one thread -- `benchmarks.run` gives any case
    whose name ends in `_1t` a `NUMBA_NUM_THREADS=1` subprocess, which has to
    happen before numba is imported. Capping the pool from inside the process
    is not enough: the idle workers still spin, and `process_time()` counts
    every thread, which made the ratio swing by 2x between runs.

    Pinned, both sides are single-threaded and the ratio isolates what a
    nonzero costs us over scipy -- almost entirely `g`. It is stable to well
    under a percent between runs and does not depend on the runner's core
    count, which is what makes it gateable.
    """
    from vsparse import RECIPES, VCSRArray

    # Deliberately larger than the cases above (6M nonzeros, not 2M). Both
    # sides here land within a small multiple of each other, so the ratio only
    # settles once each measurement is long enough to swamp scheduling jitter.
    mat = integer_counts_csr(60_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(mat.shape[1], 8))

    out: dict[str, float] = {}
    for recipe in sorted(RECIPES):
        # Each recipe builds its own 6M-nonzero delta (~150 MB). Left to the
        # allocator, the recipes measured last ran against a progressively more
        # fragmented heap and their ratios swung by 2x; dropping the previous
        # one first keeps every recipe on the same footing.
        gc.collect()
        nv = v.normalized(recipe)
        delta = _sparse_delta(nv, mat)
        offset = -(nv.col_mean * nv.col_post_scale)

        def via_view(nv=nv):
            return nv @ B

        def via_sparse(delta=delta, offset=offset):
            return delta @ B + (offset @ B)

        np.testing.assert_allclose(via_view(), via_sparse(), rtol=1e-9, atol=1e-9)
        # repeat=15: the identity-transform kernels land within ~1.5x of scipy,
        # so at the default repeat count run-to-run jitter was a large share of
        # the ratio and the gate was not reproducible across processes.
        out[f"cpu_ratio_1t_{recipe}"] = best_cpu_time(via_view, repeat=15) / best_cpu_time(
            via_sparse, repeat=15
        )
        del nv, delta, via_view, via_sparse

    return out


# -- misaligned direction of the *core* matmul path --------------------------
#
# `matvec_vs_scipy`/`matmat_vs_scipy` above run VCSR in its aligned direction.
# These run the other one -- the scatter path in `_ops` -- which is where the
# array's layout works against the product and where the serial kernel used to
# lose to scipy by up to 4x.


@fast
def misaligned_matvec_vs_scipy() -> dict[str, float]:
    """``VCSC @ x``: iterate columns, scatter into rows."""
    import scipy.sparse as sp

    from vsparse import VCSCArray

    mat = integer_counts_csr(60_000, 2_000, density=0.05)
    v = VCSCArray.from_scipy(mat)
    csc = sp.csc_array(mat)
    x = np.random.default_rng(0).normal(size=mat.shape[1])
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ x, lambda: csc @ x)}


@fast
def misaligned_matmat_vs_scipy() -> dict[str, float]:
    """``VCSC @ B``, width 8 -- the case where the accumulator is widest."""
    import scipy.sparse as sp

    from vsparse import VCSCArray

    mat = integer_counts_csr(60_000, 2_000, density=0.05)
    v = VCSCArray.from_scipy(mat)
    csc = sp.csc_array(mat)
    B = np.random.default_rng(0).normal(size=(mat.shape[1], 8))
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ B, lambda: csc @ B)}


@fast
def misaligned_rmatmat_vs_scipy() -> dict[str, float]:
    """``B @ VCSR``: same scatter, reached from the other side."""
    import scipy.sparse as sp

    from vsparse import VCSRArray

    mat = integer_counts_csr(60_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    csr = sp.csr_array(mat)
    B = np.random.default_rng(0).normal(size=(8, mat.shape[0]))
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: B @ v, lambda: B @ csr)}


# -- larger, for the scheduled job -------------------------------------------


@slow
def large_layout_and_matmul() -> dict[str, float]:
    """The same measurements an order of magnitude up."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(200_000, 3_000, density=0.02)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(mat.shape[1], 8))
    stored = v.major_ptr.nbytes + v.values.nbytes + v.value_ptr.nbytes + v.indices.nbytes
    v @ B
    return {
        "bytes_per_nonzero": stored / v.nnz,
        "time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ B, lambda: mat @ B),
    }


# -- CUDA kernels vs materializing to a sparse matrix ------------------------
#
# The question `vsparse._cuda` exists to answer: is walking the
# value-compressed layout on the device with our own kernels actually
# competitive with expanding it to one float per nonzero and handing the
# product to cuSPARSE -- which is what `to_cupy_sparse()` does, and the device
# twin of the `to_scipy_sparse()` path a caller would otherwise take.
#
# Two numbers matter and they pull opposite ways:
#
# `cuda_time_ratio_kernel_over_csr` -- steady-state throughput, our kernel over
# cuSPARSE on the already-built CSR. A ratio above 1 would be defensible: we
# recompute `g` on every nonzero at every call where the CSR baked it in once,
# against a heavily tuned library. In practice every direction measures below
# 1, because the kernel reads ~40% less memory (the layout is the whole point)
# and this is a bandwidth-bound problem -- and `dense @ sparse` is a shape
# cuSPARSE handles particularly poorly, where we measured 0.17x-0.32x.
#
# `cuda_device_bytes_ratio_vs_csr` -- device memory held, ours over the CSR's.
# This is what the layout buys, it is deterministic, and it is the reason the
# throughput comes out where it does, so unlike the timing it is gated tightly.
#
# The one-time materialization is reported as `cuda_materialize_in_matmuls`:
# how many of our matmuls the caller pays up front to reach the CSR baseline at
# all. These run only on a GPU runner (`--set cuda`).

_CUDA_ROWS, _CUDA_COLS, _CUDA_WIDTH = 60_000, 2_000, 16


def _cuda_setup(cls_name: str, recipe: str):
    """``(gpu_view, csr, means, mat)`` for a CUDA case.

    The baseline is deliberately the *best* materialized option, not the
    cheapest one to produce, so the ratio is not a strawman: CSR in every case
    (including for a VCSC-backed view, whose natural materialization is CSC --
    `B @ csc` measured ~5x `B @ csr`), and index-sorted, which
    `to_cupy_sparse` does by default (unsorted measured ~6x slower again).
    `cuda_materialize_in_matmuls` times that same full call, so both the
    conversion and the sort are counted where a caller would pay them.
    """
    import vsparse

    mat = integer_counts_csr(_CUDA_ROWS, _CUDA_COLS, density=0.05)
    nv = getattr(vsparse, cls_name).from_scipy(mat).normalized(recipe)
    gpu = nv.to_gpu()
    return gpu, gpu.to_cupy_sparse(format="csr"), gpu.means, mat


def _csr_device_bytes(csr) -> int:
    return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)


def _cuda_metrics(gpu, csr, via_kernel, via_csr) -> dict[str, float]:
    """Time both sides, check they agree, and report the three metrics."""
    import cupy as cp

    cp.testing.assert_allclose(via_kernel(), via_csr(), rtol=2e-4, atol=2e-4)
    kernel_time = best_gpu_time(via_kernel)
    return {
        "cuda_time_ratio_kernel_over_csr": kernel_time / best_gpu_time(via_csr),
        "cuda_device_bytes_ratio_vs_csr": gpu.nbytes / _csr_device_bytes(csr),
        "cuda_materialize_in_matmuls": best_gpu_time(lambda: gpu.to_cupy_sparse(format="csr"))
        / kernel_time,
    }


def _cuda_matmul_case(cls_name: str, recipe: str) -> Callable[[], dict[str, float]]:
    """``gpu @ B`` against ``csr @ B`` plus the same rank-1 correction."""

    def bench() -> dict[str, float]:
        import cupy as cp

        gpu, csr, means, mat = _cuda_setup(cls_name, recipe)
        B = cp.asarray(
            np.random.default_rng(0).normal(size=(mat.shape[1], _CUDA_WIDTH)), dtype=cp.float32
        )
        return _cuda_metrics(gpu, csr, lambda: gpu @ B, lambda: csr @ B + (-means) @ B)

    direction = "matmul" if cls_name == "VCSRArray" else "matmul_misaligned"
    bench.__name__ = f"cuda_{recipe}_{direction}_vs_csr"
    return bench


def _cuda_rmatmul_case(cls_name: str, recipe: str) -> Callable[[], dict[str, float]]:
    """``B @ gpu`` against ``B @ csr`` plus the same rank-1 correction."""

    def bench() -> dict[str, float]:
        import cupy as cp

        gpu, csr, means, mat = _cuda_setup(cls_name, recipe)
        B = cp.asarray(
            np.random.default_rng(0).normal(size=(_CUDA_WIDTH, mat.shape[0])), dtype=cp.float32
        )
        correction = B.sum(axis=1)[:, None] * (-means)[None, :]
        return _cuda_metrics(gpu, csr, lambda: B @ gpu, lambda: B @ csr + correction)

    direction = "rmatmul" if cls_name == "VCSCArray" else "rmatmul_misaligned"
    bench.__name__ = f"cuda_{recipe}_{direction}_vs_csr"
    return bench


def _register_cuda_benchmarks() -> None:
    from vsparse import RECIPES

    # Every recipe in the aligned direction, since the recipes differ only in
    # `g` and that is a per-nonzero cost the kernel pays and the CSR does not.
    for recipe in sorted(RECIPES):
        cuda(_cuda_matmul_case("VCSRArray", recipe))
    # One recipe is enough for the other three directions: they exercise the
    # kernel's structure (register accumulation vs atomicAdd), not `g`.
    cuda(_cuda_rmatmul_case("VCSCArray", "parafac2"))
    cuda(_cuda_matmul_case("VCSCArray", "parafac2"))
    cuda(_cuda_rmatmul_case("VCSRArray", "parafac2"))


_register_cuda_benchmarks()


ALL: dict[str, Callable[[], dict[str, float]]] = {**FAST, **SLOW, **CUDA}
