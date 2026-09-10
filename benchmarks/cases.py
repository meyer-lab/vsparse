from __future__ import annotations

from collections.abc import Callable

import numpy as np

from benchmarks.harness import (
    integer_counts_csr,
    peak_alloc_mb,
    ratio_vs_scipy,
)

FAST: dict[str, Callable[[], dict[str, float]]] = {}
SLOW: dict[str, Callable[[], dict[str, float]]] = {}


def fast(fn):
    FAST[fn.__name__] = fn
    return fn


def slow(fn):
    SLOW[fn.__name__] = fn
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


# -- memory ceilings ---------------------------------------------------------


@fast
def minor_sum_peak_mb() -> dict[str, float]:
    """Memory allocated by a minor-axis sum."""
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    nnz_mb = v.nnz * 8 / 1e6
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.sum(axis=0)),
        "expanded_nnz_mb": nnz_mb,  # what a per-nonzero temporary would cost
    }


@fast
def misaligned_matmul_peak_mb() -> dict[str, float]:
    """Memory allocated by the matmul direction the storage isn't aligned for."""
    from vsparse import VCSCArray

    v = VCSCArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    rng = np.random.default_rng(0)
    B = rng.normal(size=(v.shape[1], 4))
    array_mb = (v.values.nbytes + v.value_ptr.nbytes + v.indices.nbytes) / 1e6

    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.normalized() @ B),
        "array_mb": array_mb,  # what a full second copy would cost
    }


@fast
def minor_extrema_peak_mb() -> dict[str, float]:
    """Memory allocated by a minor-axis max/min."""
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.max(axis=0)),
        "expanded_nnz_mb": v.nnz * 8 / 1e6,
    }


@fast
def minor_getnnz_peak_mb() -> dict[str, float]:
    """Memory allocated by a per-minor-index stored-element count."""
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.getnnz(axis=0)),
        "indices_nnz_mb": v.nnz * 8 / 1e6,
    }


@fast
def minor_selection_peak_mb() -> dict[str, float]:
    """Memory allocated by a minor-axis selection."""
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    cols = np.arange(0, v.shape[1], 2)
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v[:, cols]),
        "indices_nnz_mb": v.nnz * 8 / 1e6,
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
    return {
        "time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ B, lambda: csc @ B),
        "peak_alloc_mb": peak_alloc_mb(lambda: v @ B),
    }


@fast
def misaligned_rmatmat_vs_scipy() -> dict[str, float]:
    """``B @ VCSR``: same scatter, reached from the other side."""
    import scipy.sparse as sp

    from vsparse import VCSRArray

    mat = integer_counts_csr(60_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    csr = sp.csr_array(mat)
    B = np.random.default_rng(0).normal(size=(8, mat.shape[0]))
    return {
        "time_ratio_vs_scipy": ratio_vs_scipy(lambda: B @ v, lambda: B @ csr),
        "peak_alloc_mb": peak_alloc_mb(lambda: B @ v),
    }


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
        "matmul_peak_alloc_mb": peak_alloc_mb(lambda: v @ B),
    }


ALL: dict[str, Callable[[], dict[str, float]]] = {**FAST, **SLOW}
