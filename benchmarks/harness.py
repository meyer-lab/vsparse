from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp


def best_time(fn: Callable[[], Any], repeat: int = 7) -> float:
    """Best wall-clock time over ``repeat`` runs, in seconds. Warms up first."""
    fn()  # JIT compile / allocate caches outside the measurement
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def best_cpu_time(fn: Callable[[], Any], repeat: int = 7) -> float:
    """Best *CPU* time over ``repeat`` runs, in seconds. Warms up first.

    Wall time alone flatters every numba kernel here: they are ``parallel=True``
    while scipy's sparse matmul is single-threaded, so a kernel can be several
    times faster on the clock while burning an order of magnitude more CPU.
    On a shared workstation -- the machine this project is aimed at -- that
    difference is what the user actually pays.
    """
    fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.process_time()
        fn()
        best = min(best, time.process_time() - start)
    return best


def ratio_vs_scipy(ours: Callable[[], Any], theirs: Callable[[], Any], repeat: int = 7) -> float:
    """``our time / scipy's time`` for the same work."""
    return best_time(ours, repeat) / best_time(theirs, repeat)


def integer_counts_csr(n_rows: int, n_cols: int, density: float, seed: int = 0) -> sp.csr_array:
    """Integer-valued sparse counts, repeated enough for the layout to dedupe."""
    rng = np.random.default_rng(seed)
    mat = sp.random_array((n_rows, n_cols), density=density, format="csr", random_state=seed)
    mat.data = np.round(rng.integers(1, 8, size=mat.data.shape[0])).astype(np.float64)
    return mat


def skewed_counts_csr(
    n_rows: int, n_cols: int, mean_nnz_per_row: int, seed: int = 0
) -> sp.csr_array:
    """Integer counts shaped like a real single-cell matrix, unlike the uniform one above.

    Gene popularity is Zipf-distributed (a few genes in most cells, most in
    few), cell depth is log-normal (heavy-tailed row lengths), and counts are
    drawn by multinomial-style sampling, so most nonzeros equal 1.
    """
    rng = np.random.default_rng(seed)
    gene_p = 1.0 / np.arange(1, n_cols + 1) ** 1.1
    cell_p = rng.lognormal(0.0, 0.8, n_rows)
    draws = int(n_rows * mean_nnz_per_row * 1.3)
    rows = rng.choice(n_rows, size=draws, p=cell_p / cell_p.sum())
    cols = rng.choice(n_cols, size=draws, p=gene_p / gene_p.sum())
    mat = sp.coo_array((np.ones(draws), (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    mat.sum_duplicates()
    return mat


def best_gpu_time(fn: Callable[[], Any], repeat: int = 7) -> float:
    """Best wall-clock time over ``repeat`` runs of a CUDA call, in seconds.

    CUDA launches are asynchronous, so timing one the way :func:`best_time`
    does measures the launch and not the work. This synchronizes on both sides
    of each run. The warm-up run also absorbs the one-time NVRTC compile of
    :mod:`vsparse._cuda`'s kernels and cuSPARSE's own first-call setup.
    """
    import cupy as cp

    device = cp.cuda.Device()
    fn()
    device.synchronize()
    best = float("inf")
    for _ in range(repeat):
        device.synchronize()
        start = time.perf_counter()
        fn()
        device.synchronize()
        best = min(best, time.perf_counter() - start)
    return best
