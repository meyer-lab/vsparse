from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp


def peak_alloc_mb(fn: Callable[[], Any]) -> float:
    """Peak memory allocated during ``fn``, in MB.

    Not RSS, which is a process-lifetime high-water mark and so reports zero
    for anything staying under the peak set while building its input. numpy
    allocations are traced, numba's internal ones are not.
    """
    fn()  # JIT compile / warm caches outside the measurement
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return max(0.0, (peak - before) / 1e6)


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
