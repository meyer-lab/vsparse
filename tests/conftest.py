from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def make_dense(rng: np.random.Generator, shape: tuple[int, int], *, low=0, high=5) -> np.ndarray:
    """Integer-valued dense matrix with plenty of repeated values and some zeros."""
    dense = rng.integers(low, high, size=shape).astype(np.float64)
    mask = rng.random(shape) < 0.4
    dense[mask] = 0.0
    return dense


@pytest.fixture(params=[(1, 1), (5, 1), (1, 7), (8, 6), (25, 40), (50, 3)])
def shape(request) -> tuple[int, int]:
    return request.param


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def dense(rng, shape) -> np.ndarray:
    return make_dense(rng, shape)


@pytest.fixture
def csc(dense) -> sp.csc_array:
    return sp.csc_array(dense)


@pytest.fixture
def csr(dense) -> sp.csr_array:
    return sp.csr_array(dense)


#: Threads the memory-limited tests run their kernels with.
#:
#: The thread-local accumulators those tests bound are
#: ``nthreads * n_minor * width * 8`` bytes, so their size follows
#: ``numba.get_num_threads()`` -- which is a property of the machine, not of
#: the code under test. Left free, the same assertion would mean something
#: different on a 4-core runner than on a 96-core one, and a limit loose
#: enough for the widest machine would be too loose to catch a regression on
#: any of them. Pinning it makes the expected allocation a number the test
#: can actually state.
MEMORY_TEST_THREADS = 4


@pytest.fixture
def pinned_threads():
    """Pin numba's thread count so accumulator sizes are machine-independent."""
    import numba

    previous = numba.get_num_threads()
    numba.set_num_threads(MEMORY_TEST_THREADS)
    try:
        yield MEMORY_TEST_THREADS
    finally:
        numba.set_num_threads(previous)
