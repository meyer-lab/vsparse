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


@pytest.fixture
def ivcs_file(tmp_path):
    """Write a dense matrix to a packed ``.h5ad`` and hand back the path.

    ``write(dense, format="csr", obs=..., var=...)``. The streaming readers are
    tested against ``load_packed`` on the same file, so the fixture exists to
    keep the two sides from drifting apart in how the file was written.
    """
    import anndata as ad

    from vsparse import VCSCAnnData

    count = 0

    def write(dense, *, format="csr", obs=None, var=None, name=None):
        nonlocal count
        count += 1
        adata = ad.AnnData(X=sp.csr_array(dense))
        if obs is not None:
            adata.obs = obs.set_index(adata.obs.index)
        if var is not None:
            adata.var = var.set_index(adata.var.index)
        path = tmp_path / (name or f"data{count}.{format}.h5ad")
        VCSCAnnData.from_anndata(adata, format=format).write_h5ad(path)
        return path

    return write


#: Threads the memory-limited tests run their kernels with. Their accumulators
#: are ``nthreads * n_minor * width * 8`` bytes, so an unpinned count would make
#: the same ceiling mean something different on every machine.
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
