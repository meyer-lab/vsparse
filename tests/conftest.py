# ruff: noqa: I001 -- the `import vsparse` below must run first, deliberately
# ahead of the numpy/pytest/scipy import block isort would otherwise merge it
# into (see the comment on that import).
from __future__ import annotations

# Import before anything below (or in any test module) can import `anndata`/
# `zarr` on its own: `zarr.core.buffer.gpu` speculatively imports `cupy`, and
# CuPy's CUDA-toolkit-header discovery misbehaves for the rest of the process
# if cupy's first-ever import in the process happens that way rather than via
# `vsparse` (see `vsparse/__init__.py`'s import-order comment, and
# `pyproject.toml`'s `-p no:zarr` addopt for the pytest-plugin-autoload half
# of this same issue). conftest.py is collected before any test module, so
# this covers every test file regardless of its own import order.
import vsparse  # noqa: F401

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
