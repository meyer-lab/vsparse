"""CUDA kernels against the CPU view they mirror.

Skipped wholesale without CuPy and a device. The GPU is float32 throughout
(:data:`vsparse._cuda.DEVICE_DTYPE`) while the host view is float64, so every
comparison here is a float32-scale tolerance, never an equality -- and in the
misaligned directions the kernels accumulate with ``atomicAdd``, so even two
GPU runs need not agree bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import RECIPES, VCSCArray, VCSRArray, cuda_is_available

pytestmark = pytest.mark.cuda

if not cuda_is_available():  # pragma: no cover - environment dependent
    pytest.skip("no CUDA device / CuPy available", allow_module_level=True)


#: Relative tolerance for a float32 product against the float64 reference.
#: The kernels sum ``nnz / n`` terms per output entry, so the bound is
#: float32 eps times a modest growth factor, not eps itself.
RTOL = 2e-4
ATOL = 2e-4


def counts(rng: np.random.Generator, shape: tuple[int, int], density: float = 0.4) -> sp.csr_array:
    """Integer counts with repeated values, so the layout actually dedupes."""
    dense = rng.integers(1, 8, size=shape).astype(np.float64)
    dense[rng.random(shape) > density] = 0.0
    return sp.csr_array(dense)


def assert_close(got, want) -> None:
    import cupy as cp

    got = cp.asnumpy(got)
    assert got.dtype == np.float32
    assert got.shape == want.shape
    # `scale` guards the empty case, where `.max()` has no identity.
    scale = max(1.0, float(np.abs(want).max())) if want.size else 1.0
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL * scale)


@pytest.fixture(scope="module")
def mat() -> sp.csr_array:
    return counts(np.random.default_rng(0), (300, 90))


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("width", [1, 3, 8, 32, 64, 130])
def test_matmul_matches_cpu(mat, recipe, cls, width):
    """``gpu @ B`` against ``cpu @ B``, in both formats and across the width chunking.

    ``width`` spans the lane-splitting cases: below a warp (lanes split across
    nonzeros), exactly a warp, and past ``ww * _ACC_MAX`` (more than one pass).
    """
    nv = cls.from_scipy(mat).normalized(recipe)
    B = np.random.default_rng(1).normal(size=(mat.shape[1], width))
    assert_close(nv.to_gpu() @ B, nv @ B)


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("width", [1, 3, 8, 32, 64, 130])
def test_rmatmul_matches_cpu(mat, recipe, cls, width):
    """``B @ gpu`` against ``B @ cpu``, in both formats and across the width chunking."""
    nv = cls.from_scipy(mat).normalized(recipe)
    B = np.random.default_rng(2).normal(size=(width, mat.shape[0]))
    assert_close(B @ nv.to_gpu(), B @ nv)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
def test_one_dimensional_operands_squeeze(mat, cls):
    """A 1-D operand comes back 1-D, on both sides, as it does on the host."""
    nv = cls.from_scipy(mat).normalized("parafac2")
    gpu = nv.to_gpu()
    rng = np.random.default_rng(3)

    x = rng.normal(size=mat.shape[1])
    got = gpu @ x
    assert got.ndim == 1
    assert_close(got, nv @ x)

    y = rng.normal(size=mat.shape[0])
    got = y @ gpu
    assert got.ndim == 1
    assert_close(got, y @ nv)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
def test_device_operand_is_accepted(mat, cls):
    """A CuPy operand is used as-is, with no host round-trip."""
    import cupy as cp

    nv = cls.from_scipy(mat).normalized("scanpy")
    B = np.random.default_rng(4).normal(size=(mat.shape[1], 8))
    assert_close(nv.to_gpu() @ cp.asarray(B, dtype=cp.float32), nv @ B)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
def test_shape_mismatch_raises(mat, cls):
    gpu = cls.from_scipy(mat).normalized("raw").to_gpu()
    with pytest.raises(ValueError, match="shape mismatch"):
        gpu @ np.zeros((mat.shape[1] + 1, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        np.zeros((4, mat.shape[0] + 1)) @ gpu


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
def test_to_cupy_sparse_reproduces_the_view(mat, recipe, cls):
    """The expanded ``Delta`` plus ``means`` is the same matrix the kernels imply.

    This is the baseline ``benchmarks/cases.py`` times the kernels against, so
    it has to compute the same thing rather than merely run.
    """
    import cupy as cp

    nv = cls.from_scipy(mat).normalized(recipe)
    gpu = nv.to_gpu()
    dense = cp.asnumpy(gpu.to_cupy_sparse().toarray()) - cp.asnumpy(gpu.means)[None, :]
    np.testing.assert_allclose(dense, nv.toarray(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
def test_transfer_keeps_the_compressed_layout(mat, cls):
    """``to_gpu`` uploads the VCS structure, not an expanded float-per-nonzero copy.

    The point of the device view: ``values`` stays ``n_unique``-long, so a
    matrix that only fits in device memory compressed still fits.
    """
    arr = cls.from_scipy(mat)
    gpu = arr.normalized("parafac2").to_gpu()
    assert gpu.values.shape[0] == arr.n_unique
    assert arr.n_unique < arr.nnz  # the fixture does dedupe, so this is a real check
    assert gpu.nnz == arr.nnz
    # An expanded CSR would need 4 bytes of data + 4 of indices per nonzero,
    # before either pointer array.
    assert gpu.nbytes < 8 * arr.nnz


def test_stats_are_float32_on_device(mat):
    """Nothing floating-point survives the transfer at float64."""
    import cupy as cp

    gpu = VCSRArray.from_scipy(mat).normalized("pearson").to_gpu()
    for name in ("values", "row_scale", "gene_scale", "col_mean", "col_post_scale"):
        assert getattr(gpu, name).dtype == cp.float32, name
    assert (
        VCSRArray.from_scipy(mat).normalized("raw").to_gpu() @ np.zeros(mat.shape[1])
    ).dtype == (cp.float32)


def test_empty_and_degenerate_columns_are_handled():
    """A zero column gives ``gene_scale == 0``; the kernels must skip, not divide."""
    dense = np.zeros((20, 6))
    dense[:, 1] = np.arange(20) % 4
    dense[:, 4] = 3.0
    nv = VCSRArray.from_scipy(sp.csr_array(dense)).normalized("parafac2")
    B = np.random.default_rng(5).normal(size=(6, 8))
    got = nv.to_gpu() @ B
    assert np.isfinite(np.asarray(got.get())).all()
    assert_close(got, nv @ B)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 1), (3, 3)])
@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("width", [0, 1, 5])
def test_degenerate_shapes_and_widths(shape, cls, width):
    """Shapes and operand widths that leave a kernel with nothing to do.

    A zero width makes the aligned kernels' chunk loop trip zero times, and a
    single-row or single-column matrix leaves the grid-stride loop with one
    slice for thousands of warps. Both should produce a correctly shaped,
    correct result rather than a launch error.
    """
    dense = np.zeros(shape)
    dense[0, 0] = 2.0
    nv = cls.from_scipy(sp.csr_array(dense)).normalized("parafac2")
    gpu = nv.to_gpu()

    assert_close(gpu @ np.ones((shape[1], width)), nv @ np.ones((shape[1], width)))
    assert_close(np.ones((width, shape[0])) @ gpu, np.ones((width, shape[0])) @ nv)
