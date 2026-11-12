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
    # before either pointer array. Narrowed uint16 indices and int32 pointers
    # (two of them, at most one entry per nonzero) leave a good deal less.
    assert gpu.nbytes < 6 * arr.nnz


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
def test_device_structure_is_narrowed(mat, cls):
    gpu = cls.from_scipy(mat).normalized("raw").to_gpu()
    assert gpu.indices.dtype == np.uint16
    assert gpu.major_ptr.dtype == gpu.value_ptr.dtype == np.int32


@pytest.mark.parametrize(
    ("n_minor", "expected"),
    [(1, np.uint16), (2**16, np.uint16), (2**16 + 1, np.uint32)],
)
def test_index_dtype_boundary(n_minor, expected):
    from vsparse._cuda import _index_dtype

    assert _index_dtype(n_minor) == np.dtype(expected)


@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("n_minor", [2**16, 2**16 + 1])
def test_matmul_across_the_uint16_boundary(cls, n_minor):
    """Indices at 65535 (still uint16) and past it (uint32) both multiply correctly."""
    rng = np.random.default_rng(1)
    shape = (20, n_minor) if cls is VCSRArray else (n_minor, 20)
    dense = np.zeros(shape)
    flat = rng.choice(dense.size, size=400, replace=False)
    dense.flat[flat] = rng.integers(1, 5, size=400)
    # Make sure the very last minor index is occupied.
    if cls is VCSRArray:
        dense[0, -1] = 3.0
    else:
        dense[-1, 0] = 3.0
    nv = cls.from_scipy(sp.csr_array(dense)).normalized("raw")
    gpu = nv.to_gpu()
    assert gpu.indices.dtype == (np.uint16 if n_minor <= 2**16 else np.uint32)
    other = rng.standard_normal((shape[1], 5))
    assert_close(gpu @ other, nv @ other)
    left = rng.standard_normal((4, shape[0]))
    assert_close(left @ gpu, left @ nv)


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


# -- accumulator specialization and slice splitting ---------------------------
#
# The kernels are compiled once per (accumulators, chunked) pair, and a chunked
# warp walks only its share of a slice. The matrices in this file are far too
# small for `_split_factor` to ever choose a split by itself, so the tests below
# force one, and separately pin down what the heuristic itself decides.


@pytest.fixture
def forced_split(monkeypatch):
    """Make every launch cut each major slice into ``n`` chunks."""
    from vsparse import _cuda

    def force(n: int) -> None:
        monkeypatch.setattr(_cuda, "_split_factor", lambda *_args: n)

    return force


def skewed_counts() -> sp.csr_array:
    """Slices whose runs and sizes are as uneven as chunk bounds can be asked to handle.

    Row 0 is dense and all one value (a single run the whole width of the
    matrix), row 1 is empty, row 2 holds one entry, row 3 is two runs of very
    different length, and the rest are sparse random counts. Column 5 is empty
    too, so both layouts have empty and one-entry slices.
    """
    rng = np.random.default_rng(7)
    dense = rng.integers(1, 4, size=(40, 30)).astype(np.float64)
    dense[rng.random(dense.shape) > 0.15] = 0.0
    dense[0, :] = 1.0
    dense[1, :] = 0.0
    dense[2, :] = 0.0
    dense[2, 17] = 5.0
    dense[3, :] = 1.0
    dense[3, :4] = 2.0
    dense[:, 5] = 0.0
    return sp.csr_array(dense)


@pytest.mark.parametrize("split", [2, 3, 7, 64])
@pytest.mark.parametrize("cls", [VCSRArray, VCSCArray])
@pytest.mark.parametrize("width", [1, 4, 8, 12, 32, 33, 65, 97, 130, 256])
@pytest.mark.parametrize("matrix", ["random", "skewed"])
def test_split_slices_match_cpu(forced_split, mat, matrix, cls, width, split):
    """Chunked kernels agree with the host in all four directions, at every accumulator count.

    ``width`` covers each accumulator count (1 up to 4, and 130 for more than
    one pass); ``split`` covers a handful of chunks and far more chunks than a
    slice has nonzeros, where most chunks are empty. The skewed matrix puts the
    chunk boundaries inside a single long run, on an empty slice and on a
    one-entry slice.
    """
    a = mat if matrix == "random" else skewed_counts()
    forced_split(split)
    nv = cls.from_scipy(a).normalized("parafac2")
    gpu = nv.to_gpu()
    rng = np.random.default_rng(1)

    B = rng.normal(size=(a.shape[1], width))
    assert_close(gpu @ B, nv @ B)
    C = rng.normal(size=(width, a.shape[0]))
    assert_close(C @ gpu, C @ nv)


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_split_slices_cover_every_recipe(forced_split, mat, recipe):
    """The chunked walk computes ``Delta`` the same way for every ``g``."""
    forced_split(4)
    for cls in (VCSRArray, VCSCArray):
        nv = cls.from_scipy(mat).normalized(recipe)
        B = np.random.default_rng(1).normal(size=(mat.shape[1], 16))
        assert_close(nv.to_gpu() @ B, nv @ B)


def test_the_heuristic_splits_a_matrix_with_few_slices():
    """No forcing: a tall, narrow VCSC is split by ``_split_factor`` itself, and still correct."""
    from vsparse import _cuda

    rng = np.random.default_rng(3)
    a = counts(rng, (20_000, 64), density=0.3)
    nv = VCSCArray.from_scipy(a).normalized("parafac2")
    gpu = nv.to_gpu()
    assert _cuda._launch_config(64, 16, gpu.nnz, gpu.sm_count).chunked

    B = rng.normal(size=(a.shape[1], 16))
    assert_close(gpu @ B, nv @ B)  # misaligned: split chunks atomicAdd into one output
    C = rng.normal(size=(16, a.shape[0]))
    assert_close(C @ gpu, C @ nv)  # aligned: split chunks write partial rows, then reduce


@pytest.mark.parametrize(("cls", "side"), [(VCSRArray, "right"), (VCSCArray, "left")])
def test_aligned_direction_stays_deterministic_when_split(forced_split, mat, cls, side):
    """The aligned kernels' documented determinism survives splitting.

    Partial rows are summed in a fixed order afterwards rather than
    ``atomicAdd``ed, so two runs agree to the bit.
    """
    forced_split(5)
    gpu = cls.from_scipy(mat).normalized("parafac2").to_gpu()
    rng = np.random.default_rng(6)
    if side == "right":  # VCSR @ B is aligned
        B = rng.normal(size=(mat.shape[1], 16))
        first, second = gpu @ B, gpu @ B
    else:  # B @ VCSC is aligned
        B = rng.normal(size=(16, mat.shape[0]))
        first, second = B @ gpu, B @ gpu
    assert np.array_equal(first.get(), second.get())


@pytest.mark.parametrize(
    ("width", "acc"),
    [
        (1, 1),
        (3, 1),
        (8, 1),
        (16, 1),
        (32, 1),
        (33, 2),
        (64, 2),
        (65, 3),
        (96, 3),
        (97, 4),
        (130, 4),
    ],
)
def test_accumulator_count_follows_the_width(width, acc):
    """A lane gets just enough accumulators for one pass, capped at ``_ACC_MAX``."""
    from vsparse import _cuda

    assert _cuda._launch_config(1000, width, 10**6, 84).acc == acc


def test_split_factor_decisions():
    """When the heuristic splits, and when it declines."""
    from vsparse import _cuda

    plan = _cuda._launch_config
    full, nnz = 60_000, 6_000_000
    assert plan(full, 16, nnz, 84).split == 1  # plenty of slices already
    few = plan(2_000, 16, nnz, 84)
    assert 1 < few.split <= _cuda._MAX_SPLIT  # starved grid: split
    assert (
        plan(2_000, 4, nnz, 84).split == 1
    )  # narrow width: lane-groups already supply the parallelism
    assert plan(2_000, 16, 2_000 * 10, 84).split == 1  # slices too short to cut
    assert plan(0, 16, 0, 84).split == 1  # nothing to split
    assert few.chunked and not plan(full, 16, nnz, 84).chunked
    # The grid always covers every (slice, chunk) pair, up to the block cap.
    assert few.grid[0] * few.block[1] >= min(2_000 * few.split, _cuda._MAX_BLOCKS * few.block[1])


@pytest.mark.parametrize(
    ("width", "vec", "ww", "acc"),
    [(1, 1, 1, 1), (6, 1, 8, 1), (4, 4, 1, 1), (8, 4, 2, 1), (16, 4, 4, 1), (128, 4, 32, 1),
     (256, 4, 32, 2), (512, 4, 32, 4), (130, 1, 32, 4), (33, 1, 32, 2)],
)  # fmt: skip
def test_aligned_launch_uses_float4_for_multiples_of_four(width, vec, ww, acc):
    """Aligned launches load ``float4`` when the width allows it; misaligned ones never do."""
    from vsparse import _cuda

    cfg = _cuda._launch_config(1000, width, 10**6, 84, aligned=True)
    assert (cfg.vec, cfg.ww, cfg.acc) == (vec, ww, acc)
    assert _cuda._launch_config(1000, width, 10**6, 84).vec == 1


def test_vector_loads_do_not_change_when_a_narrow_matrix_splits():
    """The split threshold counts columns, not lanes: width 8 splits as scalar or float4."""
    from vsparse import _cuda

    nnz = 6_000_000
    assert _cuda._launch_config(2_000, 8, nnz, 84, aligned=True).split > 1
    assert _cuda._launch_config(2_000, 4, nnz, 84, aligned=True).split == 1
