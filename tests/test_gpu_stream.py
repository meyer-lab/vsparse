"""GPU tests for block-streamed & VRAM-compressed IVCSR v2 views (#76)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSRArray, cuda_is_available, open_gpu_stream

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not cuda_is_available(), reason="CUDA device or CuPy not available"),
]


def _write_vcsr_h5ad(tmp_path, n_obs=300, n_vars=500, density=0.08, seed=0):
    rng = np.random.default_rng(seed)
    mat = sp.random_array((n_obs, n_vars), density=density, format="lil", random_state=rng)
    mat[:5] = 0
    mat[120:125] = 0
    mat[-3:] = 0
    mat[:, 10:20] = 0
    csr = mat.tocsr()
    csr.data = rng.integers(1, 25, size=csr.nnz).astype(np.float32)
    va = VCSCAnnData(
        X=VCSRArray.from_scipy(csr),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]),
    )
    path = tmp_path / f"stream_{seed}.h5ad"
    va.write_h5ad(path)
    return path, va


@pytest.mark.parametrize("recipe", ["raw", "cp10k_log1p", "parafac2", "scanpy", "pearson"])
@pytest.mark.parametrize(
    ("budget", "max_block_nnz"),
    [
        ("1GB", 2000),  # all blocks resident compressed in VRAM
        (1, 1500),  # forced disk streaming (0 resident blocks)
        (1, 1),  # single-row blocks
    ],
    ids=["resident", "streamed", "single_row"],
)
def test_gpu_stream_matches_to_gpu(tmp_path, recipe, budget, max_block_nnz):
    import cupy as cp

    path, va = _write_vcsr_h5ad(tmp_path)
    assert va.X is not None
    ref = va.X.normalized(recipe).to_gpu()

    rng = np.random.default_rng(11)
    B = rng.normal(size=(va.shape[1], 16)).astype(np.float32)
    C = rng.normal(size=(8, va.shape[0])).astype(np.float32)
    v_right = rng.normal(size=(va.shape[1],)).astype(np.float32)
    v_left = rng.normal(size=(va.shape[0],)).astype(np.float32)

    with open_gpu_stream(
        path, recipe=recipe, device_budget=budget, max_block_nnz=max_block_nnz
    ) as gv:
        assert gv.shape == va.shape
        assert gv.nnz == va.X.nnz
        if budget == "1GB":
            assert gv.n_resident_blocks == gv.n_blocks
        else:
            assert gv.n_resident_blocks == 0

        np.testing.assert_allclose(
            cp.asnumpy(gv.means), cp.asnumpy(ref.means), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(cp.asnumpy(gv @ B), cp.asnumpy(ref @ B), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(cp.asnumpy(C @ gv), cp.asnumpy(C @ ref), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(
            cp.asnumpy(gv @ v_right), cp.asnumpy(ref @ v_right), rtol=1e-4, atol=1e-3
        )
        np.testing.assert_allclose(
            cp.asnumpy(v_left @ gv), cp.asnumpy(v_left @ ref), rtol=1e-4, atol=1e-3
        )

        ab, ca = gv.matmul_both(B, C)
        np.testing.assert_allclose(cp.asnumpy(ab), cp.asnumpy(ref @ B), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(cp.asnumpy(ca), cp.asnumpy(C @ ref), rtol=1e-4, atol=1e-3)


def test_gpu_stream_large_escaped_gaps(tmp_path):
    """Verify GPU Huffman decoder handles gaps >= 255 (multi-byte LEB128 escapes)."""
    import cupy as cp

    rng = np.random.default_rng(7)
    n_obs, n_vars = 80, 5000
    mat = sp.random_array((n_obs, n_vars), density=0.005, format="csr", random_state=rng)
    mat.data = rng.integers(1, 10, size=mat.nnz).astype(np.float32)
    va = VCSCAnnData(
        X=VCSRArray.from_scipy(mat),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]),
    )
    path = tmp_path / "sparse_wide.h5ad"
    va.write_h5ad(path)

    assert va.X is not None
    ref = va.X.normalized("parafac2").to_gpu()
    B = rng.normal(size=(n_vars, 8)).astype(np.float32)
    with open_gpu_stream(path, recipe="parafac2", max_block_nnz=300) as gv:
        np.testing.assert_allclose(cp.asnumpy(gv @ B), cp.asnumpy(ref @ B), rtol=1e-4, atol=1e-3)
