"""Parity tests: GPUNormalizedVCS (CUDA) vs. CPU/Numba matmul.

Skipped entirely when no working CUDA device is available.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import RECIPES, VCSCArray, VCSRArray

cuda_mod = pytest.importorskip("vsparse._vcs_matmul_cuda")

pytestmark = pytest.mark.skipif(
    not cuda_mod.cuda_available(), reason="no working CUDA device/toolkit available"
)


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_cuda_matmul_matches_cpu(dense, vcls, recipe):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized(view=recipe)
    gv = nv.to_gpu()
    n_rows, n_cols = nv.shape

    rng = np.random.default_rng(0)
    B = rng.standard_normal((n_cols, 3))
    Bl = rng.standard_normal((5, n_rows))

    np.testing.assert_allclose(gv @ B, nv @ B, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(Bl @ gv, Bl @ nv, atol=1e-3, rtol=1e-3)


def test_to_gpu_requires_cuda_available(monkeypatch, dense, vcls):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    monkeypatch.setattr(cuda_mod, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError):
        nv.to_gpu()


def test_gpu_buffers_are_not_rebuilt_across_calls(dense, vcls, monkeypatch):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    nv = v.normalized()
    gv = nv.to_gpu()
    n_rows, n_cols = nv.shape
    rng = np.random.default_rng(2)
    B = rng.standard_normal((n_cols, 2))
    Bl = rng.standard_normal((5, n_rows))

    _ = gv @ B
    cached_native = dict(gv._aligned_cache)
    _ = gv @ B
    assert gv._aligned_cache is not None
    assert set(gv._aligned_cache) == set(cached_native)
    for fmt, aligned in cached_native.items():
        assert gv._aligned_cache[fmt] is aligned  # not rebuilt on a second call

    # The opposite-direction call regroups+caches the other alignment too.
    _ = Bl @ gv
    assert len(gv._aligned_cache) == 2
    _ = Bl @ gv
    assert len(gv._aligned_cache) == 2
