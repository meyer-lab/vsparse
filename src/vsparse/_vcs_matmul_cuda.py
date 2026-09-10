"""CUDA (CuPy ``RawKernel``) matmul for a device-resident copy of a normalized VCS view.

:class:`GPUNormalizedVCS` (built via
:meth:`~vsparse._vcs_norm._VCSNormalizedBase.to_gpu`) mirrors
:mod:`vsparse._vcs_matmul` exactly -- same ``baseline + Delta`` decomposition,
same major-alignment requirement (:class:`~vsparse.VCSRArray`-shaped for
``self @ B``, :class:`~vsparse.VCSCArray`-shaped for ``B @ self``) -- but
walks the ``major_ptr``/``values``/``value_ptr``/``indices`` arrays from
device memory in a hand-written CUDA kernel instead of a parallel Numba loop
on the host.

One CUDA block is assigned per major slice (a row for the VCSR kernel, a
column for the VCSC kernel); the block's threads split the dense operand's
width. Since a block exclusively owns the output row/column it writes into,
and a given thread only ever touches one lane of that output (``c``,
``c + blockDim.x``, ...), the accumulation needs no atomics -- exactly the
same "no cross-thread writes" property :mod:`vsparse._vcs_matmul`'s docstring
relies on for its own parallelization.

Compute happens in float32 (throughput on consumer/GeForce GPUs is a small
fraction of float64's); ``GPUNormalizedVCS.__matmul__``/``__rmatmul__``
return float64 numpy arrays, matching :mod:`vsparse._vcs_matmul`'s contract,
so callers see no difference beyond reduced precision. This is opt-in: the
plain CPU ``VCSCArrayNormalized``/``VCSRArrayNormalized.__matmul__`` never
routes here on its own, so existing behavior/precision is unaffected by
whether CuPy/CUDA happen to be installed -- callers ask for the GPU
explicitly, via :meth:`~vsparse._vcs_norm._VCSNormalizedBase.to_gpu`.

The "other-direction" regroup (building the opposite-format copy needed when
the source array doesn't match a kernel's required alignment) runs on the
CPU, reusing :meth:`~vsparse._base._VCSBase._transpose_major`, and is
uploaded once and cached on the :class:`GPUNormalizedVCS` instance -- there
is no GPU-native regroup.

Known caveat: importing this module configures ``CUDA_PATH`` (if
discoverable) and imports CuPy, in that order, since CuPy's CUDA-toolkit-
header discovery for NVRTC misbehaves for the rest of the process if CuPy's
*first* import happens some other way (e.g. ``zarr.core.buffer.gpu``, which
``anndata``'s zarr backend imports speculatively, imports CuPy on its own).
``vsparse/__init__.py`` imports this module first for that reason; a caller
that imports ``anndata``/``zarr`` before ``vsparse`` in the same process can
still hit this upstream CuPy/zarr interaction.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from vsparse._norm_common import G_IDENTITY, G_LOG1P, G_LOG1P_1000X, G_SQRT

if TYPE_CHECKING:
    from vsparse._base import _VCSBase
    from vsparse._vcs_norm import _VCSNormalizedBase

__all__ = ["HAS_CUDA", "GPUNormalizedVCS", "cuda_available"]


def _configure_cuda_path() -> None:
    """Best-effort discovery of CUDA toolkit headers, so CuPy's NVRTC JIT can find them.

    CuPy's ``RawKernel`` compiles via NVRTC, which needs
    ``cuda_runtime.h``/``crt/host_config.h`` on disk somewhere; a CUDA-capable
    machine that only has the driver + `cupy-cuda12x` wheel installed (no
    full toolkit) often has the headers at one of these standard locations
    without ``CUDA_PATH`` set.
    """
    if "CUDA_PATH" in os.environ:
        return
    for candidate in ("/usr", "/usr/lib/cuda"):
        if os.path.exists(os.path.join(candidate, "include", "cuda_runtime.h")):
            os.environ["CUDA_PATH"] = candidate
            return


_configure_cuda_path()

try:
    import cupy as cp  # ty: ignore[unresolved-import]

    HAS_CUDA = bool(cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0)
except Exception:
    cp = None  # ty: ignore[invalid-assignment]
    HAS_CUDA = False


_G_TRANSFORM_CUDA = r"""
__device__ __forceinline__ float g_transform(float x, int g_code) {
    if (g_code == 1) return log1pf(x);
    if (g_code == 2) return log10f(1.0f + 1000.0f * x);
    if (g_code == 3) return x > 0.0f ? sqrtf(x) : 0.0f;
    return x;
}
"""

_VCSR_MATMUL_DELTA_SRC = (
    _G_TRANSFORM_CUDA
    + r"""
extern "C" __global__
void vcsr_matmul_delta(
    const long long* major_ptr, const float* values, const long long* value_ptr,
    const int* indices, const float* row_scale, const float* gene_scale,
    const float* col_post_scale, int g_code, const float* B, float* out,
    int k, int n_major
) {
    int i = blockIdx.x;
    if (i >= n_major) return;
    float rs = row_scale[i];
    float* out_row = out + (size_t)i * k;
    for (long long u = major_ptr[i]; u < major_ptr[i + 1]; ++u) {
        float v = values[u];
        for (long long kk = value_ptr[u]; kk < value_ptr[u + 1]; ++kk) {
            int col = indices[kk];
            float gs = gene_scale[col];
            if (gs > 0.0f) {
                float delta = col_post_scale[col] * g_transform(v / rs / gs, g_code);
                const float* b_row = B + (size_t)col * k;
                for (int c = threadIdx.x; c < k; c += blockDim.x) {
                    out_row[c] += delta * b_row[c];
                }
            }
        }
    }
}
"""
)

_VCSC_RMATMUL_DELTA_SRC = (
    _G_TRANSFORM_CUDA
    + r"""
extern "C" __global__
void vcsc_rmatmul_delta(
    const long long* major_ptr, const float* values, const long long* value_ptr,
    const int* indices, const float* row_scale, const float* gene_scale,
    const float* col_post_scale, int g_code, const float* Bt, float* out_t,
    int p, int n_major
) {
    int j = blockIdx.x;
    if (j >= n_major) return;
    float gs = gene_scale[j];
    if (gs == 0.0f) return;
    float s = col_post_scale[j];
    float* acc = out_t + (size_t)j * p;
    for (long long u = major_ptr[j]; u < major_ptr[j + 1]; ++u) {
        float v = values[u];
        for (long long kk = value_ptr[u]; kk < value_ptr[u + 1]; ++kk) {
            int row = indices[kk];
            float delta = s * g_transform(v / row_scale[row] / gs, g_code);
            const float* brow = Bt + (size_t)row * p;
            for (int c = threadIdx.x; c < p; c += blockDim.x) {
                acc[c] += delta * brow[c];
            }
        }
    }
}
"""
)

_G_CODE_MAP = {G_IDENTITY: 0, G_LOG1P: 1, G_LOG1P_1000X: 2, G_SQRT: 3}

_kernel_lock = threading.Lock()
_kernels_compiled = False
_vcsr_kernel: Any = None
_vcsc_kernel: Any = None


def _kernels_ready() -> bool:
    """Compile both RawKernels once, lazily; return False (without raising) if that fails.

    ``cupy.RawKernel.__init__`` never actually invokes NVRTC -- compilation
    is lazy, deferred to first ``__call__``/``.kernel`` access -- so
    constructing the kernel objects can't be used to detect a missing
    toolkit; :meth:`~cupy.RawKernel.compile` is called explicitly here to
    force that lazy compilation to happen (and possibly fail) up front,
    rather than surfacing mid-fit inside :meth:`GPUNormalizedVCS.__matmul__`/
    ``__rmatmul__``. Callers use this to fall back gracefully (or refuse
    :meth:`~vsparse._vcs_norm._VCSNormalizedBase.to_gpu`) instead of
    propagating a compile error.
    """
    global _kernels_compiled, _vcsr_kernel, _vcsc_kernel
    if _kernels_compiled:
        return _vcsr_kernel is not None
    with _kernel_lock:
        if _kernels_compiled:
            return _vcsr_kernel is not None
        try:
            assert cp is not None
            vcsr_kernel = cp.RawKernel(_VCSR_MATMUL_DELTA_SRC, "vcsr_matmul_delta")
            vcsc_kernel = cp.RawKernel(_VCSC_RMATMUL_DELTA_SRC, "vcsc_rmatmul_delta")
            vcsr_kernel.compile()
            vcsc_kernel.compile()
            _vcsr_kernel = vcsr_kernel
            _vcsc_kernel = vcsc_kernel
        except Exception:
            _vcsr_kernel = None
            _vcsc_kernel = None
        _kernels_compiled = True
        return _vcsr_kernel is not None


def cuda_available() -> bool:
    """Whether the CUDA path is actually usable: a device is present *and* the kernels compiled.

    :meth:`vsparse._vcs_norm._VCSNormalizedBase.to_gpu` checks this rather
    than :data:`HAS_CUDA` directly, since a CUDA device can be present
    without a discoverable toolkit for NVRTC to compile against (see
    :func:`_kernels_ready`).
    """
    return HAS_CUDA and _kernels_ready()


_MAX_INT32 = np.iinfo(np.int32).max


class _GPUAligned:
    """Device copies of one major-aligned array's ``major_ptr``/``values``/``value_ptr``/``indices``."""

    __slots__ = ("indices", "major_ptr", "n_major", "value_ptr", "values")

    def __init__(self, arr: _VCSBase) -> None:
        if arr.n_minor > _MAX_INT32:
            raise NotImplementedError(
                "GPU kernels use 32-bit minor-axis indices; "
                f"n_minor={arr.n_minor} exceeds int32 range"
            )
        assert cp is not None
        self.major_ptr = cp.asarray(arr.major_ptr, dtype=cp.int64)
        self.values = cp.asarray(arr.values, dtype=cp.float32)
        self.value_ptr = cp.asarray(arr.value_ptr, dtype=cp.int64)
        self.indices = cp.asarray(arr.indices, dtype=cp.int32)
        self.n_major = arr.n_major


def _prep_dense_gpu(other: Any, expect_rows: int) -> tuple[Any, bool]:
    assert cp is not None
    B = cp.asarray(other, dtype=cp.float32)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return cp.ascontiguousarray(B), squeeze


class GPUNormalizedVCS:
    """A device-resident copy of a ``VCSCArrayNormalized``/``VCSRArrayNormalized`` view.

    Built via :meth:`~vsparse._vcs_norm._VCSNormalizedBase.to_gpu`, never
    directly. Uploads/regroups the source view's arrays onto the GPU once,
    at construction time, and reuses them across every subsequent ``@``/
    ``__rmatmul__`` call -- the natural shape for repeated calls against the
    same data (e.g. one ALS/NMF fit's per-round matmuls).

    Not a general-purpose array type: only ``@``/``__rmatmul__`` against a
    dense operand are supported, matching the pair of operations
    :mod:`vsparse._vcs_matmul` (the CPU counterpart) implements.
    """

    # Tell numpy to defer `ndarray @ GPUNormalizedVCS` to our __rmatmul__
    # instead of trying to broadcast this as an ndarray -- same reason
    # NormalizedViewBase/_VCSBase set this (see vsparse._norm_common).
    __array_ufunc__ = None

    __slots__ = (
        "_aligned_cache",
        "_cpu_arr",
        "col_mean",
        "col_post_scale",
        "g_code",
        "gene_scale",
        "row_scale",
        "shape",
    )

    def __init__(self, nview: _VCSNormalizedBase) -> None:
        if not cuda_available():
            raise RuntimeError(
                "no working CUDA device/toolkit available -- see vsparse._vcs_matmul_cuda.cuda_available()"
            )
        assert cp is not None
        self.shape = nview.shape
        self._cpu_arr = nview._arr
        self._aligned_cache: dict[str, _GPUAligned] = {}
        self.row_scale = cp.asarray(nview.row_scale, dtype=cp.float32)
        self.gene_scale = cp.asarray(nview.gene_scale, dtype=cp.float32)
        self.col_mean = cp.asarray(nview.col_mean, dtype=cp.float32)
        self.col_post_scale = cp.asarray(nview.col_post_scale, dtype=cp.float32)
        self.g_code = _G_CODE_MAP[nview.recipe.g_code]
        # The array's native format is uploaded eagerly (cheap: it's what
        # `nview` already has decoded); the opposite-format regroup is built
        # lazily, only if that direction's matmul is actually used.
        self._aligned_cache[self._cpu_arr._format] = _GPUAligned(self._cpu_arr)

    def _aligned(self, fmt: str) -> _GPUAligned:
        """The device-resident array in ``fmt`` ("csr" or "csc"), regrouping+uploading once."""
        cached = self._aligned_cache.get(fmt)
        if cached is not None:
            return cached
        src = self._cpu_arr._transpose_major()
        built = _GPUAligned(src)
        self._aligned_cache[fmt] = built
        return built

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<GPUNormalizedVCS shape={self.shape} dtype=float32(compute)/float64(result)>"

    def __matmul__(self, other: Any) -> np.ndarray:
        """``self @ other`` on the GPU -- CUDA counterpart of :func:`vsparse._vcs_matmul.normalized_at_dense`."""
        assert cp is not None
        n_rows, n_cols = self.shape
        B, squeeze = _prep_dense_gpu(other, n_cols)
        k = B.shape[1]
        aligned = self._aligned("csr")

        out = cp.zeros((n_rows, k), dtype=cp.float32)
        threads = min(128, max(k, 1))
        _vcsr_kernel(  # ty: ignore[call-non-callable]
            (aligned.n_major,),
            (threads,),
            (
                aligned.major_ptr,
                aligned.values,
                aligned.value_ptr,
                aligned.indices,
                self.row_scale,
                self.gene_scale,
                self.col_post_scale,
                self.g_code,
                B,
                out,
                k,
                aligned.n_major,
            ),
        )

        offset = self.col_mean * self.col_post_scale
        baseline = (-offset) @ B  # (k,)
        out += baseline[None, :]
        cp.cuda.Stream.null.synchronize()
        result = cp.asnumpy(out).astype(np.float64, copy=False)
        return result[:, 0] if squeeze else result

    def __rmatmul__(self, other: Any) -> np.ndarray:
        """``other @ self`` on the GPU -- CUDA counterpart of :func:`vsparse._vcs_matmul.dense_at_normalized`."""
        assert cp is not None
        n_rows, n_cols = self.shape
        other_gpu = cp.asarray(other, dtype=cp.float32)
        squeeze = other_gpu.ndim == 1
        B2 = other_gpu.reshape(1, -1) if squeeze else other_gpu
        if B2.ndim != 2 or B2.shape[1] != n_rows:
            raise ValueError(f"shape mismatch: expected last dimension {n_rows}, got {B2.shape}")
        B2 = cp.ascontiguousarray(B2)
        p = B2.shape[0]
        Bt = cp.ascontiguousarray(B2.T)  # (n_rows, p) -- cache-friendly stride, see _vcs_matmul

        aligned = self._aligned("csc")

        out_t = cp.zeros((n_cols, p), dtype=cp.float32)
        threads = min(128, max(p, 1))
        _vcsc_kernel(  # ty: ignore[call-non-callable]
            (aligned.n_major,),
            (threads,),
            (
                aligned.major_ptr,
                aligned.values,
                aligned.value_ptr,
                aligned.indices,
                self.row_scale,
                self.gene_scale,
                self.col_post_scale,
                self.g_code,
                Bt,
                out_t,
                p,
                aligned.n_major,
            ),
        )

        out = cp.ascontiguousarray(out_t.T)
        offset = self.col_mean * self.col_post_scale
        baseline = B2.sum(axis=1)[:, None] * (-offset)[None, :]  # (p, n_cols)
        out += baseline
        cp.cuda.Stream.null.synchronize()
        result = cp.asnumpy(out).astype(np.float64, copy=False)
        return result[0, :] if squeeze else result
