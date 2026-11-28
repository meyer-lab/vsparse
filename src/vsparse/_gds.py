"""GPU-streamed and VRAM-compressed normalized views for IVCSR v2 files (#76).

:func:`open_gpu_stream` opens an IVCSR v2 (static-Huffman) ``.h5ad`` file and
returns a :class:`CudaStreamedView` supporting ``gv @ B``, ``B @ gv``, and
single-pass ``gv.matmul_both(B, C)`` without ever holding the full unpacked
``indices`` array in host RAM or VRAM.

Architecture:

- **VRAM compressed residency cache:** As many row blocks as fit in
  ``device_budget`` keep their *Huffman-compressed* bytes resident in VRAM
  (~2x higher capacity than storing decoded ``uint16`` indices).
- **Double-buffered scratch decode:** Two preallocated decoded-``indices``
  scratch buffers and two CUDA streams overlap decoding (and disk reads for
  overflow blocks) of block *i+1* with running the existing
  :mod:`vsparse._cuda` matmul kernels on block *i*.
- **Zero-nvCOMP I/O:** Because IVCSR v2 stores ``packed_indices`` uncompressed
  and contiguous, overflow blocks read directly from NVMe into device memory
  via ``kvikio.CuFile.pread`` (GPUDirect Storage, or kvikio compat mode) when
  ``kvikio`` is installed, falling back to ``os.pread`` + async host-to-device
  copy otherwise.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import h5py
import numpy as np

from vsparse import _compression, _huffman, _io, _stats
from vsparse._cuda import (
    DEVICE_DTYPE,
    CudaNormalizedView,
    _C_TYPES,
    _index_dtype,
    _pointer_dtype,
    _prep_dense,
    _prep_dense_right,
    _require_cupy,
    _sm_count,
)
from vsparse._norm_common import (
    DEFAULT_RECIPE,
    G_LOG1P,
    G_LOG1P_1000X,
    G_SQRT,
    Recipe,
    _row_gene_scales,
    resolve_recipe,
)

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["CudaStreamedView", "open_gpu_stream"]

_DEFAULT_BLOCK_NNZ = 32_000_000
_DECODE_THREADS = 256

_HUFFMAN_DECODE_SRC = f"""
typedef {{idx_t}} idx_t;
typedef {{ptr_t}} ptr_t;

__device__ __forceinline__ void refill(
    const unsigned char* __restrict__ buf,
    long long& byte_pos,
    unsigned long long& bit_buf,
    int& nbits)
{{{{
    unsigned long long b0 = buf[byte_pos];
    unsigned long long b1 = buf[byte_pos + 1];
    unsigned long long b2 = buf[byte_pos + 2];
    unsigned long long b3 = buf[byte_pos + 3];
    bit_buf |= (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) << nbits;
    byte_pos += 4;
    nbits += 32;
}}}}

extern "C" __global__ void huffman_decode_rows(
    const ptr_t* __restrict__ major_ptr,
    const ptr_t* __restrict__ value_ptr,
    const unsigned char* __restrict__ packed,
    const long long* __restrict__ bit_offsets,
    const unsigned short* __restrict__ dec_table,
    idx_t* __restrict__ out_indices,
    const long long base_bit,
    const long long n_rows)
{{{{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         r < n_rows; r += stride) {{{{
        const long long g_start = major_ptr[r];
        const long long g_end = major_ptr[r + 1];
        if (g_start == g_end) continue;

        const long long rel_bit = bit_offsets[r] - base_bit;
        long long byte_pos = rel_bit >> 3;
        const int skip = (int)(rel_bit & 7LL);
        unsigned long long bit_buf = ((unsigned long long)packed[byte_pos]) >> skip;
        byte_pos += 1;
        int nbits = 8 - skip;

        for (long long g = g_start; g < g_end; ++g) {{{{
            long long prev = -1;
            const long long k_start = value_ptr[g];
            const long long k_end = value_ptr[g + 1];
            for (long long k = k_start; k < k_end; ++k) {{{{
                if (nbits < {_huffman.MAX_CODE_LEN}) {{{{
                    refill(packed, byte_pos, bit_buf, nbits);
                }}}}
                const unsigned short entry = dec_table[bit_buf & {(_huffman.TABLE_SIZE - 1)}u];
                const int sym = (int)(entry & 0xFFu);
                const int length = (int)(entry >> 8);
                bit_buf >>= length;
                nbits -= length;

                long long gap = sym;
                if (sym == {_huffman.ESCAPE}) {{{{
                    unsigned long long val = 0;
                    int shift = 0;
                    while (true) {{{{
                        if (nbits < 8) {{{{
                            refill(packed, byte_pos, bit_buf, nbits);
                        }}}}
                        unsigned long long b = bit_buf & 0xFFull;
                        bit_buf >>= 8;
                        nbits -= 8;
                        val |= (b & 0x7Full) << shift;
                        if ((b & 0x80ull) == 0) break;
                        shift += 7;
                    }}}}
                    gap = {_huffman.ESCAPE}LL + (long long)val;
                }}}}
                prev = prev + 1 + gap;
                out_indices[k] = (idx_t)prev;
            }}}}
        }}}}
    }}}}
}}}}
"""

_COL_STATS_SRC = f"""
typedef {{idx_t}} idx_t;
typedef {{ptr_t}} ptr_t;

__device__ __forceinline__ float apply_g(const float x, const int g_code) {{{{
    if (g_code == {G_LOG1P}) return log1pf(x);
    if (g_code == {G_LOG1P_1000X}) return log10f(1.0f + 1000.0f * x);
    if (g_code == {G_SQRT}) return x > 0.0f ? sqrtf(x) : 0.0f;
    return x;
}}}}

extern "C" __global__ void accumulate_col_stats_csr(
    const ptr_t* __restrict__ major_ptr,
    const float* __restrict__ values,
    const ptr_t* __restrict__ value_ptr,
    const idx_t* __restrict__ indices,
    const float* __restrict__ row_scale,
    const float* __restrict__ gene_scale,
    const int g_code,
    const long long n_rows,
    double* __restrict__ col_sum,
    double* __restrict__ col_sumsq)
{{{{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         r < n_rows; r += stride) {{{{
        const float rs = row_scale[r];
        const long long g_start = major_ptr[r];
        const long long g_end = major_ptr[r + 1];
        for (long long g = g_start; g < g_end; ++g) {{{{
            const float v = values[g];
            const long long k_start = value_ptr[g];
            const long long k_end = value_ptr[g + 1];
            for (long long k = k_start; k < k_end; ++k) {{{{
                const unsigned int col = (unsigned int)indices[k];
                const float gs = gene_scale[col];
                if (!(gs > 0.0f)) continue;
                const double y = (double)apply_g(v / rs / gs, g_code);
                atomicAdd(col_sum + col, y);
                atomicAdd(col_sumsq + col, y * y);
            }}}}
        }}}}
    }}}}
}}}}
"""


@functools.cache
def _decode_kernel(idx_dtype: np.dtype, ptr_dtype: np.dtype) -> Any:
    cp = _require_cupy()
    src = _HUFFMAN_DECODE_SRC.format(idx_t=_C_TYPES[idx_dtype], ptr_t=_C_TYPES[ptr_dtype])
    return cp.RawKernel(src, "huffman_decode_rows")


@functools.cache
def _col_stats_kernel(idx_dtype: np.dtype, ptr_dtype: np.dtype) -> Any:
    cp = _require_cupy()
    src = _COL_STATS_SRC.format(idx_t=_C_TYPES[idx_dtype], ptr_t=_C_TYPES[ptr_dtype])
    return cp.RawKernel(src, "accumulate_col_stats_csr")


def _parse_budget(budget: int | str) -> int:
    """Parse a byte budget such as ``20 * 1024**3`` or ``"20GB"`` / ``"512MiB"``."""
    if isinstance(budget, int):
        if budget <= 0:
            raise ValueError(f"device_budget must be positive, got {budget}")
        return budget
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*", budget)
    if not m:
        raise ValueError(f"cannot parse device_budget {budget!r}")
    val = float(m.group(1))
    unit = m.group(2).lower()
    scales = {
        "": 1,
        "b": 1,
        "kb": 10**3,
        "kib": 1 << 10,
        "mb": 10**6,
        "mib": 1 << 20,
        "gb": 10**9,
        "gib": 1 << 30,
        "tb": 10**12,
        "tib": 1 << 40,
    }
    if unit not in scales:
        raise ValueError(f"unknown device_budget unit {unit!r} in {budget!r}")
    out = int(val * scales[unit])
    if out <= 0:
        raise ValueError(f"device_budget must be positive, got {budget!r}")
    return out


class _BlockMeta(NamedTuple):
    r0: int
    r1: int
    g0: int
    g1: int
    nnz: int
    byte0: int
    byte1: int
    base_bit: int


class _ResidentBlock(NamedTuple):
    major_ptr: Any
    values: Any
    value_ptr: Any
    packed: Any
    bit_offsets: Any


class _BlockReader:
    """Read raw byte ranges of ``packed_indices`` into a device staging buffer.

    Uses ``kvikio.CuFile.pread`` (GPUDirect Storage / compat mode) when
    ``kvikio`` is importable and the dataset is contiguous on a POSIX file;
    otherwise falls back to ``os.pread`` (or ``h5py`` slicing) + async H2D copy.
    """

    def __init__(self, path: Path, ds: h5py.Dataset) -> None:
        self._path = str(path)
        self._ds = ds
        self._file_offset = _huffman._h5_contiguous_offset(ds)
        self._cufile: Any = None
        self._fd: int | None = None

        if self._file_offset is not None:
            try:
                import kvikio

                self._cufile = kvikio.CuFile(self._path, "r")
            except Exception:
                self._fd = os.open(self._path, os.O_RDONLY)

    def read_into_device(self, byte0: int, byte1: int, dst_gpu: Any, stream: Any) -> Any:
        nbytes = byte1 - byte0
        if nbytes <= 0:
            return dst_gpu[: _huffman.PAD_BYTES]
        if self._cufile is not None and self._file_offset is not None:
            future = self._cufile.pread(
                dst_gpu[:nbytes], nbytes, file_offset=self._file_offset + byte0
            )
            future.get()
        elif self._fd is not None and self._file_offset is not None:
            raw = os.pread(self._fd, nbytes, self._file_offset + byte0)
            host = np.frombuffer(raw, dtype=np.uint8, count=nbytes)
            dst_gpu[:nbytes].set(host, stream=stream)
        else:
            host = np.asarray(self._ds[byte0:byte1], dtype=np.uint8)
            dst_gpu[:nbytes].set(host, stream=stream)
        return dst_gpu[:nbytes]

    def close(self) -> None:
        if self._cufile is not None:
            self._cufile.close()
            self._cufile = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _compute_blocks(
    major_ptr: np.ndarray,
    value_ptr: np.ndarray,
    bit_offsets: np.ndarray,
    packed_len: int,
    max_nnz: int,
) -> list[_BlockMeta]:
    n_rows = major_ptr.shape[0] - 1
    if n_rows == 0:
        return []
    slice_nnz = value_ptr[major_ptr]
    blocks: list[_BlockMeta] = []
    pos = 0
    while pos < n_rows:
        target = int(slice_nnz[pos]) + max_nnz
        nxt = int(np.searchsorted(slice_nnz, target, side="right") - 1)
        if nxt <= pos:
            nxt = pos + 1
        g0 = int(major_ptr[pos])
        g1 = int(major_ptr[nxt])
        nnz = int(value_ptr[g1] - value_ptr[g0])
        b0 = int(bit_offsets[pos])
        b1 = int(bit_offsets[nxt])
        byte0 = b0 >> 3
        byte1 = min(((b1 + 7) >> 3) + _huffman.PAD_BYTES, packed_len) if b1 > b0 else byte0
        blocks.append(_BlockMeta(pos, nxt, g0, g1, nnz, byte0, byte1, byte0 * 8))
        pos = nxt
    return blocks


class CudaStreamedView:
    """Block-streamed and VRAM-compressed normalized view over an IVCSR v2 ``.h5ad`` file.

    Built by :func:`open_gpu_stream`. Implements ``self @ B``, ``B @ self``,
    and single-pass :meth:`matmul_both`.
    """

    __array_ufunc__ = None

    def __init__(
        self,
        path: Path,
        *,
        recipe: str | Recipe = DEFAULT_RECIPE,
        x_key: str = "X",
        device_budget: int | str = "16GB",
        max_block_nnz: int | None = None,
    ) -> None:
        cp = _require_cupy()
        self._cp = cp
        self._path = Path(path)
        self.recipe = resolve_recipe(recipe)
        self._budget = _parse_budget(device_budget)
        self.sm_count = _sm_count(cp.cuda.Device().id)

        self._h5 = h5py.File(self._path, "r")
        try:
            elem = self._h5[x_key]
            spec = str(elem.attrs.get("encoding-type", ""))
            if spec != "ivcsr" or "huffman_lengths" not in elem:
                raise ValueError(
                    f"open_gpu_stream requires an IVCSR v2 (Huffman) group at {x_key!r}; "
                    f"got encoding-type={spec!r}"
                )
            self.shape = _io._read_shape(elem)
            self._major_ptr = _compression.read_array(elem["major_ptr"]).astype(
                np.int64, copy=False
            )
            self._values = _compression.read_array(elem["values"]).astype(DEVICE_DTYPE, copy=False)
            self._value_ptr = _compression.read_array(elem["value_ptr"]).astype(
                np.int64, copy=False
            )
            self._bit_offsets = _compression.read_array(elem["bit_offsets"]).astype(
                np.int64, copy=False
            )
            code_lengths = _compression.read_array(elem["huffman_lengths"])
            _, _, dec_table = _huffman.build_tables(code_lengths)
            self._dec_table = cp.asarray(dec_table, dtype=cp.uint16)

            packed_ds = elem["packed_indices"]
            packed_len = int(packed_ds.shape[0])
            self._reader = _BlockReader(self._path, packed_ds)

            self._nnz = int(self._value_ptr[-1])
            self._idx_dtype = _index_dtype(self.shape[1])

            block_nnz = _DEFAULT_BLOCK_NNZ if max_block_nnz is None else int(max_block_nnz)
            if block_nnz <= 0:
                raise ValueError(f"max_block_nnz must be positive, got {block_nnz}")
            self._blocks = _compute_blocks(
                self._major_ptr, self._value_ptr, self._bit_offsets, packed_len, block_nnz
            )
            self._ptr_dtype = _pointer_dtype(max((b.nnz for b in self._blocks), default=0))

            # Preallocate two reusable scratch/staging buffer pairs and two CUDA streams.
            max_nnz_blk = max((b.nnz for b in self._blocks), default=0)
            max_bytes_blk = max((b.byte1 - b.byte0 for b in self._blocks), default=0)
            self._streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True))
            self._idx_scratch = (
                cp.empty(max(1, max_nnz_blk), dtype=self._idx_dtype),
                cp.empty(max(1, max_nnz_blk), dtype=self._idx_dtype),
            )
            self._packed_staging = (
                cp.zeros(
                    max(_huffman.PAD_BYTES, max_bytes_blk + _huffman.PAD_BYTES), dtype=cp.uint8
                ),
                cp.zeros(
                    max(_huffman.PAD_BYTES, max_bytes_blk + _huffman.PAD_BYTES), dtype=cp.uint8
                ),
            )

            # Populate VRAM compressed residency cache within remaining budget.
            overhead = (
                2 * self._idx_scratch[0].nbytes
                + 2 * self._packed_staging[0].nbytes
                + self._dec_table.nbytes
                + (self.shape[0] + 3 * self.shape[1]) * 4
            )
            self._resident: dict[int, _ResidentBlock] = {}
            self._populate_residency_cache(max(0, self._budget - overhead))

            # Derive normalization statistics (a/b from sidecar, c/s via one GPU pass if needed).
            stats = _stats.read_stats(elem, self.shape)
            if stats is None:
                stats = self._compute_basic_stats_host(elem)
            rs, gs = _row_gene_scales(stats, self.recipe)
            self.row_scale = cp.asarray(rs, dtype=DEVICE_DTYPE)
            self.gene_scale = cp.asarray(gs, dtype=DEVICE_DTYPE)
            self.col_mean, self.col_post_scale = self._compute_column_stats()
        except Exception:
            self.close()
            raise

    def _compute_basic_stats_host(self, elem: Any) -> _stats.VCSStats:
        """Fallback when a file lacks a stats sidecar: decode and compute stats on host."""
        from vsparse._base import VCSRArray

        arr = _io._make_read_ivcs_v2(VCSRArray)(elem, _reader=None)
        return arr.stats

    def _populate_residency_cache(self, remaining_budget: int) -> None:
        cp = self._cp
        used = 0
        for idx, blk in enumerate(self._blocks):
            ptr_bytes = (blk.r1 - blk.r0 + 1 + blk.g1 - blk.g0 + 1) * self._ptr_dtype.itemsize
            val_bytes = (blk.g1 - blk.g0) * 4
            packed_bytes = (blk.byte1 - blk.byte0) + _huffman.PAD_BYTES
            bo_bytes = (blk.r1 - blk.r0 + 1) * 8
            blk_bytes = ptr_bytes + val_bytes + packed_bytes + bo_bytes
            if used + blk_bytes > remaining_budget:
                break
            packed_dev = cp.zeros(packed_bytes, dtype=cp.uint8)
            self._reader.read_into_device(blk.byte0, blk.byte1, packed_dev, cp.cuda.Stream.null)
            self._resident[idx] = _ResidentBlock(
                major_ptr=cp.asarray(
                    self._major_ptr[blk.r0 : blk.r1 + 1] - blk.g0, dtype=self._ptr_dtype
                ),
                values=cp.asarray(self._values[blk.g0 : blk.g1], dtype=DEVICE_DTYPE),
                value_ptr=cp.asarray(
                    self._value_ptr[blk.g0 : blk.g1 + 1] - self._value_ptr[blk.g0],
                    dtype=self._ptr_dtype,
                ),
                packed=packed_dev,
                bit_offsets=cp.asarray(self._bit_offsets[blk.r0 : blk.r1 + 1], dtype=cp.int64),
            )
            used += blk_bytes

    def _prepare_block(self, idx: int, slot: int) -> tuple[Any, Any, Any, Any, _BlockMeta]:
        """Decode block ``idx`` into scratch ``slot`` on ``self._streams[slot]``."""
        cp = self._cp
        blk = self._blocks[idx]
        stream = self._streams[slot]
        with stream:
            if idx in self._resident:
                rb = self._resident[idx]
                mp_d, val_d, vp_d, packed_d, bo_d = (
                    rb.major_ptr,
                    rb.values,
                    rb.value_ptr,
                    rb.packed,
                    rb.bit_offsets,
                )
            else:
                mp_d = cp.asarray(
                    self._major_ptr[blk.r0 : blk.r1 + 1] - blk.g0, dtype=self._ptr_dtype
                )
                val_d = cp.asarray(self._values[blk.g0 : blk.g1], dtype=DEVICE_DTYPE)
                vp_d = cp.asarray(
                    self._value_ptr[blk.g0 : blk.g1 + 1] - self._value_ptr[blk.g0],
                    dtype=self._ptr_dtype,
                )
                bo_d = cp.asarray(self._bit_offsets[blk.r0 : blk.r1 + 1], dtype=cp.int64)
                nbytes = blk.byte1 - blk.byte0
                staging = self._packed_staging[slot]
                if nbytes > 0:
                    stream.synchronize()
                    self._reader.read_into_device(blk.byte0, blk.byte1, staging, stream)
                    staging[nbytes : nbytes + _huffman.PAD_BYTES] = 0
                packed_d = staging

            idx_d = self._idx_scratch[slot][: blk.nnz]
            n_rows_blk = blk.r1 - blk.r0
            if blk.nnz > 0:
                grid = (min(4096, max(1, -(-n_rows_blk // _DECODE_THREADS))),)
                _decode_kernel(self._idx_dtype, self._ptr_dtype)(
                    grid,
                    (_DECODE_THREADS,),
                    (
                        mp_d,
                        vp_d,
                        packed_d,
                        bo_d,
                        self._dec_table,
                        idx_d,
                        np.int64(blk.base_bit),
                        np.int64(n_rows_blk),
                    ),
                )
        return mp_d, val_d, vp_d, idx_d, blk

    def _compute_column_stats(self) -> tuple[Any, Any]:
        cp = self._cp
        n_rows, n_cols = self.shape
        needs_mean = self.recipe.center
        needs_std = self.recipe.post_scale
        if not needs_mean and not needs_std:
            return cp.zeros(n_cols, dtype=DEVICE_DTYPE), cp.ones(n_cols, dtype=DEVICE_DTYPE)

        col_sum = cp.zeros(n_cols, dtype=cp.float64)
        col_sumsq = cp.zeros(n_cols, dtype=cp.float64)
        kernel = _col_stats_kernel(self._idx_dtype, self._ptr_dtype)

        for idx in range(len(self._blocks)):
            slot = idx & 1
            mp_d, val_d, vp_d, idx_d, blk = self._prepare_block(idx, slot)
            with self._streams[slot]:
                n_rows_blk = blk.r1 - blk.r0
                if blk.nnz > 0:
                    grid = (min(4096, max(1, -(-n_rows_blk // _DECODE_THREADS))),)
                    kernel(
                        grid,
                        (_DECODE_THREADS,),
                        (
                            mp_d,
                            val_d,
                            vp_d,
                            idx_d,
                            self.row_scale[blk.r0 : blk.r1],
                            self.gene_scale,
                            np.int32(self.recipe.g_code),
                            np.int64(n_rows_blk),
                            col_sum,
                            col_sumsq,
                        ),
                    )
        for s in self._streams:
            s.synchronize()

        mean = col_sum / float(n_rows) if n_rows > 0 else cp.zeros(n_cols, dtype=cp.float64)
        col_mean = mean.astype(DEVICE_DTYPE) if needs_mean else cp.zeros(n_cols, dtype=DEVICE_DTYPE)
        if needs_std and n_rows > 0:
            var = cp.maximum(0.0, col_sumsq / float(n_rows) - mean * mean)
            std = cp.sqrt(var)
            post_scale = cp.where(std > 0.0, 1.0 / std, 0.0).astype(DEVICE_DTYPE)
        else:
            post_scale = cp.ones(n_cols, dtype=DEVICE_DTYPE)
        return col_mean, post_scale

    # -- introspection & lifecycle --------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(DEVICE_DTYPE)

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def n_blocks(self) -> int:
        return len(self._blocks)

    @property
    def n_resident_blocks(self) -> int:
        return len(self._resident)

    @property
    def means(self) -> Any:
        return self.col_mean * self.col_post_scale

    def close(self) -> None:
        if hasattr(self, "_reader") and self._reader is not None:
            self._reader.close()
        if hasattr(self, "_h5") and self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> CudaStreamedView:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- matrix products ------------------------------------------------------

    def _run_block(
        self,
        mp_d: Any,
        val_d: Any,
        vp_d: Any,
        idx_d: Any,
        blk: _BlockMeta,
        B: Any | None,
        out_ab: Any | None,
        Ct: Any | None,
        out_ca_t: Any | None,
    ) -> None:
        n_rows_blk = blk.r1 - blk.r0
        view = CudaNormalizedView._from_device(
            format="csr",
            shape=(n_rows_blk, self.shape[1]),
            recipe=self.recipe,
            major_ptr=mp_d,
            values=val_d,
            value_ptr=vp_d,
            indices=idx_d,
            row_scale=self.row_scale[blk.r0 : blk.r1],
            gene_scale=self.gene_scale,
            col_mean=self.col_mean,
            col_post_scale=self.col_post_scale,
            sm_count=self.sm_count,
        )
        if B is not None and out_ab is not None:
            view._run("matmul_delta_csr", B, out_ab[blk.r0 : blk.r1], n_rows_blk)
        if Ct is not None and out_ca_t is not None:
            view._run("rmatmul_delta_csr", Ct[blk.r0 : blk.r1], out_ca_t, n_rows_blk)

    def _stream_products(self, B: Any | None, Ct: Any | None) -> tuple[Any | None, Any | None]:
        cp = self._cp
        n_rows, n_cols = self.shape
        out_ab = cp.zeros((n_rows, B.shape[1]), dtype=DEVICE_DTYPE) if B is not None else None
        out_ca_t = cp.zeros((n_cols, Ct.shape[1]), dtype=DEVICE_DTYPE) if Ct is not None else None

        n_blocks = len(self._blocks)
        if n_blocks == 0:
            return out_ab, out_ca_t

        # Pipeline: decode block i+1 on stream (i+1)&1 while running matmul for block i on stream i&1.
        cur = self._prepare_block(0, 0)
        for i in range(n_blocks):
            slot = i & 1
            nxt = self._prepare_block(i + 1, (i + 1) & 1) if i + 1 < n_blocks else None
            with self._streams[slot]:
                self._run_block(*cur, B, out_ab, Ct, out_ca_t)
            if nxt is not None:
                cur = nxt
        for s in self._streams:
            s.synchronize()
        return out_ab, out_ca_t

    def __matmul__(self, other: Any) -> Any:
        cp = self._cp
        B, squeeze = _prep_dense(cp, other, self.shape[1])
        out, _ = self._stream_products(B, None)
        assert out is not None
        out += (-self.means) @ B
        return out[:, 0] if squeeze else out

    def __rmatmul__(self, other: Any) -> Any:
        cp = self._cp
        B2, squeeze = _prep_dense_right(cp, other, self.shape[0])
        Bt = cp.ascontiguousarray(B2.T)
        _, out_t = self._stream_products(None, Bt)
        assert out_t is not None
        out = cp.ascontiguousarray(out_t.T)
        out += B2.sum(axis=1)[:, None] * (-self.means)[None, :]
        return out[0, :] if squeeze else out

    def matmul_both(self, B: Any, C: Any) -> tuple[Any, Any]:
        """Compute ``(self @ B, C @ self)`` in a single streamed pass over the blocks."""
        cp = self._cp
        B_dev, squeeze_b = _prep_dense(cp, B, self.shape[1])
        C_dev, squeeze_c = _prep_dense_right(cp, C, self.shape[0])
        Ct = cp.ascontiguousarray(C_dev.T)

        out_ab, out_ca_t = self._stream_products(B_dev, Ct)
        assert out_ab is not None and out_ca_t is not None
        out_ab += (-self.means) @ B_dev
        out_ca = cp.ascontiguousarray(out_ca_t.T)
        out_ca += C_dev.sum(axis=1)[:, None] * (-self.means)[None, :]
        return (out_ab[:, 0] if squeeze_b else out_ab, out_ca[0, :] if squeeze_c else out_ca)


def open_gpu_stream(
    path: str | PathLike[str],
    *,
    recipe: str | Recipe = DEFAULT_RECIPE,
    x_key: str = "X",
    device_budget: int | str = "16GB",
    max_block_nnz: int | None = None,
) -> CudaStreamedView:
    """Open an IVCSR v2 ``.h5ad`` file for block-streamed GPU matrix products.

    Caches as many Huffman-compressed row blocks in VRAM as ``device_budget``
    allows and streams any remaining blocks from disk (via GPUDirect Storage
    when ``kvikio`` is installed).
    """
    return CudaStreamedView(
        Path(path),
        recipe=recipe,
        x_key=x_key,
        device_budget=device_budget,
        max_block_nnz=max_block_nnz,
    )
