# GPU streaming

`vsparse.open_gpu_stream` streams an IVCSR v2 `.h5ad` file directly to an NVIDIA GPU in
row blocks, supporting the same normalized matrix products as
{class}`~vsparse.CudaNormalizedView` (`gv @ B`, `B @ gv`) plus a fused single-pass
`gv.matmul_both(B, C)` — without ever holding the full unpacked `indices` array in host
RAM or VRAM.

## When to use it

- **Fits in VRAM unpacked:** use `arr.normalized(recipe).to_gpu()` (see the CUDA section of [Usage](usage.md)).
- **Fits in host RAM, exceeds VRAM:** you can either use `open_gpu_stream` on the `.h5ad`
  file, or iterate zero-copy row blocks from memory with
  `norm_view.iter_blocks(max_nnz, gpu=True)`.
- **Exceeds VRAM (or both host RAM and VRAM):** use `open_gpu_stream`.

## Prerequisites

1. **CUDA / CuPy:** install the `cuda` extra (`pip install vsparse[cuda]`).
2. **IVCSR v2 file:** the input `.h5ad` must store `X` (or `x_key`) as an IVCSR v2
   (static-Huffman) group, written by `VCSCAnnData.write_h5ad` when `X` is a
   {class}`~vsparse.VCSRArray`.
3. **GPUDirect Storage (optional):** if `kvikio` is installed, overflow blocks are read
   straight from NVMe into device memory with `kvikio.CuFile.pread` (or kvikio's POSIX
   compat mode when `nvidia-fs` is unavailable). Without `kvikio`, `vsparse`
   automatically falls back to `os.pread` + asynchronous host-to-device transfers.

## Quickstart

```python
import numpy as np
import vsparse

# 1. Write an IVCSR v2 file (once)
va = vsparse.VCSCAnnData.from_anndata(adata, format="csr")
va.write_h5ad("dataset.ivcsr.h5ad")

# 2. Stream normalized products on the GPU
with vsparse.open_gpu_stream(
    "dataset.ivcsr.h5ad",
    recipe="parafac2",
    device_budget="16GB",
) as gv:
    B = np.random.default_rng(0).normal(size=(gv.shape[1], 16)).astype(np.float32)
    C = np.random.default_rng(1).normal(size=(16, gv.shape[0])).astype(np.float32)

    out_right = gv @ B  # (n_rows, 16) float32 CuPy array
    out_left = C @ gv  # (16, n_cols) float32 CuPy array

    # Preferred in iterative fits (e.g. PARAFAC2 / SVD): computes both in one pass
    ab, ca = gv.matmul_both(B, C)
```

Using a `with` block (or calling `gv.close()`) closes the underlying HDF5 handle and file
descriptor when you are done.

## How compressed VRAM residency and streaming work

IVCSR v2 stores `packed_indices` as an uncompressed, contiguous static-Huffman bitstream
with a per-row `int64` bit-offset table. `open_gpu_stream` partitions the rows into
nonzero-balanced blocks (bounded by `max_block_nnz`, defaulting to 32M nonzeros per block)
and manages GPU memory in two tiers:

1. **Compressed VRAM residency cache:** after reserving two small preallocated scratch
   buffers (for double-buffered decoding), `open_gpu_stream` caches as many blocks'
   *Huffman-compressed* bytes in VRAM as fit within `device_budget`. Because Huffman-packed
   gaps are roughly ~2× smaller than decoded `uint16` indices, datasets that exceed VRAM
   when unpacked can often stay fully resident compressed and run at HBM bandwidth with
   zero disk traffic per iteration.
2. **Overlapped streaming for overflow blocks:** any blocks that do not fit in
   `device_budget` remain on disk and are streamed on demand using two CUDA streams —
   reading and GPU-decoding block *i + 1* into one scratch buffer while the existing
   `vsparse._cuda` matmul kernels run on block *i* in the other.

You can inspect how many blocks fit in VRAM:

```python
with vsparse.open_gpu_stream("dataset.ivcsr.h5ad", device_budget="16GB") as gv:
    print(f"{gv.n_resident_blocks} / {gv.n_blocks} blocks resident in VRAM")
```

### Tuning `device_budget` and `max_block_nnz`

- `device_budget` accepts an integer byte count or a string such as `"20GB"`, `"16GiB"`,
  or `"512MB"`. Leave headroom for dense operands (`B`, `C`), outputs, and your model's
  own device allocations.
- `max_block_nnz` caps the decoded scratch buffer size (`2 * max_block_nnz * 2` bytes for
  `uint16` indices). Lowering `max_block_nnz` reduces peak scratch VRAM on smaller GPUs at
  the cost of more block launches.

### Single-pass `matmul_both`

Iterative decompositions such as PARAFAC2 or randomized SVD need both `gv @ B` and
`C @ gv` every iteration. Calling them separately reads any non-resident blocks from disk
twice. `gv.matmul_both(B, C)` decodes each block once and runs both products before moving
to the next block.

## Normalization statistics

Per-cell and per-gene scales (`a` and `b`) are read from the file's `stats` sidecar.
Recipes that center or post-scale (`"parafac2"`, `"scanpy"`, `"pearson"`) compute their
column mean (`c`) and standard deviation (`s`) in a single streamed GPU pass when
`open_gpu_stream` is called.

## Precision and current limitations

- **Float32 device math:** like {class}`~vsparse.CudaNormalizedView`, all device values,
  statistics, operands, and outputs are `float32`, and `B @ gv` uses `atomicAdd` (so
  repeated runs may differ in the last bits).
- **IVCSR v2 only:** `open_gpu_stream` requires a row-major IVCSR v2 group
  (`encoding-type="ivcsr"` with `huffman_lengths`). For v1 files or column-major IVCSC
  files, re-save as IVCSR v2 with `VCSCAnnData.write_h5ad`.
- **Subsetting:** `open_gpu_stream` operates on the full stored matrix; for cell/gene
  subsets, load directly with `vsparse.load_packed(path, obs_idx=..., var_idx=...)`,
  which only reads the selected byte ranges from disk.
