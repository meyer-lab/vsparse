# vsparse

`vsparse` provides `VCSCArray`/`VCSRArray`: Value-Compressed Sparse Column/Row (VCSC/VCSR)
array types, implemented in NumPy and accelerated with [Numba](https://numba.pydata.org/). The
array types are standalone -- they can be built from and converted to SciPy sparse arrays and
used entirely without [AnnData](https://anndata.readthedocs.io). `vsparse` also ships an
optional AnnData integration (`from_anndata`, `to_layer`, and the `VCSCAnnData` subclass) for
projects that want compressed arrays backed directly into an `AnnData` object.

VCSC/VCSR are compressed-sparse layouts inspired by
[IVSparse's VCSC](https://github.com/Seth-Wolfgang/IVSparse). In addition to the usual
compressed-sparse pointer/index arrays, nonzero values within each major-axis slice
(columns for VCSC, rows for VCSR) are deduplicated: each unique value is stored once,
alongside the list of minor-axis positions that share it. This is a strict memory win
whenever values repeat heavily within a slice — as is typical for integer count
matrices, e.g. single-cell RNA-seq counts.

## Install

Install from PyPI:

```sh
pip install vsparse
# or with uv
uv add vsparse
```

For the CUDA support described below (needs an NVIDIA GPU; pulls in CuPy):

```sh
pip install "vsparse[cuda]"
# or with uv
uv add "vsparse[cuda]"
```

From source:

```sh
pip install git+https://github.com/meyer-lab/vsparse.git
# or with uv
uv add git+https://github.com/meyer-lab/vsparse.git
```

For local development:

```sh
git clone https://github.com/meyer-lab/vsparse.git
cd vsparse
uv sync --all-extras --dev
```

## Quickstart

### Working with VCSC / VCSR arrays

`VCSCArray`/`VCSRArray` are standalone array types -- they only need a SciPy sparse array
(or an AnnData object, if you have one) to be built, and none of the operations below require
AnnData at all:

```python
import vsparse
import scipy.sparse as sp

# Build directly from a SciPy sparse array -- no AnnData involved
csc = sp.random(1000, 500, density=0.1, format="csc")
v = vsparse.VCSCArray.from_scipy(csc)

# Or build from an AnnData object or SciPy sparse array
adata = ...  # an AnnData object
v = vsparse.from_anndata(adata)  # VCSCArray (column-compressed) from adata.X
vr = vsparse.from_anndata(adata, format="csr")  # VCSRArray (row-compressed)

# Transposition is zero-copy (swaps major/minor axes and shares buffers)
vr = v.T  # VCSRArray

# Scalar arithmetic & math
v2 = v * 2.0  # Scalar multiplication
v_div = v / 2.0  # Scalar division
v_neg = -v  # Negation
v_log = v.log1p()  # Elementwise log1p

# Matrix & vector products (Numba-parallelized)
y = v @ x  # Matrix-vector: (n_rows, n_cols) @ (n_cols,) -> (n_rows,)
y_left = x @ v  # Vector-matrix: (n_rows,) @ (n_rows, n_cols) -> (n_cols,)
Y = v @ B  # Matrix-matrix: (n_rows, n_cols) @ (n_cols, k) -> (n_rows, k)
Y_left = B @ v  # Matrix-matrix: (k, n_rows) @ (n_rows, n_cols) -> (k, n_cols)

# Slicing
col_slice = v[:, [1, 3, 5]]  # Fast major-axis slicing (returns VCSCArray)
sub = v[0:10, 0:10]  # 2D slicing (falls back to scipy)

# Conversion & layers
sp_csc = v.to_scipy()  # -> scipy.sparse.csc_array (or to_csr())
dense = v.toarray()  # -> numpy.ndarray
vsparse.to_layer(adata, v, key="counts_vcsc")  # Attach to AnnData layer
```

### `VCSCAnnData`: AnnData with direct VCSC/VCSR backing

For projects that want the compressed array wired directly into AnnData, `vsparse.VCSCAnnData`
is an `AnnData` subclass whose `X` (and optionally `raw_X`) is backed directly by a
`VCSCArray` or `VCSRArray`:

```python
import vsparse

va = vsparse.VCSCAnnData.from_anndata(adata)  # Compresses X and raw.X
va.X  # VCSCArray
va.raw_X  # VCSCArray (separate from anndata's .raw)

# Persist to HDF5 (.h5ad) or Zarr with default Blosc2+LZ4 compression
va.write_h5ad("compressed.h5ad")  # Read back with VCSCAnnData.read_h5ad
va2 = vsparse.VCSCAnnData.read_h5ad("compressed.h5ad")

va.write_zarr("compressed.zarr")  # Read back with VCSCAnnData.read_zarr
va3 = vsparse.VCSCAnnData.read_zarr("compressed.zarr")

# Escape hatch back to standard AnnData
plain = va.to_anndata()
```

### Byte-Packed On-Disk Format (IVCSC / IVCSR)

For smaller files, pass `format="ivcsc"` (or `"ivcsr"`) on write. This byte-packs the minor-axis indices with delta + varint encoding (inspired by [IVSparse's IVCSC](https://github.com/Seth-Wolfgang/IVSparse)).

It is purely an archival storage format: `read_h5ad` / `read_zarr` decompress the indices on load and return a standard `VCSCArray`/`VCSRArray`.

```python
# Write delta+varint byte-packed indices
va.write_h5ad("archived.h5ad", format="ivcsc")

# Reads back directly as an ordinary VCSCAnnData
va_loaded = vsparse.VCSCAnnData.read_h5ad("archived.h5ad")
```

### Fast Filtering & Depth Normalization (`load_and_normalize`)

For IVCSR-backed datasets, `vsparse.load_and_normalize` bypasses full array decompression for rapid preprocessing, reproducing the filtering and depth normalization from `parafac2.normalize.prepare_dataset`:

- **Cell filtering without decoding**: Computes cell totals in $O(n_{\text{unique}})$ time directly from group sizes without unpacking varint indices.
- **Fused filter & normalization**: Performs gene filtering, cell/gene depth-scaling, and $\log_{10}(1000x + 1)$ transform in parallel passes over the compact CSR representation.

```python
adata_norm = vsparse.load_and_normalize(
    "archived.h5ad",
    min_cell_counts=10.0,  # Filter cells with counts <= 10
    gene_threshold=0.05,  # Filter genes with counts <= 0.05 * n_cells
)
```

### CUDA (`to_gpu`)

A normalized view moves onto an NVIDIA GPU with `.to_gpu()`, keeping its value-compressed layout there rather than expanding to one float per nonzero — so a matrix that only fits in device memory compressed still fits. Everything floating-point on the device is float32.

```python
gpu = vsparse.VCSRArray.from_scipy(counts).normalized("parafac2").to_gpu()
out = gpu @ B  # float32 CuPy array; B @ gpu likewise
```

cuSPARSE reads only CSR/CSC, so walking the VCS layout on the device needs custom kernels; `vsparse._cuda` supplies four, one per (format, direction). Against the alternative — expanding to a sorted CuPy CSR and calling cuSPARSE — they measured (RTX 5080, 60k x 2k, 6M nonzeros, width 16):

| direction | kernel / cuSPARSE | device bytes vs CSR |
| --- | --- | --- |
| `VCSR @ B` | 0.71x–0.86x | 0.62x |
| `VCSC @ B` | 0.83x | 0.51x |
| `B @ VCSC` | 0.32x | 0.51x |
| `B @ VCSR` | 0.23x | 0.62x |

`CudaNormalizedView.to_cupy_sparse()` builds that CSR baseline if you want it.

See `docs/` for full usage guides and API documentation.

## Development

```sh
uv sync --all-extras --dev     # add --no-extra cuda to skip the ~1 GB CuPy wheel
uv run pytest
uv run ruff check .
uv run ty check
uv run sphinx-build -b html docs docs/_build/html
```

The CUDA tests and benchmarks need an NVIDIA GPU; without one `tests/test_cuda.py`
skips itself. On a GPU machine:

```sh
uv run pytest -m cuda
uv run python -m benchmarks.run --set cuda
```
