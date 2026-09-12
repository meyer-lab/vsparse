# Usage

`VCSCArray`/`VCSRArray` are standalone array types built on top of SciPy sparse arrays --
see [Converting from/to scipy sparse](#converting-fromto-scipy-sparse) to use them without
AnnData at all. The sections below also cover `vsparse`'s optional AnnData integration.

## Converting an AnnData object

```python
import anndata as ad
import vsparse

adata = ad.read_h5ad("data.h5ad")

# From adata.X, as a VCSCArray (column-compressed)
v = vsparse.from_anndata(adata)

# From adata.raw.X, as a VCSRArray (row-compressed)
v = vsparse.from_anndata(adata, use_raw=True, format="csr")

# From a specific layer
v = vsparse.from_anndata(adata, layer="counts")
```

## Converting from/to scipy sparse

```python
import scipy.sparse as sp
from vsparse import VCSCArray

csc = sp.csc_array(dense_or_sparse_matrix)
v = VCSCArray.from_scipy(csc)

back = v.to_scipy()  # -> scipy.sparse.csc_array
dense = v.toarray()  # -> numpy.ndarray
```

## Supported operations

```python
v.T  # transpose (free: returns a VCSRArray sharing buffers)
v * 2.0  # scalar multiplication
v / 2.0  # scalar division
v.log1p()  # elementwise log1p
v @ x  # matrix-vector product, x: 1-D array of length n_cols
x @ v  # vector-matrix product, x: 1-D array of length n_rows
v @ B  # matrix-matrix product, B: 2-D array with B.shape[0] == n_cols
B @ v  # matrix-matrix product, B: 2-D array with B.shape[1] == n_rows
v[:, [1, 3, 5]]  # major-axis (column, for VCSCArray) indexing
v[0:2, 0:2]  # general 2-D indexing (falls back to scipy internally)
```

Attach a decompressed result back onto an `AnnData` object as a layer:

```python
vsparse.to_layer(adata, v, key="vcsc_roundtrip")
```

## `VCSCAnnData`: X backed directly by a VCSCArray

`vsparse.VCSCAnnData` is an `AnnData` subclass whose `X` (and, separately, `raw_X`) *is* a
`VCSCArray`/`VCSRArray`, not a scipy array:

```python
import vsparse

va = vsparse.VCSCAnnData.from_anndata(adata)  # compresses X and raw.X
va.X  # a VCSCArray
va.raw_X  # a VCSCArray (kept separately from anndata's own `.raw`)

# Persist -- read back with the matching classmethod, not anndata.read_h5ad/read_zarr
va.write_h5ad("compressed.h5ad")
va2 = vsparse.VCSCAnnData.read_h5ad("compressed.h5ad")

va.write_zarr("compressed.zarr")
va3 = vsparse.VCSCAnnData.read_zarr("compressed.zarr")

# Escape hatch: decompress to a normal, fully-featured AnnData
plain = va.to_anndata()
```

Standard `AnnData` validates every array assigned to `X`/`layers`/etc. against a fixed
allowlist of types, so a plain `AnnData` cannot hold a `VCSCArray` in `X`.

### On-disk compression and the IVCSC/IVCSR storage format

`write_h5ad`/`write_zarr` compress every array with Blosc2+LZ4 by default
(`h5py`/`hdf5plugin` for `.h5ad`, `numcodecs`/zarr's native Blosc codec for
zarr). Pass `dataset_kwargs={}` to write uncompressed, or your own
`dataset_kwargs` to use a different codec.

`X`/`raw_X` are stored in the VCSC/VCSR layout by default (`format="vcsc"`).
Passing `format="ivcsc"` instead stores them as **IVCSC/IVCSR** -- the same
layout, but with the minor-axis `indices` array delta+varint byte-packed
(inspired by [IVSparse's IVCSC](https://github.com/Seth-Wolfgang/IVSparse)).
This is a *file-storage-only* format: it trades extra CPU on write/read for a
smaller file, and there is no in-memory IVCSC array type to compute with --
`read_h5ad`/`read_zarr` always hand back an ordinary VCSCArray/VCSRArray,
decompressed from IVCSC/IVCSR immediately on load.

```python
va.write_h5ad("archived.h5ad", format="ivcsc")
va4 = vsparse.VCSCAnnData.read_h5ad("archived.h5ad")  # va4.X is a plain VCSCArray
```
`VCSCAnnData` works around this by overriding the `X` property; as a consequence,
operations that need anndata's normal per-element type dispatch on `X` -- slicing into
views, concatenation, most of the scanpy/anndata ecosystem -- are **not** supported while
`X` is VCSC/VCSR-backed. Call `.to_anndata()` first if you need those.

`VCSCArray`/`VCSRArray` are also registered with anndata's on-disk IO registry
directly, so they round-trip through `write_elem`/`read_elem` even when nested inside
a plain `AnnData`'s `.uns`, independent of `VCSCAnnData`.

## Fast IVCSR loading and normalization

When working with large IVCSR-stored `.h5ad` files, `vsparse.load_and_normalize` provides
a fused loader and depth-normalization kernel reproducing the preprocessing in
`parafac2.normalize.prepare_dataset`:

```python
import vsparse

adata = vsparse.load_and_normalize(
    "archived.ivcsr.h5ad",
    min_cell_counts=10.0,
    gene_threshold=0.05,
)
```

This bypasses generic `AnnData` indexing and decompression overhead:
1. Cell counts are computed in $O(n_{\text{unique}})$ time from unique-value group sizes without touching or decoding the packed `indices` byte stream.
2. The delta+varint indices are unpacked in parallel across CPU cores.
3. Row/gene filtering, compaction, and depth normalization ($(\text{cell\_scale}) \times (\text{gene\_sums})$ followed by $\log_{10}(1000x + 1)$) are executed in fused parallel passes directly into the output CSR representation.
