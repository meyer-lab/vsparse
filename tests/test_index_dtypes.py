from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSCArray, VCSRArray
from vsparse._indexutils import smallest_index_dtype
from vsparse._rapid_load import _filter_and_compact

INT32_MAX = np.iinfo(np.int32).max


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _with_int64_indices(mat):
    out = mat.copy()
    out.indices = out.indices.astype(np.int64)
    out.indptr = out.indptr.astype(np.int64)
    return out


@pytest.mark.parametrize(("n", "expected"), [(INT32_MAX, np.int32), (INT32_MAX + 1, np.int64)])
def test_dtype_switches_at_the_int32_boundary(n, expected):
    assert smallest_index_dtype(n) == np.dtype(expected)


def test_from_scipy_narrows_int64_indices(dense, vcls):
    """An int64-indexed input is stored as int32 when the minor axis fits."""
    mat = _with_int64_indices(_scipy_for(vcls, dense))
    v = vcls.from_scipy(mat)

    assert v.indices.dtype == np.int32
    assert v.indices.nbytes == 4 * v.nnz
    np.testing.assert_allclose(v.toarray(), dense)


def test_minor_axis_beyond_int32_keeps_int64(vcls):
    """Narrowing an axis that genuinely needs int64 would truncate the indices."""
    minor_idx = np.array([0, INT32_MAX + 5, 1, INT32_MAX + 9], dtype=np.int64)
    shape = (INT32_MAX + 10, 2) if vcls is VCSCArray else (2, INT32_MAX + 10)
    mat_cls = sp.csc_array if vcls is VCSCArray else sp.csr_array
    mat = mat_cls(
        (np.array([1.0, 2.0, 3.0, 4.0]), minor_idx, np.array([0, 2, 4], dtype=np.int64)),
        shape=shape,
    )

    v = vcls.from_scipy(mat)
    assert v.indices.dtype == np.int64
    np.testing.assert_array_equal(np.sort(v.indices), np.sort(minor_idx))


def test_construction_never_widens_narrower_indices(vcls):
    shape = (4, 3) if vcls is VCSCArray else (3, 4)
    v = vcls(
        shape,
        major_ptr=np.array([0, 1, 2, 3], dtype=np.int64),
        values=np.array([1.0, 2.0, 3.0]),
        value_ptr=np.array([0, 1, 2, 3], dtype=np.int64),
        indices=np.array([0, 1, 2], dtype=np.int16),
    )
    assert v.indices.dtype == np.int16


def test_out_of_bounds_index_raises_rather_than_truncating(vcls):
    """Narrowing happens after validation, so a bad index cannot wrap silently."""
    shape = (4, 1) if vcls is VCSCArray else (1, 4)
    with pytest.raises(ValueError, match="out of bounds"):
        vcls(
            shape,
            major_ptr=np.array([0, 1], dtype=np.int64),
            values=np.array([1.0]),
            value_ptr=np.array([0, 1], dtype=np.int64),
            indices=np.array([2**32 + 1], dtype=np.int64),
        )


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda v, d: v.astype(np.float32), id="astype"),
        pytest.param(lambda v, d: v[:, ::2], id="select_minor"),
        pytest.param(lambda v, d: v + v, id="add"),
        pytest.param(lambda v, d: v * 2.0, id="scalar_mul"),
        pytest.param(lambda v, d: v._transpose_major(), id="transpose_major"),
    ],
)
def test_derived_arrays_stay_narrow(vcls, dense, op):
    """Narrowing lives in __init__, so every op building a new array inherits it."""
    v = vcls.from_scipy(_with_int64_indices(_scipy_for(vcls, dense)))
    assert op(v, dense).indices.dtype == np.int32


@pytest.mark.parametrize("fmt", ["vcsc", "ivcsc"])
def test_roundtrip_preserves_values_and_stores_narrow_indices(tmp_path, dense, fmt):
    import anndata as ad

    va = VCSCAnnData.from_anndata(ad.AnnData(X=sp.csr_array(dense)), format="csr")
    path = tmp_path / f"data.{fmt}.h5ad"
    va.write_h5ad(path, format=fmt)

    back = VCSCAnnData.read_h5ad(path)
    assert isinstance(back.X, VCSRArray)
    assert back.X.indices.dtype == np.int32
    np.testing.assert_allclose(back.X.toarray(), dense)


def test_packed_write_narrows_a_wide_in_memory_array(tmp_path, dense):
    """The write path re-derives the dtype rather than trusting the array it is handed."""
    import anndata as ad

    va = VCSCAnnData.from_anndata(ad.AnnData(X=sp.csr_array(dense)), format="csr")
    assert isinstance(va.X, VCSRArray)
    va.X.indices = va.X.indices.astype(np.int64)

    path = tmp_path / "wide.h5ad"
    va.write_h5ad(path, format="ivcsc")

    back = VCSCAnnData.read_h5ad(path)
    assert isinstance(back.X, VCSRArray)
    assert back.X.indices.dtype == np.int32
    np.testing.assert_allclose(back.X.toarray(), dense)


def _small_filter_inputs():
    dense = np.array(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0], [5.0, 0.0, 6.0, 0.0]],
        dtype=np.float32,
    )
    return sp.csr_array(dense), np.array([True, False, True]), np.array([True, False, True, False])


def test_filter_and_compact_uses_int32_when_both_bounds_fit():
    X, cell_mask, gene_mask = _small_filter_inputs()
    new_indptr, out_indices, out_data, kept_rows, n_kept = _filter_and_compact(
        X.indptr, X.indices, X.data, cell_mask, gene_mask
    )

    assert new_indptr.dtype == np.int32
    assert out_indices.dtype == np.int32
    assert n_kept == 2
    np.testing.assert_array_equal(kept_rows, [0, 2])
    np.testing.assert_allclose(out_data, [1.0, 2.0, 5.0, 6.0])
    np.testing.assert_array_equal(out_indices, [0, 1, 0, 1])


def test_gene_indices_stay_int32_when_pointers_need_int64(monkeypatch):
    """Forced, since a >INT32_MAX-nonzero matrix cannot be allocated in a test."""
    import vsparse._rapid_load as rapid_load

    X, cell_mask, gene_mask = _small_filter_inputs()
    nnz_in = int(X.indices.shape[0])
    real = rapid_load.smallest_index_dtype

    monkeypatch.setattr(
        rapid_load,
        "smallest_index_dtype",
        lambda n: np.dtype(np.int64) if n == nnz_in else real(n),
    )
    new_indptr, out_indices, _, _, _ = _filter_and_compact(
        X.indptr, X.indices, X.data, cell_mask, gene_mask
    )

    assert new_indptr.dtype == np.int64
    assert out_indices.dtype == np.int32
    np.testing.assert_array_equal(out_indices, [0, 1, 0, 1])
