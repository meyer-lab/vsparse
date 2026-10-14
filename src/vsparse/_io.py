"""Registers VCSCArray/VCSRArray with anndata's on-disk IO registry.

This lets a VCSCArray/VCSRArray be written/read anywhere anndata's generic
element writer (``anndata.io.write_elem``/``read_elem``) is used -- e.g.
nested inside ``adata.uns``, or as ``X``/a custom top-level key on a
:class:`~vsparse.VCSCAnnData` -- and survive a round trip through HDF5 or zarr
without any manual (de)serialization.

Both formats store the array's statistics sidecar (see :mod:`vsparse._stats`)
as a ``stats`` subgroup next to the array data. Readers attach it when it's
present and valid, and otherwise leave the array to recompute it on demand, so
files without one -- written by earlier versions, or by other tools -- still
read normally.

This relies on ``anndata._io.specs.registry``, an internal (not yet public)
extension mechanism -- the same one anndata itself uses to support
dask/awkward/cupy arrays. It is not guaranteed stable across anndata
versions; if this module fails to import, it means anndata's registry
internals changed shape.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import h5py
import numpy as np
import zarr

from vsparse import _ivcsc, _stats
from vsparse._base import VCSCArray, VCSRArray, _VCSBase
from vsparse._compression import read_array
from vsparse._indexutils import smallest_index_dtype

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata._io.specs.registry import Reader, Writer

    GroupStorageType = h5py.Group | zarr.Group

__all__ = ["write_ivcs_elem"]

_ARRAY_KEYS = ("major_ptr", "values", "value_ptr", "indices")
_SPEC_VERSION = "0.1.0"
_IVCS_SPEC_VERSION = "0.1.0"
_IVCS_SPEC_NAMES: dict[type[_VCSBase], str] = {VCSCArray: "ivcsc", VCSRArray: "ivcsr"}


def _write_vcs(
    f: GroupStorageType,
    k: str,
    v: _VCSBase,
    *,
    _writer: Writer,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    g = f.require_group(k)
    g.attrs["shape"] = v.shape
    for name in _ARRAY_KEYS:
        _writer.write_elem(g, name, getattr(v, name), dataset_kwargs=dataset_kwargs)
    _stats.write_stats(g, v.stats, write_elem=_writer.write_elem, dataset_kwargs=dataset_kwargs)


def _read_shape(elem: GroupStorageType) -> tuple[int, int]:
    shape_vals = [int(s) for s in np.asarray(elem.attrs["shape"]).tolist()]
    return (shape_vals[0], shape_vals[1])


def _with_stats(arr: _VCSBase, elem: GroupStorageType) -> _VCSBase:
    """Attach the sidecar stored under ``elem``, if there's a usable one -- see :mod:`vsparse._stats`."""
    arr._stats = _stats.read_stats(elem, arr.shape)
    return arr


def _make_read_vcs(cls: type[_VCSBase]):
    def _read(elem: GroupStorageType, *, _reader: Reader) -> _VCSBase:
        arrays = {name: read_array(elem[name]) for name in _ARRAY_KEYS}
        arr = cls(
            _read_shape(elem),
            arrays["major_ptr"],
            arrays["values"],
            arrays["value_ptr"],
            arrays["indices"],
        )
        return _with_stats(arr, elem)

    return _read


def write_ivcs_elem(
    f: GroupStorageType,
    k: str,
    v: _VCSBase,
    *,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    """Write ``v`` in the byte-packed IVCSC/IVCSR on-disk format.

    This trades write/read cost for a smaller file: ``indices`` is delta+
    varint packed via :mod:`vsparse._ivcsc` instead of stored as a plain int
    array. It is meant for archival storage only -- reading it back (through
    the registry, e.g. via ``anndata.io.read_elem``) always reconstructs a
    normal VCSCArray/VCSRArray with a plain ``indices`` array.

    ``v``'s statistics sidecar (:attr:`~vsparse.VCSCArray.stats`) is written
    alongside, computing it first if ``v`` doesn't carry it yet.

    Unlike the VCSC/VCSR codec registered below, this is not reachable
    through anndata's generic ``write_elem`` type dispatch (which can only
    route by the Python type of ``v``, and both formats share the same
    VCSCArray/VCSRArray types) -- call it directly, e.g. from
    :meth:`~vsparse.VCSCAnnData.write_h5ad`.
    """
    g = f.require_group(k)
    g.attrs["shape"] = v.shape
    # Record the narrowest dtype that can address the minor axis rather than
    # whatever the in-memory array happens to carry.
    in_memory = v.indices.dtype
    narrowest = smallest_index_dtype(v.n_minor)
    stored_dtype = narrowest if narrowest.itemsize < in_memory.itemsize else in_memory
    g.attrs["indices_dtype"] = stored_dtype.name
    for name in ("major_ptr", "values", "value_ptr"):
        ad.io.write_elem(g, name, getattr(v, name), dataset_kwargs=dataset_kwargs)
    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    ad.io.write_elem(g, "packed_indices", packed, dataset_kwargs=dataset_kwargs)
    _stats.write_stats(g, v.stats, write_elem=ad.io.write_elem, dataset_kwargs=dataset_kwargs)
    g.attrs["encoding-type"] = "ivcsc" if v._format == "csc" else "ivcsr"
    g.attrs["encoding-version"] = _IVCS_SPEC_VERSION


def _make_read_ivcs(cls: type[_VCSBase]):
    def _read(elem: GroupStorageType, *, _reader: Reader) -> _VCSBase:
        major_ptr = read_array(elem["major_ptr"])
        values = read_array(elem["values"])
        value_ptr = read_array(elem["value_ptr"])
        packed = read_array(elem["packed_indices"])
        dtype = np.dtype(cast(str, elem.attrs["indices_dtype"]))
        indices = _ivcsc.unpack_indices(value_ptr, packed, dtype)
        arr = cls(_read_shape(elem), major_ptr, values, value_ptr, indices)
        return _with_stats(arr, elem)

    return _read


def _register() -> None:
    from anndata._io.specs.registry import _REGISTRY, IOSpec

    for store_type in (h5py.Group, zarr.Group):
        for cls, spec_name in ((VCSCArray, "vcsc"), (VCSRArray, "vcsr")):
            spec = IOSpec(spec_name, _SPEC_VERSION)
            if not _REGISTRY.has_write(store_type, cls, frozenset()):
                _REGISTRY.register_write(store_type, cls, spec)(_write_vcs)
            if not _REGISTRY.has_read(store_type, spec, frozenset()):
                _REGISTRY.register_read(store_type, spec, frozenset())(_make_read_vcs(cls))
        for cls, spec_name in _IVCS_SPEC_NAMES.items():
            spec = IOSpec(spec_name, _IVCS_SPEC_VERSION)
            if not _REGISTRY.has_read(store_type, spec, frozenset()):
                _REGISTRY.register_read(store_type, spec, frozenset())(_make_read_ivcs(cls))


_register()
