"""An AnnData subclass whose X (and optionally raw X) is VCSC/VCSR-backed."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vsparse import _compression, _io
from vsparse._base import VCSCArray, VCSRArray, _VCSBase
from vsparse._norm_common import DEFAULT_RECIPE, Recipe, resolve_recipe
from vsparse._vcs_norm import VCSCArrayNormalized, VCSRArrayNormalized

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

__all__ = ["VCSCAnnData"]

_VCS_TYPES = (VCSCArray, VCSRArray)
_AnyVCS = _VCSBase
_DF_KEYS = ("obs", "var")
_MAPPING_KEYS = ("obsm", "varm", "obsp", "varp", "layers", "uns")
_FIELD_KEYS = (*_DF_KEYS, *_MAPPING_KEYS)
_STORE_FORMATS = ("vcsc", "ivcsc")

# Names statistics are recorded under in .obs/.varm/.uns -- see .normalized().
_VSPARSE_UNS_KEY = "vsparse"
_VSPARSE_OBS_A = "vsparse_a"
_VSPARSE_VARM_B = "vsparse_b"
_VSPARSE_VARM_C = "vsparse_c"
_VSPARSE_VARM_S = "vsparse_s"


def _as_slice_index(idx: Any, n: int) -> Any:
    """Turn a bare int index into a length-1 slice, matching anndata's own convention."""
    if isinstance(idx, int | np.integer):
        i = int(idx)
        if i < 0:
            i += n
        return slice(i, i + 1, 1)
    return idx


def _subset_1d(v: Any, idx: Any) -> Any:
    if isinstance(v, pd.DataFrame):
        return v.iloc[idx]
    return v[idx]


def _subset_2d(v: Any, oidx: Any, vidx: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, _VCS_TYPES):
        result = v[oidx, vidx]
        # VCSCArray/VCSRArray.__getitem__ only converts to a plain scipy
        # array when both axes collapse to a scalar (oidx/vidx are always
        # slices/arrays here, never bare ints -- see _as_slice_index above),
        # but re-wrap defensively so X/raw_X always stay VCS-backed.
        if not isinstance(result, _VCS_TYPES):
            result = type(v).from_scipy(result)
        return result
    if sp.issparse(v):
        return v[oidx, :][:, vidx]
    return np.asarray(v)[oidx][:, vidx]


def _check_vcs_type(value: Any, name: str) -> None:
    if value is not None and not isinstance(value, _VCS_TYPES):
        raise TypeError(
            f"{name} must be a VCSCArray or VCSRArray, got {type(value).__name__}. "
            f"Build one with VCSCArray.from_scipy(...) or vsparse.from_anndata(...)."
        )


class VCSCAnnData(ad.AnnData):
    """An :class:`~anndata.AnnData` whose ``X`` is a VCSCArray/VCSRArray.

    Standard :class:`~anndata.AnnData` validates every array assigned to
    ``X``/``layers``/etc. against a fixed allowlist of types (dense/sparse/
    dask), so a plain ``AnnData`` cannot hold a :class:`~vsparse.VCSCArray`
    directly. This subclass overrides the ``X`` property to store one
    without going through that validation. A "raw" VCSC/VCSR matrix, if any,
    is kept as ``.raw_X`` -- a plain attribute, *not* wired into anndata's own
    ``.raw``/``Raw`` machinery, which has the same restriction.

    Because of this, most operations that need anndata's normal per-element
    type dispatch on ``X`` -- concatenation, most of scanpy/anndata's
    ecosystem -- are **not** supported while ``X`` is VCSC/VCSR-backed. Call
    :meth:`to_anndata` first to get a fully-featured, ordinary
    ``AnnData``. Indexing (``adata[obs_idx, var_idx]``) *is* supported (see
    :meth:`__getitem__`), but always as an eager copy, not a lazy view --
    anndata's view machinery bypasses the ``X``/``raw_X`` overrides here.

    Persist with :meth:`write_h5ad`/:meth:`write_zarr` and
    :meth:`read_h5ad`/:meth:`read_zarr` (not the top-level
    ``anndata.read_h5ad``/``read_zarr``, which always reconstruct a plain
    ``AnnData`` and would fail validating a VCSC-typed ``X``).
    """

    def __init__(
        self,
        X: _AnyVCS | None = None,
        *,
        raw_X: _AnyVCS | None = None,
        **kwargs: Any,
    ) -> None:
        _check_vcs_type(X, "X")
        _check_vcs_type(raw_X, "raw_X")
        if "raw" in kwargs:
            raise TypeError(
                "VCSCAnnData does not support the standard `raw=` argument; pass `raw_X=` instead."
            )
        self._vcs_X: _AnyVCS | None = None
        self._vcs_raw_X: _AnyVCS | None = None
        self._vcs_norm_cache: dict[Any, Any] = {}
        shape = kwargs.pop("shape", None)
        if shape is None and X is None and "obs" not in kwargs:
            shape = (0, 0)
        elif X is not None or ("obs" in kwargs and "var" in kwargs):
            shape = None
        super().__init__(X=None, shape=shape, **kwargs)
        self._vcs_X = X
        self._vcs_raw_X = raw_X

    # -- X / raw_X ------------------------------------------------------------

    @property
    def X(self) -> _AnyVCS | None:
        return self._vcs_X

    @X.setter
    def X(self, value: Any) -> None:
        if value is not None and not isinstance(value, _VCS_TYPES):
            if sp.issparse(value) or isinstance(value, np.ndarray):
                vcls = VCSCArray if isinstance(value, sp.csc_array | sp.csc_matrix) else VCSRArray
                value = vcls.from_scipy(value)
            else:
                _check_vcs_type(value, "X")
        if (
            value is not None
            and hasattr(self, "_obs")
            and hasattr(self, "_var")
            and value.shape != self.shape
        ):
            raise ValueError(f"X shape {value.shape} does not match adata shape {self.shape}")
        self._vcs_X = value

    @property
    def raw_X(self) -> _AnyVCS | None:
        """The raw/X matrix, as a VCSCArray/VCSRArray (see class docstring)."""
        return self._vcs_raw_X

    @raw_X.setter
    def raw_X(self, value: Any) -> None:
        if value is not None and not isinstance(value, _VCS_TYPES):
            if sp.issparse(value) or isinstance(value, np.ndarray):
                vcls = VCSCArray if isinstance(value, sp.csc_array | sp.csc_matrix) else VCSRArray
                value = vcls.from_scipy(value)
            else:
                _check_vcs_type(value, "raw_X")
        self._vcs_raw_X = value

    # -- indexing / view creation ---------------------------------------------

    def __getitem__(self, index: Any) -> VCSCAnnData:  # ty: ignore[invalid-method-override]
        """Subset by obs/var index (labels, ints, slices, boolean masks), as with plain AnnData.

        Unlike plain AnnData, this always returns a new, eagerly-copied
        object rather than a lazy, memory-sharing view -- anndata's built-in
        view machinery slices the private ``_X`` attribute directly, which
        this class doesn't use (see the class docstring), so it can't be
        reused here. ``X``/``raw_X`` stay VCSC/VCSR-backed either way, via
        that array type's own indexing.
        """
        oidx, vidx = self._normalize_indices(index)
        oidx = _as_slice_index(oidx, self.n_obs)
        vidx = _as_slice_index(vidx, self.n_vars)

        obs = cast(pd.DataFrame, self.obs).iloc[oidx].copy()
        var = cast(pd.DataFrame, self.var).iloc[vidx].copy()

        # obs["vsparse_a"]/varm["vsparse_b"/"c"/"s"] get windowed along with
        # obs/varm below, same as any other per-cell/per-gene column -- each
        # kept cell/gene keeps its own precomputed factor. But population-level
        # aspects of those statistics (e.g. the median depth or per-gene mean
        # baked into them) were derived from the *pre-subset* population, so
        # they no longer reflect this narrower one -- mark them stale rather
        # than silently pretending they were computed fresh. Call
        # .normalized(recalculate=True) to recompute for this subset, or
        # recalculate=False to keep using these (stale) values.
        uns = self.uns
        if _VSPARSE_UNS_KEY in uns:
            uns = {**uns, _VSPARSE_UNS_KEY: {**uns[_VSPARSE_UNS_KEY], "stale": True}}

        return VCSCAnnData(
            X=_subset_2d(self._vcs_X, oidx, vidx),
            raw_X=_subset_2d(self._vcs_raw_X, oidx, vidx),
            obs=obs,
            var=var,
            uns=uns,
            obsm={k: _subset_1d(v, oidx) for k, v in self.obsm.items() if k is not None},
            varm={k: _subset_1d(v, vidx) for k, v in self.varm.items() if k is not None},
            obsp={k: _subset_2d(v, oidx, oidx) for k, v in self.obsp.items() if k is not None},
            varp={k: _subset_2d(v, vidx, vidx) for k, v in self.varp.items() if k is not None},
            layers={k: _subset_2d(v, oidx, vidx) for k, v in self.layers.items() if k is not None},
        )

    # -- normalization ----------------------------------------------------------

    def normalized(self, view: str | Recipe = DEFAULT_RECIPE, *, recalculate: bool = True) -> Any:
        """A normalized view of ``X`` -- see :meth:`vsparse._base._VCSBase.normalized`.

        Also records the recipe's statistics -- per-cell ``a`` in
        ``obs["vsparse_a"]``, per-gene ``b``/``c``/``s`` in
        ``varm["vsparse_b"]``/``["vsparse_c"]``/``["vsparse_s"]``, and the
        active recipe name plus a ``stale`` flag in ``uns["vsparse"]`` -- so
        they persist across :meth:`write_h5ad`/:meth:`write_zarr` and can be
        reused instead of recomputed.

        ``recalculate=False`` reuses whatever is already recorded for
        ``view``: first an in-memory view already built this session (however
        it was built), else what's stored in ``obs``/``varm``/``uns`` (built
        fresh, but skipping the ``O(nnz)`` statistics passes) if it matches
        ``view`` and the current shape. ``uns["vsparse"]["stale"]`` is set to
        ``True`` after indexing (see :meth:`__getitem__`), since population
        statistics like a depth median or a per-gene mean no longer reflect
        the subset -- ``recalculate=False`` reuses them anyway; the default
        ``recalculate=True`` always recomputes fresh (and clears ``stale``).
        """
        if self._vcs_X is None:
            raise ValueError("normalized() requires X to be set")
        recipe = resolve_recipe(view)
        cache = self._vcs_norm_cache
        if not recalculate:
            cached = cache.get(recipe)
            if cached is not None:
                return cached
            stored = self.uns.get(_VSPARSE_UNS_KEY)
            if (
                stored is not None
                and stored.get("recipe") == recipe.name
                and _VSPARSE_OBS_A in self.obs
                and len(self.obs[_VSPARSE_OBS_A]) == self.n_obs
                and _VSPARSE_VARM_B in self.varm
                and _VSPARSE_VARM_C in self.varm
                and _VSPARSE_VARM_S in self.varm
                and len(self.varm[_VSPARSE_VARM_B]) == self.n_vars
            ):
                nview_cls = (
                    VCSCArrayNormalized
                    if isinstance(self._vcs_X, VCSCArray)
                    else VCSRArrayNormalized
                )
                nview = nview_cls.from_stats(
                    self._vcs_X,
                    recipe,
                    a=np.asarray(self.obs[_VSPARSE_OBS_A], dtype=np.float64),
                    b=np.asarray(self.varm[_VSPARSE_VARM_B], dtype=np.float64).reshape(-1),
                    c=np.asarray(self.varm[_VSPARSE_VARM_C], dtype=np.float64).reshape(-1),
                    s=np.asarray(self.varm[_VSPARSE_VARM_S], dtype=np.float64).reshape(-1),
                    stale=bool(stored.get("stale", False)),
                )
                cache[recipe] = nview
                return nview

        nview = self._vcs_X.normalized(recipe, recalculate=True)
        cache[recipe] = nview
        self.obs[_VSPARSE_OBS_A] = nview.a
        self.varm[_VSPARSE_VARM_B] = nview.b
        self.varm[_VSPARSE_VARM_C] = nview.c
        self.varm[_VSPARSE_VARM_S] = nview.s
        self.uns[_VSPARSE_UNS_KEY] = {"recipe": recipe.name, "stale": False}
        return nview

    # -- conversion -------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        format: str = "csc",
        raw_format: str | None = None,
        *,
        include_raw: bool = True,
    ) -> VCSCAnnData:
        """Build from a regular :class:`~anndata.AnnData`, compressing X (and raw.X)."""
        vcls = VCSCArray if format == "csc" else VCSRArray
        X = vcls.from_scipy(adata.X) if adata.X is not None else None
        raw_X = None
        if include_raw and adata.raw is not None:
            rcls = VCSCArray if (raw_format or format) == "csc" else VCSRArray
            raw_X = rcls.from_scipy(adata.raw.X)
        return cls(
            X=X,
            raw_X=raw_X,
            obs=cast(pd.DataFrame, adata.obs).copy(),
            var=cast(pd.DataFrame, adata.var).copy(),
            uns=adata.uns,
            obsm={k: v for k, v in adata.obsm.items() if k is not None},
            varm={k: v for k, v in adata.varm.items() if k is not None},
            obsp={k: v for k, v in adata.obsp.items() if k is not None},
            varp={k: v for k, v in adata.varp.items() if k is not None},
            layers={k: v for k, v in adata.layers.items() if k is not None},
        )

    def to_anndata(self) -> ad.AnnData:
        """Decompress to a regular, fully-featured :class:`~anndata.AnnData`."""
        obs = cast(pd.DataFrame, self.obs)
        var = cast(pd.DataFrame, self.var)
        out = ad.AnnData(
            X=self._vcs_X.to_scipy() if self._vcs_X is not None else None,
            obs=obs.copy(),
            var=var.copy(),
            uns=self.uns,
            obsm=cast(Any, {k: v for k, v in self.obsm.items() if k is not None}),
            varm=cast(Any, {k: v for k, v in self.varm.items() if k is not None}),
            obsp=cast(Any, {k: v for k, v in self.obsp.items() if k is not None}),
            varp=cast(Any, {k: v for k, v in self.varp.items() if k is not None}),
            layers=cast(Any, {k: v for k, v in self.layers.items() if k is not None}),
        )
        if self._vcs_raw_X is not None:
            out.raw = ad.AnnData(X=self._vcs_raw_X.to_scipy(), obs=obs.copy(), var=var.copy())
        return out

    # -- persistence --------------------------------------------------------------
    #
    # Implemented by hand (field-by-field, via anndata.io.write_elem/read_elem)
    # rather than delegating to anndata.write_h5ad/write_zarr/read_h5ad/read_zarr:
    # those either construct a plain AnnData (crashes coerce_array on a VCSC-typed
    # X) or dispatch on the exact Python type of `self` (VCSCAnnData isn't
    # registered as an "anndata"-encoded type, only AnnData is).

    def _write_group(
        self,
        g: Any,
        *,
        format: str = "vcsc",
        convert_strings_to_categoricals: bool = True,
        dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        if format not in _STORE_FORMATS:
            raise ValueError(f"format must be one of {_STORE_FORMATS}, got {format!r}")
        if convert_strings_to_categoricals:
            # Writing field-by-field skips what anndata's own writers do here,
            # leaving low-cardinality columns as one string per row.
            self.strings_to_categoricals()
        write_array = ad.io.write_elem if format == "vcsc" else _io.write_ivcs_elem
        if self._vcs_X is not None:
            write_array(g, "X", self._vcs_X, dataset_kwargs=dataset_kwargs)
        if self._vcs_raw_X is not None:
            write_array(g, "raw_X", self._vcs_raw_X, dataset_kwargs=dataset_kwargs)
        for key in _DF_KEYS:
            ad.io.write_elem(g, key, getattr(self, key), dataset_kwargs=dataset_kwargs)
        for key in _MAPPING_KEYS:
            mapping = {k: v for k, v in getattr(self, key).items() if k is not None}
            ad.io.write_elem(g, key, mapping, dataset_kwargs=dataset_kwargs)
        g.attrs["encoding-type"] = "anndata"
        g.attrs["encoding-version"] = "0.1.0"

    @classmethod
    def _read_group(cls, g: Any) -> VCSCAnnData:
        # X/raw_X read back as plain VCSCArray/VCSRArray regardless of whether
        # they were stored "vcsc" or "ivcsc" -- the registry dispatches on the
        # encoding-type attr each group was written with, not on how it's read.
        kwargs = {k: ad.io.read_elem(g[k]) for k in _FIELD_KEYS if k in g}
        X = cast("_AnyVCS | None", ad.io.read_elem(g["X"]) if "X" in g else None)
        raw_X = cast("_AnyVCS | None", ad.io.read_elem(g["raw_X"]) if "raw_X" in g else None)
        return cls(X=X, raw_X=raw_X, **kwargs)

    def write_h5ad(  # ty: ignore[invalid-method-override]
        self,
        filename: str | PathLike[str],
        *,
        format: str = "vcsc",
        convert_strings_to_categoricals: bool = True,
        dataset_kwargs: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Write to ``.h5ad``. Read back with :meth:`read_h5ad`.

        Parameters
        ----------
        format
            ``"vcsc"`` (default) stores ``X``/``raw_X`` with plain int arrays
            for the minor-axis indices. ``"ivcsc"`` (IVCSC/IVCSR) instead
            byte-packs them (delta + varint encoding) for a smaller file, at
            the cost of extra work on write/read. Either way, ``X``/``raw_X``
            come back from :meth:`read_h5ad` as ordinary VCSCArray/VCSRArray
            objects -- ``"ivcsc"`` is purely an on-disk storage format.
        convert_strings_to_categoricals
            Convert ``obs``/``var`` string columns to categorical in place
            before writing, as ``anndata``'s own writers do. Only columns
            with fewer categories than rows are converted.
        dataset_kwargs
            Passed to ``h5py.Group.create_dataset`` for every array written.
            Defaults to Blosc2+LZ4 compression; pass ``{}`` to store
            uncompressed. Either way, compression is only ever applied to
            numeric arrays -- see :func:`vsparse._compression.numeric_only_compression`.
        """
        import h5py

        if dataset_kwargs is None:
            dataset_kwargs = _compression.h5_dataset_kwargs()
        with (
            _compression.numeric_only_compression("h5"),
            h5py.File(filename, "w") as f,
        ):
            self._write_group(
                f,
                format=format,
                convert_strings_to_categoricals=convert_strings_to_categoricals,
                dataset_kwargs=dataset_kwargs,
            )

    @classmethod
    def read_h5ad(cls, filename: str | PathLike[str]) -> VCSCAnnData:
        """Read a file written by :meth:`write_h5ad`."""
        import h5py

        with h5py.File(filename, "r") as f:
            return cls._read_group(f)

    def write_zarr(
        self,
        store: Any,
        *,
        format: str = "vcsc",
        convert_strings_to_categoricals: bool = True,
        dataset_kwargs: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Write to a zarr store. Read back with :meth:`read_zarr`.

        See :meth:`write_h5ad` for ``format``/``convert_strings_to_categoricals``/
        ``dataset_kwargs`` (including the numeric-only compression behavior);
        the default compression here is Blosc+LZ4 via ``numcodecs``.
        """
        import zarr

        if dataset_kwargs is None:
            dataset_kwargs = _compression.zarr_dataset_kwargs()
        with _compression.numeric_only_compression("zarr"):
            f = zarr.open_group(store, mode="w")
            self._write_group(
                f,
                format=format,
                convert_strings_to_categoricals=convert_strings_to_categoricals,
                dataset_kwargs=dataset_kwargs,
            )

    @classmethod
    def read_zarr(cls, store: Any) -> VCSCAnnData:
        """Read a store written by :meth:`write_zarr`."""
        import zarr

        f = zarr.open_group(store, mode="r")
        return cls._read_group(f)
