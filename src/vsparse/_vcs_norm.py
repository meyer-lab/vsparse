"""Normalized *views* of :class:`~vsparse.VCSCArray`/:class:`~vsparse.VCSRArray`.

:class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized` wrap a plain
(unpacked) VCSC/VCSR array and behave like a normalized matrix -- read-depth
normalized, transformed, optionally centered/scaled, per one of
:data:`vsparse._norm_common.RECIPES` (the default, ``"parafac2"``, matches
what :func:`vsparse._rapid_load.load_and_normalize` builds) -- without ever
materializing it. See :mod:`vsparse._norm_common` for the shared recipe/
statistics/materialization logic (:class:`~vsparse._norm_common.NormalizedViewBase`)
and :mod:`vsparse._vcs_matmul` for the matmul kernels these use (a direct
per-nonzero walk over the already-decoded ``indices`` array).

Call :meth:`~_VCSNormalizedBase.to_gpu` for a device-resident counterpart
(:class:`~vsparse._vcs_matmul_cuda.GPUNormalizedVCS`) whose own ``@``/
``__rmatmul__`` run CUDA kernels in float32 -- see
:mod:`vsparse._vcs_matmul_cuda`. This is opt-in, not automatic: ``@``/
``__rmatmul__`` on the CPU view here always use the float64 Numba kernels,
regardless of whether a CUDA device is present, so existing behavior and
precision are unaffected by whether CuPy/CUDA happen to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vsparse._norm_common import DEFAULT_RECIPE, NormalizedViewBase, Recipe

if TYPE_CHECKING:
    from vsparse._base import _VCSBase

__all__ = ["VCSCArrayNormalized", "VCSRArrayNormalized"]


class _VCSNormalizedBase(NormalizedViewBase):
    """Shared implementation for :class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized`."""

    __slots__ = ("_dual_arr",)

    _dual_arr: _VCSBase | None

    def __init__(
        self, arr: _VCSBase, recipe: str | Recipe = DEFAULT_RECIPE, *, stale: bool = False
    ) -> None:
        super().__init__(arr, recipe, stale=stale)
        self._init_extra()

    def _init_extra(self) -> None:
        # Opposite-format copy of `arr`, cached by vsparse._vcs_matmul when
        # regrouping the whole array fits one chunk's budget.
        self._dual_arr = None

    def __matmul__(self, other: Any) -> Any:
        """``self @ other`` for a dense ``other`` -- see :mod:`vsparse._vcs_matmul`."""
        from vsparse._vcs_matmul import normalized_at_dense

        return normalized_at_dense(self, other)

    def __rmatmul__(self, other: Any) -> Any:
        """``other @ self`` for a dense ``other`` -- see :mod:`vsparse._vcs_matmul`."""
        from vsparse._vcs_matmul import dense_at_normalized

        return dense_at_normalized(self, other)

    def to_gpu(self) -> Any:
        """A device-resident :class:`~vsparse._vcs_matmul_cuda.GPUNormalizedVCS` copy of this view.

        Explicit and one-way: building it uploads/regroups this view's
        arrays onto the GPU once (cached on the returned object across
        repeated ``@``/``__rmatmul__`` calls against it), but nothing about
        the CPU view itself changes, and nothing here is created implicitly
        just because a CUDA device happens to be available. Requires CuPy
        and a working CUDA device -- see
        :func:`vsparse._vcs_matmul_cuda.cuda_available`.
        """
        from vsparse._vcs_matmul_cuda import GPUNormalizedVCS

        return GPUNormalizedVCS(self)


class VCSCArrayNormalized(_VCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of a :class:`~vsparse.VCSCArray`."""

    __slots__ = ()
    _format = "csc"


class VCSRArrayNormalized(_VCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of a :class:`~vsparse.VCSRArray`."""

    __slots__ = ()
    _format = "csr"
