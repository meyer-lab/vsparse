"""vsparse: Value-Compressed Sparse Column/Row (VCSC/VCSR) arrays, with optional AnnData integration.

See :class:`~vsparse.VCSCArray` and :class:`~vsparse.VCSRArray` for the array
types, :func:`~vsparse.from_anndata` to build one from an
:class:`anndata.AnnData` object, and :class:`~vsparse.VCSCAnnData` for an
AnnData subclass that holds a VCSC/VCSR array as ``X`` directly.
"""

# Import first, before anything below pulls in `anndata` -> `zarr`, whose
# `zarr.core.buffer.gpu` speculatively imports `cupy` on its own. CuPy caches
# its CUDA-toolkit-headers path globally the *first* time it's imported in
# the process (see `cupy._environment.get_cuda_path`'s `_cuda_path` sentinel
# cache); if that first import happens via zarr, before
# `_vcs_matmul_cuda._configure_cuda_path` has set `CUDA_PATH`, CUDA kernel
# compilation is broken for the rest of the process even if `CUDA_PATH` is
# set correctly afterward. Importing this module first -- which sets
# `CUDA_PATH` (if discoverable) and then imports `cupy` itself -- ensures
# that first import happens with the right environment already in place.
from vsparse import _vcs_matmul_cuda as _vcs_matmul_cuda  # noqa: I001

from vsparse._anndata import from_anndata, to_layer
from vsparse._anndata_class import VCSCAnnData
from vsparse._base import VCSCArray, VCSRArray
from vsparse._norm_common import RECIPES, Recipe
from vsparse._rapid_load import load_and_normalize, load_packed
from vsparse._vcs_norm import VCSCArrayNormalized, VCSRArrayNormalized

__all__ = [
    "RECIPES",
    "Recipe",
    "VCSCAnnData",
    "VCSCArray",
    "VCSCArrayNormalized",
    "VCSRArray",
    "VCSRArrayNormalized",
    "from_anndata",
    "load_and_normalize",
    "load_packed",
    "to_layer",
]

__version__ = "0.1.0"
