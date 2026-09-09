"""vsparse: Value-Compressed Sparse Column/Row (VCSC/VCSR) arrays, with optional AnnData integration.

See :class:`~vsparse.VCSCArray` and :class:`~vsparse.VCSRArray` for the array
types, :func:`~vsparse.from_anndata` to build one from an
:class:`anndata.AnnData` object, and :class:`~vsparse.VCSCAnnData` for an
AnnData subclass that holds a VCSC/VCSR array as ``X`` directly.
"""

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
