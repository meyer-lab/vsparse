"""CUDA transfer and matmul kernels for the normalized VCS views.

:meth:`~vsparse._norm_common.NormalizedViewBase.to_gpu` moves a normalized
view onto an NVIDIA GPU as a :class:`CudaNormalizedView`, which supports the
same ``@``/``__rmatmul__`` the CPU view does. The array keeps its
value-compressed layout on the device -- the transfer is the same five arrays
the host holds (``major_ptr``/``values``/``value_ptr``/``indices`` plus the
``O(n_rows + n_cols)`` statistics), never an expanded one-float-per-nonzero
copy. That matters more on a GPU than on a host, since device memory is the
scarcer resource: the whole reason to hold a count matrix in VCS form is that
the expanded form does not fit.

Keeping the layout is also what forces custom kernels. cuSPARSE speaks CSR and
CSC only, so nothing in CuPy can walk the two-level
``major_ptr -> (values, value_ptr) -> indices`` structure; the four kernels
below do, and :meth:`CudaNormalizedView.to_cupy_sparse` is the expanded
alternative they are measured against (see ``benchmarks/cases.py``).

Everything floating-point on the device is ``float32``
-- see :data:`DEVICE_DTYPE` and the precision note on :func:`to_gpu`.

Kernel structure
----------------

As on the host (:mod:`vsparse._vcs_matmul`), ``A_norm = Delta + 1 (x) (-c*s)``
with ``Delta`` nonzero only on the stored entries, so only ``Delta @ B`` /
``B @ Delta`` need a kernel; the rank-1 term is a CuPy one-liner.

Each of the four kernels is one warp per major slice, grid-strided (or per
*chunk* of a slice, see below). The warp's 32 lanes split two ways: ``ww`` lanes (the smallest power of two at or above
the dense width, capped at 32) cover the width, and the remaining
``groups = 32 / ww`` lane-groups walk the slice's nonzeros in stride. A
PARAFAC2-shaped width of 8 would otherwise leave 24 of 32 lanes idle on every
nonzero; splitting recovers them.

The four are the two independent binary choices the host kernels also make:

- *which axis the statistics hang off* -- VCSR's major slice is a row, so
  ``row_scale`` is uniform across the slice and ``gene_scale``/``col_post_scale``
  vary per nonzero; VCSC is the reverse.
- *aligned or not* -- when the output's disjoint axis is the major axis
  (VCSR for ``self @ B``, VCSC for ``B @ self``) each warp owns its output row
  outright, so it accumulates in registers, reduces across lane-groups with
  ``__shfl_down_sync``, and stores once. In the misaligned direction the output
  index varies per nonzero, so the kernel ``atomicAdd``s instead.

The host's answer to that misaligned direction -- a private per-thread
accumulator, capped by :func:`~vsparse._ops.accumulator_threads` -- has no
analogue here and needs none: ``nthreads`` on a GPU is in the tens of
thousands, so a private copy each is out of the question, while float32
``atomicAdd`` is a single hardware instruction and contention is low when the
scattered index is a cell id. The cost is that the misaligned kernels sum in
nondeterministic order, so repeated runs can differ in the last bits. The
aligned kernels are deterministic.

Two things are chosen per launch, not per kernel, from profiling these kernels
on a 60,000 x 2,000 matrix at 5% density (see ``benchmarks/cases.py``):

- *Accumulators.* An aligned kernel keeps ``acc`` partial sums per lane, enough
  for the width to be covered in one pass (:func:`_accumulators`). It used to
  always keep four, predicated off past the width: at a width of 16 that was
  three dead multiply-adds per nonzero and enough registers (50) to cap the SM
  at four blocks, where one accumulator fits in 40 and allows six. The kernels
  are compiled once per accumulator count in use, so each width gets its own.
- *Chunks.* A matrix with few major slices -- a few thousand genes in a VCSC --
  launches under one wave of warps, and a kernel that spends its time waiting
  on gathers needs many resident warps to hide them: ``B @ VCSC`` (2,000
  slices) achieved 48% occupancy where the kernel allowed 83%.
  :func:`_split_factor` cuts each slice
  into chunks of its nonzeros, one warp each, when that is the case. The aligned
  kernels then write one partial row per chunk, summed afterwards in a fixed
  order, so they stay deterministic; the misaligned ones just ``atomicAdd`` as
  they already did. Matrices with enough slices keep the original one-warp-per-
  slice kernel, a separate compilation, because the chunk bounds cost ~25% at
  the widest widths for a split of one where they do nothing.
- *Vector loads.* The aligned kernels gather one dense-operand row per nonzero,
  and when the width is a multiple of four they do it with ``float4`` loads
  (:data:`_VECTOR_WIDTH`), so ``width / 4`` lanes cover the width. That is a
  third compilation axis, and it shifts what "lanes across the width" means for
  the accumulator count and the split threshold, which are both expressed in the
  vector units actually loaded.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vsparse._norm_common import _G_CODES, G_IDENTITY, G_LOG1P, G_LOG1P_1000X, G_SQRT

if TYPE_CHECKING:
    from vsparse._norm_common import NormalizedViewBase

__all__ = ["DEVICE_DTYPE", "CudaNormalizedView", "cuda_is_available", "to_gpu"]

#: Every floating-point array on the device, without exception: values, the
#: statistics, the dense operand and the result.
DEVICE_DTYPE = np.float32

#: Threads per block, as (lanes, warps). One warp handles one major slice.
_WARP = 32
_WARPS_PER_BLOCK = 8

#: Width accumulators each lane may hold in the aligned kernels. A warp covers
#: ``ww * _ACC_MAX`` columns of the dense operand per pass over the slice, so
#: any width up to 128 is a single pass; wider needs one pass per chunk. This
#: is only the cap: the kernels are compiled once per accumulator count
#: actually needed (see :func:`_accumulators`), so a width that needs one
#: accumulator pays for one, not for four predicated-off ones.
_ACC_MAX = 4

#: Blocks to launch. Capped rather than one-per-slice so a matrix with millions
#: of cells doesn't build an enormous grid for slices that are mostly empty;
#: the kernels grid-stride, so any cap is correct.
_MAX_BLOCKS = 4096

#: Splitting a major slice across several warps. A matrix with few slices (a
#: 2,000-gene VCSC, say) otherwise launches a few hundred blocks, under a
#: single wave, and the kernels -- which spend their time waiting on gathers --
#: need many resident warps to hide that. :func:`_split_factor` asks for about
#: ``_TARGET_WARPS_PER_SM`` work items per SM (a couple of waves, so the tail
#: balances), capped at ``_MAX_SPLIT`` chunks per slice and kept from cutting a
#: slice below ``_MIN_CHUNK_NNZ`` nonzeros, where the per-chunk setup (a binary
#: search) would cost more than the parallelism buys. And a slice is not split
#: at all when ``ww`` (the lanes covering the width) is below ``_MIN_SPLIT_WW``:
#: the other ``32 / ww`` lane-groups are already walking nonzeros side by side,
#: so the warp is not short of loads in flight, and splitting measured 10-15%
#: slower at widths 1-4 against 10-20% faster from width 6 up.
_TARGET_WARPS_PER_SM = 128
_MAX_SPLIT = 16
_MIN_CHUNK_NNZ = 256
_MIN_SPLIT_WW = 8

#: The aligned kernels gather each dense-operand row with ``float4`` loads when
#: the width is a multiple of this: a quarter of the load instructions, which
#: measured 1.5-2.4x faster at widths 8-64 and ~10-30% at 128-256. CuPy
#: allocations are 256-byte aligned, so with ``width * 4`` a multiple of 16
#: bytes every row is too. Other widths keep scalar loads. The misaligned
#: kernels stay scalar: ``atomicAdd(float4*)`` needs sm_90 and gained <10%,
#: since the atomics, not the ``B`` loads, dominate them.
_VECTOR_WIDTH = 4


def cuda_is_available() -> bool:
    """Whether CuPy is importable and a CUDA device is actually present."""
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - driver present but unusable
        return False


def _require_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "vsparse CUDA support requires CuPy; install it with `pip install vsparse[cuda]`"
        ) from exc
    return cp


# -- kernel source ------------------------------------------------------------
#
# Assembled from fragments rather than written out four times: the four kernels
# differ only in the two substitutions described in the module docstring, and
# spelling each one out in full would be ~200 lines of near-identical CUDA.

_PREAMBLE = f"""
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float apply_g(const float x, const int g_code) {{
    if (g_code == {G_LOG1P}) return log1pf(x);
    if (g_code == {G_LOG1P_1000X}) return log10f(1.0f + 1000.0f * x);
    if (g_code == {G_SQRT}) return x > 0.0f ? sqrtf(x) : 0.0f;
    return x;
}}

// The aligned kernels' inner loop is written once over a vector type VT: a
// plain float, or a float4 when the width is a multiple of four (see VEC).
__device__ __forceinline__ void fma_into(float& a, const float d, const float b) {{ a += d * b; }}
__device__ __forceinline__ void fma_into(float4& a, const float d, const float4 b) {{
    a.x += d * b.x; a.y += d * b.y; a.z += d * b.z; a.w += d * b.w;
}}
__device__ __forceinline__ float zero_of(float*) {{ return 0.0f; }}
__device__ __forceinline__ float4 zero_of(float4*) {{ return make_float4(0.f, 0.f, 0.f, 0.f); }}
__device__ __forceinline__ float shfl_down(const float a, const int off) {{
    return __shfl_down_sync(FULL_MASK, a, off);
}}
__device__ __forceinline__ float4 shfl_down(const float4 a, const int off) {{
    return make_float4(__shfl_down_sync(FULL_MASK, a.x, off), __shfl_down_sync(FULL_MASK, a.y, off),
                       __shfl_down_sync(FULL_MASK, a.z, off), __shfl_down_sync(FULL_MASK, a.w, off));
}}
__device__ __forceinline__ void add_into(float& a, const float b) {{ a += b; }}
__device__ __forceinline__ void add_into(float4& a, const float4 b) {{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}}
"""

#: The transforms ``apply_g`` implements. A ``g_code`` missing from here would
#: fall through its final ``return x`` and silently compute the identity, so
#: this asserts against the host's own set instead: adding a transform to
#: :data:`~vsparse._norm_common.RECIPES` without teaching the kernels about it
#: fails at import, not in someone's results.
_CUDA_G_CODES = frozenset({G_IDENTITY, G_LOG1P, G_LOG1P_1000X, G_SQRT})
if _CUDA_G_CODES != _G_CODES:
    raise RuntimeError(
        f"vsparse._cuda's apply_g implements {sorted(_CUDA_G_CODES)} but "
        f"vsparse._norm_common defines {sorted(_G_CODES)}; add the missing "
        "transform to the CUDA _PREAMBLE"
    )

#: Arguments, identical for all four kernels. ``B``/``out`` are both
#: ``(*, width)`` and C-contiguous, whichever direction the product runs in.
_SIGNATURE = """
extern "C" __global__ void {name}(
    const long long* __restrict__ major_ptr,
    const float* __restrict__ values,
    const long long* __restrict__ value_ptr,
    const int* __restrict__ indices,
    const float* __restrict__ row_scale,
    const float* __restrict__ gene_scale,
    const float* __restrict__ col_post_scale,
    const int g_code,
    const float* __restrict__ B,
    float* __restrict__ out,
    const long long n_major,
    const int width,
    const int ww,
    const int groups,
    const int split)
"""

#: Per-warp lane bookkeeping, shared by every variant. ``wl`` indexes the
#: width, ``nl`` the lane-group.
_WARP_HEAD = """
{
    const int lane = threadIdx.x;
    const int wl = lane & (ww - 1);
    const int nl = lane / ww;
    const long long stride = (long long)gridDim.x * blockDim.y;
"""

#: Two ways to hand slices to warps, and the two walks that go with them. Each
#: variant is a ``(slices, walk, orow)`` triple of source fragments; a kernel is
#: compiled for one of them (see :func:`_build_source`).
#:
#: *Whole*: one warp per major slice, grid-strided -- the original scheme, and
#: the right one whenever there are enough slices to fill the device.
#:
#: *Chunked*: a warp's unit of work is a (major slice, chunk) pair, ``split``
#: chunks to a slice, so a matrix with few slices still fills the device. The
#: chunk is a contiguous share ``[ks, ke)`` of the slice's *flat* index range,
#: split by position rather than by value-run because the runs are wildly
#: uneven (a count matrix is mostly one value). ``ustart`` is the run containing
#: ``ks``, found by binary search over the slice's ``value_ptr`` entries; chunk
#: 0 starts at the slice start and skips the search.
#:
#: They are separate kernels, not one with ``split == 1`` as the fast case,
#: because the chunk bounds are not free: clipping every run to ``[ks, ke)`` and
#: carrying ``ks``/``ke``/``ustart`` through the loop measured ~25% slower at a
#: width of 128 (four accumulators, so register pressure shows), for a split of
#: one where it does nothing.
_SLICES_WHOLE = """
    for (long long maj = (long long)blockIdx.x * blockDim.y + threadIdx.y;
         maj < n_major; maj += stride) {
        const long long mstart = major_ptr[maj];
        const long long mstop = major_ptr[maj + 1];
"""

_SLICES_CHUNKED = """
    const long long n_items = n_major * split;
    for (long long item = (long long)blockIdx.x * blockDim.y + threadIdx.y;
         item < n_items; item += stride) {
        const long long maj = item / split;
        const int chunk = (int)(item - maj * split);
        const long long mstart = major_ptr[maj];
        const long long mstop = major_ptr[maj + 1];
        const long long kbeg = value_ptr[mstart];
        const long long span = value_ptr[mstop] - kbeg;
        const long long ks = kbeg + span * chunk / split;
        const long long ke = kbeg + span * (chunk + 1) / split;
        long long ustart = mstart;
        if (ks > kbeg) {
            long long lo = mstart, hi = mstop - 1;
            while (lo < hi) {
                const long long mid = lo + (hi - lo + 1) / 2;
                if (value_ptr[mid] <= ks) lo = mid; else hi = mid - 1;
            }
            ustart = lo;
        }
"""

#: VCSR: the major slice is a row, so ``row_scale`` is uniform over it and the
#: per-gene statistics are looked up per nonzero (and a zero-variance gene is
#: skipped one nonzero at a time).
_SCALES_MAJOR_IS_ROW = {
    "hoist": """
        const float rs = row_scale[maj];
""",
    "delta": """
            const float gs = gene_scale[mi];
            if (!(gs > 0.0f)) continue;
            const float d = col_post_scale[mi] * apply_g(v / rs / gs, g_code);
""",
}

#: VCSC: the major slice is a gene, so the per-gene statistics are uniform over
#: it -- including the zero-variance test, which skips the whole slice -- and
#: ``row_scale`` is the per-nonzero lookup.
_SCALES_MAJOR_IS_COL = {
    "hoist": """
        const float gs = gene_scale[maj];
        if (!(gs > 0.0f)) continue;
        const float s = col_post_scale[maj];
""",
    "delta": """
            const float d = s * apply_g(v / row_scale[mi] / gs, g_code);
""",
}

#: The per-nonzero walk itself, shared by both bodies so the two cannot drift
#: apart in how they stride the layout. Opens two loops -- over the slice's
#: unique values, then over the indices sharing each -- which ``_WALK_CLOSE``
#: closes, and leaves ``mi`` (the minor index) and ``d`` (the ``Delta`` entry)
#: in scope between them. Lane-group ``nl`` starts ``nl`` into each run and
#: steps by ``groups``, so the groups split the run between them.
_WALK_WHOLE = """
            for (long long u = mstart; u < mstop; ++u) {{
                const float v = values[u];
                const long long kstop = value_ptr[u + 1];
                for (long long kk = value_ptr[u] + nl; kk < kstop; kk += groups) {{
                    const int mi = indices[kk];
{delta}"""

#: The same walk over this warp's chunk ``[ks, ke)`` only: it starts at the run
#: containing ``ks``, stops at the first run starting at or past ``ke``, and
#: clips the runs at either end to the chunk.
_WALK_CHUNKED = """
            for (long long u = ustart; u < mstop; ++u) {{
                const long long ubeg = value_ptr[u];
                if (ubeg >= ke) break;
                const float v = values[u];
                const long long kbegin = ubeg > ks ? ubeg : ks;
                const long long uend = value_ptr[u + 1];
                const long long kstop = uend < ke ? uend : ke;
                for (long long kk = kbegin + nl; kk < kstop; kk += groups) {{
                    const int mi = indices[kk];
{delta}"""

_WALK_CLOSE = """
                }}
            }}"""

#: Aligned: the warp owns output row ``maj`` (or, chunked, one partial row of
#: it, ``split`` of them to a slice, summed afterwards in a fixed order), so it
#: accumulates the width in registers, reduces across lane-groups, and stores
#: once. The dense operand is indexed by the per-nonzero minor index.
_BODY_ALIGNED = """
{hoist}
        // VT is float, or float4 when VEC == 4: `width` is then a multiple of
        // four and every row of B and of the output is 16-byte aligned, so each
        // gather is one 128-bit load instead of four 32-bit ones. `wv` is the
        // width in VT's, and `ww` counts lanes across those.
        VT* const orow = (VT*)(out + {orow_index} * (long long)width);
        const VT* const Bv = (const VT*)B;
        const int wv = width / VEC;
        for (int cbase = 0; cbase < wv; cbase += ww * ACC_MAX) {{
            VT acc[ACC_MAX];
            #pragma unroll
            for (int a = 0; a < ACC_MAX; ++a) acc[a] = zero_of((VT*)0);
{walk_open}
                    const VT* const brow = Bv + (long long)mi * wv;
                    #pragma unroll
                    for (int a = 0; a < ACC_MAX; ++a) {{
                        const int c = cbase + a * ww + wl;
                        if (c < wv) fma_into(acc[a], d, brow[c]);
                    }}{walk_close}

            // Lanes sharing a `wl` sit `ww` apart, so folding by ww, 2*ww, ...
            // lands each column's total in the nl == 0 group.
            #pragma unroll
            for (int a = 0; a < ACC_MAX; ++a) {{
                for (int off = ww; off < {warp}; off <<= 1)
                    add_into(acc[a], shfl_down(acc[a], off));
            }}
            if (nl == 0) {{
                #pragma unroll
                for (int a = 0; a < ACC_MAX; ++a) {{
                    const int c = cbase + a * ww + wl;
                    if (c < wv) orow[c] = acc[a];
                }}
            }}
        }}
    }}
}}
"""

#: Misaligned: the output row varies per nonzero, so there is nothing to
#: accumulate in a register and the kernel scatters with atomicAdd. The dense
#: operand is the one indexed by ``maj`` here, so its row is loop-invariant.
_BODY_MISALIGNED = """
{hoist}
        const float* const brow = B + maj * (long long)width;
{walk_open}
                    float* const orow = out + (long long)mi * width;
                    for (int c = wl; c < width; c += ww)
                        atomicAdd(orow + c, d * brow[c]);{walk_close}
    }}
}}
"""

#: ``name -> (body template, scale fragments)``. The names encode the format
#: the kernel walks and the direction of the product.
_KERNELS = {
    "matmul_delta_csr": (_BODY_ALIGNED, _SCALES_MAJOR_IS_ROW),
    "rmatmul_delta_csc": (_BODY_ALIGNED, _SCALES_MAJOR_IS_COL),
    "matmul_delta_csc": (_BODY_MISALIGNED, _SCALES_MAJOR_IS_COL),
    "rmatmul_delta_csr": (_BODY_MISALIGNED, _SCALES_MAJOR_IS_ROW),
}


def _build_source(acc: int, chunked: bool, vec: int) -> str:
    """The full CUDA source for ``acc`` accumulators per lane, ``vec`` floats wide.

    All four kernels, assembled from the fragments above, for either the whole
    or the chunked way of handing slices to warps. ``acc`` only matters to the
    aligned kernels; the misaligned ones loop over the width instead and
    ignore ``vec``.
    """
    slices, walk, orow_index = (
        (_SLICES_CHUNKED, _WALK_CHUNKED, "(maj * split + chunk)")
        if chunked
        else (_SLICES_WHOLE, _WALK_WHOLE, "maj")
    )
    vt = "float4" if vec == 4 else "float"
    parts = [f"#define ACC_MAX {acc}\n#define VEC {vec}\n#define VT {vt}\n", _PREAMBLE]
    for name, (body, scales) in _KERNELS.items():
        # The walk is spliced in before formatting, not passed to it, because
        # it carries a `{delta}` of its own that the same pass has to expand.
        spliced = body.replace("{walk_open}", walk).replace("{walk_close}", _WALK_CLOSE)
        parts.append(_SIGNATURE.format(name=name))
        parts.append(_WARP_HEAD)
        parts.append(slices)
        parts.append(spliced.format(warp=_WARP, orow_index=orow_index, **scales))
    return "".join(parts)


_module_cache: dict[tuple[int, bool, int], Any] = {}


def _module(acc: int, chunked: bool, vec: int) -> Any:
    """The compiled :class:`cupy.RawModule` for a variant, built once per process.

    Variants are compiled lazily, so a process only pays for the accumulator
    counts and slice schemes its matrices and widths actually reach.
    """
    key = (acc, chunked, vec)
    if key not in _module_cache:
        cp = _require_cupy()
        # --use_fast_math: measured 24% off the parafac2 kernel (every recipe
        # but "raw" puts a transcendental on every nonzero) while the worst
        # relative error against the float64 host view moved from 2.3e-7 to
        # 3.4e-7 -- nothing, next to the float32 the whole device path is
        # already committed to. Its denormal flushing is if anything the safer
        # behaviour here: a denormal `gene_scale` fails the `> 0.0f` guard and
        # its column is skipped, rather than dividing into an infinity.
        source = _build_source(acc, chunked, vec)
        _module_cache[key] = cp.RawModule(code=source, options=("--use_fast_math",))
    return _module_cache[key]


@functools.cache
def _sm_count(device_id: int) -> int:
    """Streaming multiprocessors on a device (cached: the attribute query is a driver call)."""
    cp = _require_cupy()
    attr = cp.cuda.runtime.cudaDevAttrMultiProcessorCount
    return int(cp.cuda.runtime.deviceGetAttribute(attr, device_id))


def _accumulators(width: int, ww: int) -> int:
    """Accumulators a lane needs to cover ``width`` in one pass, capped at ``_ACC_MAX``."""
    return min(_ACC_MAX, max(1, -(-width // ww)))


def _split_factor(n_major: int, nnz: int, sm_count: int, columns: int) -> int:
    """Chunks to cut each major slice into; 1 when the grid is already full or ``columns`` is narrow.

    ``columns`` is the dense columns the width lanes cover in one load: ``ww``
    for scalar loads, ``4 * ww`` for float4 ones.
    """
    if n_major <= 0 or columns < _MIN_SPLIT_WW:
        return 1
    wanted = -(-(sm_count * _TARGET_WARPS_PER_SM) // n_major)
    affordable = nnz // (n_major * _MIN_CHUNK_NNZ)
    return max(1, min(wanted, affordable, _MAX_SPLIT))


class _Launch(NamedTuple):
    """Everything a kernel launch needs beyond the data."""

    grid: tuple[int]
    block: tuple[int, int]
    ww: int
    groups: int
    acc: int
    split: int
    vec: int

    @property
    def chunked(self) -> bool:
        """Whether slices are cut into chunks, which selects the chunked kernels."""
        return self.split > 1


def _launch_config(
    n_major: int, width: int, nnz: int, sm_count: int, *, aligned: bool = False
) -> _Launch:
    """The launch for a slice count, a dense width, and a device of ``sm_count`` SMs.

    ``aligned`` marks the register-accumulating kernels, which load ``float4``
    when ``width`` is a multiple of four (:data:`_VECTOR_WIDTH`), so ``width / 4``
    lanes cover the width. The misaligned kernels ``atomicAdd`` scalars and
    always take ``vec == 1``.
    """
    vec = _VECTOR_WIDTH if aligned and width % _VECTOR_WIDTH == 0 else 1
    lanes = width // vec
    ww = 1
    while ww < lanes and ww < _WARP:
        ww <<= 1
    split = _split_factor(n_major, nnz, sm_count, ww * vec)
    blocks = min(_MAX_BLOCKS, max(1, -(-(n_major * split) // _WARPS_PER_BLOCK)))
    return _Launch(
        (blocks,),
        (_WARP, _WARPS_PER_BLOCK),
        ww,
        _WARP // ww,
        _accumulators(lanes, ww),
        split,
        vec,
    )


# -- the device-resident view -------------------------------------------------


class CudaNormalizedView:
    """A :class:`~vsparse._norm_common.NormalizedViewBase` resident on a CUDA device.

    Built by :meth:`~vsparse._norm_common.NormalizedViewBase.to_gpu`, not
    directly. Holds the wrapped array's value-compressed structure and the
    recipe's statistics as CuPy arrays, and computes ``self @ B`` / ``B @ self``
    with the kernels in this module. All floating-point state is
    :data:`DEVICE_DTYPE` (``float32``).

    The object pins its device memory for as long as it is alive; drop the
    reference (and, if you need the memory back immediately,
    ``cupy.get_default_memory_pool().free_all_blocks()``) to release it.
    """

    # Tell numpy to defer `ndarray @ CudaNormalizedView` to our __rmatmul__
    # instead of trying to broadcast us, exactly as the host types do.
    __array_ufunc__ = None

    __slots__ = (
        "_format",
        "col_mean",
        "col_post_scale",
        "gene_scale",
        "indices",
        "major_ptr",
        "recipe",
        "row_scale",
        "shape",
        "sm_count",
        "value_ptr",
        "values",
    )

    def __init__(self, view: NormalizedViewBase) -> None:
        cp = _require_cupy()
        arr = view._arr
        self._format = view._format
        self.shape = view.shape
        self.recipe = view.recipe
        # Of the device the arrays below land on, read once: the launch plan
        # needs it on every call, and asking the driver costs microseconds a
        # kernel that takes a few hundred.
        self.sm_count = _sm_count(cp.cuda.Device().id)

        # Structure. The pointer arrays index into `values`/`indices`, which can
        # exceed 2**31 entries, so they stay 64-bit; `indices` holds a row or
        # column number and is narrowed to int32 for the kernels to share one
        # signature (a 2**31-row matrix is far past what fits on a device).
        self.major_ptr = cp.asarray(arr.major_ptr, dtype=cp.int64)
        self.value_ptr = cp.asarray(arr.value_ptr, dtype=cp.int64)
        self.indices = cp.asarray(arr.indices, dtype=cp.int32)
        self.values = cp.asarray(arr.values, dtype=DEVICE_DTYPE)

        # Statistics: O(n_rows + n_cols), negligible next to the structure.
        self.row_scale = cp.asarray(view.row_scale, dtype=DEVICE_DTYPE)
        self.gene_scale = cp.asarray(view.gene_scale, dtype=DEVICE_DTYPE)
        self.col_mean = cp.asarray(view.col_mean, dtype=DEVICE_DTYPE)
        self.col_post_scale = cp.asarray(view.col_post_scale, dtype=DEVICE_DTYPE)

    # -- introspection --------------------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(DEVICE_DTYPE)

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    @property
    def nbytes(self) -> int:
        """Device bytes held, structure plus statistics."""
        return sum(
            int(a.nbytes)
            for a in (
                self.major_ptr,
                self.value_ptr,
                self.indices,
                self.values,
                self.row_scale,
                self.gene_scale,
                self.col_mean,
                self.col_post_scale,
            )
        )

    @property
    def means(self) -> Any:
        """Per-gene mean-correction vector, ``col_post_scale * col_mean``."""
        return self.col_mean * self.col_post_scale

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<{type(self).__name__} shape={self.shape} dtype={self.dtype} "
            f"recipe={self.recipe.name!r} nnz={self.nnz}>"
        )

    # -- kernels --------------------------------------------------------------

    def _kernel_args(self, g_code: int) -> tuple[Any, ...]:
        return (
            self.major_ptr,
            self.values,
            self.value_ptr,
            self.indices,
            self.row_scale,
            self.gene_scale,
            self.col_post_scale,
            np.int32(g_code),
        )

    def _run(self, name: str, B: Any, out: Any, n_major: int) -> None:
        cp = _require_cupy()
        width = int(B.shape[1])
        aligned = _KERNELS[name][0] is _BODY_ALIGNED
        cfg = _launch_config(n_major, width, self.nnz, self.sm_count, aligned=aligned)
        # An aligned kernel stores rather than adds, so split chunks of one
        # slice can't share its output row: each writes a partial row of its
        # own (zero-initialised, so a skipped slice stays zero), and the
        # partials are summed below. The sum is a fixed-order reduction, so
        # the aligned direction stays deterministic. A misaligned kernel
        # atomicAdds into `out` whatever the split, and needs no scratch.
        scratch = aligned and cfg.chunked
        target = cp.zeros((n_major * cfg.split, width), dtype=DEVICE_DTYPE) if scratch else out
        kernel = _module(cfg.acc, cfg.chunked, cfg.vec).get_function(name)
        kernel(
            cfg.grid,
            cfg.block,
            (
                *self._kernel_args(self.recipe.g_code),
                B,
                target,
                np.int64(n_major),
                np.int32(width),
                np.int32(cfg.ww),
                np.int32(cfg.groups),
                np.int32(cfg.split),
            ),
        )
        if scratch:
            cp.sum(target.reshape(n_major, cfg.split, width), axis=1, out=out)

    def __matmul__(self, other: Any) -> Any:
        """``self @ other`` for a dense ``other``; returns a float32 CuPy array."""
        cp = _require_cupy()
        n_rows, n_cols = self.shape
        B, squeeze = _prep_dense(cp, other, n_cols)
        out = cp.zeros((n_rows, B.shape[1]), dtype=DEVICE_DTYPE)

        if self._format == "csr":
            self._run("matmul_delta_csr", B, out, n_rows)
        else:
            self._run("matmul_delta_csc", B, out, n_cols)

        out += (-self.means) @ B  # every row's implicit-zero contribution
        return out[:, 0] if squeeze else out

    def __rmatmul__(self, other: Any) -> Any:
        """``other @ self`` for a dense ``other``; returns a float32 CuPy array."""
        cp = _require_cupy()
        n_rows, n_cols = self.shape
        B2, squeeze = _prep_dense_right(cp, other, n_rows)
        # (n_rows, p), so the kernels' inner loop over the width walks one
        # contiguous row -- the same reason the host kernels transpose.
        Bt = cp.ascontiguousarray(B2.T)
        out_t = cp.zeros((n_cols, Bt.shape[1]), dtype=DEVICE_DTYPE)

        if self._format == "csc":
            self._run("rmatmul_delta_csc", Bt, out_t, n_cols)
        else:
            self._run("rmatmul_delta_csr", Bt, out_t, n_rows)

        out = cp.ascontiguousarray(out_t.T)
        out += B2.sum(axis=1)[:, None] * (-self.means)[None, :]
        return out[0, :] if squeeze else out

    # -- the expanded alternative ---------------------------------------------

    def to_cupy_sparse(self, format: str | None = None, *, sort_indices: bool = True) -> Any:
        """The uncentered ``Delta`` term as a CuPy CSR/CSC matrix, one float per nonzero.

        The device-side counterpart of
        :meth:`~vsparse._norm_common.NormalizedViewBase.to_scipy_sparse`, and
        the baseline the kernels are benchmarked against: it hands the product
        to cuSPARSE, at the cost of expanding every value-compressed run and so
        giving up the layout's whole memory advantage. :attr:`means` is left to
        subtract externally, exactly as on the host.

        Parameters
        ----------
        format : {"csr", "csc"}, optional
            Sparse format to build. Defaults to whichever one matches the
            wrapped array's own major axis, which is the one that costs
            nothing to assemble. Worth overriding: cuSPARSE is markedly
            faster with a CSR operand in *both* directions -- measurably so
            for ``dense @ sparse``, where CSC ran ~5x slower in
            ``benchmarks/cases.py`` -- so a caller multiplying repeatedly
            should pay the one-time conversion and pass ``"csr"``.
        sort_indices : bool, default True
            Whether to sort each slice's indices ascending before returning.
            A VCS slice stores its indices grouped by the value they share,
            never in index order, so the expanded matrix is *not* canonical
            unless this sorts it -- and cuSPARSE measured ~6x slower on the
            unsorted form. That penalty is why this defaults to ``True``
            where the host's
            :meth:`~vsparse._norm_common.NormalizedViewBase.to_scipy_sparse`,
            whose result usually goes somewhere less picky, returns the
            unsorted order. Pass ``False`` to skip the sort when the result
            is used once, or by something that does not care.
        """
        cp = _require_cupy()
        import cupyx.scipy.sparse as cps

        indptr = self.value_ptr[self.major_ptr]
        raw = cp.repeat(self.values, cp.diff(self.value_ptr))
        major = cp.repeat(cp.arange(len(indptr) - 1, dtype=cp.int32), cp.diff(indptr))
        if self._format == "csr":
            row, col, ctor = major, self.indices, cps.csr_matrix
        else:
            row, col, ctor = self.indices, major, cps.csc_matrix

        gs = self.gene_scale[col]
        scaled = cp.where(gs > 0, raw / self.row_scale[row] / gs, DEVICE_DTYPE(0))
        data = (self.col_post_scale[col] * _g_cupy(cp, scaled, self.recipe.g_code)).astype(
            DEVICE_DTYPE
        )
        out = ctor((data, self.indices.astype(cp.int32), indptr.astype(cp.int32)), shape=self.shape)
        if format not in (None, "csr", "csc"):
            raise ValueError(f"format must be 'csr', 'csc' or None, got {format!r}")
        if format is not None and format != self._format:
            out = out.tocsr() if format == "csr" else out.tocsc()
        elif sort_indices:
            out.sort_indices()
        return out


def _g_cupy(cp: Any, x: Any, g_code: int) -> Any:
    """Vectorized ``g`` on a CuPy array -- the device twin of ``_g_np``."""
    if g_code == G_LOG1P:
        return cp.log1p(x)
    if g_code == G_LOG1P_1000X:
        return cp.log10(1.0 + 1000.0 * x)
    if g_code == G_SQRT:
        return cp.sqrt(cp.clip(x, 0.0, None))
    return x


def _prep_dense(cp: Any, other: Any, expect_rows: int) -> tuple[Any, bool]:
    """A C-contiguous float32 device ``(expect_rows, k)``, and whether it was 1-D."""
    B = cp.asarray(other, dtype=DEVICE_DTYPE)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return cp.ascontiguousarray(B), squeeze


def _prep_dense_right(cp: Any, other: Any, expect_cols: int) -> tuple[Any, bool]:
    """A C-contiguous float32 device ``(p, expect_cols)``, and whether it was 1-D."""
    B = cp.asarray(other, dtype=DEVICE_DTYPE)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(1, -1)
    if B.ndim != 2 or B.shape[1] != expect_cols:
        raise ValueError(f"shape mismatch: expected last dimension {expect_cols}, got {B.shape}")
    return cp.ascontiguousarray(B), squeeze


def to_gpu(view: NormalizedViewBase) -> CudaNormalizedView:
    """Move ``view``'s structure and statistics onto the current CUDA device.

    Backs :meth:`~vsparse._norm_common.NormalizedViewBase.to_gpu`.

    The statistics are computed on the host, in float64, and narrowed on
    transfer: the ``O(nnz)`` passes that derive them run once, where the matmul
    the device is here for runs many times. Everything on the device is then
    float32, so a product will not agree with the host's float64 one to better
    than roughly ``1e-6`` relative -- and in the misaligned directions
    (``VCSC @ B``, ``B @ VCSR``) the atomic accumulation also makes it
    nondeterministic in the last bits.
    """
    return CudaNormalizedView(view)
