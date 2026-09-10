"""Peak-RSS ceilings for the operations that must not scale with ``nnz``.

Each bound below is stated together with the dataset it was measured on and
with what the *unfixed* implementation would have cost, because a ceiling that
both the old and the new code pass is not testing anything. The numbers come
from ``tests/memory_ceiling.py``, which measures a whole fresh process rather
than numpy allocations -- see that module for why ``peak_alloc_mb`` in the
benchmark suite is not a substitute.
"""

from __future__ import annotations

import pytest
from memory_ceiling import linux_only, peak_rss_growth_mb

# A 100k x 2k matrix at 5% density: 10M stored nonzeros.
_COUNTS_10M = """
import numpy as np, scipy.sparse as sp
rng = np.random.default_rng(0)
mat = sp.random_array((100_000, 2_000), density=0.05, format="csr", random_state=0)
mat.data = rng.integers(1, 8, size=mat.data.shape[0]).astype(np.float64)
"""

# 60k x 2k at 5%: 6M nonzeros, whose full regroup is 6e6 * (8 + 4) = 72 MB.
_VCSC_6M = """
import numpy as np, scipy.sparse as sp
import vsparse._vcs_matmul as vm
from vsparse import VCSCArray
rng = np.random.default_rng(0)
mat = sp.random_array((60_000, 2_000), density=0.05, format="csr", random_state=0)
mat.data = rng.integers(1, 8, size=mat.data.shape[0]).astype(np.float64)
v = VCSCArray.from_scipy(mat)
del mat
B = rng.normal(size=(v.shape[1], 4))
"""


@linux_only
def test_minor_sums_does_not_allocate_an_nnz_temporary():
    """Regression for #30.

    ``sum(axis=0)`` used to do ``np.repeat(values.astype(np.float64), sizes)``
    -- one float64 per stored nonzero. On the 10M-nonzero matrix below that is
    80 MB; the thread-local accumulators that replaced it are
    ``n_threads * n_cols * 8``, under 1 MB for any sane thread count.

    Ceiling 32 MB. Verified against both implementations on this fixture:
    the pre-#30 spelling (``np.repeat`` + ``bincount``) grows peak RSS by
    **153.1 MB** and fails; the current one grows it by **0.31 MB**. A 494x
    separation, so the bound is nowhere near either side by accident.
    """
    setup = (
        _COUNTS_10M
        + """
from vsparse import VCSRArray
v = VCSRArray.from_scipy(mat)
del mat
v.sum(axis=0)   # compile the kernel before the high-water mark is reset
"""
    )
    growth, was_reset = peak_rss_growth_mb(setup, "v.sum(axis=0)")
    if not was_reset:
        pytest.skip("could not reset VmHWM; the measurement would only be an upper bound")
    assert growth < 32.0, f"sum(axis=0) grew peak RSS by {growth:.1f} MB (ceiling 32 MB)"


@linux_only
def test_misaligned_matmul_peak_is_set_by_the_chunk_budget_not_by_nnz():
    """Regression for #32.

    The misaligned direction regroups the array into the opposite format. Doing
    that globally costs a second copy -- 72 MB for the 6M-nonzero matrix here,
    and the prototype measured +13.4 GiB on a real cohort slice, extrapolating
    to ~160 GiB at full scale. The chunked path holds one chunk at a time
    instead.

    The budget is shrunk to 8 MiB so a small, fast fixture still separates the
    two. Verified by forcing each path on this fixture: raising the budget so
    the whole array fits one chunk (the global-dual branch of
    ``_aligned_source``) grows peak RSS by **255.6 MB** and fails; the chunked
    path grows it by **3.39 MB**. Ceiling 24 MB, a 75x separation.
    """
    setup = (
        _VCSC_6M
        + """
vm._CHUNK_BUDGET_BYTES = 8 << 20
assert len(vm._chunk_bounds(v, vm._CHUNK_BUDGET_BYTES)) > 1, "fixture must not fit one chunk"
nv = v.normalized()
nv @ B          # compile the kernels before the high-water mark is reset
nv2 = v.normalized()
"""
    )
    growth, was_reset = peak_rss_growth_mb(setup, "nv2 @ B")
    if not was_reset:
        pytest.skip("could not reset VmHWM; the measurement would only be an upper bound")
    assert growth < 24.0, f"misaligned matmul grew peak RSS by {growth:.1f} MB (ceiling 24 MB)"


@linux_only
@pytest.mark.parametrize("density", [0.025, 0.10])
def test_misaligned_matmul_costs_a_fraction_of_a_full_regroup(density):
    """The claim behind the ceiling above, checked across a 4x range of ``nnz``.

    Note what is *not* asserted: that the peak is flat. It isn't, quite --
    measured 1.74 / 3.43 / 5.96 MB across a 0.025 / 0.05 / 0.10 sweep. That
    drift is the allocator holding freed chunks rather than any single large
    transient
    (``_chunk_bounds`` caps every chunk at
    ``_CHUNK_BUDGET_BYTES // _TRANSPOSE_BYTES_PER_NNZ`` nonzeros, so the
    transient itself is constant), and asserting flatness would be asserting
    something the implementation does not promise.

    What it does promise is that the peak stays a small fraction of the second
    copy a global regroup would need -- 36, 72 and 144 MB respectively here.
    Measured, every point lands near 5%; 25% is the ceiling, which a
    reintroduced global dual (255.6 MB at the middle size) blows through
    everywhere. Two densities rather than three, to keep the subprocess cost
    down while still spanning 4x in ``nnz``.
    """
    n_rows, n_cols = 60_000, 2_000
    regroup_mb = n_rows * n_cols * density * 12 / 1e6
    setup = f"""
import numpy as np, scipy.sparse as sp
import vsparse._vcs_matmul as vm
from vsparse import VCSCArray
rng = np.random.default_rng(0)
mat = sp.random_array(({n_rows}, {n_cols}), density={density}, format="csr", random_state=0)
mat.data = rng.integers(1, 8, size=mat.data.shape[0]).astype(np.float64)
v = VCSCArray.from_scipy(mat)
del mat
B = rng.normal(size=(v.shape[1], 4))
vm._CHUNK_BUDGET_BYTES = 8 << 20
assert len(vm._chunk_bounds(v, vm._CHUNK_BUDGET_BYTES)) > 1, "fixture must not fit one chunk"
nv = v.normalized()
nv @ B
nv2 = v.normalized()
"""
    growth, was_reset = peak_rss_growth_mb(setup, "nv2 @ B")
    if not was_reset:
        pytest.skip("could not reset VmHWM; the measurement would only be an upper bound")
    assert growth < 0.25 * regroup_mb, (
        f"peak grew {growth:.2f} MB, over 25% of the {regroup_mb:.0f} MB a full regroup costs"
    )
