"""Property tests: every operation against a dense NumPy reference.

The unit tests elsewhere pick their shapes by hand, which is how a duplicate
minor-axis index silently returned zeros for a while (#35) -- no test happened
to use one. This module generates the awkward cases instead of hoping someone
remembers them: empty axes, all-zero data, a single distinct value, repeated
and unsorted index keys, negative indices, empty selections.

Every test here has the same shape. Build a dense integer matrix, put it
through both a VCSC/VCSR array and plain NumPy, and require the answers to
match. The dense side is deliberately the dumbest possible implementation --
it is the specification, so it must be obviously correct rather than clever.

Counts, not floats. `values` are integers because that is what the layout is
for: it dedupes repeated values, and float data with every value distinct
exercises a different (and much less interesting) regime. `_counts` enforces
that rather than leaving it to convention.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from vsparse import VCSCArray, VCSRArray

# Deadlines off: the first call into any numba kernel pays JIT compilation,
# which has nothing to do with the property under test and trips the default
# 200ms deadline on whichever example happens to come first.
_SETTINGS = settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

# Shapes include zero on either axis. A 0-row or 0-column matrix has no
# nonzeros to iterate, so it is exactly where an off-by-one in a pointer array
# or an unguarded `max()` over an empty axis shows up.
_dims = st.integers(min_value=0, max_value=12)


@st.composite
def _counts(draw, max_value: int = 6) -> np.ndarray:
    """A dense integer-count matrix, across the structural space that matters.

    Hypothesis drives the *structure* -- shape, density, how many distinct
    values -- while the cell contents come from a seeded NumPy generator.

    That split is deliberate. Drawing each cell from ``st.integers(0, top)``
    lets hypothesis' shrinking bias dominate: it pulls every element toward 0,
    so 47% of examples came out all-zero and only 2% fully dense, and the suite
    spent most of its budget re-testing the empty case. Densities here are
    hit in the proportions written down, and the seed still shrinks, so a
    failure still reduces to a small reproducible example.
    """
    n_rows = draw(_dims)
    n_cols = draw(_dims)
    density = draw(st.sampled_from([0.0, 0.1, 0.3, 0.6, 0.9, 1.0]))
    # 1 distinct value is the layout's best case (one value group for the whole
    # array); `max_value` spreads the groups out.
    top = draw(st.sampled_from([1, 2, max_value]))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    rng = np.random.default_rng(seed)
    dense = rng.integers(1, top + 1, size=(n_rows, n_cols)).astype(np.float64)
    if density < 1.0:
        dense[rng.random((n_rows, n_cols)) >= density] = 0.0
    return dense


@st.composite
def _index_key(draw, n: int):
    """One axis's worth of indexing key, in every spelling the array accepts."""
    kinds = ["slice", "empty", "all"]
    if n > 0:
        kinds += ["list", "repeated", "unsorted", "negative", "bool", "int"]
    kind = draw(st.sampled_from(kinds))
    if kind == "all":
        return slice(None)
    if kind == "empty":
        return []
    if kind == "slice":
        start = draw(st.integers(min_value=0, max_value=max(n, 1)))
        stop = draw(st.integers(min_value=0, max_value=max(n, 1)))
        step = draw(st.sampled_from([1, 2]))
        return slice(start, stop, step)
    if kind == "int":
        return draw(st.integers(min_value=0, max_value=n - 1))
    if kind == "bool":
        return draw(npst.arrays(dtype=bool, shape=(n,), elements=st.booleans()))
    idx = draw(st.lists(st.integers(min_value=0, max_value=n - 1), min_size=1, max_size=6))
    if kind == "repeated":
        # The #35 case: the same minor index named more than once must fan out,
        # not silently drop to zeros.
        idx = [idx[0]] * draw(st.integers(min_value=2, max_value=4)) + idx
    elif kind == "unsorted":
        idx = sorted(idx, reverse=True)
    elif kind == "negative":
        idx = [i - n for i in idx]
    return idx


def _both(dense: np.ndarray):
    """The same matrix as a VCSC array and a VCSR array."""
    return (
        VCSCArray.from_scipy(sp.csc_array(dense)),
        VCSRArray.from_scipy(sp.csr_array(dense)),
    )


# -- round trip ---------------------------------------------------------------


@_SETTINGS
@given(dense=_counts())
def test_scipy_round_trip_is_exact(dense):
    for v in _both(dense):
        np.testing.assert_array_equal(v.toarray(), dense)
        np.testing.assert_array_equal(v.to_scipy().toarray(), dense)
        np.testing.assert_array_equal(v.to_csr().toarray(), dense)
        np.testing.assert_array_equal(v.to_csc().toarray(), dense)


@_SETTINGS
@given(dense=_counts())
def test_transpose_matches_numpy(dense):
    for v in _both(dense):
        np.testing.assert_array_equal(v.transpose().toarray(), dense.T)


@_SETTINGS
@given(dense=_counts())
def test_copy_is_independent(dense):
    for v in _both(dense):
        c = v.copy()
        np.testing.assert_array_equal(c.toarray(), dense)
        assert c.values is not v.values


@_SETTINGS
@given(dense=_counts())
def test_ivcs_codec_round_trips(dense):
    """The byte-packed index codec must be lossless for every shape."""
    from vsparse import _ivcsc

    for v in _both(dense):
        packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
        back = _ivcsc.unpack_indices(v.value_ptr, packed, v.indices.dtype)
        np.testing.assert_array_equal(back, v.indices)


# -- reductions ---------------------------------------------------------------


@_SETTINGS
@given(dense=_counts())
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum_matches_numpy(dense, axis):
    for v in _both(dense):
        np.testing.assert_allclose(np.asarray(v.sum(axis=axis)), dense.sum(axis=axis))


@_SETTINGS
@given(dense=_counts())
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean_matches_numpy(dense, axis):
    if dense.size == 0:
        return  # numpy warns and returns nan; the empty-mean contract is not this test's subject
    for v in _both(dense):
        np.testing.assert_allclose(np.asarray(v.mean(axis=axis)), dense.mean(axis=axis))


@_SETTINGS
@given(dense=_counts())
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("op", ["max", "min"])
def test_extrema_match_numpy(dense, axis, op):
    """Implicit zeros count: a column of stored 5s and one absent entry maxes at 5, mins at 0."""
    if dense.shape[axis] == 0 or dense.shape[1 - axis] == 0:
        return
    for v in _both(dense):
        got = np.asarray(getattr(v, op)(axis=axis))
        np.testing.assert_allclose(got, getattr(dense, op)(axis=axis))


@_SETTINGS
@given(dense=_counts())
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_getnnz_matches_numpy(dense, axis):
    expected = (dense != 0).sum(axis=axis)
    for v in _both(dense):
        np.testing.assert_array_equal(np.asarray(v.getnnz(axis=axis)), expected)


@_SETTINGS
@given(dense=_counts())
def test_count_nonzero_matches_numpy(dense):
    for v in _both(dense):
        assert v.count_nonzero() == int((dense != 0).sum())


# -- elementwise --------------------------------------------------------------


@_SETTINGS
@given(dense=_counts())
def test_log1p_matches_numpy(dense):
    for v in _both(dense):
        np.testing.assert_allclose(v.log1p().toarray(), np.log1p(dense))


@_SETTINGS
@given(dense=_counts(), factor=st.integers(min_value=-4, max_value=4))
def test_scalar_multiply_matches_numpy(dense, factor):
    for v in _both(dense):
        np.testing.assert_allclose(v.multiply(factor).toarray(), dense * factor)


@_SETTINGS
@given(dense=_counts())
@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64])
def test_astype_matches_numpy(dense, dtype):
    for v in _both(dense):
        np.testing.assert_allclose(v.astype(dtype).toarray(), dense.astype(dtype))


@_SETTINGS
@given(dense=_counts())
def test_elementwise_add_matches_numpy(dense):
    for v in _both(dense):
        np.testing.assert_allclose((v + v).toarray(), dense + dense)


# -- matmul, both directions --------------------------------------------------


@_SETTINGS
@given(dense=_counts(), k=st.integers(min_value=1, max_value=3))
def test_matmul_matches_numpy(dense, k):
    rng = np.random.default_rng(0)
    B = rng.normal(size=(dense.shape[1], k))
    x = rng.normal(size=dense.shape[1])
    for v in _both(dense):
        np.testing.assert_allclose(v @ B, dense @ B, atol=1e-9)
        np.testing.assert_allclose(v @ x, dense @ x, atol=1e-9)


@_SETTINGS
@given(dense=_counts(), p=st.integers(min_value=1, max_value=3))
def test_rmatmul_matches_numpy(dense, p):
    rng = np.random.default_rng(0)
    B = rng.normal(size=(p, dense.shape[0]))
    x = rng.normal(size=dense.shape[0])
    for v in _both(dense):
        np.testing.assert_allclose(B @ v, B @ dense, atol=1e-9)
        np.testing.assert_allclose(x @ v, x @ dense, atol=1e-9)


# -- indexing -----------------------------------------------------------------


def _dense_take(dense: np.ndarray, row_key, col_key) -> np.ndarray:
    """NumPy's answer for the same selection, always as a 2-D array."""
    rows = [row_key] if isinstance(row_key, int | np.integer) else row_key
    cols = [col_key] if isinstance(col_key, int | np.integer) else col_key
    return dense[rows, :][:, cols]


@_SETTINGS
@given(data=st.data(), dense=_counts())
def test_indexing_matches_numpy(data, dense):
    """Slices, masks, negatives, unsorted keys, repeats and empty selections, on both axes."""
    row_key = data.draw(_index_key(dense.shape[0]), label="rows")
    col_key = data.draw(_index_key(dense.shape[1]), label="cols")
    expected = _dense_take(dense, row_key, col_key)
    for v in _both(dense):
        got = v[row_key, col_key]
        got = got.toarray() if hasattr(got, "toarray") else np.asarray(got)
        np.testing.assert_array_equal(got.reshape(expected.shape), expected)


@_SETTINGS
@given(data=st.data(), dense=_counts())
def test_repeated_minor_index_fans_out(data, dense):
    """Regression property for #35, which a hand-written suite missed entirely."""
    assume(dense.shape[1] > 0)
    col = data.draw(st.integers(min_value=0, max_value=dense.shape[1] - 1))
    repeats = data.draw(st.integers(min_value=2, max_value=5))
    key = [col] * repeats
    expected = dense[:, key]
    for v in _both(dense):
        got = v[:, key]
        np.testing.assert_array_equal(got.toarray(), expected)
        assert got.shape == expected.shape


# -- normalized view ----------------------------------------------------------


@_SETTINGS
@given(dense=_counts())
def test_normalized_view_matches_a_dense_reference(dense):
    """`(g(x * a * b) - c)` computed the dumb way, on every generated shape."""
    assume(dense.shape[0] > 0 and dense.shape[1] > 0)
    assume(dense.sum() > 0)

    depth = dense.sum(axis=1)
    median = float(np.median(depth))
    assume(median > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(depth > 0, median / depth, 1.0)
        scaled = dense * a[:, None]
        gsum = scaled.sum(axis=0)
        b = np.where(gsum > 0, 1.0 / gsum, 0.0)
        g = np.log10(1.0 + 1000.0 * scaled * b[None, :])
    expected = g - g.mean(axis=0)

    for v in _both(dense):
        np.testing.assert_allclose(v.normalized().toarray(), expected, atol=1e-8)


@_SETTINGS
@given(dense=_counts(), k=st.integers(min_value=1, max_value=3))
def test_normalized_matmul_agrees_with_its_own_toarray(dense, k):
    """The kernels take the rank-1 shortcut; `toarray` does not. They must agree."""
    assume(dense.shape[0] > 0 and dense.shape[1] > 0)
    assume(dense.sum() > 0)
    assume(float(np.median(dense.sum(axis=1))) > 0)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(dense.shape[1], k))
    C = rng.normal(size=(k, dense.shape[0]))
    for v in _both(dense):
        nv = v.normalized()
        full = nv.toarray()
        np.testing.assert_allclose(nv @ B, full @ B, atol=1e-8)
        np.testing.assert_allclose(C @ nv, C @ full, atol=1e-8)


# -- degenerate shapes, named rather than hoped for ---------------------------
#
# The generator above reaches these, but only with whatever probability the
# strategy happens to give them. They are the cases that actually break
# pointer arithmetic, so they get named examples that run every time.

_DEGENERATE = {
    "0x0": np.zeros((0, 0)),
    "0 rows": np.zeros((0, 4)),
    "0 cols": np.zeros((4, 0)),
    "all zero": np.zeros((3, 4)),
    "one value": np.full((3, 4), 2.0),
    "single cell": np.array([[7.0]]),
    "single row": np.array([[0.0, 3.0, 0.0, 1.0]]),
    "single col": np.array([[0.0], [5.0], [0.0]]),
    "one dense col": np.array([[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]),
    "one dense row": np.array([[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]]),
}


@pytest.mark.parametrize("name", sorted(_DEGENERATE))
def test_degenerate_shapes_round_trip(name):
    dense = _DEGENERATE[name]
    for v in _both(dense):
        np.testing.assert_array_equal(v.toarray(), dense)
        np.testing.assert_array_equal(v.transpose().toarray(), dense.T)
        assert v.nnz == int((dense != 0).sum())


@pytest.mark.parametrize("name", sorted(_DEGENERATE))
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_degenerate_shapes_reduce(name, axis):
    dense = _DEGENERATE[name]
    for v in _both(dense):
        np.testing.assert_allclose(np.asarray(v.sum(axis=axis)), dense.sum(axis=axis))
        np.testing.assert_array_equal(np.asarray(v.getnnz(axis=axis)), (dense != 0).sum(axis=axis))


@pytest.mark.parametrize("name", sorted(_DEGENERATE))
def test_degenerate_shapes_matmul(name):
    dense = _DEGENERATE[name]
    rng = np.random.default_rng(0)
    B = rng.normal(size=(dense.shape[1], 2))
    C = rng.normal(size=(2, dense.shape[0]))
    for v in _both(dense):
        np.testing.assert_allclose(v @ B, dense @ B, atol=1e-9)
        np.testing.assert_allclose(C @ v, C @ dense, atol=1e-9)


@pytest.mark.parametrize("name", sorted(_DEGENERATE))
def test_degenerate_shapes_empty_selection(name):
    """An empty key on either axis: the shape must survive even with no data."""
    dense = _DEGENERATE[name]
    for v in _both(dense):
        np.testing.assert_array_equal(v[[], :].toarray(), dense[[], :])
        np.testing.assert_array_equal(v[:, []].toarray(), dense[:, []])
        np.testing.assert_array_equal(v[[], []].toarray(), dense[[], :][:, []])
