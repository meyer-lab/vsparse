# Benchmarks

A small suite CI runs as a regression gate, plus larger cases for running by
hand.

```sh
uv run python -m benchmarks.run --set fast            # run + compare (what CI does)
uv run python -m benchmarks.run --set slow            # the larger cases
uv run python -m benchmarks.run --case matvec_vs_scipy
uv run python -m benchmarks.run --set fast --record   # rewrite baselines.json
```

Exits nonzero if a gated metric exceeds its ceiling.

## Metrics

Correctness tests do not catch cost regressions, so the suite measures two
things that move silently:

**Layout size and memory allocated by an operation.** Both deterministic and
comparable across machines, so they are gated tightly. Memory uses
`tracemalloc` rather than `ru_maxrss`, which is a process-lifetime high-water
mark and reports zero for an operation staying under the peak set while
building its input.

**Throughput relative to scipy**, never absolute seconds. The same work is
timed through scipy in the same process and the ratio recorded, which cancels
most of the difference between machines. Still the noisiest metric, so its
gate is much looser.

Each case runs in its own subprocess, since measurement state and JIT warm-up
leak between them otherwise.

## Baselines

`baselines.json` holds a ceiling per gated metric, regenerated with
`--record` by multiplying a fresh measurement by that metric's margin. Only
metrics named in `margins` are gated; anything else a case returns is
recorded for context.

The checked-in ceilings were recorded before the memory fixes landed, so the
memory ones are deliberately generous and should be re-recorded as those
merge. On a 4M-nonzero array, for results of length `n_minor`:

| metric | recorded | with the fix |
|---|---|---|
| `minor_sum_peak_mb` | 66 MB | 0.8 MB |
| `minor_extrema_peak_mb` | 66 MB | 1.6 MB |
| `minor_getnnz_peak_mb` | 32 MB | 0.8 MB |
| `minor_selection_peak_mb` | 62 MB | 4.8 MB |
| `misaligned_matmul_peak_mb` | 204 MB | 115 MB |

## Adding a case

Write a function returning `{metric: value}` in `cases.py`, decorated with
`@fast` (runs on every PR, keep it under a minute) or `@slow`. Add any new
gated metric to `margins` in `baselines.json`, then `--record`.

## Memory: what belongs here and what belongs in the tests

Memory shows up in both places, measuring different things, and the split is
deliberate.

**The test suite owns the ceilings.** `pytest-memray` marks
(`limit_memory`, `limit_leaks`) assert that an operation's allocation is
bounded by its accumulator rather than by `nnz` -- a property that either
holds or does not, with no baseline to record. `memray` intercepts the
allocator itself, so it sees numba's allocations, including the thread-local
accumulators inside a `parallel=True` kernel; `tracemalloc` sees those too,
but only as Python-level allocations, and neither sees resident-set effects
(see below). Those tests pin `numba.set_num_threads` so the expected number
is a property of the code rather than of the runner, and build their inputs
in fixtures, since a mark measures only the test body.

**The benchmarks own the numbers.** `peak_alloc_mb` records what an operation
allocated so a change in it is visible over time and against a baseline. It
stays on `tracemalloc`: these run outside pytest, where the marks do not
apply.

Neither measures RSS, and so neither catches allocator *fragmentation* -- many
variably-sized alloc/free cycles driving the resident set far above the live
set, which is what #49 hit at ~1,150 chunks per pass. Both tools report the
live high-water mark, which stays small throughout such a run. That failure
mode is real but is not reliably gateable: RSS moves with the allocator, the
runner and the thread count. Diagnose it with `/proc/self/status` `VmHWM`
around a workload when it is suspected, rather than asserting on it in CI.
