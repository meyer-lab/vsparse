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
most of the difference between machines. It does not cancel all of it -- a
shared CI runner has measured 5x what a workstation does on the same commit --
so this is much the noisiest metric and its gate is correspondingly loose.

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
