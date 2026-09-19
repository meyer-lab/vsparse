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

**Layout size.** `bytes_per_nonzero` and `indices_bytes_per_nonzero` -- what
the compressed format actually costs per stored value. Deterministic and
comparable across machines, so they are gated tightly, and nothing else in
the project guards them.

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

Memory is no longer measured here; see below.

## Adding a case

Write a function returning `{metric: value}` in `cases.py`, decorated with
`@fast` (runs on every PR, keep it under a minute) or `@slow`. Add any new
gated metric to `margins` in `baselines.json`, then `--record`.

## Memory is a test, not a benchmark

Memory is not measured here. `pytest-memray` ceilings in `tests/` assert it
instead, because the claims are structural: "a minor-axis reduction must not
allocate anything nnz-sized" either holds or it does not. A ceiling fails the
moment it stops being true, where a recorded number only shows it drifting,
against a baseline needing a re-record whenever the bound legitimately moves.

Those tests pin `numba.set_num_threads`, so a ceiling means the same thing on
a 4-core runner and a 96-core one. A recorded figure could not: the
accumulators being bounded are sized by the runner's core count.

Neither tool sees RSS, so neither catches allocator *fragmentation* -- many
variably-sized alloc/free cycles driving the resident set far above the live
set. Both report the live high-water mark, which stays small throughout such a
run. That failure mode is real but not reliably gateable, since RSS moves with
the allocator, the runner and the thread count; diagnose it with
`/proc/self/status` `VmHWM` around a workload when it is suspected.
