"""Run benchmark cases and compare them against the checked-in baselines.

    python -m benchmarks.run --set fast              # run and compare (CI does this)
    python -m benchmarks.run --set fast --record     # rewrite baselines.json
    python -m benchmarks.run --case matvec_vs_scipy  # one case

Exits nonzero if any metric regresses past its recorded ceiling. Each case
runs in its own subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASELINES = Path(__file__).with_name("baselines.json")


def _run_one_in_subprocess(name: str) -> dict[str, float]:
    env = os.environ.copy()
    if name.endswith("_1t"):
        # Has to be set before numba is imported, which is why these cases are
        # named rather than configured: capping the pool at runtime leaves the
        # idle workers spinning, and process_time() counts them.
        env["NUMBA_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.run", "--emit", name],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"case {name!r} failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _compare(results: dict[str, dict[str, float]], baselines: dict) -> list[str]:
    """Metrics that exceeded their ceiling."""
    failures = []
    for case, metrics in results.items():
        limits = baselines.get("cases", {}).get(case, {})
        for metric, value in metrics.items():
            ceiling = limits.get(metric)
            if ceiling is None:
                continue  # recorded for context, not gated
            if value > ceiling:
                failures.append(f"{case}.{metric}: {value:.4g} exceeds ceiling {ceiling:.4g}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["fast", "slow", "all"], default="fast")
    parser.add_argument("--case", help="run a single case by name")
    parser.add_argument("--record", action="store_true", help="rewrite baselines.json")
    parser.add_argument("--emit", help=argparse.SUPPRESS)  # internal: run one, print JSON
    args = parser.parse_args()

    from benchmarks import cases as case_module

    if args.emit:
        print(json.dumps(case_module.ALL[args.emit]()))
        return 0

    if args.case:
        names = [args.case]
    else:
        chosen = {"fast": case_module.FAST, "slow": case_module.SLOW, "all": case_module.ALL}
        names = list(chosen[args.set])

    results = {}
    for name in names:
        results[name] = _run_one_in_subprocess(name)
        rendered = "  ".join(f"{k}={v:.4g}" for k, v in results[name].items())
        print(f"{name:32s} {rendered}")

    if args.record:
        existing = json.loads(BASELINES.read_text()) if BASELINES.exists() else {"cases": {}}
        margins = existing.get("margins", {})
        for case, metrics in results.items():
            ceilings = {}
            for metric, value in metrics.items():
                margin = margins.get(metric)
                if margin is None:
                    continue  # not a gated metric: recorded for context only
                ceilings[metric] = round(value * margin, 4)
            existing.setdefault("cases", {})[case] = ceilings
        BASELINES.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"\nrecorded ceilings to {BASELINES.name}")
        return 0

    if not BASELINES.exists():
        print("\nno baselines.json; run with --record first", file=sys.stderr)
        return 1

    failures = _compare(results, json.loads(BASELINES.read_text()))
    if failures:
        print("\nregressions:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("\nno regressions past recorded ceilings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
