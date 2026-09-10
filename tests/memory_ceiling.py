"""Peak-RSS ceilings for operations that must not build an ``nnz``-sized temporary.

Complements ``benchmarks/harness.py``'s ``peak_alloc_mb``, which is a
``tracemalloc`` measurement and therefore sees only Python/numpy allocations --
numba's own are invisible to it, and it is not RSS. This measures the whole
process's high-water mark instead, which is the number a user actually runs out
of, and the only one that can express a claim like "the working set does not
grow with ``n_cells``".

Each measurement runs in a fresh interpreter, so the mark is attributable to
one operation rather than to whatever the test session did earlier. Inside it:

1. ``setup`` builds the input,
2. the peak is reset (``/proc/self/clear_refs``, which zeroes ``VmHWM`` back
   to the current ``VmRSS``), so building the input doesn't count against the
   ceiling,
3. ``work`` runs, and ``VmHWM`` is read back.

Step 2 is what makes the number meaningful: without it, every ceiling would be
dominated by the cost of the array under test. Where ``clear_refs`` isn't
writable the helper falls back to subtracting the current ``VmRSS``, which is
correct as long as ``setup`` left no transient above ``work``'s own peak -- so
the fallback is reported and the tests skip on it rather than silently
measuring something weaker.

Linux only; ``VmHWM`` has no portable equivalent. The tests using this are
skipped elsewhere.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

__all__ = ["linux_only", "peak_rss_growth_mb"]

linux_only = pytest.mark.skipif(
    not Path("/proc/self/status").exists(),
    reason="peak-RSS ceilings need /proc/self/status (Linux)",
)

_RUNNER = """\
import gc, json
from pathlib import Path

def _kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1])
    raise RuntimeError(field + " missing from /proc/self/status")

{setup}

# Drop anything setup left unreferenced, then zero the high-water mark so the
# ceiling below is about `work` alone and not about the array it runs on.
gc.collect()
try:
    Path("/proc/self/clear_refs").write_text("5")
    reset = _kb("VmHWM:") <= _kb("VmRSS:") + 1024
except OSError:
    reset = False
base = _kb("VmHWM:") if reset else _kb("VmRSS:")

{work}

print(json.dumps({{"growth_mb": (_kb("VmHWM:") - base) / 1024.0, "reset": reset}}))
"""


def peak_rss_growth_mb(setup: str, work: str) -> tuple[float, bool]:
    """Peak RSS ``work`` adds on top of ``setup``, in MB, measured in a fresh process.

    ``setup`` and ``work`` are top-level source, not expressions -- ``setup``
    binds whatever names ``work`` needs.

    Returns ``(growth_mb, peak_was_reset)``. A False second element means the
    process could not zero ``VmHWM`` and the number is only an upper bound.
    """
    script = _RUNNER.format(
        setup=textwrap.dedent(setup).strip(), work=textwrap.dedent(work).strip()
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"measurement process failed:\n{proc.stdout}\n{proc.stderr}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    return float(result["growth_mb"]), bool(result["reset"])
