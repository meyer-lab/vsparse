# CLAUDE.md

Guidance for AI coding agents (and human contributors) working in this repository.

## Maintainability checks for new/changed code

This project's dev dependency group (`uv sync --group dev`) includes several
static-analysis tools that are not wired into pre-commit/CI, so they won't
run automatically. Whenever you add or substantially modify Python code in
`src/vsparse/`, run the relevant tool(s) below yourself and address anything
they flag before considering the change done.

### Dead code -- vulture

```sh
uv run vulture src/vsparse/
```

Flags functions, variables, and imports that appear to be unused. Before
deleting a reported item, confirm with `grep`/`git grep` that it really has
no callers (vulture's confidence score is a heuristic, and a few things --
e.g. manually-invoked one-off scripts -- are meant to have no in-repo
callers). If something is a deliberate exception, prefer leaving a short
comment explaining why over silencing the tool.

### Duplicated code -- jscpd

```sh
npx jscpd src/vsparse/ --min-lines 5 --min-tokens 50
```

(No install needed beyond Node/npx; jscpd is not a Python dependency.) Flags
copy-pasted blocks. If a new module duplicates an existing block of 10+
lines, prefer extracting a shared helper instead of copy-pasting.

### Cyclomatic complexity -- radon + xenon

```sh
uv run radon cc src/vsparse/ -n C -s   # list functions ranked C or worse
uv run radon mi src/vsparse/ -s        # maintainability index per file
uv run xenon --max-absolute B --max-modules A --max-average A src/vsparse/
```

`xenon` exits non-zero and prints every function/module exceeding the given
rank thresholds (A best -- F worst). Treat a new function ranked C or worse
as a signal to break it up: extract the branchy/loop-heavy interior into one
or more named helper functions (as opposed to introducing more parameters or
flags to the same function). A handful of pre-existing functions still
exceed these thresholds; it's fine to leave those alone unless you're
already modifying them, but don't add new ones.

## Why these aren't pre-commit hooks

`ruff` and `codespell` are fast and have an unambiguous pass/fail; they run
on every commit. `vulture`, `jscpd`, and `xenon` are noisier and require
judgment calls (a flagged function may be an intentional exception), so
they're kept as tools to run and reason about manually rather than hard
commit gates.
