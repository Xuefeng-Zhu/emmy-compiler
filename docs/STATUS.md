# Build status

Latest: **p00 and p01 passed official PTX assembly, RTX 5090 execution and exact
output comparison** on October 24, 2026 (America/Los_Angeles). See
[runner results](RUNNER_RESULTS.md). Both cases required one attempt; no compiler
fixes were needed. The 39 local tests passed again in 0.611 seconds.

The initial local-only checkpoint below is retained as historical context.

Checkpoint: October 24, 2026 (America/Los_Angeles).
Project: `/Users/frank/mygit/emmy-compiler`.

## Delivered

- Independent root CLI, typed official IR loading and metadata ABI validation.
- Our direct PTX 8.0/sm_89 lowering for the bounded 1-D f32 add slice.
- Official p00 and p01 compile locally; entry and shapes are read from input.
- Explicit rejection of unsupported features, invalid accesses and incomplete writes.
- English setup, architecture, challenge boundaries and third-party notices.
- Isolated working CPython 3.12.12 environment (`.python/` + `.venv312/`).
- All 37 dependencies pinned in requirements.lock; no global installation changes.

## Executed validation

From the project directory:

| Command | Observed result |
| --- | --- |
| `.venv312/bin/python tools/fetch_reference.py` | Verified 13 SHA-256-pinned official files at commit ec8cde1943f88efea3827a95bda6085553d2d407 |
| `.venv312/bin/python -B -m unittest discover -s tests -v` | 39 tests passed; final run 0.595 seconds |
| `.venv312/bin/python compile.py .reference/problems/p00_scalar_add/k_add_pointwise.ir.json > build/p00.ptx` | Exit 0; ASCII PTX emitted with 64-cell guard |
| `.venv312/bin/python compile.py .reference/problems/p01_vector_add/k_add_pointwise.ir.json > build/p01.ptx` | Exit 0; ASCII PTX emitted with 512-cell guard |
| `UV_CACHE_DIR=/tmp/emmy-uv-cache uv pip check --python .venv312/bin/python` | All 37 installed packages compatible |

Local test output is saved in `build/tests.txt`; generated PTX is in `build/`.
These artifacts are ignored by Git and reproducible using the commands above.
Tests cover actual official deserialization, repeated deterministic compilation,
entry renaming, changed small shapes, metadata argument reordering, global thread
index and bounds guard, loads/adds/stores, CLI arity, missing/malformed files,
unsupported operations/types, invalid SSA, address range, zero extent, overflowing
index arithmetic, overlapping/incomplete writes and clean failure output.

The tests above describe the initial local CPU/parser/text checks. Subsequent
explicitly authorized runner validation passed p00/p01, as recorded separately.
The Mac still has no ptxas or NVIDIA GPU; assembly and execution ran on the official
service through its reviewed client in an isolated `.venv-runner` environment.

## Environment issue resolved locally

The existing Homebrew Python 3.13 and 3.14 installations fail to import sqlite3
because `/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib` is missing. Emmy imports
sqlite3 while loading its compiler package. A project-contained CPython 3.12.12
runtime solved this prerequisite without changing Homebrew or system libraries.
The failed temporary Python 3.13 venv created for this task was removed; the
working `.venv312` remains.

## Remaining work

1. Completed: reviewed official client and verified p00/p01 through assembly,
   hardware execution and expected-output comparison. No failures or retries.
2. Add constants as buffer arguments and further pointwise arithmetic for p02/p03.
3. Add reduction statements, shuffle/shared-memory synchronization and coverage
   of all four reduction tasks. The official MVP totals eight tasks.
4. Validate shape/operation variations; benchmark only after correctness.
5. Before any public entry, confirm licensing/IP terms, select an original-code
   license, publish only authorized material and prepare the participant's own
   3–5 minute voice explanation.

The bounded setup and two authorized runner tests are complete. This does not
authorize further GPU jobs, a full-competition build, publication or submission.
The runner results and helper are committed locally only; no new push was made.
