# Build status

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

All tests above are local CPU/parser/text checks. **No PTX assembly, GPU numerical
execution, official runner request or official problem pass is claimed.**
The installed Mac environment has no ptxas or NVIDIA GPU. No official upstream
repository script was executed.

## Environment issue resolved locally

The existing Homebrew Python 3.13 and 3.14 installations fail to import sqlite3
because `/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib` is missing. Emmy imports
sqlite3 while loading its compiler package. A project-contained CPython 3.12.12
runtime solved this prerequisite without changing Homebrew or system libraries.
The failed temporary Python 3.13 venv created for this task was removed; the
working `.venv312` remains.

## Remaining work

1. After user authorization, review the official runner client and use its
   ptxas/GPU feedback for p00 and p01. Resolve any assembly/ABI/numerical failures
   before extending the supported subset.
2. Add constants as buffer arguments and further pointwise arithmetic for p02/p03.
3. Add reduction statements, shuffle/shared-memory synchronization and coverage
   of all four reduction tasks. The official MVP totals eight tasks.
4. Validate shape/operation variations; benchmark only after correctness.
5. Before any public entry, confirm licensing/IP terms, select an original-code
   license, publish only authorized material and prepare the participant's own
   3–5 minute voice explanation.

The current bounded setup is complete. This checkpoint does not authorize an
ongoing full-competition build, remote GPU use, publication or submission.
