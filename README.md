# Emmy Compiler

An independent Python compiler for the RiftStack Emmy Hackathon 2026. It reads
Emmy Kernel IR with `emmy-ml==0.3.11` and emits PTX through our own lowering.
No Emmy renderer, nvcc or other compiler generates the PTX.

**All 15 public problems passed official PTX assembly, RTX 5090 execution and
expected-array comparison. The final local suite has 169 passing tests.**
This includes all eight basic problems. Public-fixture success does not establish
hidden-test coverage, universal f32 behavior or competitive performance.

See [build status](docs/STATUS.md), [actual GPU results](docs/RUNNER_RESULTS.md),
[architecture](docs/ARCHITECTURE.md) and [submission readiness](docs/SUBMISSION_READINESS.md).

## Setup

Python 3.12+ is required. The validated local environment uses **CPython 3.12.12**
on macOS arm64 and the full pinned [requirements.lock](requirements.lock).
No torch, CUDA toolkit, GPU, account or credentials are needed for these tests.
Emmy declares additional dependencies; the lock records all 37 installed packages.

With an existing healthy Python 3.12 installation:

```sh
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.lock
.venv312/bin/python tools/fetch_reference.py
.venv312/bin/python -m unittest discover -s tests -v
```

On this Mac, Homebrew Python cannot import SQLite. The working interpreter was
installed inside this project using the already-installed `uv`; global Python
was not changed. To reproduce that isolated setup:

```sh
UV_CACHE_DIR=/tmp/emmy-uv-cache UV_PYTHON_INSTALL_DIR="$PWD/.python" uv python install 3.12.12 --no-bin
UV_CACHE_DIR=/tmp/emmy-uv-cache UV_PYTHON_INSTALL_DIR="$PWD/.python" uv venv --python 3.12.12 .venv312
UV_CACHE_DIR=/tmp/emmy-uv-cache uv pip install --python .venv312/bin/python -r requirements.lock
.venv312/bin/python tools/fetch_reference.py
.venv312/bin/python -m unittest discover -s tests -v
```

The reference fetcher is our small, reviewable downloader. It fetches only data
and documentation at a fixed official commit, verifies SHA-256 hashes, and never
runs upstream repository scripts. `.reference/` is local and excluded from Git.
Run it before tests in a new checkout. The lock was validated on this Mac;
other platforms must supply compatible wheels and have not been tested.

## Compile

The root entry point follows the official contract:

```sh
source .venv312/bin/activate
python compile.py .reference/problems/p00_scalar_add/k_add_pointwise.ir.json > /tmp/emmy-p00.ptx
python compile.py .reference/problems/r04_row_sum/k_sum_1_reduce.ir.json > /tmp/emmy-r04.ptx
python compile.py .reference/problems/f02_row_softmax/k_softmax_05263d.ir.json > /tmp/emmy-f02.ptx
python compile.py .reference/problems/m01_matmul_512/k_matmul_2967be.ir.json > /tmp/emmy-m01.ptx
```

Keep each IR next to its official `meta.json`. Entry names come from IR; pointer
argument order and launch contracts come from the matching metadata row. Runtime
constants stay in pointer buffers. Successful stdout is ASCII PTX only. Invalid
or unsupported contracts produce a nonzero exit, a stderr diagnostic and no
partial PTX. Only trusted official or self-generated IR is supported: Emmy's
required deserializer uses Python expression evaluation internally. This is not
an untrusted JSON upload service or a security sandbox.

## Coverage

| Category | Public cases | Official GPU result |
| --- | --- | --- |
| Pointwise | p00–p03 | 4/4 PASS |
| Reduction | r01–r04 | 4/4 PASS |
| Fused normalization / softmax | f01–f03 | 3/3 PASS |
| Attention | a01–a02 | 2/2 PASS |
| Matrix multiplication | m01–m02 | 2/2 PASS |

Backends are selected from IR structure, with sizes, bindings, loops and parameter
order derived from the actual graph. Tests change shapes, names, ABI order,
runtime values and schedules. Unsupported types, dynamic shapes and unproven
index/synchronization contracts fail explicitly; this is not all possible Emmy IR.

## Project map

- `compile.py`: required CLI.
- `emmy_ptx/loader.py`: official deserialization, buffer shapes and launch ABI.
- `emmy_ptx/ptx.py`: pointwise lowering and structural backend dispatch.
- `emmy_ptx/structured.py`: bounded indices, structured loops, scalar operations
  and ordinary output-coverage proofs.
- `emmy_ptx/reduction.py`, `cooperative.py`, `matmul.py`: sum/mean/row reduction,
  multi-state fused reductions, and shared-memory/async-copy contractions.
- `tests/`: deterministic parser, ABI, CLI, numeric-model and transport regressions.
- `tools/fetch_reference.py`: hash-pinned public data downloader.
- `tools/run_official.py`: bounded wrapper around the unchanged reviewed official
  client. It uploads PTX and public descriptors over HTTP and needs authorization.
- `docs/NUMERICS.md`: rounding, approximate exp, atomic subnormals and unresolved
  exceptional-input semantics.
- `THIRD_PARTY_NOTICES.md`: provenance and unresolved publication/licensing terms.

The Mac has no NVIDIA GPU or local ptxas. Local CPU instruction models do not
replace hardware evidence; the separate runner receipts document actual GPU
execution. Each public case has one successful GPU launch with no warmup and one
iteration, so recorded timings are indicative, not benchmarks. m01 first failed
assembly; its corrected module-scope shared declaration then passed. Both receipts
are retained.

The existing [private repository](https://github.com/Xuefeng-Zhu/emmy-compiler) was
last pushed through `519fc60`. Subsequent results and implementation commits
remain local. No public publication, registration or competition submission has
been performed. The participant still needs to decide licensing/publication and
record the required 3–5 minute explanation in their own voice.
