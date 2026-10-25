# Emmy Compiler

An independent, local Python compiler for the RiftStack Emmy Hackathon 2026.
It reads Emmy Kernel IR with `emmy-ml==0.3.11` and emits PTX using our own lowering.
The current slice handles one-dimensional, statically sized f32 addition,
multiplication and ReLU, with runtime constant buffers. All four official
pointwise cases (`p00` through `p03`) compile locally. A validated single-block
f32 sum path also compiles `r01` and `r02`, including rank-0 result buffers.

**Status: all four public pointwise cases passed official PTX assembly, GPU
execution and expected-array comparison; 85 local tests pass. r01/r02 have
local PTX generation and cooperative CPU-model checks only.**
The GPU runs cover the supplied finite inputs. Exceptional-input behavior and
hidden variants are not validated; the CPU model remains a test helper.
This is an initial working slice, not a completed competition MVP. See
[build status](docs/STATUS.md) and [architecture and supported scope](docs/ARCHITECTURE.md).

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
python compile.py .reference/problems/p01_vector_add/k_add_pointwise.ir.json > /tmp/emmy-p01.ptx
python compile.py .reference/problems/p02_scale_shift/k_add_pointwise.ir.json > /tmp/emmy-p02.ptx
python compile.py .reference/problems/p03_relu_mul/k_mul_pointwise.ir.json > /tmp/emmy-p03.ptx
```

Keep each IR next to its official `meta.json`. Kernel entry names come from the
IR; pointer parameter order comes from the matching metadata row. Its `args`
descriptors must agree with `arg_order`, buffer types/shapes and input/output roles.
Constants remain pointer arguments whose values are supplied by the runtime. Successful
stdout is ASCII PTX only. Unsupported operations and invalid input produce a
nonzero exit with a diagnostic on stderr and no partial PTX.

Only compile trusted official or self-generated IR: Emmy's required deserializer
uses Python expression evaluation internally. This program is not an untrusted
JSON upload service or a security sandbox.

## Project map

- `compile.py`: command-line entry point.
- `emmy_ptx/loader.py`: official IR deserialization and launch/ABI validation.
- `emmy_ptx/ptx.py`: typed statement walk, virtual registers and direct PTX emission.
- `emmy_ptx/reduction.py`: validated cooperative sum plan and direct PTX emission.
- `tests/`: parser/ABI/CLI regressions, numeric/address-flow models, shuffle/barrier
  checks and mocked transport tests; no GPU or runner calls.
- `docs/REDUCTIONS.md`: reduction topology, synchronization and local evidence.
- `docs/NUMERICS.md`: rounding, ReLU exceptional values and the reference discrepancy.
- `tools/fetch_reference.py`: immutable public reference downloads.
- `tools/run_official.py`: bounded wrapper around the reviewed official runner client;
  requires separate authorization before uploading PTX or running GPU tests.
- `requirements-runner.lock`: optional isolated runner-client environment.
- `docs/CHALLENGE.md`: official boundaries and later submission prerequisites.
- `THIRD_PARTY_NOTICES.md`: source provenance and unresolved licensing questions.

All four authorized public pointwise cases passed on an RTX 5090 with zero
maximum absolute error. See [runner results](docs/RUNNER_RESULTS.md) for exact
commands, stdout, receipts and limits. Each case ran once; the timings are not
performance benchmarks. The [private source repository](https://github.com/Xuefeng-Zhu/emmy-compiler)
was updated through commit `519fc60`. The subsequent p02/p03 runner results and
wrapper adaptation remain local and unpushed.

Local implementation is continuing through mean and row sum, then the remaining
public categories. New GPU validation, hidden-set validation and competition
submission remain pending. Local continuation does not authorize new uploads.
