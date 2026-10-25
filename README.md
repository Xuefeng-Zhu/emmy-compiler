# Emmy Compiler

An independent Python compiler for the RiftStack Emmy Hackathon 2026. It reads
Emmy Kernel IR with `emmy-ml==0.3.11` and emits PTX directly. Emmy provides typed
IR deserialization; our code implements the lowering.

**15/15 public problems passed official PTX assembly, RTX 5090 execution and
expected-array comparison with the original tolerances.**
[GPU receipts and results](docs/RUNNER_RESULTS.md) preserve every pass and the
corrected matmul assembly failure. The local suite also checks changed shapes,
operation combinations, ABI order and invalid IR.
The latest [exp accuracy fix](docs/EXP_ACCURACY_FIX.md) passes **222 local tests**
and **14/14 fresh GPU checks**. The three changed public artifacts (f02/a01/a02)
passed again; the other twelve public PTX artifacts are byte-identical to their
GPU-passed versions.

The initial [CPU CI/reproduction checks](docs/CI.md) added 18 integrity and failure-
propagation tests: **240 local tests pass** in a fresh Python 3.12.12 environment.
The learning material adds three checks, bringing the local suite to **243 passing
tests**. The compiler and previously validated PTX are unchanged.

## Learn the compiler

Start with the [engineering walkthrough](docs/LEARNING_GUIDE.md), or open the
[offline interactive field guide](docs/explainer/index.html) in a browser after
cloning/downloading the repository. Step through real generated PTX, inspect
thread ownership, and explore the saved exp GPU error data. GitHub displays the
HTML source; this project does not deploy a website.

Use the [3–5 minute own-voice outline](docs/WALKTHROUGH.md) to prepare a video.
The [explainer notes](docs/explainer/README.md) cover provenance, regeneration,
accessibility, screenshots and exact validation commands.

## Setup

Use Python 3.12+ with `venv` and `pip`. The pinned environment was validated with
CPython 3.12.12 on macOS arm64; compilation and local tests need no GPU, CUDA
toolkit, torch, account or credentials.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip --isolated --disable-pip-version-check install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.lock
python -m pip --isolated check
python tools/fetch_reference.py
python tools/fetch_reference.py --verify-only
python tools/run_tests.py
python tools/check_cli.py
```

The reference fetcher downloads public fixtures and documentation from a fixed
official commit and verifies their SHA-256 hashes. It runs no upstream scripts.
`.reference/` is excluded from Git. The [CPU CI workflow](.github/workflows/ci.yml)
repeats installation, fixture verification, the complete suite and all 15 CLI
emissions on Python 3.12.12 / Linux. It rejects empty or skipped test suites and
does not contact the GPU runner. See [clean-environment reproduction](docs/CI.md).
For an interpreter with a broken SQLite
module, see the optional [development environment notes](docs/DEVELOPMENT.md).

## Compile

The root entry point follows the official contract:

```sh
python compile.py .reference/problems/p00_scalar_add/k_add_pointwise.ir.json > /tmp/emmy-p00.ptx
python compile.py .reference/problems/r04_row_sum/k_sum_1_reduce.ir.json > /tmp/emmy-r04.ptx
python compile.py .reference/problems/f02_row_softmax/k_softmax_05263d.ir.json > /tmp/emmy-f02.ptx
python compile.py .reference/problems/m01_matmul_512/k_matmul_2967be.ir.json > /tmp/emmy-m01.ptx
```

Keep each IR beside its official `meta.json`. The IR supplies entry names and
operations; metadata supplies pointer argument order and launch geometry. Runtime
constants stay in pointer buffers. Successful stdout contains ASCII PTX only;
invalid or unsupported IR returns a nonzero exit, a stderr diagnostic and no
partial PTX.

Use trusted official or self-generated IR. Emmy's required deserializer evaluates
Python expressions internally, so this CLI is not suitable for hostile uploads.

## Results

| Category | Public cases | Official GPU result |
| --- | --- | --- |
| Pointwise | p00–p03 | 4/4 PASS |
| Reduction | r01–r04 | 4/4 PASS |
| Fused normalization / softmax | f01–f03 | 3/3 PASS |
| Attention | a01–a02 | 2/2 PASS |
| Matrix multiplication | m01–m02 | 2/2 PASS |

The saved GPU runs cover these public fixtures. The latest common-exp change
also passed seven activation cases, two guarded attention cases, the 4097-value
wide-range diagnostic and a 1019-value finite-boundary diagnostic. The wide-range
diagnostic now has zero failures, down from 251, with its original threshold.
The finite-boundary sample includes zero/subnormal outputs; it does not establish
universal f32 accuracy. Official tolerances and `equal_nan=False` are unchanged.
Single-iteration timings are illustrative. See the
[exp accuracy report](docs/EXP_ACCURACY_FIX.md),
[latest evidence archive](docs/validation/2026-10-25-exp-fix/) and preserved
[earlier regression](docs/GENERALIZATION_VALIDATION.md).

Local variants cover exp/tanh/sigmoid/SiLU/GELU and abs compositions, rank-2 add,
row maximum, transformed reductions such as sum of squares, and conditional
attention with changed key lengths and rectangular shapes.

## How it works

`compile.py` loads the typed graph and validates its ABI, then lowers each
supported statement to PTX. Common code handles scalar arithmetic, addresses,
registers, loops and output coverage. Cooperative reductions and shared-memory
contractions add schedule-specific shuffle, barrier and copy handling. Sizes,
bindings and loop bounds come from the graph, without problem-name dispatch.

- [`loader.py`](emmy_ptx/loader.py): deserialization, shapes and launch ABI.
- [`ptx.py`](emmy_ptx/ptx.py): common scalar arithmetic, register allocation and
  backend dispatch.
- [`structured.py`](emmy_ptx/structured.py): scalar statements, bounded indices,
  control flow and output-coverage checks.
- [`reduction.py`](emmy_ptx/reduction.py), [`cooperative.py`](emmy_ptx/cooperative.py)
  and [`matmul.py`](emmy_ptx/matmul.py): cooperative and shared-memory schedules.
- [`tests/`](tests): parser, CLI, ABI, numerical and synchronization regressions.
- [`tools/run_official.py`](tools/run_official.py): bounded wrapper around the
  unchanged official GPU client.

See [architecture](docs/ARCHITECTURE.md) for dispatch and invariants,
[numerics](docs/NUMERICS.md) for floating-point semantics, and
[the four-minute walkthrough](docs/WALKTHROUGH.md) for a presentation outline.

The [source repository](https://github.com/Xuefeng-Zhu/emmy-compiler) is public.
Original code is licensed under [Apache-2.0](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) record dependency and reference
provenance. The remaining entry materials are tracked in
[submission readiness](docs/SUBMISSION_READINESS.md).
