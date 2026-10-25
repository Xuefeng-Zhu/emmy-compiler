# Emmy Compiler

An independent Python compiler for the RiftStack Emmy Hackathon 2026. It reads
Emmy Kernel IR with `emmy-ml==0.3.11` and emits PTX directly. Emmy provides typed
IR deserialization; our code implements the lowering.

**15/15 public problems passed official PTX assembly, RTX 5090 execution and
expected-array comparison with the original tolerances.**
[GPU receipts and results](docs/RUNNER_RESULTS.md) preserve every pass and the
corrected matmul assembly failure. The local suite also checks changed shapes,
operation combinations, ABI order and invalid IR.
The current [local validation](docs/STATUS.md#shape-and-operation-generalization)
passes **194 tests**.

## Setup

Use Python 3.12+ with `venv` and `pip`. The pinned environment was validated with
CPython 3.12.12 on macOS arm64; compilation and local tests need no GPU, CUDA
toolkit, torch, account or credentials.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python tools/fetch_reference.py
python -m unittest discover -s tests -v
```

The reference fetcher downloads public fixtures and documentation from a fixed
official commit and verifies their SHA-256 hashes. It runs no upstream scripts.
`.reference/` is excluded from Git. For an interpreter with a broken SQLite
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

The saved GPU runs cover these public fixtures. Changed operations and shapes are
checked locally; their hardware validation is tracked separately. Each recorded
GPU launch used no warmup and one iteration, so timings are illustrative.

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
