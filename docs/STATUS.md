# Build status

Checkpoint: October 25, 2026 (UTC).
Project: `/Users/frank/mygit/emmy-compiler`.

**All four public pointwise cases passed PTX assembly, RTX 5090 execution and
official expected-array comparison. The latest local suite has 108 tests.**

p02/p03 each passed once with max absolute error 0 and client/wrapper exit 0.
Their single-launch timings were 14.976/15.328 microseconds, not benchmarks. No
compiler fixes or retries were needed. These results apply to the supplied finite
arrays; exceptional-input semantics and hidden variants remain unverified.

## Basic-eight local checkpoint

All eight basic public cases now compile and pass their local CPU models. Four
pointwise cases have prior GPU evidence; r01–r04 have no assembly/GPU evidence.
The new extension adds runtime-divisor mean and one-CTA-per-row sum while
preserving the four pointwise PTX hashes and r01/r02 behavior.

`PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > build/basic-eight-local-tests.txt 2>&1`

Result: **108 tests in 12.929 seconds, OK; exit 0**. The reference manifest now
verifies 82 immutable public files, including later categories prepared for local
work. [Full log](validation/2026-10-25/basic-eight-local-tests.txt) and
[result record](validation/2026-10-25/basic-eight-local-results.json).

| Case | Local model max absolute error | NVIDIA assembly / GPU |
| --- | --- | --- |
| r03_mean | 9.313225746154785e-10 | Not run |
| r04_row_sum | 5.7220458984375e-6 | Not run |

The multi-CTA model checks independent shared/register state, barrier participation
and unique output ownership. Tests vary rows, columns, block size, names, ABI
order and the runtime divisor independently of input length. All modeled inputs
of both official fixtures are read once and outputs written once.

Next local slice: reusable fused row normalization and online softmax, followed by
attention and matrix multiplication. New GPU uploads and pushes remain separate.

## Current local reduction checkpoint

Reusable single-block f32 sum lowering now covers r01/r02 at this historical checkpoint. It supports rank-0
outputs through the official pointer ABI, lengths derived from IR, full-warp
blocks from 32 to 1024 threads, idle lanes/tails and unique shared-memory writers.
All four pointwise PTX artifacts remain byte-identical to their GPU-tested hashes.

Command: `PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > build/reduction-local-tests.txt 2>&1`.
Result: **85 tests in 8.122 seconds, OK; exit 0**. All 36 immutable reference files
verified offline. [Test log](validation/2026-10-25/reduction-local-tests.txt),
[result record](validation/2026-10-25/reduction-local-results.json), and
[reduction scope](REDUCTIONS.md). The model also rejects missing synchronization,
uninitialized shared reads, incomplete participation, shared write races and
unknown instructions. It is not an assembler or GPU emulator.

| Case | Local CLI / cooperative CPU model | NVIDIA assembly / GPU |
| --- | --- | --- |
| r01_array_sum | Pass; max absolute error 3.814697265625e-6 | Not run |
| r02_big_sum | Pass; max absolute error 1.52587890625e-5 | Not run |
| r03_mean, r04_row_sum | See newer local checkpoint above | Not run |

The following sections retain historical pointwise checkpoints and their exact
commands. At that checkpoint coverage was four public GPU cases plus two local-only cases,
not a completed eight-problem MVP. Local implementation is continuing; new GPU
uploads and pushes need separate authorization.

## Delivered in this local extension

- Strict agreement between `args`, `arg_order`, pointer signatures and output roles.
- Clear CompileError diagnostics for malformed Assign operators/operands, including
  `op=None` and `args=None`, with empty CLI stdout and no traceback.
- ConstantOp buffers supplied through runtime pointers, including indexed arrays;
  values, buffer names and dimensions are not hardcoded from public cases.
- General f32 multiply and unary ReLU, with explicit rounding and NaN/zero choices.
- Tests for changed lengths/constants, multiple-element constants, parameter
  reordering, unsupported types/operations, exceptional values and accidental FMA.
- English architecture/numerics notes and a hash-pinned manifest of 26 upstream
  files. Reference files, virtual environments and build output remain ignored.

## Evidence and limits

| Official case | CLI generation | CPU model vs official expected.npz | PTX assembly / GPU comparison |
| --- | --- | --- | --- |
| p00_scalar_add | Exit 0 | Passed | Prior PASS, RTX 5090 |
| p01_vector_add | Exit 0 | Passed | Prior PASS, RTX 5090 |
| p02_scale_shift | Exit 0 | Passed, rtol/atol 1e-6 | PASS, RTX 5090 |
| p03_relu_mul | Exit 0 | Passed, rtol/atol 0 | PASS, RTX 5090 |
| r01/r02 reductions | Exit 0 | Passed, original tolerances | Not run |
| r03/r04 reductions | Exit 0 | Passed, original tolerances | Not run |

The official basic scope is eight problems: four pointwise plus four reductions.
This is not yet the competition MVP. There are 15 public problems overall.

Latest runner validation used the two serial commands in RUNNER_RESULTS.md.
Final local regression:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > build/runner-pointwise-regression.txt 2>&1
```

Result: **67 tests in 6.005 seconds, OK; exit 0**. This includes three mocked
transport tests (seven scenarios) that validate the exact public payloads, output
names/completeness and rejection of NaN results without contacting the runner.
[Full log](validation/2026-10-25/runner-regression.txt) and
[result record](validation/2026-10-25/runner-regression.json).
The real p02/p03 receipts were verified intact after the mocked tests.

Prior comparison-policy correction checkpoint, executed from the project directory:

```sh
.venv312/bin/python -B -m unittest discover -s tests -v > build/comparison-policy-tests.txt 2>&1
```

Result: **64 tests in 13.219 seconds, OK; exit 0**. Official-array comparisons now
explicitly use `equal_nan=False` with unchanged metadata tolerances. Regressions
reject NaN/NaN, NaN/zero and zero/NaN, and accept signed zeros. A separate semantic
test checks ReLU followed by multiply against the official expected-generation
expression `np.maximum(x, 0) * y`, including NaN classification and exact non-NaN
bits. All four finite official expected arrays still match in the CPU model.
The [correction test log](validation/2026-10-25/comparison-policy-tests.txt) and
[result record](validation/2026-10-25/comparison-policy-results.json) preserve this run.
Compiler emission, reference metadata, tolerances and previous GPU receipts are
unchanged. No GPU requests, dependencies or pushes were added.

The preceding pointwise-extension checkpoint is retained below. Executed from
the project directory:

```sh
.venv312/bin/python -B -m unittest discover -s tests -v > build/local-pointwise-tests.txt 2>&1
.venv312/bin/python -B compile.py .reference/problems/p00_scalar_add/k_add_pointwise.ir.json > build/p00.ptx
.venv312/bin/python -B compile.py .reference/problems/p01_vector_add/k_add_pointwise.ir.json > build/p01.ptx
.venv312/bin/python -B compile.py .reference/problems/p02_scale_shift/k_add_pointwise.ir.json > build/p02.ptx
.venv312/bin/python -B compile.py .reference/problems/p03_relu_mul/k_mul_pointwise.ir.json > build/p03.ptx
```

The test command exited 0: **62 tests in 4.463 seconds, OK**. Each compiler command
exited 0 with ASCII PTX and empty stderr. An offline SHA-256 check verified all 26
cached files against reference-manifest.json, pinned to upstream commit
`ec8cde1943f88efea3827a95bda6085553d2d407`. The saved
[test output](validation/2026-10-25/local-tests.txt) and
[local result record](validation/2026-10-25/local-results.json) contain details.
No new dependencies, runner requests, GPU jobs or repository pushes were made.

The numeric test module now has eleven methods covering CPU-model value/address
flow, output write coverage and the acceptance comparator. They cover scalar
lengths 1, 3, 255, 256, 257, 1023 and 1025;
width-four shapes 4, 12 and 1028; runtime constants; a nine-element constant array;
parameter reordering; special float values; and separate multiply/add rounding.
This model is deliberately limited and is not a PTX assembler or GPU emulator.

Generated p00/p01 PTX hashes are byte-identical to the artifacts used for the
previous authorized GPU tests. Those tests each passed once on an RTX 5090 with
zero maximum absolute error. Their 23.584/20.352 microsecond measurements used
zero warmups and one iteration; they are not a stable performance benchmark.
Original commands and receipts remain unchanged in [RUNNER_RESULTS.md](RUNNER_RESULTS.md).

## Environment and remaining scope

The Mac has no NVIDIA GPU or ptxas. CPython 3.12.12 and the 37 pinned dependencies
remain isolated in `.python/` and `.venv312/`; the optional runner environment is
separate. The project-local runtime avoids the existing Homebrew SQLite issue.

Pointwise support remains static one-dimensional f32 buffers and one ordinary
Tile. The per-row f32 sum topology additionally supports rank-0 outputs, rank-2 row
inputs, runtime division,
strided accumulation, shuffles and synchronized shared storage. Other reduction
topologies and shapes remain explicit errors at this checkpoint.
The NaN conflict between the official expected-generation function and CUDA
reference is recorded in [NUMERICS.md](NUMERICS.md). The current official checker
also rejects NaN/NaN. Hidden exceptional-input requirements remain unresolved;
this is not a claim of universal f32 support. Deserialization accepts trusted
IR only.

The r03/r04 local implementation is complete. New reduction
GPU uploads are pending separate approval; local work can proceed independently.

## Completed p02/p03 GPU payload

After separate explicit approval, exactly these generated PTX strings and public
launch/input descriptions were sent, one execution each (`warmup=0`, `iters=1`):

| Case | Entry | Grid / block | Ordered f32 arguments |
| --- | --- | --- | --- |
| p02 | k_add_pointwise | (4,1,1) / (256,1,1) | mul_c1[1] data=[2.0], add_c1[1] data=[1.0], x[4096] seed=0, add[4096] output |
| p03 | k_mul_pointwise | (2,1,1) / (256,1,1) | x[2048] seed=0, y[2048] seed=1, mul[2048] output |

Both used zero shared memory and no output zeroing. The PTX hashes match the
prior local-results.json and the new per-case receipts. The official HTTP service
assembled and executed PTX; expected.npz comparison happened locally with the
original tolerances and equal_nan=False. No Python source, repository content,
credentials or private input data was sent to the runner. The constrained wrapper
now recognizes p00–p03; that does not authorize further runs.

The four previously authorized commits through 519fc60 were pushed to the existing
private repository; remote main SHA and privacy were verified. The subsequent
p02/p03 runner results and wrapper/test changes remain local and unpushed. No
public publication, competition registration, paid resources, other project's GPU
quota or submission actions occurred. Before a
public entry, original-code licensing and competition IP terms need review; the
participant must supply the required 3–5 minute explanation in their own voice.
