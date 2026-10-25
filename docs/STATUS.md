# Build status

Checkpoint: October 25, 2026 (UTC).
Project: `/Users/frank/mygit/emmy-compiler`.

**64 local tests pass. All four public pointwise cases compile and match official
expected arrays in our CPU model. GPU PASS evidence remains limited to p00/p01.**

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
| p02_scale_shift | Exit 0 | Passed, rtol/atol 1e-6 | Not run |
| p03_relu_mul | Exit 0 | Passed, rtol/atol 0 | Not run |
| r01–r04 reductions | Unsupported | Not run | Not run |

The official basic scope is eight problems: four pointwise plus four reductions.
This is not yet the competition MVP. There are 15 public problems overall.

Latest verification correction, executed from the project directory:

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

Only static one-dimensional f32 buffers and one ordinary Tile are supported.
Rank-0 buffers, other types, partial grouped tails, control flow, reductions,
shared-memory synchronization and cooperative scheduling remain explicit errors.
The NaN conflict between the official expected-generation function and CUDA
reference is recorded in [NUMERICS.md](NUMERICS.md). The current official checker
also rejects NaN/NaN. Hidden exceptional-input requirements remain unresolved;
this is not a claim of universal f32 support. Deserialization accepts trusted
IR only.

Next implementation after pointwise GPU validation: a general first reduction,
including scalar result buffers, reduction loops/accumulators and the required
shuffle/shared-memory synchronization. No reduction work was started here.

## Minimal next GPU validation payload (not sent)

A separate authorization would cover exactly these generated PTX strings and
public launch/input descriptions, one execution each (`warmup=0`, `iters=1`):

| Case | Entry | Grid / block | Ordered f32 arguments |
| --- | --- | --- | --- |
| p02 | k_add_pointwise | (4,1,1) / (256,1,1) | mul_c1[1] data=[2.0], add_c1[1] data=[1.0], x[4096] seed=0, add[4096] output |
| p03 | k_mul_pointwise | (2,1,1) / (256,1,1) | x[2048] seed=0, y[2048] seed=1, mul[2048] output |

Both require zero shared memory and no output zeroing. PTX hashes are recorded in
local-results.json. The official HTTP runner would assemble and execute PTX;
expected.npz comparison stays local. No Python source, repository content,
credentials or private input data are needed. The existing runner wrapper still
allows only p00/p01; it has not been expanded or invoked by this change.

All new commits remain local. No publication, competition registration, paid
resources, other project's GPU quota or submission actions occurred. Before a
public entry, original-code licensing and competition IP terms need review; the
participant must supply the required 3–5 minute explanation in their own voice.
