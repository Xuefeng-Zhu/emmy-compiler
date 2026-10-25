# Build status

## Latest exp accuracy checkpoint — October 25, 2026 (UTC)

Only common natural-exp lowering changed from source baseline `ee40e27`.
**222 local tests passed in 149.783 seconds, exit 0**; the focused run passed
25 tests in 0.842 seconds, exit 0. These are local CPU/static checks.

All **14 fresh GPU checks passed** on NVIDIA GeForce RTX 5090 with unchanged
tolerances and `equal_nan=False`: f02/a01/a02, seven activation cases (s03–s09),
two guarded attention cases (s13/s14), and two exp diagnostics (s15/s16).
The other twelve public PTX artifacts remain byte-identical to GPU-passed
versions, preserving evidence for all 15 current public artifacts.

The 4097-value wide-range exp diagnostic improved from 251 failures to zero;
maximum relative error fell from 3.6186774e-6 to 1.52530531e-7. The 1019-value
finite-boundary diagnostic also passed, with matching zero classifications,
maximum observed 2 ULP and maximum relative error 1.6858741e-7 over normal
expected outputs. This sampled result is not a universal accuracy guarantee;
NaN/infinity classifications remain local-only checks.

Exactly 14 compile and 14 GPU requests completed, with zero retries and zero
uncertain outcomes. See the [exp accuracy report](EXP_ACCURACY_FIX.md) and
[retained evidence](validation/2026-10-25-exp-fix/). The historical failure and
prior checkpoint below remain part of the record.

## Historical pre-fix regression checkpoint — October 25, 2026 (UTC)

Compiler source remains at `0e16a83235a8b2180a9c40dc4ede06b8aa33d511`.
The existing 194-test suite passed again in 134.921 seconds (exit 0). With eight
new CPU checks and eight mocked-client tests, **210 tests passed in 138.357
seconds, exit 0**.

Current r01–r04 artifacts each passed official assembly, GPU execution and the
original expected-array comparison. Eleven other public PTX artifacts are
byte-identical to their earlier GPU-passed versions. This restores evidence for
all 15 current public artifacts without claiming that all 15 were rerun.

Fourteen normal synthetic cases passed on the RTX 5090: two odd-shaped matrix adds,
seven activation cases, three reductions and two guarded attention cases. The
separate 4097-value exp diagnostic over [-80, 80] failed at 251 GPU values under
unchanged rtol=2e-6/atol=2e-7; the CPU model failed at 257 values. The diagnostic
threshold comes from existing local tests, not a new official activation
problem. No threshold was weakened, and no compiler fix was applied.

All 19 new compile requests and 19 GPU requests completed with known outcomes:
zero retries and zero uncertain requests. They include the diagnostic failure;
completion does not mean every numerical comparison passed. See
[GENERALIZATION_VALIDATION.md](GENERALIZATION_VALIDATION.md) for the exact scope,
commands, exp counterexample and proposed next fix, and the
[latest evidence archive](validation/2026-10-25-regression/) for retained records.
Hidden-set and universal-input correctness remain unproven.

## Historical public GPU checkpoint — October 25, 2026 (UTC)

**15/15 public problems passed official PTX assembly, GPU execution and expected
output comparison. The corresponding local checkpoint passed 169 tests.** The observed runner was an NVIDIA
GeForce RTX 5090 (`sm_120`); emitted PTX remains version 8.0 targeting sm_89.
Original problem tolerances and `equal_nan=False` were preserved.

| Category | Local parsing/emission/numeric model | Official GPU |
| --- | --- | --- |
| p00–p03 pointwise | Pass | 4/4 PASS |
| r01–r04 reductions | Pass | 4/4 PASS |
| f01–f03 fused operations | Pass | 3/3 PASS |
| a01–a02 attention | Pass | 2/2 PASS |
| m01–m02 matrix multiplication | Pass | 2/2 PASS |

The first two categories contain eight problems, all now verified on the public
fixtures. There are no remaining public-case compiler failures or uncertain
runner requests.

## Shape and operation generalization

Historical checkpoint: October 24, 2026 (America/Los_Angeles). **194 tests passed
in 152.886 seconds**. This adds 25 tests to the public GPU checkpoint, covering
activation compositions, rank-2 add, transformed/cooperative max reductions,
conditional attention, branch scopes and guarded output ownership. An additional
focused conditional run passed all eight tests, including conjunction/disjunction
and equality guards. Original public tolerances remain unchanged.

At that checkpoint the CLI compiled all 15 public fixtures. Eleven PTX artifacts
remained
byte-identical to their GPU-passed versions. The four reduction artifacts had
changed to common structured address/arithmetic emission; their local models
passed, while the saved hardware receipts still described earlier artifacts.
That local checkpoint made no remote GPU requests. The latest regression above
supersedes this hardware-evidence gap with fresh r01–r04 GPU passes.

[Full test log](validation/2026-10-24-generalization/local-tests.txt),
[focused conditional log](validation/2026-10-24-generalization/conditional-tests.txt),
[source hashes and results](validation/2026-10-24-generalization/results.json),
[current PTX comparison](validation/2026-10-24-generalization/ptx-regression.json).

Separate fresh-IR probes build attention graphs directly and run Emmy's kernel
materialization passes, without its renderer. Both emission-order and offline
greedy schedules compile for `(rows, keys, depth, columns)` equal to
`(256, 256, 32, 32)`, `(96, 200, 32, 72)` and `(5, 9, 7, 11)`: **6/6 compile**.
The greedy irregular case generates 12 actual Cond nodes with compound `&&`
boundary guards. Both irregular schedules pass the independent PTX CPU model
against a float64 attention oracle, with maximum absolute error
`1.7881393432617188e-7` and one write per output. Larger cases were compiled only.
The probes disable golden/online scheduling evidence and bound structural pricing
to five seconds; they do not exercise every possible Emmy schedule.

[Fresh IR, PTX hashes and results](validation/2026-10-24-generalization/generated-attention/summary.json),
[reproduction script](validation/2026-10-24-generalization/generated-attention/check.py).
Run from the repository root:

```sh
python docs/validation/2026-10-24-generalization/generated-attention/check.py --output /tmp/emmy-attention-probe
```

## Local validation at the public GPU checkpoint

```sh
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > build/all-public-local-tests-final.txt 2>&1
```

Result: **169 tests in 145.803 seconds, OK; exit 0**. The suite includes actual
typed-IR parsing, argument/metadata validation, renamed/resized kernels, boundary
indices/tails, accumulator lifetimes, shared-memory/rendezvous models, numerical
comparisons against every public expected.npz, malformed IR and mocked transport.
It performs no network or GPU requests.

All 82 cached public reference files match the immutable hash manifest. The
declaration-scope correction changed only m01 PTX: all other 14 GPU-tested artifacts
were regenerated and verified byte-identical. No new dependencies were installed.

[Final test log](validation/2026-10-25/remaining-public/all-public-local-tests.txt),
[PTX hash regression](validation/2026-10-25/remaining-public/ptx-regression.json),
[all-public result record](validation/2026-10-25/all-public-results.json).
The preceding 168-test run is also retained beside the final log.

## Hardware validation and corrected failure

The remaining eleven cases were executed serially after explicit approval using
the pinned official submit.py through the reviewed bounded wrapper. This session
made twelve compile requests and eleven GPU launches. m01's first compile was
rejected because `.extern .shared` appeared inside the entry body; no GPU launch
occurred for that attempt. The general header emitter now places external shared
storage at module scope. Regression tests and one m01 retry confirmed the fix.

All successful launches used zero warmups and one iteration. The reported times
are not stable benchmarks. See [RUNNER_RESULTS.md](RUNNER_RESULTS.md) for exact
commands, per-case errors, register counts, timings and saved failure/pass receipts.

## Implemented scope and limits

The compiler handles the public pointwise, per-row reduction, multi-state fused,
structured attention and shared-copy contraction schedules. Dimensions, bindings,
loop parameters and runtime constants come from IR/metadata; no public case names
or expected values select compiler behavior. The architecture and explicit
unsupported domains are described in [ARCHITECTURE.md](ARCHITECTURE.md),
[REDUCTIONS.md](REDUCTIONS.md) and [ADVANCED_IR.md](ADVANCED_IR.md).

The Mac has no NVIDIA GPU or ptxas. Python 3.12.12, its 37 pinned dependencies and
the optional runner environment remain isolated. Models do not establish every
hardware schedule or exceptional-value behavior. Exp approximation, finite max
identity, atomic subnormals and the upstream ReLU NaN conflict remain documented
in [NUMERICS.md](NUMERICS.md).

The [repository](https://github.com/Xuefeng-Zhu/emmy-compiler) is public, as verified
through GitHub's repository API. Original code uses [Apache-2.0](../LICENSE).
Remaining entry work includes recording a personal voice explanation and
completing final submission.
See [SUBMISSION_READINESS.md](SUBMISSION_READINESS.md).
