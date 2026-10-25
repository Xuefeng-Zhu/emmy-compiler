# Generalization validation — October 25, 2026 (UTC)

> Historical checkpoint, superseded by the [exp accuracy fix](EXP_ACCURACY_FIX.md)
> and its [new evidence archive](validation/2026-10-25-exp-fix/). Range reduction
> fixes the sampled wide-range failure under unchanged thresholds. The original
> results and failure receipts below remain unchanged as pre-fix evidence.

Compiler under test: `0e16a83235a8b2180a9c40dc4ede06b8aa33d511`. Production compiler files were unchanged throughout this test task.

The original local suite passed **194 tests in 134.921 seconds**. After adding eight CPU-model checks and eight mocked-transport checks, the expanded suite passed **210 tests in 138.357 seconds**, exit 0. These local suites perform no network or GPU work.

All four changed public reduction artifacts passed fresh official assembly, GPU execution and expected-array comparison. The other eleven public artifacts were regenerated and verified byte-identical to their earlier GPU-passed PTX. This establishes current artifact evidence for all 15 public fixtures, with only four public cases rerun in this session.

Fourteen bounded synthetic cases passed. A separate wide-range exp diagnostic failed its unchanged accuracy threshold. This is a retained numerical limitation, not a passing hidden test or a fixed defect.

[Machine-readable summary](validation/2026-10-25-regression/summary.json), [current public PTX hashes](validation/2026-10-25-regression/current-public-artifacts.json), [210-test log](validation/2026-10-25-regression/expanded-local-tests.txt), and [evidence hashes](validation/2026-10-25-regression/evidence-manifest.json).

## Actual GPU results

The observed device was **NVIDIA GeForce RTX 5090, sm_120**. The reviewed pinned official client made exactly **19 compile requests and 19 successful GPU launches**, serially, with no retries or uncertain outcomes. Eighteen numeric comparisons passed and one failed. Every launch used zero warmups and one timed iteration; the times below are indicative device measurements, not benchmark claims.

| Case | Assembly / execution | Numeric result | Max absolute error | One iteration (us) |
| --- | --- | --- | ---: | ---: |
| [r01_array_sum](validation/2026-10-25-regression/official/r01_array_sum/r01_array_sum.json) | PASS / PASS | PASS | 3.81469727e-06 | 21.120 |
| [r02_big_sum](validation/2026-10-25-regression/official/r02_big_sum/r02_big_sum.json) | PASS / PASS | PASS | 1.52587891e-05 | 37.280 |
| [r03_mean](validation/2026-10-25-regression/official/r03_mean/r03_mean.json) | PASS / PASS | PASS | 9.31322575e-10 | 28.736 |
| [r04_row_sum](validation/2026-10-25-regression/official/r04_row_sum/r04_row_sum.json) | PASS / PASS | PASS | 5.7220459e-06 | 21.312 |
| [s01_add_7x13](validation/2026-10-25-regression/synthetic/s01_add_7x13/report.json) | PASS / PASS | PASS | 0 | 21.312 |
| [s02_add_17x19](validation/2026-10-25-regression/synthetic/s02_add_17x19/report.json) | PASS / PASS | PASS | 0 | 28.256 |
| [s03_exp_193](validation/2026-10-25-regression/synthetic/s03_exp_193/report.json) | PASS / PASS | PASS | 0.000732421875 | 20.224 |
| [s04_tanh_193](validation/2026-10-25-regression/synthetic/s04_tanh_193/report.json) | PASS / PASS | PASS | 1.1920929e-07 | 27.360 |
| [s05_sigmoid_193](validation/2026-10-25-regression/synthetic/s05_sigmoid_193/report.json) | PASS / PASS | PASS | 5.96046448e-08 | 26.880 |
| [s06_silu_193](validation/2026-10-25-regression/synthetic/s06_silu_193/report.json) | PASS / PASS | PASS | 2.38418579e-07 | 20.256 |
| [s07_erf_193](validation/2026-10-25-regression/synthetic/s07_erf_193/report.json) | PASS / PASS | PASS | 2.38418579e-07 | 20.320 |
| [s08_gelu_193](validation/2026-10-25-regression/synthetic/s08_gelu_193/report.json) | PASS / PASS | PASS | 4.76837158e-07 | 20.832 |
| [s09_gelu_tanh_193](validation/2026-10-25-regression/synthetic/s09_gelu_tanh_193/report.json) | PASS / PASS | PASS | 4.76837158e-07 | 20.832 |
| [s10_sum_1031_t256](validation/2026-10-25-regression/synthetic/s10_sum_1031_t256/report.json) | PASS / PASS | PASS | 0 | 20.256 |
| [s11_sum_squares_4099_t256](validation/2026-10-25-regression/synthetic/s11_sum_squares_4099_t256/report.json) | PASS / PASS | PASS | 0 | 20.640 |
| [s12_max_negative_73_t64](validation/2026-10-25-regression/synthetic/s12_max_negative_73_t64/report.json) | PASS / PASS | PASS | 0 | 20.384 |
| [s13_guarded_attention_5x11x7x6](validation/2026-10-25-regression/synthetic/s13_guarded_attention_5x11x7x6/report.json) | PASS / PASS | PASS | 1.78813934e-07 | 21.152 |
| [s14_generated_greedy_attention_5x9x7x11](validation/2026-10-25-regression/synthetic/s14_generated_greedy_attention_5x9x7x11/report.json) | PASS / PASS | PASS | 1.78813934e-07 | 26.752 |
| [s15_exp_wide_4097](validation/2026-10-25-regression/synthetic/s15_exp_wide_4097/report.json) | PASS / PASS | FAIL | 1.38649284e+29 | 15.456 |

All four public comparisons use the original `meta.json` tolerances and `equal_nan=False`. Synthetic additions compare exactly; synthetic reductions retain r01 tolerances `rtol=1e-4, atol=1e-4`, and attention retains `rtol=1e-3, atol=1e-4`. Standalone new activations have no corresponding public official tolerance. Their thresholds are the existing local regression thresholds: exp/tanh/sigmoid/SiLU `2e-6, 2e-7`; erf `2e-6, 6e-7`; GELU variants `2e-5, 1e-6`. None was loosened after a result.

The 14 normal cases cover seeded odd rank-2 addition, seven activations with random finite inputs and transition/zero/subnormal witnesses, tail sums, sum of squares, an all-negative maximum with the last tail element as maximum, a guarded rectangular attention schedule, and the previously materialized fresh greedy attention schedule with compound boundary conditions. Inputs, metadata, expected arrays and emitted PTX have separate hashes. Each full returned array is saved; the generator reproduces all 45 core fixture files byte-for-byte.

## Retained exp accuracy failure

For 4097 evenly spaced float32 inputs in `[-80, 80]`, the independent float64 exp oracle is rounded to float32 before comparison. With the existing small-range threshold `rtol=2e-6, atol=2e-7`, the GPU result fails **251/4097 values**, and the CPU PTX model fails 257. Assembly and execution both succeeded; the official client and wrapper returned **exit 1** for numeric failure. Full output and failure stdout are retained.

The maximum relative error across all nonzero outputs is `3.6186773989542874e-6` on the GPU. One failing witness is `x=44.609375`: expected `2.363770999225398e19`, actual `2.363763522546329e19`, relative error `3.163030205263915e-6`. The large absolute error reflects the magnitude of exp outputs. The largest relative-error point itself is very small and passes the absolute-tolerance part of the checker.

The shared [exp emitter](../emmy_ptx/ptx.py) multiplies x by a rounded float32 log2(e), rounds that product, then uses `ex2.approx.f32`. The CPU model already exceeds the threshold before the GPU approximation is considered, so loss from range scaling is a demonstrated contributor. The public finite fixtures and sampled `[-8,8]` activation case still pass. Existing documentation did not promise a universal f32 exp bound; this test establishes a concrete wider-domain accuracy gap.

The smallest focused implementation proposal is to add range reduction inside the common exp emitter: choose an integer binary exponent, calculate a small residual using split ln(2) constants, evaluate exp2 on that residual and rescale. Keep the same threshold, include these witnesses, and validate rounding, overflow and subnormal boundaries before extending the claimed domain. This proposal was **not implemented** in this test-only task. Do not fix the failure by changing expected values or widening tolerances.

## Reproduction and protocol

The official [client](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/submit.py) accepts a custom problem folder; the [API](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/docs/API.md) supports seeded or explicit buffer inputs. Launch values are derived from each validated IR schedule and its adjacent metadata. Synthetic cases are our test inputs, not the organizers' hidden set.

Actual local commands, from the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > /tmp/emmy-0e16a83-local-tests.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > /tmp/emmy-0e16a83-expanded-tests.txt 2>&1
.venv312/bin/python -B tools/generate_synthetic_fixtures.py --output build/validation-0e16a83/reproduced-fixtures
```

The generator finishes all 15 cases and exits 1 because it deliberately reports the exp diagnostic failure. Its output must be a new directory. Regenerating does not contact a server. The expanded suite remains green because the additional diagnostic is reported separately rather than silently skipped or weakened.

The four public GPU commands used the archived [capture wrapper](validation/2026-10-25-regression/capture_official.py), which only records complete responses around the existing bounded `tools/run_official.py`. Each was invoked as `.venv-runner/bin/python -B /Users/frank/Documents/Codex/2026-10-24/task/emmy-validation-tools/capture_official.py CASE`, with CASE in order `r01_array_sum`, `r02_big_sum`, `r03_mean`, `r04_row_sum`.

Exact synthetic commands, paths and individual exit codes are in [synthetic-batch.json](validation/2026-10-25-regression/synthetic-batch.json). Their common form was:

```sh
.venv-runner/bin/python -B tools/run_synthetic.py CASE_FOLDER --project "$PWD" --manifest CASE_FOLDER/manifest.json --attempt NEW_ATTEMPT_DIRECTORY
```

Only generated PTX and public/synthetic launch/input descriptions were sent to the same disclosed `http://runner.riftstack.ai:48701` service. Source files, manifests, oracles, credentials and repository metadata remain local. The wrapper pins the official client, freezes hashed fixture files, rejects unexpected destinations/payloads/redirects, disables inherited proxy/auth/cookies, and persists request state before transport. It requires both official PASS/exit 0 and an independent complete-output comparison. The mock suite covers malformed/truncated/extra outputs, NaNs, mismatch, busy limits, timeout uncertainty and assembly/local-generation failures.

Raw working snapshots, complete frozen input cases and previous receipts remain under ignored `build/validation-0e16a83/`. The selected evidence archive preserves outputs, PTX, source/input/output hashes and exact stdout/stderr. Official reference source is not vendored. No dependencies were installed, no compiler behavior changed, no service was purchased, and no GitHub push or formal submission was performed. Future remote tests and publication require their own authorization.
