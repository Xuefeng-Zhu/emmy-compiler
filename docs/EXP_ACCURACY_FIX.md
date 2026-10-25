# Shared exp accuracy fix — October 25, 2026 (UTC)

The shared natural-exp lowering now range-reduces its input before evaluating exp2. On the exact saved 4097-point `[-80,80]` diagnostic, GPU mismatches fell from **251 to 0** at the unchanged `rtol=2e-6, atol=2e-7, equal_nan=False`. Maximum relative error over the nonzero outputs fell from **3.6186773989542874e-6 to 1.5253053108309107e-7**. Input, metadata and expected-array hashes are identical before and after.

The complete local suite passed **222 tests in 149.783 seconds**, exit 0; the focused preflight passed 25 tests in 0.842 seconds. All **14 approved GPU regressions passed** assembly, execution and the original expected-array comparison, with client/wrapper exit 0. There were exactly 14 compile and 14 run requests, no retries and no uncertain requests.

[Summary and per-case hashes](validation/2026-10-25-exp-fix/summary.json), [full local log](validation/2026-10-25-exp-fix/all-local-tests.txt), [all current public artifact hashes](validation/2026-10-25-exp-fix/current-public-artifacts.json), [original failure](validation/2026-10-25-regression/synthetic/s15_exp_wide_4097/report.json). The original failure archive is unchanged.

## Diagnosis and bounded change

The old sequence evaluated `ex2.approx.f32(round_f32(x * log2(e)))`. At the saved failing input `x=44.609375`, its rounded exponent is `64.35771942138672`, while a float64 calculation gives approximately `64.3577240896561`. The CPU model, which uses mathematical exp2 rather than the GPU approximation, already fails the original grid at 257 points. This demonstrates a range-scaling rounding contribution; it does not attribute every observed hardware error to a single instruction.

Only the common exp branch in `emmy_ptx/ptx.py` changes production behavior. The exp_fast alias and dependent activations inherit the same implementation. No public problem names, expected values, individual counterexample values or shapes select behavior.

1. Clamp with NaN-propagating min/max to `[-104,89]`. Outside this safe interval the f32 exponential rounds to zero or overflows; finite bounding also avoids `inf-inf` in the reduction.
2. Use `cvt.rni.f32.f32` to obtain a nearest integral-valued exponent n from x*log2(e).
3. Compute `r = (x - n*ln2_hi) - n*ln2_lo`, with separately rounded instructions. `ln2_hi=0f3f317200=22713/32768`, and `ln2_lo=0f35bfbe8e`. With n in `[-150,128]`, n*hi is exactly representable. The remaining residual is small.
4. Evaluate exp2(r*log2(e)). Split n with `cvt.rzi.f32.f32` into two half exponents in `[-75,64]`; evaluate their normal-range exp2 scales. Multiply as `(partial * first) * second` so the scale product cannot overflow or underflow prematurely.

Each arithmetic instruction retains explicit rounding; no user multiplication/addition is fused. The two new CPU-model conversion branches preserve float32 and signed zero, and independent hand-written PTX probes exercise ties-to-even, truncation and exceptional classifications. The prior failure grid is now a permanent emitted-PTX regression in `tests/test_exp_emitted.py`.

There remain **three approximate exp2 evaluations**; integer-input scales are not claimed exact. NVIDIA documents conversion semantics and an exp2 primary-range fractional error bound, not a correct-rounding guarantee for this composed natural exponential. The design is justified by range reduction and verified samples, not a universal f32 proof. Primary references: [PTX 8 cvt](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt), [PTX 8 ex2](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#floating-point-instructions-ex2).

## GPU coverage

Observed device: NVIDIA GeForce RTX 5090, sm_120. Every launch used zero warmups and one timed iteration. Timings are indicative; this accuracy change adds instructions and does not establish a performance improvement.

| Case | Result | Max absolute error | Single iteration (us) |
| --- | --- | ---: | ---: |
| [s15_exp_wide_4097](validation/2026-10-25-exp-fix/synthetic/s15_exp_wide_4097/report.json) | PASS | 4.95176016e+27 | 15.104 |
| [s03_exp_193](validation/2026-10-25-exp-fix/synthetic/s03_exp_193/report.json) | PASS | 0.000122070312 | 22.176 |
| [s04_tanh_193](validation/2026-10-25-exp-fix/synthetic/s04_tanh_193/report.json) | PASS | 1.1920929e-07 | 20.544 |
| [s05_sigmoid_193](validation/2026-10-25-exp-fix/synthetic/s05_sigmoid_193/report.json) | PASS | 5.96046448e-08 | 20.416 |
| [s06_silu_193](validation/2026-10-25-exp-fix/synthetic/s06_silu_193/report.json) | PASS | 2.98023224e-08 | 21.088 |
| [s07_erf_193](validation/2026-10-25-exp-fix/synthetic/s07_erf_193/report.json) | PASS | 2.38418579e-07 | 20.544 |
| [s08_gelu_193](validation/2026-10-25-exp-fix/synthetic/s08_gelu_193/report.json) | PASS | 4.76837158e-07 | 20.480 |
| [s09_gelu_tanh_193](validation/2026-10-25-exp-fix/synthetic/s09_gelu_tanh_193/report.json) | PASS | 4.76837158e-07 | 20.448 |
| [s13_guarded_attention_5x11x7x6](validation/2026-10-25-exp-fix/synthetic/s13_guarded_attention_5x11x7x6/report.json) | PASS | 1.78813934e-07 | 20.384 |
| [s14_generated_greedy_attention_5x9x7x11](validation/2026-10-25-exp-fix/synthetic/s14_generated_greedy_attention_5x9x7x11/report.json) | PASS | 1.78813934e-07 | 32.928 |
| [f02_row_softmax](validation/2026-10-25-exp-fix/official/f02_row_softmax/report.json) | PASS | 7.4505806e-09 | 14.976 |
| [a01_attention](validation/2026-10-25-exp-fix/official/a01_attention/report.json) | PASS | 6.55651093e-07 | 142.496 |
| [a02_attention_128](validation/2026-10-25-exp-fix/official/a02_attention_128/report.json) | PASS | 5.06639481e-07 | 53.984 |
| [s16_exp_boundary_finite](validation/2026-10-25-exp-fix/synthetic/s16_exp_boundary_finite/report.json) | PASS | 5.0706024e+30 | 20.992 |

Only f02, a01 and a02 changed among the 15 public PTX artifacts, and all three have fresh GPU PASS receipts. The other 12 are byte-identical to current previously passed artifacts. This does not mean all 15 were rerun.

The synthetic cases include seven activation variants, two guarded attention schedules, the unchanged 4097-point failing grid, and a new 1019-point finite boundary fixture. The latter samples signed zeros, tiny inputs, normal/subnormal/zero output transitions, neighboring reduction-switch inputs and upper finite inputs through 87. Its zero classification matched the oracle, all returned values were finite, and its largest observed f32 ULP distance was 2. Its maximum normal-output relative error was 1.6858741113968294e-7. Absolute tolerance alone would hide tiny-output errors, so zero classification was checked separately.

Public tolerances remain their original metadata values. Synthetic activation and finite-boundary thresholds retain the existing local test values; they are not organizer-defined standalone activation tolerances. Local tests additionally exercise NaN, infinities and overflow classification. These exceptional inputs were **not** sent through the runner; NaN semantics and the official checker's rejection of NaN/NaN remain separate issues. No actual hidden tests were available or claimed passed.

## Exact source and artifact binding

Baseline before the fix: `ee40e27a4069d5eed5901690fafd24a20de0221a`. Each GPU report stores the exact nine production-source file hashes used during its run. The final canonical source-map SHA256 is:

ea2b5bb7db37b6a77945cb2af206459c526794375a180ed30255c26089114a7f

Canonicalization is JSON with sorted keys and separators `,` and `:`, over the relative-path-to-SHA256 map in [final-compiler-source.json](validation/2026-10-25-exp-fix/final-compiler-source.json). The original GPU-time source map is retained separately. One explanatory comment was corrected after the runs from 22711/32768 to 22713/32768; instruction constants and behavior did not change. All 14 tested PTX artifacts were regenerated from final source and verified byte-identical.

Wide diagnostic PTX SHA256 before:

88d38e7e42a18c623aa30e4e59d1f21d254b645b28799fbc9039778665934ed1

Wide diagnostic PTX SHA256 after:

4a9225e2ee48952928ed720355cdca9836b35ba1216890dbac8eb89a32e03f16

Every saved input manifest, expected array, generated PTX and returned output has its own hash. [evidence-manifest.json](validation/2026-10-25-exp-fix/evidence-manifest.json) covers the retained archive. Compiler source was never uploaded.

## Reproduction

Actual focused and full local test commands, run from the project root:

```sh
PYTHONPATH=.:tests .venv312/bin/python -B -m unittest test_float_rounding_model test_pointwise_extended test_extra_generalization -v
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v
.venv312/bin/python -B tools/make_exp_boundary_fixture.py --output build/exp-fix/reproduced-boundary
```

The boundary generator requires a new output directory and makes no network requests. Its three core fixture files reproduced the GPU-tested inputs byte-for-byte. The earlier synthetic fixture generator remains available for the unchanged activation and guarded-attention cases.

The wide diagnostic was run first with:

```sh
.venv-runner/bin/python -B tools/run_synthetic.py build/validation-0e16a83/reproduced-fixtures/s15_exp_wide_4097 --project /Users/frank/mygit/emmy-compiler --manifest build/validation-0e16a83/reproduced-fixtures/s15_exp_wide_4097/manifest.json --attempt build/exp-fix/synthetic/s15_exp_wide_4097
```

The other 13 exact commands and exit codes are in [batch.json](validation/2026-10-25-exp-fix/batch.json). They use the pinned official client through the same reviewed wrapper and destination. Official inputs were also checked against the immutable upstream manifest before execution. Only generated PTX and public/synthetic input/launch descriptions went to `http://runner.riftstack.ai:48701`; no credentials, source, payment, Modal resource or formal submission was involved. No public push is part of this task.
