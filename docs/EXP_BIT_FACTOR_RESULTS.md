# Exp bit-factor experiment: no demonstrated speed improvement

This branch publishes a completed negative experiment. **The compiler remains
the public main baseline `c5f9aa9`; the candidate is an unapplied patch.** It
preserved precision in the measured scope, but the repeated timings did not
support adopting it. The earlier simple-exp implementation is not restored.
The video still describes its frozen `c5f9aa9` version and was not changed.

## Candidate and correctness

The candidate keeps the clamp, ln2 high/low range reduction, exponent split and
`(partial * first) * second` order. It replaces only the two integer-input
`ex2.approx.f32` scale evaluations with signed conversion, exponent-bit
construction and a bitcast. Clamping bounds `n` to `[-150,128]`; both split
exponents stay in `[-75,64]`, safely inside the normal-power encoding range
`[-126,127]`. No `.ftz`, tolerance, global precision or FMA policy changes.
Scalar PTX count increases **17 → 23**, while exp2 count falls **3 → 1**;
these counts do not predict machine-code latency.

All **20 completed candidate GPU checks passed** their unchanged comparisons
through the pinned official Emmy client on an RTX 5090. They include public
p00/p01, f02/a01/a02, activations, attention variants and synthetic diagnostics.
The 18 paired current-baseline checks passed and all 18 current/candidate output
hashes were identical. The old simple-exp variant passed 16/18; the wide and
finite-boundary diagnostics are its retained negative controls. The local
candidate suite passed **250 tests**, plus 15 public CLI emissions and two
invalid-input checks. These are historical candidate results, not additional
GPU runs or the test count of this documentation-only branch.

| 4097-point `[-80,80]` diagnostic | Old simple exp | Current baseline | Candidate |
| --- | ---: | ---: | ---: |
| Mismatches at unchanged `rtol=2e-6, atol=2e-7` | 251 | 0 | 0 |
| Maximum relative error | 3.6186774e-6 | 1.5253053e-7 | 1.5253053e-7 |
| Maximum sampled ULP distance | 57 | 2 | 2 |

The unchanged 1019-point finite-boundary fixture has matching zero
classification and maximum 2 ULP for both current and candidate. Maximum
relative error over normal outputs is 1.6858741e-7. The old variant fails 46
values and differs in zero classification. This preserves improved precision
on these samples; it is not a universal precision guarantee. NaN, infinities,
signed zero, extrema and overflow transitions were checked locally; exceptional
GPU inputs were not tested. Official tolerances and `equal_nan=False` remain
unchanged. [Sanitized correctness and source fingerprints](validation/2026-10-25-exp-bit-factor-results/correctness-summary.json)
record the completed scope without input/output arrays or runner access handles.

## Repeated timings

Each workload uses all six permutations of old/current/candidate, in this order:
`O/C/N`, `O/N/C`, `C/O/N`, `C/N/O`, `N/O/C`, `N/C/O`. Each variant occupies each
position twice. Every request uses identical data and geometry, 5 warmups and
31 timed iterations. There are **90 completed timing requests**: six returned
medians per variant/workload, 2790 timed iterations and 450 warmups overall.

| Case | Old µs | Current µs | Candidate µs | Current / candidate range µs |
| --- | ---: | ---: | ---: | --- |
| f02_row_softmax | 3.024 | 4.736 | 4.752 | 4.704–4.800 / 4.736–4.768 |
| a01_attention | 98.992 | 131.792 | 131.712 | 131.744–131.808 / 131.680–131.744 |
| a02_attention_128 | 39.584 | 43.680 | 43.712 | 43.648–43.712 / 43.712–44.352 |
| Synthetic attention 256×256×32×32 | 76.448 | 84.656 | 84.656 | 84.608–84.672 / 84.608–84.672 |
| Synthetic softmax 7×1025 | 4.784 | 6.832 | 6.832 | 6.784–6.848 / 6.784–6.848 |

Entries are medians of six server-reported medians. No useful speed improvement
was demonstrated. The runner does not lock clocks, and individual iteration
samples were not returned. The six aggregate medians are the observations;
the 186 timed iterations per variant/workload are not treated as independent
samples. No confidence interval was captured or inferred. [Sanitized timing statistics](validation/2026-10-25-exp-bit-factor-results/timing-statistics.json)
retain all six medians, order, range, sample SD, IQR and MAD.

The attention stress is newly synthetic and is not established identical to
the fixture behind an earlier single-run `s_a_attn_256` timing. Its name does
not make those earlier timings comparable. The softmax stress widens scores
and includes a non-divisible tail.

## One unresolved request

Historical requests produced 57 successful compile responses and 146 successful
run responses, including the 90 timing requests. One old simple-exp
`s_exp_262145` correctness `POST /run` ended with a client connection error and
no HTTP response. Its compile succeeded, but admission, execution, completion
and whether a launch remains active are unknown. It was **not retried**;
current/candidate large-exp runs and that workload's timings were not sent.

The pinned and rechecked current [official API documentation](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/docs/API.md)
and client provide no read-only per-request status/result lookup. Health,
subsequent independent requests and the documented launch time limit cannot
resolve that request. No other users' jobs or speculative endpoints were
queried. [Sanitized status record](validation/2026-10-25-exp-bit-factor-results/request-status.json)
preserves this limitation without launch-capable identifiers. Publication
involved no additional GPU runs, paid services, Modal, H200 resources, official
submission or video changes.

## CPU reproduction and publication scope

The [candidate source/tests patch](validation/2026-10-25-exp-bit-factor-results/candidate-source-tests.patch)
contains the exact experimental compiler/model/test changes; it is **not
applied to this branch**. It uses zero context, so the explicit
`--unidiff-zero` option and the stated baseline are required. Apply it only in a separate disposable checkout based
on this branch. With the pinned environment prepared as in the
[setup instructions](../README.md#setup):

```sh
git apply --check --unidiff-zero docs/validation/2026-10-25-exp-bit-factor-results/candidate-source-tests.patch
git apply --unidiff-zero docs/validation/2026-10-25-exp-bit-factor-results/candidate-source-tests.patch
python tools/fetch_reference.py
python tools/build_explainer.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B tools/run_tests.py
python -B tools/check_cli.py
```

The explainer command refreshes source-hash provenance after applying the patch;
the historical GPU curves remain historical. [The CPU-only stress generator](validation/2026-10-25-exp-bit-factor-results/make_exp_stress_fixtures.py)
uses existing public typed-schedule helpers, explicit deterministic synthetic
inputs and independent float64 oracles rounded to f32. Run it from that
disposable checkout with `--output build/reproduced-exp-stress` and BLAS/OMP
thread limits. It performs no network or GPU work. Generating the large-exp
fixture does not resolve or authorize retrying the historical uncertain run.

This publication adds only this report and its small supporting files under
`docs/`. It includes no official arrays, generated PTX, raw runner payloads or
responses, compiled-binary handles, credentials or local absolute paths. Source
fingerprints bind the historical implementation without granting runner access.
The full reviewed experiment remains a local archive; it was not merged into
main. [Publication checks and file hashes](validation/2026-10-25-exp-bit-factor-results/publication-checks.json)
describe the verification of this results-only record.
