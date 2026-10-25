# Official runner validation

## Final public coverage — October 25, 2026 (UTC)

**All 15 public cases passed PTX assembly, GPU execution and the unchanged official
expected-array comparison.** Every successful official-client and wrapper exit
code is 0. Original tolerances and `equal_nan=False` remain unchanged.

The observed device was NVIDIA GeForce RTX 5090, compute capability 12.0; the
server assembled for sm_120 by default. Generated PTX targets sm_89. The table's
time is the server-reported device measurement with warmup=0 and iters=1, not
network latency or a stable performance benchmark. Each final receipt links to
exact stdout/stderr files in the same directory.

| Case / receipt | Final outcome | Registers | Max absolute error | Single iteration, us |
| --- | --- | ---: | ---: | ---: |
| [p00_scalar_add](validation/2026-10-24/p00_scalar_add.json) | PASS | 18 | 0 | 23.584 |
| [p01_vector_add](validation/2026-10-24/p01_vector_add.json) | PASS | 18 | 0 | 20.352 |
| [p02_scale_shift](validation/2026-10-25/p02_scale_shift.json) | PASS | 21 | 0 | 14.976 |
| [p03_relu_mul](validation/2026-10-25/p03_relu_mul.json) | PASS | 22 | 0 | 15.328 |
| [r01_array_sum](validation/2026-10-25/remaining-public/r01_array_sum.json) | PASS | 17 | 3.81469727e-06 | 22.112 |
| [r02_big_sum](validation/2026-10-25/remaining-public/r02_big_sum.json) | PASS | 17 | 1.52587891e-05 | 36.672 |
| [r03_mean](validation/2026-10-25/remaining-public/r03_mean.json) | PASS | 17 | 9.31322575e-10 | 28.064 |
| [r04_row_sum](validation/2026-10-25/remaining-public/r04_row_sum.json) | PASS | 18 | 5.7220459e-06 | 20.864 |
| [f01_rmsnorm](validation/2026-10-25/remaining-public/f01_rmsnorm.json) | PASS | 21 | 1.90734863e-06 | 15.488 |
| [f02_row_softmax](validation/2026-10-25/remaining-public/f02_row_softmax.json) | PASS | 22 | 7.4505806e-09 | 22.272 |
| [f03_l2norm](validation/2026-10-25/remaining-public/f03_l2norm.json) | PASS | 20 | 1.49011612e-08 | 16.576 |
| [a01_attention](validation/2026-10-25/remaining-public/a01_attention.json) | PASS | 140 | 6.55651093e-07 | 108.896 |
| [a02_attention_128](validation/2026-10-25/remaining-public/a02_attention_128.json) | PASS | 80 | 4.76837158e-07 | 49.024 |
| [m01_matmul_512](validation/2026-10-25/remaining-public/m01_matmul_512.json) | PASS | 36 | 8.77380371e-05 | 113.696 |
| [m02_matmul_rect](validation/2026-10-25/remaining-public/m02_matmul_rect.json) | PASS | 31 | 1.14440918e-05 | 15.296 |

[Aggregate result record](validation/2026-10-25/all-public-results.json),
[latest health](validation/2026-10-25/remaining-public/health.json),
[final 169-test local log](validation/2026-10-25/remaining-public/all-public-local-tests.txt).

The new session ran only the remaining eleven public cases, sequentially. It made
12 compile requests and 11 GPU launches; there are no uncertain outcomes. The
previous four pointwise cases were preserved, not rerun. No Python source, private
repository files, credentials or private inputs were uploaded. Requests contained
only generated PTX and pinned public launch/input descriptions over the disclosed
HTTP endpoint `http://runner.riftstack.ai:48701`. No service was purchased.

The unchanged official client is pinned to repository commit
`ec8cde1943f88efea3827a95bda6085553d2d407` and SHA-256
`b3c2fac2e05e33a8d21a24c4ba7d2dfc23c9b71911646437910e0d6fd0e66dbf`.
Our wrapper validates destinations and exact payloads, disables inherited auth/
proxy/cookies and redirects, journals requests before transport, checks complete
output names/shapes/counts and independently verifies the comparison. It requests
enough values for matrix outputs (262144 for m01), within official service limits.
The scalar expected shape `()` is deliberately compared with the descriptor's
one-element buffer after reshape. No expected array or tolerance was changed.

### Exact new commands and the corrected assembly failure

Executed serially from the project root:

```sh
.venv-runner/bin/python -B tools/run_official.py r01_array_sum
.venv-runner/bin/python -B tools/run_official.py r02_big_sum
.venv-runner/bin/python -B tools/run_official.py r03_mean
.venv-runner/bin/python -B tools/run_official.py r04_row_sum
.venv-runner/bin/python -B tools/run_official.py f03_l2norm
.venv-runner/bin/python -B tools/run_official.py f01_rmsnorm
.venv-runner/bin/python -B tools/run_official.py f02_row_softmax
.venv-runner/bin/python -B tools/run_official.py a02_attention_128
.venv-runner/bin/python -B tools/run_official.py a01_attention
.venv-runner/bin/python -B tools/run_official.py m02_matmul_rect
.venv-runner/bin/python -B tools/run_official.py m01_matmul_512
```

The last command's first attempt returned exit 1, stage `ptxas`: the external
shared declaration at line 16 was incorrectly inside the entry body. It made no
GPU launch. The [original receipt](validation/2026-10-25/m01-attempt-1/m01_matmul_512.json),
[stdout](validation/2026-10-25/m01-attempt-1/m01_matmul_512.stdout.txt),
[stderr](validation/2026-10-25/m01-attempt-1/m01_matmul_512.stderr.txt) and
[rejected PTX](validation/2026-10-25/m01-attempt-1/m01_matmul_512.ptx) are preserved.

Commit `e7af7b2` adds module-scope external declarations and a placement regression.
The final local suite passed 169 tests in 145.803 seconds. All other 14 passed
PTX artifacts were unchanged. Only then was the final command repeated once:

```sh
.venv-runner/bin/python -B tools/run_official.py m01_matmul_512
```

It returned exit 0 with 36 registers and PASS over all 262144 output values. The
launch successfully used the prescribed 73728-byte dynamic shared allocation.
The final PTX SHA-256 is
`03478118931c84bed72baaea6d1e056dd37573059f8c19e6c7348f42fcd91722`.
The tested client wrapper and local failure-capture correction are in `7ad6d98`.
No other new case needed a retry, and no failed/queued result was counted as PASS.

These finite public inputs do not validate every hidden shape, exceptional value,
atomic ordering or performance scenario. The historical checkpoints below remain
for provenance; their older coverage counts are superseded by the table above.

## p02 and p03 — October 25, 2026 (UTC)

Both newly authorized cases passed on the first attempt. Compiler commit tested:
`519fc60fb6eac9ba421a8c726ba27f6f17964b82`. No compiler fixes were required.
The existing reviewed official client and isolated runner environment were reused;
only our transport wrapper was extended for these case/output names.

| Case | Local generation | Server ptxas | GPU execution | Local expected.npz comparison | Client / wrapper exit |
| --- | --- | --- | --- | --- | --- |
| p02_scale_shift | exit 0 | accepted, 21 registers | success, 4096 values | PASS, max abs error 0; rtol/atol 1e-6 | 0 / 0 |
| p03_relu_mul | exit 0 | accepted, 22 registers | success, 2048 values | PASS, max abs error 0; rtol/atol 0 | 0 / 0 |

Health at 05:09:35 UTC reported an NVIDIA GeForce RTX 5090 (sm_120), ready for
assembly and execution, with default_arch=sm_120. The wrapper passed no arch
override. p02 ran at 05:12:15–16 UTC, followed by p03 at 05:12:28–29 UTC. Each made
one /compile and one /run request, all HTTP 200 with ok=true. No retries, pending
requests or uncertain outcomes remain.

Commands executed, serially from the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-runner/bin/python -B tools/run_official.py p02_scale_shift
PYTHONDONTWRITEBYTECODE=1 .venv-runner/bin/python -B tools/run_official.py p03_relu_mul
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -B -m unittest discover -s tests -v > build/runner-pointwise-regression.txt 2>&1
```

The preflight health request used GET to the documented /health endpoint, timeout
(10,30), redirects disabled and Session.trust_env=False. The official client and
all ten p02/p03 fixture files matched the pinned hashes before execution. No new
dependencies or credentials were installed.

Exact official stdout:

```text
[k_add_pointwise] 21 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (14.976 us)
[k_mul_pointwise] 22 registers
[k_mul_pointwise] PASS  mul: max abs err 0.000e+00  (15.328 us)
```

Both stderr files are empty. The reported 14.976/15.328 microseconds are device
timings with warmup=0 and iters=1, not stable benchmarks or network latency. The
unchanged official client compares outputs locally using np.allclose; the wrapper
also checks exact metadata tolerances with explicit equal_nan=False. It requires
the proper PASS output name, official exit 0 and successful local comparison.
The approved public payloads followed the pinned official metadata; no compiler source or
private repository data was sent to the runner.

[Health](validation/2026-10-25/health-p02-p03.json),
[p02 receipt](validation/2026-10-25/p02_scale_shift.json),
[p03 receipt](validation/2026-10-25/p03_relu_mul.json), and exact stdout/stderr
are preserved under validation/2026-10-25/. The final local regression passed
67 tests in 6.005 seconds, exit 0; [full log](validation/2026-10-25/runner-regression.txt).
Three of those tests exercise seven mocked transport scenarios, including output
name/payload validation and rejection of malformed or NaN results; those mock
executions neither contacted the service nor overwrote real receipts.

The four earlier commits through 519fc60 were pushed to the existing private
repository and its main SHA/privacy verified before these runs. This new wrapper
and result commit remains local; no later push is included. At that checkpoint public pointwise
coverage was four of four, while the eight-problem MVP still needed reductions.
All GPU evidence covers finite public fixtures only. The NaN contract conflict in
NUMERICS.md remains unresolved; no exceptional-input GPU test ran.

## Earlier p00 and p01 validation

Completed October 24, 2026, 21:31 America/Los_Angeles
(October 25, 04:31 UTC). Compiler commit tested:
`8f915c532c0daa5f590ee36afe068ba2bd4fd69a`.
No compiler fixes were needed. Test setup and results were recorded in bef05e5,
initially local and subsequently included in the authorized push through 519fc60.

| Case | Local PTX generation | Server ptxas | GPU execution | Local comparison | Client / wrapper exit |
| --- | --- | --- | --- | --- | --- |
| p00_scalar_add | success | accepted, 18 registers | success, 128 output values | PASS, max abs error 0 | 0 / 0 |
| p01_vector_add | success | accepted, 18 registers | success, 1024 output values | PASS, max abs error 0 | 0 / 0 |

Health returned `NVIDIA GeForce RTX 5090 (compute capability 12.0, sm_120)` with
`can_compile=true`, `can_run=true`, `default_arch=sm_120`. Each case made one
`/compile` and one `/run` request, serially. No retries or other cases ran.
No request has an uncertain or pending outcome.

The runner reported `median_us=23.584` for p00 and `20.352` for p01, each with
`warmup=0`, `iters=1`. These are indicative single-launch timings, not a controlled
benchmark or end-to-end network latency. Both metadata files specify
`rtol=0`, `atol=0`; all expected `add` values were returned and compared locally.

### Source and environment

Official repository pinned to `ec8cde1943f88efea3827a95bda6085553d2d407`:
[client](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/submit.py),
[API](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/docs/API.md).
The unchanged reviewed client has SHA-256
`b3c2fac2e05e33a8d21a24c4ba7d2dfc23c9b71911646437910e0d6fd0e66dbf`.
Fixture and client hashes are in `reference-manifest.json`; downloads remain
untracked in `.reference/`.

A separate `.venv-runner` contains 40 pinned packages, recorded in
`requirements-runner.lock`. The existing compiler `.venv312` was preserved.
The initial installation commands were:

```sh
UV_CACHE_DIR=/tmp/emmy-uv-cache /Users/frank/.local/bin/uv venv --python .venv312/bin/python .venv-runner
UV_CACHE_DIR=/tmp/emmy-uv-cache /Users/frank/.local/bin/uv pip install --python .venv-runner/bin/python -r requirements.lock 'requests==2.34.2'
```

For reproduction, install the resulting complete `requirements-runner.lock`.
`tools/fetch_reference.py` now fetches/verifies the three additional pinned files:
`submit.py` and the two `expected.npz` files. No global package installation or
credentials were created.

### Commands executed

Working directory: the emmy-compiler project root. First, a read-only health
request used `requests.Session(trust_env=False)` configuration (set as an attribute)
and `GET http://runner.riftstack.ai:48701/health`, timeout `(10, 30)`, redirects
disabled. Its sanitized response is preserved in `validation/2026-10-24/health.json`.

Then, in order:

```sh
.venv-runner/bin/python -B tools/run_official.py p00_scalar_add
.venv-runner/bin/python -B tools/run_official.py p01_vector_add
.venv312/bin/python -B -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/emmy-uv-cache /Users/frank/.local/bin/uv pip check --python .venv-runner/bin/python
```

The wrapper invokes the unchanged official client with equivalent arguments:

```sh
.venv-runner/bin/python .reference/submit.py p00_scalar_add --compiler compile.py --server http://runner.riftstack.ai:48701
.venv-runner/bin/python .reference/submit.py p01_vector_add --compiler compile.py --server http://runner.riftstack.ai:48701
```

Only the wrapper commands were executed; the equivalent direct commands were not
run separately. The wrapper additionally performs local PTX preflight, restricts
request destinations/payloads, disables environment credentials/proxies and
redirects, clears cookies, caps busy retries and validates output completeness.
It requires both actual `PASS  add:` and client exit 0. It preserves the official
numerical comparison logic. Future runs still require authorization.

Exact official stdout, p00:

```text
[k_add_pointwise] 18 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (23.584 us)
```

Exact official stdout, p01:

```text
[k_add_pointwise] 18 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (20.352 us)
```

Both stderr files are empty. Sanitized receipts and exact stdout/stderr are
committed in `validation/2026-10-24/`; original working copies and generated PTX
remain under ignored `build/runner/`. Receipts record HTTP 200 and `ok=true` for
assembly and execution, hashes, output counts and service timings. The receipt
`exit_code` field is the official client's code; both wrapper commands also exited 0.

The local suite again passed all 39 tests (0.611 seconds). The optional runner
venv's 40 dependencies passed `uv pip check`. No further broad tests were run.

### Boundaries of the earlier p00/p01 session

Only generated PTX and official launch/input descriptions for the two approved
cases were sent to the documented HTTP service. No Python source, repository,
personal information or credentials were sent. No paid service, signup, legal
agreement, broader test suite, OKC resource or Modal job was used.

That first checkpoint covered two problems. The later p02/p03 results above bring
public pointwise GPU coverage to four, still short of the eight-problem MVP.
Reductions, later categories, hidden variants and exceptional-value GPU behavior
remain unverified. Future remote tests and pushes require separate authorization.
