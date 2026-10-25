# Official runner validation: p00 and p01

Completed October 24, 2026, 21:31 America/Los_Angeles
(October 25, 04:31 UTC). Compiler commit tested:
`8f915c532c0daa5f590ee36afe068ba2bd4fd69a`.
No compiler fixes were needed. Test setup and results are a later local-only commit.

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

## Source and environment

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

## Commands executed

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

## Boundaries

Only generated PTX and official launch/input descriptions for the two approved
cases were sent to the documented HTTP service. No Python source, repository,
personal information or credentials were sent. No paid service, signup, legal
agreement, broader test suite, OKC resource or Modal job was used.

Two passing problems do not satisfy the eight-problem official MVP. p02/p03,
all reductions, later categories and hidden variants remain unverified. Next work,
if requested, is broader pointwise support. No new push is included in this task.
