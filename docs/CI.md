# CPU CI and clean-environment reproduction

The `CPU validation` workflow runs on pushes to `main` and pull requests using
standard `ubuntu-24.04` GitHub-hosted runners and CPython 3.12.12. It has only
`contents: read` permission, does not persist checkout credentials, and pins the
two official GitHub Actions to commit SHAs. It uses no secrets, deployment,
package cache, artifact upload, GPU service or optional runner environment.

## Reproduce the checks

From a fresh checkout, with a working Python 3.12.12 interpreter:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple -r requirements.lock
.venv/bin/python -m pip --isolated check
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/fetch_reference.py --verify-only
.venv/bin/python tools/run_tests.py
.venv/bin/python tools/check_cli.py
```

Installation uses only PyPI wheels at the versions already listed in
`requirements.lock`; no source-distribution build scripts run. The lock fixes
versions, not distribution hashes or the operating system image, so this is a
repeatable version-pinned setup, not a fully hermetic build. A missing compatible
wheel or unavailable registry makes installation fail rather than silently
selecting another version.

Reference acquisition is a separate network step. All 82 files come from the
fixed official commit in `reference-manifest.json`, and each file must match its
SHA-256 before being saved. `--verify-only` checks the cache without network or
writes and fails for missing or corrupt bytes. Upstream fixtures and expected
arrays remain in ignored `.reference/`; CI does not publish them as artifacts.
Upstream scripts are never executed by the fetcher. The existing transport unit
tests execute the previously reviewed, hash-pinned official client with mocked
network calls.

The test runner rejects an empty suite, skips, failures and errors. Its negative
regressions deliberately launch tiny failing/error/empty/skipped suites in child
processes and require nonzero exit codes. Workflow steps use Bash's error and
pipeline failure handling, without `continue-on-error` or success-masking pipes.
The CLI check compiles every public manifest-listed IR, checks clean ASCII PTX
stdout and empty stderr, and prints the generated PTX SHA-256. It also requires
invalid invocation to return nonzero with empty stdout and a stderr diagnostic.

These are parser, numerical CPU-model, transport-mock and PTX-text checks. They
do not assemble PTX, execute GPU kernels, rerun the competition's hidden tests,
or replace the separate GPU evidence. Compiler algorithms and numerical
tolerances are unchanged by this CI work.

## Verification record

The initial local clean-room check starts from source commit
`56b8797b8b61fe511372325b7b306e0b1d1ac149` plus the CI-only changes. The checkout
contains only tracked files; the virtual environment and reference cache are
created from scratch. **240 tests passed in 148.376 seconds (exit 0), with no skips**, on macOS arm64.
All 37 pinned dependencies installed from wheels, and `pip check` passed. All
82 reference files were fetched into an empty cache and verified. All 15 CLI
outputs and two invalid-input checks passed; their PTX hashes match the previous
GPU-validated artifacts byte for byte. No GPU requests were made.
[Exact local logs and hashes](validation/2026-10-25-ci/) are retained with this change.
The corresponding GitHub Actions run is authoritative for Linux validation;
a local macOS pass alone is not a remote CI pass.

## Maintenance

Update dependency versions or action pins deliberately and rerun the same
checks. This workflow grants no repository write access and does not configure
branch protection or account billing. It uses the standard public-repository
runner tier, documented as free by GitHub; no paid runner is selected.

Primary documentation: [Python setup](https://docs.github.com/actions/guides/building-and-testing-python),
[workflow syntax and permissions](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions),
[standard runner billing](https://docs.github.com/en/actions/concepts/billing-and-usage).
