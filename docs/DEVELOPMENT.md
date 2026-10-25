# Optional development environment notes

The README uses a normal Python 3.12 virtual environment. These notes preserve
the workaround used on the original macOS development machine, whose Homebrew
Python could not import `sqlite3`.

If your Python has the same problem, a project-local interpreter installed by
`uv` provides an alternative without changing global Python:

```sh
UV_CACHE_DIR=/tmp/emmy-uv-cache UV_PYTHON_INSTALL_DIR="$PWD/.python" uv python install 3.12.12 --no-bin
UV_CACHE_DIR=/tmp/emmy-uv-cache UV_PYTHON_INSTALL_DIR="$PWD/.python" uv venv --python 3.12.12 .venv312
UV_CACHE_DIR=/tmp/emmy-uv-cache uv pip install --python .venv312/bin/python -r requirements.lock
.venv312/bin/python tools/fetch_reference.py
.venv312/bin/python -m unittest discover -s tests -v
```

`requirements.lock` records the complete environment resolved on macOS arm64 with
CPython 3.12.12. Other platforms need compatible package wheels. The compiler
itself uses Emmy for typed IR loading and emits PTX without invoking a CUDA
compiler.

## Evidence

Local tests execute parser/ABI checks and deterministic CPU instruction models.
They require the hash-verified `.reference/` fixtures but make no GPU requests.
Official assembly and GPU receipts are preserved in `docs/validation/` and
summarized in [RUNNER_RESULTS.md](RUNNER_RESULTS.md).

The optional `.venv-runner` environment isolates the official submit client's
transport dependencies. `tools/run_official.py` sends generated PTX and public
launch descriptors to the official service. A new runner case or rerun requires
separate authorization; ordinary local tests do not contact it.
