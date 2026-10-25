# Offline compiler field guide

Open [index.html](index.html) from a downloaded/cloned repository in a browser.
It is one self-contained page: no backend, external scripts, fonts, analytics,
network fetches or persistent storage. The surrounding repository supplies the
optional documentation/source links. GitHub shows HTML source, not a deployed
website. No GitHub Pages or other hosting is configured.

The page includes:

- A six-step explanation of an original ten-element addition and its **real
  generated PTX**, with complete serialized IR and metadata available.
- Selectable thread ownership for two eight-thread blocks, including six idle
  tail threads. This is a CPU teaching illustration, not a GPU measurement.
- A range-reduction table with explicit f32 rounding, labeled as a CPU
  illustration; JavaScript `Math.pow` does not emulate NVIDIA exp2 approximation.
- All **4097 saved GPU outputs before and after** the exp fix, plotted as relative
  error or normalized tolerance error. Both curves use the same expected array
  and unchanged synthetic-test thresholds. No official reference arrays are
  included.

[provenance.json](provenance.json) binds the page to compiler sources, saved
receipts, raw outputs and the synthetic oracle using SHA-256. The builder checks
those receipt hashes, reconstructs the deterministic input grid, verifies the
oracle and recomputes the 251/0 failure counts. It does not execute an upstream
script or contact a GPU service.

## Edit and regenerate

Edit `template.html` for layout/text or `tools/build_explainer.py` for derived
educational data. From the repository root, using the normal pinned environment:

```sh
.venv/bin/python tools/build_explainer.py
.venv/bin/python tools/build_explainer.py --check
.venv/bin/python tools/check_docs.py
.venv/bin/python tools/run_tests.py
```

The generated sample is constructed directly from typed Emmy nodes, without
copying an official benchmark. Check it independently:

```sh
.venv/bin/python compile.py docs/explainer/sample/learn_add.ir.json > /tmp/learn_add.ptx
cmp /tmp/learn_add.ptx docs/explainer/sample/learn_add.ptx
```

Three local regression tests check the sample's exact CPU-model result and tail
ownership, deterministic generation/evidence provenance, and local links/offline
resource policy. They are part of the existing CPU CI suite. No browser package
is needed to run them.

## Browser checks

[browser-check.js](browser-check.js) is an optional Playwright CLI `run-code`
callback, not a page dependency. Use an already installed Playwright CLI and an
isolated session. Serve only this directory temporarily for testing:

```sh
python3 -m http.server 8769 --bind 127.0.0.1 --directory docs/explainer
playwright-cli -s=emmy-learning open http://127.0.0.1:8769/index.html --browser chrome
```

Pass the callback file contents as the CLI's `run-code` argument without shell
interpolation, for example with Python's `subprocess.run` argument list. Close
only that test session with `playwright-cli -s=emmy-learning close` and stop the
server you started. Do not close other browser sessions.

The checks cover all six trace steps, keyboard buttons/slider boundaries, active
and idle thread selection, both chart measures, the embedded GPU failure counts,
1440/768/360/320-pixel layouts, reduced motion, and operation after the browser's
network is disabled. Network events must contain only the local page load, with
no subresource requests or JavaScript errors. The test harness blocks direct
`file:` navigation, so browser automation uses loopback HTTP; the page itself
has no server requirement. Offline interaction is verified after initial load,
not claimed as an automated direct-file launch.

## Accessibility and evidence boundaries

Native controls provide keyboard operation and visible focus. Dynamic selections
have polite live text, charts have descriptions and an exact selected-value table,
and line patterns/text supplement color. The layout reflows to narrow screens
and respects reduced-motion preferences. These checks are useful evidence, not
a full assistive-technology certification.

Screenshots and exact completed check results are recorded in
[visual validation](../VISUAL_VALIDATION.md). The [learning guide](../LEARNING_GUIDE.md)
explains pointwise, reduction, matmul and attention mechanisms; the
[video outline](../WALKTHROUGH.md) offers a four-minute route through them.
