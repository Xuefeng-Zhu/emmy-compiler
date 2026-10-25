# Submission readiness and participant explanation

This checklist records work and remaining entry materials. Official sources were
rechecked on October 25, 2026 (UTC): [challenge website](https://riftstack.ai/hackathon),
[task repository](https://github.com/cloudrift-ai/emmy-hackathon-2026), and the
pinned TASK.md/MVP.md listed in CHALLENGE.md.

## Engineering readiness

- Root `compile.py` accepts the required IR path and adjacent `meta.json`.
- Setup uses isolated Python 3.12 and pinned packages; public fixtures have a
  fixed commit and hashes. No upstream generation scripts are executed.
- Original code performs direct IR-to-PTX lowering; unsupported contracts fail.
- Local parser, ABI, address, numeric-model and malformed-IR tests are available
  with `python -m unittest discover -s tests -v`.
- All 15 current public artifacts have matching GPU evidence: the latest session
  reran r01–r04 successfully and verified the other eleven PTX hashes unchanged
  from earlier passes. Original official tolerances remain intact.
- Latest local validation: 210 tests passed in 138.357 seconds, exit 0. This
  includes eight new CPU checks and eight mocked-client tests.
- Fourteen normal synthetic GPU cases passed. A separate wide-range exp
  diagnostic failed at 251 of 4097 values under its unchanged local threshold;
  the shared range-reduced-exp proposal is not implemented.
- [Generalization validation](GENERALIZATION_VALIDATION.md) and the
  [latest evidence archive](validation/2026-10-25-regression/) record this bounded
  test checkpoint. Compiler source remained at
  `0e16a83235a8b2180a9c40dc4ede06b8aa33d511`; no production change or push was made.
- Hidden tests, universal exceptional-input behavior and sustained performance
  remain unverified. The exp diagnostic must not be presented as fixed.

## User-owned prerequisites

1. Original code uses [Apache-2.0](../LICENSE). Review the separate competition
   IP terms for entry; challenge reference materials remain outside Git.
2. The [source repository](https://github.com/Xuefeng-Zhu/emmy-compiler) is public.
   Review and push any subsequent local changes before submitting its URL.
3. Record a 3–5 minute explanation in the participant's own voice, demonstrating
   understanding. Do not use generated voice to impersonate the participant.
4. Confirm participant/team details, accessible repository/video links, eligibility
   and any submission terms, then explicitly authorize final submission.

The recorded official deadline is October 27, 2026 at 23:59 AoE. The website permits
several public Git hosts; the repository README requests GitHub. The minimum
engineering goal is all eight problems in the first two categories; all eight
have GPU PASS receipts.

## Suggested 4-minute explanation outline

- **0:00–0:35 — Problem and interface.** Show one Kernel IR and adjacent metadata.
  Explain that metadata controls argument order and launch geometry.
- **0:35–1:15 — Own lowering.** Trace typed Load/Assign/Write into registers,
  addresses and PTX. Show a changed input size/name to illustrate generality.
- **1:15–2:05 — Reductions and synchronization.** Draw a row split across lanes,
  neutral values for tails, warp shuffle, shared partials and unconditional
  barriers. Explain why only one lane writes each shared slot.
- **2:05–2:50 — Stateful and shared operations.** Explain online-softmax state
  rescaling and accumulator lifetime, then the async-copy ring and split-K
  zero-before-atomic contract. Choose one detail you understand well.
- **2:50–3:35 — Evidence.** Run local tests; distinguish a CPU instruction model
  from PTX assembly/GPU output comparison. Show actual retained GPU receipts.
- **3:35–4:00 — What the evidence covers.** Distinguish current public-artifact
  GPU evidence, the 14 passing synthetic GPU cases and CPU-model checks. Show
  the wide-range exp counterexample and explain that the proposed fix is pending.

The [Chinese walkthrough](WALKTHROUGH.md) expands this outline around the common
emission layer and schedule-specific extensions. A recording still needs to be
made in the participant's own voice.
