# Submission readiness and participant explanation

This checklist records work and remaining decisions. It does not register,
publish, upload a video, accept terms or submit an entry. Official sources were
rechecked on October 25, 2026: [challenge website](https://riftstack.ai/hackathon),
[task repository](https://github.com/cloudrift-ai/emmy-hackathon-2026), and the
pinned TASK.md/MVP.md listed in CHALLENGE.md.

## Engineering readiness

- Root `compile.py` accepts the required IR path and adjacent `meta.json`.
- Setup uses isolated Python 3.12 and pinned packages; public fixtures have a
  fixed commit and hashes. No upstream generation scripts are executed.
- Original code performs direct IR-to-PTX lowering; unsupported contracts fail.
- Local parser, ABI, address, numeric-model and malformed-IR tests are available
  with `python -m unittest discover -s tests -v`.
- NVIDIA validation is complete for all 15 public fixtures, including the eight
  basic problems. The remaining eleven were run after separate explicit approval.
- Hidden tests, GPU exceptional values and sustained performance remain untested.

## User-owned prerequisites

1. Review original-code licensing, third-party redistribution permissions and
   competition IP terms. No public license is chosen automatically.
2. Decide whether to push the new local commits and publish the currently private
   repository. Neither action follows automatically from local development.
3. Record a 3–5 minute explanation in the participant's own voice, demonstrating
   understanding. Do not use generated voice to impersonate the participant.
4. Confirm participant/team details, accessible repository/video links, eligibility
   and any submission terms, then explicitly authorize final submission.

The website lists October 27, 2026 at 23:59 AoE as the deadline and permits
several public Git hosts; the repository README requests GitHub. The minimum
engineering goal is all eight problems in the first two categories. The public eight-problem goal now has GPU evidence; hidden-set success and
competition acceptance are not established by it.

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
- **3:35–4:00 — Limits and next steps.** Mention approximate exp, special-float
  ambiguity, unsupported schedules and untested hidden variants honestly.

Use this as an outline, not a claim that a recording exists or has been submitted.
