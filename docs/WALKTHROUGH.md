# A 3–5 minute own-voice walkthrough

Aim for about four minutes. Use this as a route through the project, not a script
to read verbatim. Adapt the phrasing to what you can explain comfortably. Do not
claim to have personally designed or discovered something unless that is true.
The final recording should use your own voice and understanding.

Keep the [offline field guide](explainer/index.html),
[learning guide](LEARNING_GUIDE.md) and [results](RUNNER_RESULTS.md) ready. Avoid
launching new remote GPU jobs during recording; existing receipts are enough to
show what was tested.

| Time | Show | Explain in your own words |
| --- | --- | --- |
| 0:00–0:35 | Pipeline diagram and sample IR/metadata | The input is scheduled Emmy Kernel IR. Emmy deserializes it; this project's own checks and lowering produce PTX. Metadata supplies ordered pointers and launch geometry. |
| 0:35–1:25 | Trace steps 2–6, then thread 9 and idle thread 10 | How one thread obtains an index, checks its bound, computes byte addresses, adds in f32 and writes once. Why an output-ownership proof matters as well as bounds checks. |
| 1:25–2:10 | Reduction diagram in the learning guide | Private partials, 32-lane shuffle, shared partials and block barriers. An idle lane still participates in the cooperative reduction. This is different from an independent pointwise early return. |
| 2:10–2:45 | Choose attention **or** matmul | Attention: inner state resets while outer softmax state survives and is rescaled. Matmul: async-copy waits and barriers protect shared reads; split-K output starts at zero before launch. |
| 2:45–3:35 | Exp chart, witness button, error-budget view | The preserved failure had 251 out-of-tolerance values. Range reduction reduced this to zero on the same synthetic sample and tolerance. Distinguish the GPU curve from the CPU arithmetic illustration. |
| 3:35–4:10 | Evidence table and exact CI run | Public GPU evidence covers 15 artifacts. CPU tests, PTX hashes and GPU receipts answer different questions. Timings are single-shot; hidden and exceptional-value coverage remains bounded. |

For a three-minute version, omit the advanced example and shorten the exp
explanation. For a five-minute version, inspect one relevant source function and
one rejected-input test. Do not try to describe every file.

## Optional local terminal demonstration

From the repository root with the documented environment installed:

```sh
.venv/bin/python compile.py docs/explainer/sample/learn_add.ir.json > /tmp/learn_add.ptx
cmp /tmp/learn_add.ptx docs/explainer/sample/learn_add.ptx
.venv/bin/python tools/check_docs.py
```

This compiles an original teaching example and checks documentation. It does not
assemble or execute PTX on a GPU. If you show the full suite, run it before the
recording and display its actual completed output and commit, rather than waiting
through the whole run on camera.

## Questions worth being able to answer

- Why are all parameters pointers, including a one-element scalar result?
- Why is a pointwise tail guard safe while a reduction lane cannot skip a barrier?
- What comes from the input schedule, and what does this compiler implement?
- Why can a CPU exp model pass while approximate GPU instructions differ?
- Which result is a measurement, which is a model, and which is still untested?

Use precise, ordinary claims: “This recorded artifact passed these inputs,” or
“The current code rejects this unsupported schedule.” Avoid “all f32 is correct,”
“the hidden tests pass,” or “the fastest kernel” without supporting evidence.

Recording and final submission remain user-owned steps. This documentation does
not create a recording, publish a video or submit a competition entry.
