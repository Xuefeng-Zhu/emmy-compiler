# Challenge boundaries

Official sources checked October 24, 2026 (America/Los_Angeles):
[website](https://riftstack.ai/hackathon),
[task repository](https://github.com/cloudrift-ai/emmy-hackathon-2026),
[TASK.md](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/TASK.md),
[MVP.md](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/MVP.md).
Reference provenance is recorded in `reference-manifest.json`.

There are 15 public problems in five categories. The minimum goal is **all eight
problems in categories 1 and 2**, not two individual tests:

| Category | Public problems |
| --- | --- |
| Pointwise | p00_scalar_add, p01_vector_add, p02_scale_shift, p03_relu_mul |
| Reduction | r01_array_sum, r02_big_sum, r03_mean, r04_row_sum |

Matmul, fused and attention-like kernels are subsequent categories. Hidden
problems vary shapes and operations within these categories. Hardcoding filenames,
shapes or expected outputs is not a valid solution.

The required CLI is `python compile.py path/to/kernel.ir.json > out.ptx` with the
program at the repository root. Use Emmy 0.3.11 to load IR and implement the
translation ourselves. The metadata defines argument order, launch geometry and
tolerance. Official success requires ptxas acceptance and correct GPU outputs.
The official runner and submit.py exist; neither was invoked by this initial work.

The stated deadline is October 27, 2026 at 23:59 AoE. Submission requires a public
source repository and a 3–5 minute explanation video in the participant's own
voice. AI assistance is allowed, but the participant must understand the compiler.
The website permits several public Git hosts; the repository README asks for
GitHub. Resolve that distinction before any later submission.

This setup authorizes local implementation and tests only. Registration,
agreements, credentials, repository publication, external uploads, runner/GPU
use, paid resources and submission remain separate user decisions. The separate
Open Kernel Challenge workspace, accounts, sessions and quotas are outside this
project. Entry IP terms and redistribution permissions are not assumed settled.
