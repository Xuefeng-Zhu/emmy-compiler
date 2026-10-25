# Numerical semantics

All supported buffers and arithmetic are float32. The emitter follows explicit
IR operation order and preserves intermediate rounding:

| IR operation | PTX | Chosen behavior |
| --- | --- | --- |
| add | `add.rn.f32` | Round to nearest, ties to even |
| multiply | `mul.rn.f32` | Round to nearest, ties to even |
| relu | `max.NaN.f32 x, +0` | Propagate NaN; negative values and -0 become +0 |

No `.ftz` modifier is emitted, so subnormals are preserved. Explicit `.rn`
modifiers constrain optimizer contraction of multiply/add into FMA; the compiler
does not fuse or reassociate arithmetic. `max.NaN` is available in PTX 7.0+ on
sm_80+, within our PTX 8.0/sm_89 target. It returns canonical NaN; preservation of
NaN payloads is not promised. These choices were checked against NVIDIA's
[add](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add),
[mul](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul)
and [max](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max)
instruction reference.

The direct expected-generation expression for official p03 is
`np.maximum(x, 0) * y`, defined in
[expressions_open.py lines 41-43](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/tools/expressions_open.py#L41-L43).
[generate.py lines 131-140](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/tools/generate.py#L131-L140)
applies that function to seeded inputs to construct expected.npz. It propagates
NaN, as does the pinned Emmy host operation. This is the reason to retain
`max.NaN`, independently of the host implementation.

An upstream exceptional-value conflict remains unresolved: the official
[p03 CUDA reference](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/problems/p03_relu_mul/k_mul_pointwise.ref.cu#L7)
uses `fmaxf(0.0f, x)`, which selects zero when x alone is NaN. For x=NaN, y=1,
the expected-generation function and our PTX produce NaN, while this CUDA
expression produces +0. The public p03 seeded inputs and expected outputs are
finite; no general rule explicitly excludes NaN inputs. Finite public data do
not settle the hidden exceptional-input contract. We do not claim universal f32
input compatibility or treat this documented conflict as fixed.

Acceptance comparisons and operation-semantic tests have different purposes.
Official expected-array tests use the exact metadata tolerances and explicit
`equal_nan=False`, matching the NaN policy in
[submit.py line 150](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/submit.py#L150).
NaN/NaN and NaN/zero fail that policy; signed zeros compare equal. Regression
cases exercise the same helper used by all four official fixture comparisons.
The previous default of numpy.testing.assert_allclose allowed paired NaNs and
has been corrected; problem tolerances have not changed.

Separately, the ReLU/multiply semantic test directly evaluates the official
`np.maximum(x, 0) * y` expression for special values, comparing NaN classification
and the bits of every non-NaN result. A semantic NaN match is not a runner PASS:
the current official checker rejects NaN even against a NaN expectation. The
ReLU-only test also checks the pinned Emmy host operation. In this environment,
both map -0 to +0 before multiplication; multiplication can produce -0 again.

Tests compare finite result bits (including signed zero), NaN classification,
infinities and the smallest float32 subnormal. A separate-rounding witness uses
`a = 1 + 2^-23`, `b = -1 + 2^-24`: float32 `a*b + a` produces approximately
1.1920929e-7 with separate rounding, versus 5.960465e-8 when fused. The emitted
instruction sequence and CPU model are checked against the separate result.

The CPU model validates operand/address flow and float32 expectations. It does
not assemble PTX, emulate all NVIDIA behavior or prove optimizer behavior. The
p00–p03 GPU receipts cover only the supplied finite public arrays. Exceptional
values, NaN policy and hidden variants remain outside that GPU evidence; the
upstream semantic conflict above is not resolved by those passing runs.

## Shared scalar operations and activations

`Emitter.emit_math` is shared by pointwise, structured, cooperative and
contraction lowering. Rank and thread schedule do not change an operation's
scalar implementation. Alongside add/multiply/ReLU, it supports subtract,
divide, minimum/maximum, copy, reciprocal, sqrt, rsqrt, negative, abs, exp, tanh,
sigmoid, SiLU, erf and both GELU forms. `absolute`, `true_divide` and `exp_fast`
map to their corresponding operations.

`abs.f32` clears the sign bit, including on negative zero. Divide and sqrt use
explicit `.rn`; rsqrt computes rounded sqrt followed by rounded reciprocal.
Natural exp multiplies by the f32 value of log2(e), then uses `ex2.approx.f32`.
Sigmoid follows `1 / (1 + exp(-x))`, and SiLU follows
`x / (1 + exp(-x))`, matching the pinned Emmy operation formulas.

For `abs(x) > 0.125`, tanh computes
`(1 - exp(-2*abs(x))) / (1 + exp(-2*abs(x)))`, then copies x's sign. Smaller
inputs use a short odd polynomial to avoid cancellation near zero, preserving
signed zero and tiny inputs. Erf similarly uses an odd polynomial near zero and
the Abramowitz–Stegun 7.1.26 polynomial elsewhere, with f32 intermediates and the
same exp lowering. `gelu` uses `0.5*x*(1 + erf(x/sqrt(2)))`; `gelu_tanh` uses
`0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`. These are approximate
implementations; polynomial error, intermediate rounding and GPU exp2 error all
contribute. No universal error bound is asserted for the emitted sequence.

The emitted `copysign.f32` takes its sign from the first source operand and
magnitude from the second, as specified by NVIDIA's
[copysign reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign).
An independent hand-written PTX probe checks this operand convention in the model.

Local activation regressions compare finite samples and composed kernels against
NumPy and Python's `math.erf`, including changed shapes and rank-2 addressing. The CPU models use
mathematical float32 exp2, so these tests do not measure NVIDIA's approximation
error. The saved public GPU receipts predate these added activation variants.

## Single-block sum

Every lane starts at +0 and accumulates its strided inputs with `add.rn.f32`.
Five XOR shuffle stages reduce each full warp; a halving shared-memory tree
combines warp partials. Idle lanes contribute +0 and participate in every
rendezvous. This follows the validated IR schedule rather than serial summation.
The NumPy oracle accumulates at higher precision before conversion; different
rounding order is expected and the original per-case tolerances are unchanged.
The r01/r02 CPU-model errors are 3.814697265625e-6 and 1.52587890625e-5.
All four public reduction fixtures subsequently passed NVIDIA assembly and GPU
comparison. Subnormal and exceptional reduction behavior remains unverified.

Mean applies `div.rn.f32` to the completed f32 sum and a value loaded from its
runtime divisor buffer; no constant reciprocal is substituted. Row sum uses the
same arithmetic independently per row. Local CPU-model errors against the r03
and r04 float64-based oracles are 9.313225746154785e-10 and 5.7220458984375e-6;
original tolerances remain unchanged. The corresponding official GPU fixtures passed with the same errors; this does
not imply exceptional-input coverage.

The same cooperative schedule also lowers multiply, maximum and minimum, and
allows scalar transforms before Accum. Each loop update, warp shuffle combine
and shared-tree combine uses the same operation. Idle lanes use the pinned
identity: 1 for multiply, -1e30 for maximum/amax and +1e30 for minimum. Local
regressions cover all-negative maxima, tails, sum of squares and composed
activations; these added variants have no new hardware receipts.

## Fused, attention and atomic reductions

The shared scalar layer emits explicit `sub.rn.f32`, `div.rn.f32`, `sqrt.rn.f32`,
NaN-propagating min/max and a natural-exp sequence `mul.rn.f32 log2(e)` followed
by `ex2.approx.f32`. The local models calculate mathematical float32 exp2; they
do not model the exact approximation error of GPU exp2. All finite public fused/attention fixtures subsequently passed GPU comparison;
that is still not a bound on every input or GPU target. RMSNorm reads count, epsilon and weights from runtime
buffers; L2Norm does not inject an epsilon absent from IR.

Emmy's maximum identity is finite `-1e30`. Online-softmax state initialization
preserves that value. Extremely negative inputs, infinities, all-zero L2 rows
and other exceptional cases are not a universal supported-input promise. Zero
L2 rows can produce NaNs under the direct expression, while the official checker
rejects NaN/NaN. The earlier CUDA/expected-generation maximum discrepancy remains.

Split-K emits `red.global.add.f32`, whose global-memory form flushes subnormal
inputs/results. Its addition order also depends on CTA scheduling; the CPU model
checks a deterministic order and does not enumerate every hardware ordering.
Ordinary multiply/add instructions remain separately rounded and unfused. The
finite public matmul tolerances remain unchanged. See NVIDIA's
[exp2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2)
and [reduction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red) references.

The final hardware run covered every public fixture on an RTX 5090 with original
tolerances. a01's observed GPU error was 6.556510925292969e-7, while its mathematical
CPU exp2 model had a slightly different error. This is consistent with the stated
model limitation; no GPU approximation was substituted into the expected arrays.
