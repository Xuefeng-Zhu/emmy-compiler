# Numerical semantics

All supported buffers and arithmetic are float32. The emitter follows explicit
IR operation order and preserves intermediate rounding:

| IR operation | PTX | Chosen behavior |
| --- | --- | --- |
| add | `add.rn.f32` | Round to nearest, ties to even |
| multiply | `mul.rn.f32` | Round to nearest, ties to even |
| relu | `max.NaN.f32 x, +0` | Propagate NaN; negative values and -0 become +0 |

Ordinary add/multiply use `.rn` without `.ftz`, preserving subnormals for those
instructions on the target. Their explicit rounding constrains contraction into
FMA; emitted arithmetic remains separately rounded and is not reassociated.
This is not a blanket claim about every operation: global atomic addition has
different subnormal behavior, described below. `max.NaN` is available in PTX 7.0+ on
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
and exact non-NaN values (signed zeros compare equal). The ReLU-only test checks
the pinned Emmy host operation, which uses `np.maximum(0.0, x)`. Its zero-sign
choice differed between our macOS arm64 and Linux x64 runs: the first Linux CI
run returned -0 for the host ReLU of -0, while our PTX model returned +0.
Host NumPy bits are therefore not used as a portable signed-zero oracle.

Independent explicit bit patterns still require the PTX result to be +0 for
`max.NaN.f32 -0, +0`, following NVIDIA's rule that +0 > -0. The composed test
also checks that multiplying this +0 by a negative operand produces -0. It
retains bit checks for subnormals, normal values and infinity. This is a test
oracle correction; emitted PTX, official tolerances and `equal_nan=False` are
unchanged. A semantic NaN match is not a runner PASS: the official checker rejects
NaN even against a NaN expectation.

Tests compare PTX result bits (including signed zero), NaN classification,
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
Natural exp uses range reduction in this shared path. It first clamps x to
[-104, 89] with NaN-propagating min/max, then computes:

1. `n = cvt.rni.f32.f32(x * log2(e))`, rounding to an integral float with ties
   to even.
2. `r = (x - n * ln2_hi) - n * ln2_lo`, with separately rounded f32 operations;
   `ln2_hi = 0f3f317200` and `ln2_lo = 0f35bfbe8e`.
3. `partial = ex2.approx.f32(r * log2(e))`.
4. `half = cvt.rzi.f32.f32(n / 2)` and `rest = n - half`; evaluate the two
   scale factors with `ex2.approx.f32`, preserving multiplication order
   `(partial * first_scale) * second_scale`.

The two scale exponents stay in [-75, 64], keeping both scale factors normal
before the final products. All three exp2 instructions are approximate; integer
arguments are not assumed to produce exact powers of two. The sequence avoids
an overflowing single scale factor and retains separate rounding without FMA.
It does not promise correctly rounded exp or a universal ULP bound. See NVIDIA's
[cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
and [ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2)
references and the [implementation validation](EXP_ACCURACY_FIX.md).

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
error. Seven finite activation GPU cases and two guarded attention cases passed
again after the common-exp change, alongside the changed public f02/a01/a02
artifacts. The [exp accuracy report](EXP_ACCURACY_FIX.md) and
[latest archive](validation/2026-10-25-exp-fix/) retain the results. These bounded
samples do not establish accuracy across the full f32 range.

### Wide-range exp diagnostic and fix — October 25, 2026 (UTC)

Before range reduction, on 4097 values spanning [-80, 80], GPU exp failed at 251 values under the unchanged
rtol=2e-6/atol=2e-7 threshold; the CPU model failed at 257. These thresholds come
from existing local activation tests: there is no official tolerance for these
new activation cases. The maximum relative GPU error over the full array was
3.6186774e-6. One counterexample is x=44.609375: expected 2.363770999e19,
actual 2.3637635225e19, relative error approximately 3.163e-6.

Range reduction now fixes this sampled failure under the same thresholds:
zero of 4097 GPU values fail, and maximum relative error falls to
1.52530531e-7. A separate 1019-value finite-boundary GPU sample passes, including
matching zero classifications and sampled subnormal outputs. Its maximum
observed error is 2 ULP, and maximum relative error over normal expected outputs
is 1.6858741e-7. Its upper finite input is 87; these observations are not a
universal 2-ULP or full-overflow-boundary guarantee.

NaN and infinity classification checks remain local only. Original tolerances
and `equal_nan=False` are unchanged. The [earlier report](GENERALIZATION_VALIDATION.md)
and failure receipts preserve the pre-fix evidence; the
[new report](EXP_ACCURACY_FIX.md) records the implementation and fresh GPU results.

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
activations. The October 25 regression adds three synthetic reduction GPU
passes, including maximum and sum-of-squares variants; their receipts are in the
[latest archive](validation/2026-10-25-regression/). These samples do not prove
all extreme-value or synchronization variants.

## Fused, attention and atomic reductions

The shared scalar layer emits explicit `sub.rn.f32`, `div.rn.f32`, `sqrt.rn.f32`,
NaN-propagating min/max and the range-reduced natural-exp sequence described
above. The local models calculate mathematical float32 exp2; they do not model
the exact approximation error of GPU exp2. The changed public softmax/attention
artifacts (f02/a01/a02) passed fresh GPU comparison after this change; that is
still not a bound on every input or GPU target. RMSNorm reads count, epsilon and weights from runtime
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

The original October 25 public checkpoint covered every public fixture on an
RTX 5090 with original tolerances. a01's observed GPU error was
6.556510925292969e-7, while its mathematical
CPU exp2 model had a slightly different error. This is consistent with the stated
model limitation; no GPU approximation was substituted into the expected arrays.
