# Pointwise numerical semantics

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
