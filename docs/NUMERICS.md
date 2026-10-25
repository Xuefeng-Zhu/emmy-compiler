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

There is an unresolved source-level semantic discrepancy for exceptional inputs.
The installed pinned Emmy 0.3.11 `ElementwiseImpl('relu')` uses NumPy maximum;
in our pinned NumPy environment it propagates NaN and maps -0 to +0. The official
[p03 CUDA reference](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/problems/p03_relu_mul/k_mul_pointwise.ref.cu)
uses `fmaxf(0.0f, x)`, which selects zero when x alone is NaN. We choose the typed
Emmy host operation for this slice and explicitly test it. The public expected
arrays contain finite inputs and do not settle hidden-test requirements for NaN.
This is not a claim of identical behavior to every reference backend.

Tests compare finite result bits (including signed zero), NaN classification,
infinities and the smallest float32 subnormal. A separate-rounding witness uses
`a = 1 + 2^-23`, `b = -1 + 2^-24`: float32 `a*b + a` produces approximately
1.1920929e-7 with separate rounding, versus 5.960465e-8 when fused. The emitted
instruction sequence and CPU model are checked against the separate result.

The CPU model validates operand/address flow and float32 expectations. It does
not assemble PTX, emulate all NVIDIA behavior or prove optimizer behavior. These
new operations and exceptional-value cases still need separately authorized GPU
validation. Prior p00/p01 GPU receipts cover their original finite arrays only.
