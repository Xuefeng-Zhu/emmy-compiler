# Architecture and supported scope

The pipeline is `trusted JSON + meta.json -> Emmy typed graph -> validated kernel
contract -> our PTX emitter -> stdout`. Emmy is used for deserialization and typed
IR objects. No Emmy renderer, nvcc, ptxas or other lowering backend is invoked.

The loader accepts exactly one KernelOp plus InputOp and ConstantOp leaves. Graph
activation inputs must match InputOp outputs; kernel inputs include both activation
and constant buffers. It checks the matching metadata entry/node, f32 types,
positive static one-dimensional shapes, buffer roles and launch dimensions.
`args` must have the same names in exactly the same order as `arg_order`.
By-value scalars, intermediate buffers and in-place outputs are rejected.

Constant buffers use the same pointer ABI and Load path as other inputs. Their
metadata must supply numeric `data` with the declared length and no `seed`.
Values are never embedded in PTX. Both broadcast scalar buffers and indexed
multi-element buffers are tested. ConstantOp source/context loading is not part
of this compiler: metadata must already describe the bound buffer contents.

The emitter accepts one ordinary one-dimensional Tile with a positive static
extent, f32 Load, Assign(add/multiply/relu), and one non-atomic, unswizzled Write per
output. It validates operator identity, arity and SSA names before emission.
Grouped memory operations of width 1, 2 or 4 become scalar loads/stores, preserving
consecutive value semantics without vector alignment assumptions.

Index trees support the tile axis, nonnegative integer literals, addition and
multiplication by a constant. Affine bounds prove indices fit within 31 bits and
accesses fit declared buffers. Writes must cover each output once with contiguous
per-thread cells, ruling out races and silent partial output in this slice.
Scalar cells handle odd sizes and block tails; grouped partial cells requiring
per-element predicates are unsupported and rejected. Rank-0/2-D buffers, other
types, nonlinear indices, reductions, control flow, shared memory and cooperative
or rasterized scheduling remain unsupported.

Each thread computes `blockIdx.x * blockDim.x + threadIdx.x`; the Tile extent
guard protects excess threads. Metadata must launch enough threads. The same path
handles all four public pointwise cases and independently varied shapes/names.
Virtual registers are allocated by type. Address offsets use `mul.wide.u32`
before 64-bit pointer addition. Numeric choices are documented in NUMERICS.md.
The output remains PTX 8.0, sm_89, with 64-bit addressing.

Emission finishes in memory before stdout is written, so validation errors cannot
leave partial PTX. Official deserialization uses Python expression evaluation;
only trusted official or self-generated IR is supported, not hostile uploads.

Validation levels remain distinct: 64 local tests (including a limited CPU PTX
model), prior NVIDIA assembly and GPU output comparison for p00/p01, and pending
assembly/GPU checks for p02/p03. CPU-model agreement cannot establish PTX legality
or NVIDIA hardware behavior. See STATUS.md and RUNNER_RESULTS.md for evidence.
