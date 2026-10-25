# Architecture and initial scope

The pipeline is `trusted JSON + meta.json -> Emmy typed graph -> validated kernel
contract -> our PTX emitter -> stdout`. Emmy is used only for deserialization and
typed IR objects. We do not invoke its renderer, command-line compiler, nvcc,
ptxas, or another lowering backend.

The loader locates exactly one KernelOp, matches both its entry and node in
metadata, and validates buffer identities, f32 types, static one-dimensional
shapes and launch dimensions. Parameter order is taken directly from `arg_order`.
It never guesses the ABI from graph insertion order.

The emitter supports one ordinary one-dimensional Tile with a positive static
extent; InputOp buffers; f32 Load; Assign(add) with two SSA values; and one
non-atomic, unswizzled Write per output. Grouped memory operations of width 1, 2
or 4 become individual scalar loads/stores. This preserves the consecutive value
semantics without requiring vector alignment assumptions. It is not vector PTX
optimization.

Integer index trees support the tile axis, nonnegative integer literals, addition
and multiplication by a constant. Affine coefficients prove indices fit within
31 bits and accesses fit the declared buffers. Writes must cover the full output
once, with consecutive per-thread cells. This restriction rules out write races
and silent partial results in the supported slice. Non-affine expressions,
other arithmetic, reductions, constants as graph nodes, shared memory, extra
scheduling features and all other statements fail explicitly.

Each thread computes `blockIdx.x * blockDim.x + threadIdx.x`. A guard derived
from the Tile extent prevents excess threads from accessing memory. The prescribed
launch must cover all logical cells. p00 has 64 cells and 256 launched threads;
p01 has 512 cells and two blocks of 256. Both use the identical lowering path.

Virtual registers are allocated by type and declared after the body has been
built. Address offsets use `mul.wide.u32` before 64-bit pointer addition. Floating
addition uses explicit `add.rn.f32`. Output targets PTX 8.0, sm_89, 64-bit addressing.
Emission completes in memory before stdout is written, so rejected input cannot
leave a plausible partial kernel.

Validation has three distinct levels: local typed-input/structure tests (39 passing),
NVIDIA PTX assembly (p00/p01 passed), and execution with official tolerances and launch
metadata (p00/p01 passed on the official RTX 5090). Other cases remain unvalidated.
Text tests alone cannot establish GPU correctness; see RUNNER_RESULTS.md for receipts.
