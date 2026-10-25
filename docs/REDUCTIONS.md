# Cooperative scalar reductions

The reduction emitter validates the single-block schedule shared by
[r01](https://github.com/cloudrift-ai/emmy-hackathon-2026/tree/ec8cde1943f88efea3827a95bda6085553d2d407/problems/r01_array_sum) and
[r02](https://github.com/cloudrift-ai/emmy-hackathon-2026/tree/ec8cde1943f88efea3827a95bda6085553d2d407/problems/r02_big_sum).
It does not select behavior by filename, entry name or public input values.

1. Each thread processes input indices `tid + k * block_size`, starting with the
   reduction operation's identity.
2. All lanes reach five full-warp XOR shuffles, including lanes without inputs.
3. One lane per warp stores its partial in statically declared shared memory.
4. All threads synchronize; one lane per participating warp combines disjoint
   slots in each halving stage, with an unconditional block barrier after it.
5. Thread zero writes the result through the scalar output pointer.

The validator checks every node and semantic field in this topology. Supported
blocks are powers of two from 32 through 1024, enforced in PTX with `.reqntid`.
Input lengths are positive static integers below 2^31; tails need no padding.
The official scalar IR shape is `[]`, while metadata uses a one-element `[1]`
buffer; it is never passed by value. The public cases use 128 threads and 16 bytes
of static shared memory, while metadata `smem_bytes=0` describes dynamic storage.

Supported single-state operations are add/sum, multiply/prod, maximum/amax and
minimum. The loop can contain multiple scalar Load, Assign and Let statements
with one Accum. Its scalar work reuses the common emitter, so sum of squares,
two-input transforms and activation/reduction compositions need no separate
backend. Loads address the full reduction domain or a one-element parameter
buffer at index zero. The shuffle and tree combiners must match the accumulator.

Identities follow the pinned Emmy operation exactly: add uses 0, multiply 1,
maximum/amax -1e30, and minimum +1e30. The finite maximum/minimum identities also
apply to idle lanes; they are not replaced with infinities.

Partial warps, non-power-of-two warp counts, multiple blocks per row, non-f32
state, atomic output, custom barriers, segmented trees and empty inputs fail
explicitly. Names and argument order may change, and tests vary both lengths and
block sizes independently.

## Mean and row results

The same plan now supports an optional divisor epilogue and a row axis. Mean
loads a one-element runtime input and applies `div.rn.f32` after summation; changing
the metadata value changes runtime data without changing PTX. Row sums preserve
the input dimensions `(rows, columns)` and map one CTA to each row. Addresses
use `(row * columns + column) * 4`; row results use `row * 4`. A one-row matrix
still retains its rank-2 input/rank-1 output contract. Grid size must equal rows.

The same plan supports row maximum and transformed row sums without changing
the ownership mapping. Tests cover rectangular row maximum, all-negative inputs,
scalar and row sum of squares, two-input transforms, scalar parameters,
activation/reduction combinations, column tails, idle lanes, changed row counts
and blocks of 32–1024 threads.
The CPU model resets all private/shared state between CTAs and rejects multiple
threads or CTAs writing one ordinary output element.

## Evidence

The cooperative CPU model executes the emitted subset, checks global address
coverage, waits for warp/block participants, and rejects uninitialized shared
reads and conflicting accesses between block barriers. Every input of both
public cases is read once and the scalar output is written once. Their original
tolerances pass with `equal_nan=False`. Rank-0 expected arrays are compared after
reshaping the one-element modeled output, without changing the tolerance.

This checks our emitted control/data flow, not PTX assembler acceptance, NVIDIA
warp scheduling, the full hardware memory model or performance. The wrapper was subsequently adapted and independently tested for scalar
expected shapes and all public outputs. All four reduction fixtures now have
official GPU PASS receipts, with the same errors as their local models.

The authorized validation used exactly the public launch descriptors preserved
in the historical local records, with no warmup and one iteration. r01/r02 use
grid 1, block 128, x[1024]/x[65536] and scalar output[1]. r03 adds its runtime
divisor; r04 uses x[64,512], output[64], grid64 and block128. Dynamic shared memory
is zero; static warp partials live in PTX. No Python source or private input data
was uploaded. See [RUNNER_RESULTS.md](RUNNER_RESULTS.md) for actual receipts.

PTX synchronization was checked against NVIDIA's
[shuffle](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync) and
[barrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar) references.
