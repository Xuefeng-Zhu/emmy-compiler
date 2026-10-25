# Cooperative sum reduction

The first reduction slice recognizes a validated single-block schedule shared by
[r01](https://github.com/cloudrift-ai/emmy-hackathon-2026/tree/ec8cde1943f88efea3827a95bda6085553d2d407/problems/r01_array_sum) and
[r02](https://github.com/cloudrift-ai/emmy-hackathon-2026/tree/ec8cde1943f88efea3827a95bda6085553d2d407/problems/r02_big_sum).
It does not select behavior by filename, entry name or public input values.

1. Each thread accumulates input indices `tid + k * block_size`, starting at +0.
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

A sum with a different combiner, partial warp, non-power-of-two warp count,
multiple blocks per row, non-f32 state, atomic output, custom barrier,
segmented tree or empty input fails explicitly. Names and argument order may
change, and tests vary both lengths and block sizes independently.

## Mean and row sum

The same plan now supports an optional divisor epilogue and a row axis. Mean
loads a one-element runtime input and applies `div.rn.f32` after summation; changing
the metadata value changes runtime data without changing PTX. Row sums preserve
the input dimensions `(rows, columns)` and map one CTA to each row. Addresses
use `(row * columns + column) * 4`; row results use `row * 4`. A one-row matrix
still retains its rank-2 input/rank-1 output contract. Grid size must equal rows.

Both extensions preserve the same synchronized intra-CTA reduction. Tests cover
column tails, idle lanes, changed row counts and blocks of 32–1024 threads.
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
warp scheduling, the full hardware memory model or performance. The existing
runner wrapper only permits p00–p03 and must be separately adapted for scalar
expected shapes before reduction validation. No reduction PTX has been uploaded.

A later minimal request can contain each generated PTX string plus the public
launch descriptors preserved in
[the local result record](validation/2026-10-25/reduction-local-results.json):
`k_sum_1_reduce`, grid `(1,1,1)`, block `(128,1,1)`, no dynamic shared memory,
x f32[1024] or f32[65536] with seed 0, and output sum_1 f32[1]. Use one iteration
and no warmup; upload no Python source, private data or credentials. That request
would require separate authorization for the official HTTP runner.

PTX synchronization semantics were checked against NVIDIA's
[shuffle reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync) and
[barrier reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar).

The mean/row-sum [local record](validation/2026-10-25/basic-eight-local-results.json)
preserves their exact proposed public launch descriptors. These also remain
unsent: r03 uses x[32768], runtime count[1], scalar output[1], grid 1 and block
128; r04 uses x[64,512], output[64], grid 64 and block 128. Both use dynamic shared
memory 0. Approving local development does not approve these runner requests.
