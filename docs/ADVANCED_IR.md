# Structured, fused and shared-memory lowering

All backends use Emmy 0.3.11 only to deserialize typed IR. Our code selects a
backend from node structure and emits PTX directly. It never selects answers or
algorithms by public case name, and does not invoke Emmy rendering, nvcc or ptxas.

## Shared scalar/control-flow layer

`structured.py` implements static multidimensional Tile decoding, row-major
addresses, grouped raster order (including a partial final group), Loop and
StridedLoop, f32 Load/Write, Assign, Let, Init and Accum. Vector memory operations
are scalarized into consecutive cells with a proof that they do not cross a row.

An accumulator belongs to its immediate reduction loop. A seeded loop initializes
all its states before iteration; nested dot-product states reset each outer
iteration, while outer softmax states persist. `seed=False` requires an existing
initialized state. `Accum.base` replaces the prior operand with the named,
rescaled value, preserving online-softmax semantics. Combiner Assign operations
run in IR order. No arithmetic fusion is introduced.

Integer expression bounds and affine provenance prove supported addresses inside
static buffers and signed int32 arithmetic. Output writes are checked separately
from floating-point work: ordinary schedules must cover every output exactly
once. This proof currently accepts at most one million logical Tile cells and
one million total output elements, with a four-million-statement traversal
budget. Larger/irregular domains fail clearly instead of bypassing the check.

This layer handles both official attention schedules, including multiple output
cells per thread and grouped raster order. It does not generate tensor-core code.

## Fused row reductions

`cooperative.py` validates a reusable row topology: one strided reduction pass,
full-warp shuffle, shared partials, a full-block barrier, a halving tree, scalar
epilogue operations and a second strided output pass. It supports multiple f32
states and arbitrary supported Assign combiners. Every peer state is shuffled
before any state update. Tree stages similarly load all state pairs first.
Only lane zero writes each tree slot; all threads execute each block barrier.

The topology covers RMSNorm, row softmax and L2Norm with runtime count/epsilon/
weight buffers. Each CTA owns one row. The validator proves the final scalar
stores use that row and the full second-pass column index, rejecting fixed-row
races, missing outputs and duplicate writes. Power-of-two blocks of 32–1024
threads and non-multiple column tails are supported. Idle lanes use the pinned
operation's identity and still participate in every shuffle/barrier.

## Shared-memory contractions

`matmul.py` adds Smem, CpAsyncCopy/Commit/Wait, Sync, shared loads and atomic-add
outputs to the structured layer. Copy size, alignment, index bounds and declared
shared storage are checked. Complete CTAs and uniform waits/barriers are required.
A wait followed by a block barrier precedes shared reads. Index domains refine
ring-buffer and clamped-prefetch bounds without public sizes embedded in code.

The rectangular matmul uses static shared storage and split-K atomic adds; its
output must be named in metadata `zero_outputs`. Initialization belongs to the
host before launch; no CTA clears another CTA's partial result. The square
matmul uses one external dynamic shared pool. Its packed layout must match
metadata exactly (73,728 bytes for the public case); no duplicate static pool is
allocated. The driver must permit the requested per-block dynamic allocation.

Pipelined schedules can leave redundant prefetch groups after their last consumed
stage. Our emitter adds a final wait-group-zero drain before return, after output
stores. Ordinary outputs must have exactly one writer; atomic outputs must have
a uniform positive contribution count, a zeroing contract, and no mixed writers.

These checks are not a complete proof of arbitrary shared-memory race freedom.
Current support excludes swizzled copies, grouped shared-memory raster schedules,
partial CTAs, named/count-limited barriers and arbitrary asynchronous programs.
The independent CPU model checks actual public copy queues, initialization,
visibility, ring reuse and output contributions. It does not establish hardware
ordering or assembler acceptance.

## Numerical and hardware limits

Natural exponential is emitted as explicit f32 multiplication by log2(e), then
`ex2.approx.f32`. CPU models use mathematical float32 exp2 and cannot reproduce
the GPU instruction's approximation error. Maximum identity follows the pinned
Emmy operation (`-1e30`), not a substituted mathematical negative infinity.
Global f32 atomic add flushes subnormal inputs/results, unlike ordinary scalar
`add.rn.f32` without `.ftz`. See NUMERICS.md for the remaining exceptional-input
conflicts. Finite public CPU-model agreement is not universal f32 compatibility.

All 15 public fixtures now have official assembly/execution/comparison evidence.
The first m01 assembly attempt rejected an external shared declaration inside
the entry body. It was moved to module scope, a regression was added, and the
second compile plus its first GPU launch passed. The CPU model did not detect
this syntax error, which illustrates its limits. See RUNNER_RESULTS.md.
These runs do not settle hidden shapes or exceptional-value requirements.
