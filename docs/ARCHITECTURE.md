# Architecture and supported scope

The pipeline is `trusted JSON + meta.json -> Emmy typed graph -> validated kernel
contract -> our PTX emitter -> stdout`. Emmy supplies deserialization and typed
objects only. No Emmy renderer, nvcc or other lowering backend is invoked.

## Loader and ABI

The loader accepts one KernelOp plus InputOp/ConstantOp leaves. It matches graph
inputs/outputs, validates f32 positive static rank-1/rank-2 shapes and rank-0
outputs, and preserves both tensor dimensions and total element counts. A scalar
IR result `[]` maps to metadata buffer `[1]`, passed through a `.u64` pointer.

`args` must agree exactly with `arg_order`, dimensions, dtype and input/output
roles. Runtime constants use pointers; metadata values never enter generated PTX.
By-value scalars, intermediate graphs and in-place outputs are rejected. Launch
geometry is one-dimensional; explicit cooperative schedules require complete
matching blocks. Dynamic shared-memory bytes and output-zeroing names are
validated by the loader, then checked against the chosen backend's contract.

## Backend selection

Dispatch depends on IR node structure, never filenames or problem identifiers.

- `ptx.py`: common `emit_math` operations and register allocation, plus the
  scalar/grouped rank-1 pointwise path with affine bounds and exact contiguous
  output coverage. Rank-2 pointwise graphs use the structured path. The original
  four pointwise GPU-tested PTX artifacts are unchanged.
- `reduction.py`: single-state add/product/max/min reductions with scalar
  transforms, neutral tail lanes, full-warp shuffle and unique shared-tree
  writers. See REDUCTIONS.md.
- `structured.py`: bounded row-major indices, Tile and raster decoding, loops,
  integer-predicate conditionals, state lifetimes and output-coverage proofs. Scalar Assign/Accum nodes
  call the same inherited `emit_math` used by the rank-1 path.
- `cooperative.py`: validated multi-state per-row reductions and a second output
  pass, reusing structured scalar operations.
- `matmul.py`: shared allocation, async-copy groups, alignment/domain reasoning,
  uniform synchronization and zero-initialized atomic output accumulation.

External dynamic shared declarations appear at module scope, while registers and
static shared buffers remain inside the entry. This distinction was confirmed by
a real assembler failure and corrected m01 retry, retained with the result record.
PTX is version 8.0, target sm_89, with 64-bit global addresses and 32-bit shared
addresses. Explicit arithmetic rounding avoids unintended multiply/add fusion.

## Validation boundaries

The supported schedules use f32, positive static dimensions, bounded integer
indices and known scalar operators. Other dtypes, dynamic dimensions, unknown
nodes/operators and unproven address or synchronization contracts fail clearly.
Grouped pointwise partial cells, swizzled shared layouts, arbitrary async races,
partial cooperative warps and general multi-kernel graphs remain unsupported.
Conditionals have independent SSA scopes and may update existing carried
accumulators. Integer comparisons refine branch index bounds; output coverage
follows the branch taken for each logical coordinate. Shared-memory contractions
currently reject Cond to preserve their uniform synchronization checks.
The ordinary structured output proof has explicit finite-domain resource limits.
See ADVANCED_IR.md for exact limits and NUMERICS.md for floating-point choices.

Emission completes in memory before stdout is written, so rejected IR cannot
leave partial PTX. Official deserialization uses expression evaluation; only
trusted official or self-generated IR is supported, not hostile uploads.

Deterministic local tests and four PTX CPU models check values,
addresses, state lifetimes, synchronization and invalid inputs. Separate official
receipts establish assembly/execution/expected-array comparison for all 15 public
fixtures. That hardware evidence does not cover every shape, special value,
interleaving, hidden test or performance scenario.
