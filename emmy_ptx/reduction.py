"""Validated single-block f32 sum lowering, derived from cooperative Kernel IR.

This deliberately supports one coherent reduction topology, not arbitrary control
flow. Every statement/semantic field is checked before an equivalent schedule is
emitted. Buffer names, length, thread count and parameter order come from the IR.
"""
from dataclasses import dataclass

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write

from .loader import Kernel, require
from .ptx import Emitter


def check(condition, message):
    require(condition, 'reduction: ' + message)


def items(value, length, description):
    check(isinstance(value, (tuple, list)) and len(value) == length, description)
    return tuple(value)


def name(value):
    check(isinstance(value, str) and bool(value), 'names must be nonempty strings')
    return value


def f32(dtype):
    return getattr(dtype, 'name', None) == 'f32'


def integer(expr, value):
    return isinstance(expr, Literal) and expr.dtype == 'int' and type(expr.value) is int and expr.value == value


def variable(expr, value):
    return isinstance(expr, Var) and expr.name == value


def combine(stmt, state):
    check(f32(stmt.dtype), 'combine dtype must be f32')
    check(items(stmt.state, 1, 'only one reduction state supported') == (state,), 'combine state does not match accumulator')
    peer, = items(stmt.state_b, 1, 'only one peer state supported')
    check(name(peer) != state, 'peer state must differ from accumulator')
    assign, = items(stmt.combine_states, 1, 'only a single add combiner supported')
    check(isinstance(assign, Assign), 'combine must contain Assign')
    check(isinstance(assign.op, ElementwiseImpl) and assign.op.name == 'add', 'only sum combiners supported')
    check(assign.name == state and items(assign.args, 2, 'add combiner needs two operands') == (state, peer), 'combiner operands must be accumulator and peer')
    check(assign.dtype is None or f32(assign.dtype), 'combiner Assign must inherit f32 or declare f32')


def conditional_write(stmt, axis):
    check(isinstance(stmt, Cond), 'expected a conditional write')
    check(items(stmt.else_body, 0, 'else branches are unsupported') == (), 'else branches are unsupported')
    pred = stmt.cond
    check(isinstance(pred, BinaryExpr) and pred.op == '==' and variable(pred.left, axis) and integer(pred.right, 0), 'write must be guarded by ' + axis + ' == 0')
    write, = items(stmt.body, 1, 'conditional body must contain one Write')
    check(isinstance(write, Write) and f32(write.value_dtype), 'conditional body must write f32')
    check(write.atomic is False and write.swizzle == 'NONE', 'atomic or swizzled writes unsupported')
    return write


@dataclass(frozen=True)
class SumPlan:
    size: int
    threads: int
    input: str
    output: str

    @property
    def warps(self):
        return self.threads // 32


def validate_sum(kernel: Kernel) -> SumPlan:
    tile, = items(kernel.body, 1, 'expected one cooperative Tile')
    threads = tile.block_threads
    check(type(threads) is int and 32 <= threads <= 1024 and threads & (threads - 1) == 0, 'thread count must be a power of two from 32 to 1024')
    check(kernel.grid == (1, 1, 1) and kernel.block == (threads, 1, 1), 'only one block matching Tile.block_threads is supported')
    check(tile.aux_threads == 0 and tile.raster_axes is None and tile.raster_group is None and tile.raster_orient == 'm', 'auxiliary/rasterized scheduling unsupported')
    axis, = items(tile.axes, 1, 'only one Tile axis supported')
    check(isinstance(axis, Axis) and axis.window is None and axis.extent.is_static and axis.extent.as_static() == threads, 'Tile extent must match the participating block')
    tile_axis = name(axis.name)
    check(tile_axis not in ('lane', 'warp'), 'Tile axis cannot shadow lane or warp')
    check(len(kernel.inputs) == 1 and len(kernel.outputs) == 1 and kernel.scalar_outputs == kernel.outputs, 'sum requires one input and one rank-0 output')
    input_name, = kernel.inputs
    output_name, = kernel.outputs
    loop, shuffle, shared, partial, sync, tree, final = items(tile.body, 7, 'unsupported sum topology; expected loop/shuffle/shared/write/sync/tree/write')
    check(isinstance(loop, StridedLoop), 'expected StridedLoop')
    check(isinstance(loop.axis, Axis) and loop.axis.window is None and loop.axis.extent.is_static, 'loop needs a static unwindowed axis')
    loop_axis = name(loop.axis.name)
    check(loop_axis not in (tile_axis, 'lane', 'warp'), 'loop axis must be distinct from thread coordinates')
    size = loop.axis.extent.as_static()
    check(type(size) is int and 0 < size < 2**31 and size == kernel.shapes[input_name], 'loop extent must cover the positive input length exactly')
    check(variable(loop.start, tile_axis) and integer(loop.step, threads), 'loop must start at its thread coordinate and stride by block size')
    check(loop.seed is True and loop.end is None and loop.unroll is False, 'custom loop seeds, bounds or unrolling unsupported')
    load, accum = items(loop.body, 2, 'sum loop must contain one Load and one Accum')
    check(isinstance(load, Load) and f32(load.dtype) and load.input == input_name, 'sum must load its f32 input')
    loaded, = items(load.names, 1, 'only scalar reduction loads supported')
    name(loaded)
    index, = items(load.index, 1, 'reduction input needs one index')
    check(variable(index, loop_axis), 'sum load index must be the strided loop axis')
    check(isinstance(accum, Accum) and f32(accum.dtype), 'sum needs an f32 Accum')
    state = name(accum.name)
    check(state != loaded and accum.value == loaded, 'Accum must fold the loaded value into a distinct state')
    check(isinstance(accum.op, ElementwiseImpl) and accum.op.name == 'add', 'only add accumulators supported')
    check(accum.base is None and items(accum.axes, 1, 'Accum must declare its reduction axis') == (loop_axis,), 'rescaled or mismatched accumulator unsupported')
    check(isinstance(shuffle, WarpShuffle) and type(shuffle.length) is int and shuffle.length == 32, 'only full-warp shuffle reductions supported')
    combine(shuffle, state)
    check(isinstance(shared, Smem) and shared.dtype == 'float' and type(shared.align) is int and shared.align in (0, 4), 'shared buffer must have natural f32 alignment')
    shared_name = name(shared.name)
    check(shared_name not in kernel.shapes and shared_name not in (tile_axis, loop_axis, 'lane', 'warp', loaded, state), 'shared buffer name aliases another binding')
    warps = threads // 32
    extents = items(shared.extents, 1, 'shared buffer must be one-dimensional')
    check(type(extents[0]) is int and extents[0] == warps, 'shared buffer must contain one element per warp')
    partial_write = conditional_write(partial, 'lane')
    partial_index, = items(partial_write.index, 1, 'warp partial needs one shared index')
    check(partial_write.output == shared_name and variable(partial_index, 'warp') and items(partial_write.values, 1, 'warp partial must write one value') == (state,), 'warp leader must write its accumulator to its shared slot')
    check(isinstance(sync, Sync) and type(sync.barrier_id) is int and sync.barrier_id == 0 and sync.count is None and sync.warp is False, 'a full-block Sync(0) is required before the tree')
    check(isinstance(tree, TreeHalve) and type(tree.length) is int and tree.length == warps and tree.tid_var == 'warp', 'tree must combine one partial per warp')
    check(items(tree.bufs, 1, 'only one shared tree supported') == (shared_name,), 'tree buffer does not match partial storage')
    check(type(tree.barrier_id) is int and tree.barrier_id == 0 and tree.barrier_count is None and tree.inner is None, 'named barriers and segmented trees unsupported')
    combine(tree, state)
    output_write = conditional_write(final, tile_axis)
    check(output_write.output == output_name and items(output_write.index, 0, 'rank-0 output requires an empty index') == () and items(output_write.values, 1, 'scalar output needs one value') == (state,), 'only thread zero may store the final scalar sum')
    return SumPlan(size, threads, input_name, output_name)


class ReductionEmitter(Emitter):
    def run(self):
        k = self.kernel
        plan = validate_sum(k)
        for buffer in k.arg_order:
            param, ptr = self.reg('rd'), self.reg('rd')
            self.put(f'ld.param.u64 {param}, [param_{buffer}];')
            self.put(f'cvta.to.global.u64 {ptr}, {param};')
            self.pointers[buffer] = ptr
        tid, lane, warp, index = [self.reg('r') for _ in range(4)]
        acc, loaded, peer = [self.reg('f') for _ in range(3)]
        pred = self.reg('p')
        self.put(f'mov.u32 {tid}, %tid.x;')
        self.put(f'and.b32 {lane}, {tid}, 31;')
        self.put(f'shr.u32 {warp}, {tid}, 5;')
        # Identity initialization precedes the loop, including every idle lane.
        self.put(f'mov.f32 {acc}, 0f00000000;')
        self.put(f'mov.u32 {index}, {tid};')
        self.lines.append('SUM_LOOP:')
        self.put(f'setp.ge.u32 {pred}, {index}, {plan.size};')
        self.put(f'@{pred} bra SUM_LOOP_DONE;')
        offset, address = self.reg('rd'), self.reg('rd')
        self.put(f'mul.wide.u32 {offset}, {index}, 4;')
        self.put(f'add.u64 {address}, {self.pointers[plan.input]}, {offset};')
        self.put(f'ld.global.f32 {loaded}, [{address}];')
        self.put(f'add.rn.f32 {acc}, {acc}, {loaded};')
        self.put(f'add.u32 {index}, {index}, {plan.threads};')
        self.put('bra SUM_LOOP;')
        self.lines.append('SUM_LOOP_DONE:')
        bits, shuffled = self.reg('r'), self.reg('r')
        for delta in (16, 8, 4, 2, 1):
            self.put(f'mov.b32 {bits}, {acc};')
            self.put(f'shfl.sync.bfly.b32 {shuffled}, {bits}, {delta}, 31, 0xffffffff;')
            self.put(f'mov.b32 {peer}, {shuffled};')
            self.put(f'add.rn.f32 {acc}, {acc}, {peer};')
        shared_base, shared_offset, shared_addr = [self.reg('r') for _ in range(3)]
        self.put(f'mov.u32 {shared_base}, sum_smem;')
        self.put(f'mul.lo.u32 {shared_offset}, {warp}, 4;')
        self.put(f'add.u32 {shared_addr}, {shared_base}, {shared_offset};')
        self.put(f'setp.ne.u32 {pred}, {lane}, 0;')
        self.put(f'@{pred} bra PARTIALS_DONE;')
        self.put(f'st.shared.f32 [{shared_addr}], {acc};')
        self.lines.append('PARTIALS_DONE:')
        self.put('bar.sync 0;')
        # One lane per warp writes each tree slot; every thread joins every barrier.
        other_addr = self.reg('r')
        half = plan.warps // 2
        while half:
            label = f'TREE_DONE_{half}'
            self.put(f'setp.ne.u32 {pred}, {lane}, 0;')
            self.put(f'@{pred} bra {label};')
            self.put(f'setp.ge.u32 {pred}, {warp}, {half};')
            self.put(f'@{pred} bra {label};')
            self.put(f'ld.shared.f32 {acc}, [{shared_addr}];')
            self.put(f'add.u32 {other_addr}, {shared_addr}, {half * 4};')
            self.put(f'ld.shared.f32 {peer}, [{other_addr}];')
            self.put(f'add.rn.f32 {acc}, {acc}, {peer};')
            self.put(f'st.shared.f32 [{shared_addr}], {acc};')
            self.lines.append(label + ':')
            self.put('bar.sync 0;')
            half //= 2
        self.put(f'ld.shared.f32 {acc}, [{shared_base}];')
        self.put(f'setp.ne.u32 {pred}, {tid}, 0;')
        self.put(f'@{pred} bra DONE;')
        self.put(f'st.global.f32 [{self.pointers[plan.output]}], {acc};')
        self.lines.extend(['DONE:', '    ret;'])
        params = ',\n'.join(f'    .param .u64 param_{name}' for name in k.arg_order)
        registers = '\n'.join(f'    .reg .{dtype} %{kind}<{self.counts[kind]}>;' for kind, dtype in [('p', 'pred'), ('r', 'b32'), ('rd', 'b64'), ('f', 'f32')] if self.counts[kind])
        shared = f'    .shared .align 4 .b8 sum_smem[{plan.warps * 4}];'
        return (f'.version 8.0\n.target sm_89\n.address_size 64\n\n.visible .entry {k.name}(\n{params}\n)\n'
                f'.reqntid {plan.threads}, 1, 1\n{{\n{registers}\n{shared}\n' + '\n'.join(self.lines) + '\n}\n')
