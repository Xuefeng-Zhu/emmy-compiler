"""Validated per-row f32 reductions, derived from cooperative Kernel IR.

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
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Write

from .loader import Kernel, require
from .structured import StructuredEmitter, float_literal


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


def combine(stmt, state, operation):
    check(f32(stmt.dtype), 'combine dtype must be f32')
    check(items(stmt.state, 1, 'only one reduction state supported') == (state,), 'combine state does not match accumulator')
    peer, = items(stmt.state_b, 1, 'only one peer state supported')
    check(name(peer) != state, 'peer state must differ from accumulator')
    assign, = items(stmt.combine_states, 1, 'only a single scalar combiner supported')
    check(isinstance(assign, Assign), 'combine must contain Assign')
    check(isinstance(assign.op, ElementwiseImpl) and assign.op.name in ('add', 'sum', 'multiply', 'prod', 'maximum', 'amax', 'minimum')
          and assign.op.reduce_canon == operation, 'combiner must match the accumulator operation')
    check(assign.name == state and items(assign.args, 2, 'reduction combiner needs two operands') == (state, peer), 'combiner operands must be accumulator and peer')
    check(assign.dtype is None or f32(assign.dtype), 'combiner Assign must inherit f32 or declare f32')


def conditional_body(stmt, axis):
    check(isinstance(stmt, Cond), 'expected a conditional write')
    check(items(stmt.else_body, 0, 'else branches are unsupported') == (), 'else branches are unsupported')
    pred = stmt.cond
    check(isinstance(pred, BinaryExpr) and pred.op == '==' and variable(pred.left, axis) and integer(pred.right, 0), 'write must be guarded by ' + axis + ' == 0')
    return stmt.body


def conditional_write(stmt, axis):
    write, = items(conditional_body(stmt, axis), 1, 'conditional body must contain one Write')
    check(isinstance(write, Write) and f32(write.value_dtype), 'conditional body must write f32')
    check(write.atomic is False and write.swizzle == 'NONE', 'atomic or swizzled writes unsupported')
    return write


@dataclass(frozen=True)
class ReductionPlan:
    size: int
    threads: int
    body: tuple
    accumulator: Accum
    operation: str
    tile_axis: str
    loop_axis: str
    output: str
    rows: int = 1
    row_axis: str | None = None
    divisor: str | None = None

    @property
    def warps(self):
        return self.threads // 32


def validate_reduction(kernel: Kernel) -> ReductionPlan:
    tile, = items(kernel.body, 1, 'expected one cooperative Tile')
    threads = tile.block_threads
    check(type(threads) is int and 32 <= threads <= 1024 and threads & (threads - 1) == 0, 'thread count must be a power of two from 32 to 1024')
    check(kernel.block == (threads, 1, 1), 'block must match Tile.block_threads')
    check(tile.aux_threads == 0 and tile.raster_axes is None and tile.raster_group is None and tile.raster_orient == 'm', 'auxiliary/rasterized scheduling unsupported')
    check(isinstance(tile.axes, (tuple, list)) and len(tile.axes) in (1, 2), 'expected thread axis with optional row axis')
    axis = tile.axes[-1]
    row_axis, rows = None, 1
    if len(tile.axes) == 2:
        row = tile.axes[0]
        check(isinstance(row, Axis) and row.window is None and row.extent.is_static, 'row axis must be static and unwindowed')
        rows = row.extent.as_static()
        check(type(rows) is int and 0 < rows < 2**31, 'row extent must be positive')
        row_axis = name(row.name)
    check(kernel.grid == (rows, 1, 1), 'launch requires exactly one block per output row')
    check(isinstance(axis, Axis) and axis.window is None and axis.extent.is_static and axis.extent.as_static() == threads, 'Tile extent must match the participating block')
    tile_axis = name(axis.name)
    check(tile_axis not in ('lane', 'warp', row_axis) and row_axis not in ('lane', 'warp'), 'Tile axes must be distinct from lane and warp')
    check(len(kernel.outputs) == 1, 'reduction requires one output')
    output_name, = kernel.outputs
    loop, shuffle, shared, partial, sync, tree, final = items(tile.body, 7, 'unsupported reduction topology; expected loop/shuffle/shared/write/sync/tree/write')
    check(isinstance(loop, StridedLoop), 'expected StridedLoop')
    check(isinstance(loop.axis, Axis) and loop.axis.window is None and loop.axis.extent.is_static, 'loop needs a static unwindowed axis')
    loop_axis = name(loop.axis.name)
    check(loop_axis not in (tile_axis, row_axis, 'lane', 'warp'), 'loop axis must be distinct from thread coordinates')
    size = loop.axis.extent.as_static()
    check(type(size) is int and 0 < size < 2**31, 'loop extent must be a positive 31-bit integer')
    check(variable(loop.start, tile_axis) and integer(loop.step, threads), 'loop must start at its thread coordinate and stride by block size')
    check(loop.seed is True and loop.end is None and loop.unroll is False, 'custom loop seeds, bounds or unrolling unsupported')
    check(isinstance(loop.body, (tuple, list)) and loop.body, 'reduction loop body must be a nonempty sequence')
    check(all(isinstance(stmt, (Load, Assign, Let, Accum)) for stmt in loop.body),
          'reduction loop supports scalar Loads, Assigns, Lets and one Accum')
    accumulators = [stmt for stmt in loop.body if isinstance(stmt, Accum)]
    accum, = items(accumulators, 1, 'reduction loop must carry exactly one accumulator')
    check(f32(accum.dtype), 'reduction needs an f32 Accum')
    state = name(accum.name)
    check(state not in (tile_axis, row_axis, loop_axis, 'lane', 'warp'), 'accumulator aliases a reduction coordinate')
    check(isinstance(accum.op, ElementwiseImpl) and accum.op.reduce_canon in ('add', 'multiply', 'maximum', 'minimum')
          and accum.op.identity is not None, 'unsupported accumulator operation or identity')
    operation = accum.op.reduce_canon
    check(accum.base is None and items(accum.axes, 1, 'Accum must declare its reduction axis') == (loop_axis,),
          'rescaled or mismatched accumulator unsupported')
    input_dims = (size,) if row_axis is None else (rows, size)
    output_dims = () if row_axis is None else (rows,)
    check(kernel.dimensions[output_name] == output_dims, 'reduction output dimensions must match its Tile axes')
    loaded_names, used_inputs = set(), set()
    full_input = False
    for load in (stmt for stmt in loop.body if isinstance(stmt, Load)):
        check(f32(load.dtype) and isinstance(load.input, str) and load.input in kernel.inputs, 'reduction must load f32 input buffers')
        loaded, = items(load.names, 1, 'only scalar reduction loads supported')
        check(name(loaded) != state, 'Load must not replace the accumulator')
        loaded_names.add(loaded)
        used_inputs.add(load.input)
        dimensions = kernel.dimensions[load.input]
        indices = items(load.index, len(dimensions), 'reduction input index rank mismatch')
        if (dimensions == input_dims and variable(indices[-1], loop_axis)
                and (row_axis is None or variable(indices[0], row_axis))):
            full_input = True
        else:
            check(dimensions == (1,) and integer(indices[0], 0), 'reduction input dimensions must match its axes or be a scalar buffer')
    check(full_input, 'reduction must load at least one input matching its loop domain')
    check(isinstance(shuffle, WarpShuffle) and type(shuffle.length) is int and shuffle.length == 32, 'only full-warp shuffle reductions supported')
    combine(shuffle, state, operation)
    check(isinstance(shared, Smem) and shared.dtype == 'float' and type(shared.align) is int and shared.align in (0, 4), 'shared buffer must have natural f32 alignment')
    shared_name = name(shared.name)
    check(shared_name not in kernel.shapes and shared_name not in (tile_axis, row_axis, loop_axis, 'lane', 'warp', state) and shared_name not in loaded_names, 'shared buffer name aliases another binding')
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
    combine(tree, state, operation)
    epilogue = conditional_body(final, tile_axis)
    check(isinstance(epilogue, (tuple, list)) and len(epilogue) in (1, 3), 'reduction epilogue must write directly or load/divide/write')
    divisor, result = None, state
    if len(epilogue) == 3:
        denominator, division, output_write = epilogue
        check(isinstance(denominator, Load) and f32(denominator.dtype) and isinstance(denominator.input, str) and denominator.input in kernel.inputs, 'division epilogue must load an f32 divisor buffer')
        divisor = denominator.input
        check(divisor not in used_inputs and kernel.dimensions[divisor] == (1,), 'divisor must be a separate one-element input buffer')
        denominator_name, = items(denominator.names, 1, 'divisor load must produce one value')
        check(name(denominator_name) != state and denominator_name not in loaded_names, 'divisor load shadows a reduction binding')
        denominator_index, = items(denominator.index, 1, 'divisor load needs one index')
        check(integer(denominator_index, 0), 'divisor must be loaded at index zero')
        check(isinstance(division, Assign) and f32(division.dtype) and isinstance(division.op, ElementwiseImpl) and division.op.name == 'divide', 'epilogue must divide the sum by its loaded divisor')
        check(items(division.args, 2, 'divide requires two operands') == (state, denominator_name), 'divide operands must be sum and divisor')
        result = name(division.name)
        check(result not in (state, denominator_name) and result not in loaded_names, 'divide result shadows an existing binding')
    else:
        output_write, = epilogue
    expected_inputs = frozenset(used_inputs | ({divisor} if divisor else set()))
    check(kernel.inputs == expected_inputs, 'unused or unsupported reduction inputs')
    check(isinstance(output_write, Write) and f32(output_write.value_dtype) and output_write.atomic is False and output_write.swizzle == 'NONE', 'reduction output must be an ordinary f32 Write')
    output_indices = items(output_write.index, len(output_dims), 'reduction output index rank mismatch')
    check(row_axis is None or variable(output_indices[0], row_axis), 'row result must use its Tile row index')
    check(output_write.output == output_name and items(output_write.values, 1, 'reduction output needs one value') == (result,), 'only the row leader may store its final reduction result')
    return ReductionPlan(size, threads, tuple(loop.body), accum, operation, tile_axis, loop_axis, output_name, rows, row_axis, divisor)


class ReductionEmitter(StructuredEmitter):
    def run(self):
        k = self.kernel
        plan = validate_reduction(k)
        for buffer in k.arg_order:
            param, ptr = self.reg('rd'), self.reg('rd')
            self.put(f'ld.param.u64 {param}, [param_{buffer}];')
            self.put(f'cvta.to.global.u64 {ptr}, {param};')
            self.pointers[buffer] = ptr
        tid, lane, warp, index = [self.reg('r') for _ in range(4)]
        acc, peer = [self.reg('f') for _ in range(2)]
        pred = self.reg('p')
        self.put(f'mov.u32 {tid}, %tid.x;')
        self.put(f'and.b32 {lane}, {tid}, 31;')
        self.put(f'shr.u32 {warp}, {tid}, 5;')
        # Identity initialization precedes the loop, including every idle lane.
        self.put(f'mov.f32 {acc}, {float_literal(plan.accumulator.op.identity)};')
        self.ints[plan.tile_axis] = self.integer_symbol(tid, 0, plan.threads - 1)
        self.ints[plan.loop_axis] = self.integer_symbol(index, 0, plan.size - 1)
        self.put(f'mov.u32 {index}, {tid};')
        row = None
        if plan.row_axis is not None:
            row = self.reg('r')
            self.put(f'mov.u32 {row}, %ctaid.x;')
            self.ints[plan.row_axis] = self.integer_symbol(row, 0, plan.rows - 1)
        self.definitions.update(self.ints)
        self.bind(plan.accumulator.name, acc)
        self.lines.append('SUM_LOOP:')
        self.put(f'setp.ge.u32 {pred}, {index}, {plan.size};')
        self.put(f'@{pred} bra SUM_LOOP_DONE;')
        for statement in plan.body:
            if isinstance(statement, Accum):
                self.emit_math(plan.operation, [acc, self.value(statement.value)], acc)
            else:
                self.statement(statement)
        self.put(f'add.u32 {index}, {index}, {plan.threads};')
        self.put('bra SUM_LOOP;')
        self.lines.append('SUM_LOOP_DONE:')
        bits, shuffled = self.reg('r'), self.reg('r')
        for delta in (16, 8, 4, 2, 1):
            self.put(f'mov.b32 {bits}, {acc};')
            self.put(f'shfl.sync.bfly.b32 {shuffled}, {bits}, {delta}, 31, 0xffffffff;')
            self.put(f'mov.b32 {peer}, {shuffled};')
            self.emit_math(plan.operation, [acc, peer], acc)
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
            self.emit_math(plan.operation, [acc, peer], acc)
            self.put(f'st.shared.f32 [{shared_addr}], {acc};')
            self.lines.append(label + ':')
            self.put('bar.sync 0;')
            half //= 2
        self.put(f'ld.shared.f32 {acc}, [{shared_base}];')
        self.put(f'setp.ne.u32 {pred}, {tid}, 0;')
        self.put(f'@{pred} bra DONE;')
        if plan.divisor is not None:
            denominator = self.reg('f')
            self.put(f'ld.global.f32 {denominator}, [{self.pointers[plan.divisor]}];')
            self.put(f'div.rn.f32 {acc}, {acc}, {denominator};')
        output_ptr = self.pointers[plan.output]
        if row is not None:
            output_offset, output_address = self.reg('rd'), self.reg('rd')
            self.put(f'mul.wide.u32 {output_offset}, {row}, 4;')
            self.put(f'add.u64 {output_address}, {output_ptr}, {output_offset};')
            output_ptr = output_address
        self.put(f'st.global.f32 [{output_ptr}], {acc};')
        self.lines.extend(['DONE:', '    ret;'])
        params = ',\n'.join(f'    .param .u64 param_{name}' for name in k.arg_order)
        registers = '\n'.join(f'    .reg .{dtype} %{kind}<{self.counts[kind]}>;' for kind, dtype in [('p', 'pred'), ('r', 'b32'), ('rd', 'b64'), ('f', 'f32')] if self.counts[kind])
        shared = f'    .shared .align 4 .b8 sum_smem[{plan.warps * 4}];'
        return (f'.version 8.0\n.target sm_89\n.address_size 64\n\n.visible .entry {k.name}(\n{params}\n)\n'
                f'.reqntid {plan.threads}, 1, 1\n{{\n{registers}\n{shared}\n' + '\n'.join(self.lines) + '\n}\n')
