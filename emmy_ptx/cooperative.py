"""Validated per-row cooperative reductions with reusable multi-state combiners.

Scalar/loop semantics are shared with StructuredEmitter. This backend validates
one full-warp row topology before lowering its synchronization and shared tree.
"""
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Write

from .loader import require
from .reduction import conditional_body, integer, variable
from .structured import StructuredEmitter, f32, sequence, valid_name


class CooperativeEmitter(StructuredEmitter):
    def __init__(self, kernel):
        super().__init__(kernel)
        self.dimensions = dict(self.dimensions)
        self.shared = {}

    def check_loop(self, loop, columns, reduction):
        require(isinstance(loop, StridedLoop), 'cooperative rows require StridedLoop')
        axis = loop.axis
        require(isinstance(axis, Axis) and axis.window is None and axis.extent.is_static,
                'cooperative loop axis must be static and unwindowed')
        require(axis.extent.as_static() == columns, 'cooperative loop extent must match the row width')
        require(variable(loop.start, self.co_name) and integer(loop.step, self.threads),
                'cooperative loop must stride from its lane coordinate by the block size')
        require(loop.seed is True and loop.end is None and loop.unroll is False,
                'custom cooperative loop seeds, bounds and unrolling are unsupported')
        allowed = (Let, Load, Assign, Accum) if reduction else (Let, Load, Assign, Write)
        require(all(isinstance(stmt, allowed) for stmt in sequence(loop.body, 'cooperative loop body')),
                'unsupported node inside cooperative row loop')
        return valid_name(axis.name)

    def check_combine(self, node):
        states = sequence(node.state, 'reduction states')
        peers = sequence(node.state_b, 'reduction peer states')
        require(states and len(states) == len(peers), 'state and peer counts must agree')
        for name in states + peers:
            valid_name(name)
        require(len(set(states + peers)) == len(states + peers), 'reduction states and peers must be distinct')
        require(f32(node.dtype), 'cooperative states must be f32')
        assigned = []
        for statement in sequence(node.combine_states, 'reduction combiner'):
            require(isinstance(statement, Assign), 'cooperative combiner supports Assign nodes only')
            if statement.name in states:
                assigned.append(statement.name)
        require(len(assigned) == len(states) and set(assigned) == set(states),
                'combiner must update every reduction state exactly once')
        return states, peers

    def setup_tile(self, tile):
        axes = sequence(tile.axes, 'cooperative Tile axes')
        require(len(axes) == 2 and all(isinstance(axis, Axis) and axis.window is None
                and axis.extent.is_static for axis in axes), 'cooperative Tile needs row and thread axes')
        row, co = axes
        self.row_name, self.co_name = valid_name(row.name), valid_name(co.name)
        require(len({self.row_name, self.co_name, 'lane', 'warp'}) == 4,
                'cooperative coordinates must be distinct')
        rows, self.threads = row.extent.as_static(), co.extent.as_static()
        require(type(rows) is int and rows > 0 and type(self.threads) is int
                and 32 <= self.threads <= 1024 and self.threads & (self.threads - 1) == 0,
                'cooperative block must contain a power-of-two number of full warps')
        require(type(tile.block_threads) is int and tile.block_threads == self.threads,
                'cooperative thread extent must match block_threads')
        require(self.kernel.grid == (rows, 1, 1) and self.kernel.block == (self.threads, 1, 1),
                'cooperative launch requires exactly one full block per row')
        require(tile.aux_threads == 0 and type(tile.aux_threads) is int
                and tile.raster_axes is None and tile.raster_group is None and tile.raster_orient == 'm',
                'cooperative auxiliary/raster scheduling unsupported')
        require(not getattr(self.kernel, 'smem_bytes', 0) and not getattr(self.kernel, 'zero_outputs', ()),
                'cooperative rows use static shared memory and ordinary output stores')
        outputs = [self.dimensions[name] for name in self.kernel.outputs]
        require(outputs and all(len(dims) == 2 and dims[0] == rows for dims in outputs),
                'cooperative fused outputs must be row matrices')
        columns = outputs[0][1]
        require(all(dims[1] == columns for dims in outputs), 'cooperative outputs must have one common row width')
        body = sequence(tile.body, 'cooperative Tile body')
        require(len(body) >= 8 and isinstance(body[1], WarpShuffle), 'unsupported cooperative reduction topology')
        loop_axis = self.check_loop(body[0], columns, True)
        shuffle = body[1]
        states, peers = self.check_combine(shuffle)
        require(type(shuffle.length) is int and shuffle.length == 32, 'only full-warp shuffles supported')
        accumulators = [node for node in body[0].body if isinstance(node, Accum)]
        for accumulator in accumulators:
            valid_name(accumulator.name)
        require(len(accumulators) == len(states) and {node.name for node in accumulators} == set(states),
                'shuffle states must match loop accumulators')
        require(loop_axis not in (self.row_name, self.co_name, 'lane', 'warp'), 'loop coordinate aliases Tile coordinates')
        self.warps = self.threads // 32
        shared_nodes = body[2:2 + len(states)]
        require(len(shared_nodes) == len(states) and all(isinstance(node, Smem) for node in shared_nodes),
                'one shared array is required per reduction state')
        shared_names = tuple(valid_name(node.name) for node in shared_nodes)
        require(len(set(shared_names)) == len(shared_names), 'duplicate cooperative shared binding')
        require(not set(shared_names) & (set(self.kernel.arg_order) | set(states + peers)
                | {self.row_name, self.co_name, loop_axis, 'lane', 'warp'}), 'shared binding aliases a value or coordinate')
        for node in shared_nodes:
            require(node.dtype == 'float' and type(node.align) is int and node.align in (0, 4)
                    and sequence(node.extents, 'shared extents') == (self.warps,)
                    and all(type(extent) is int for extent in node.extents),
                    'shared array must contain one naturally aligned f32 per warp')
        index = 2 + len(states)
        require(len(body) >= index + 4, 'incomplete cooperative topology')
        partial, sync, tree = body[index:index + 3]
        writes = sequence(conditional_body(partial, 'lane'), 'warp partial writes')
        require(len(writes) == len(states), 'warp partial write count differs from state count')
        actual_partials = {}
        for write in writes:
            require(isinstance(write, Write) and f32(write.value_dtype) and write.atomic is False
                    and write.swizzle == 'NONE', 'warp partial must be an ordinary f32 Write')
            indices, values = sequence(write.index, 'partial indices'), sequence(write.values, 'partial values')
            require(len(indices) == 1 and variable(indices[0], 'warp') and len(values) == 1,
                    'warp partial must store one state at its warp index')
            require(isinstance(write.output, str) and write.output not in actual_partials, 'duplicate partial buffer')
            actual_partials[write.output] = values[0]
        require(isinstance(sync, Sync) and type(sync.barrier_id) is int and sync.barrier_id == 0
                and sync.count is None and sync.warp is False, 'full-block Sync(0) is required before the shared tree')
        require(isinstance(tree, TreeHalve) and type(tree.length) is int and tree.length == self.warps
                and tree.tid_var == 'warp' and type(tree.barrier_id) is int and tree.barrier_id == 0
                and tree.barrier_count is None and tree.inner is None, 'unsupported cooperative shared tree contract')
        tree_states, tree_peers = self.check_combine(tree)
        buffers = sequence(tree.bufs, 'tree buffers')
        require(all(isinstance(buffer, str) for buffer in buffers), 'tree buffer names must be strings')
        require(set(tree_states) == set(states) and len(buffers) == len(tree_states)
                and len(set(buffers)) == len(buffers) and set(buffers) == set(shared_names),
                'tree states and shared buffers must match warp partials')
        require(actual_partials == dict(zip(buffers, tree_states)), 'shared partial binding does not match tree state order')
        output_axis = self.check_loop(body[-1], columns, False)
        output_writes = [node for node in body[-1].body if isinstance(node, Write)]
        require(all(isinstance(node.output, str) for node in output_writes), 'output names must be strings')
        require(len(output_writes) == len(self.kernel.outputs)
                and {node.output for node in output_writes} == set(self.kernel.outputs),
                'epilogue must write each output exactly once per row element')
        for write in output_writes:
            indices = sequence(write.index, 'cooperative output indices')
            require(len(indices) == 2 and variable(indices[0], self.row_name)
                    and variable(indices[1], output_axis)
                    and len(sequence(write.values, 'cooperative output values')) == 1,
                    'cooperative output must cover its own row with scalar loop-index stores')
        require(all(isinstance(node, (Load, Assign, Let)) for node in body[index + 3:-1]),
                'unsupported fused reduction epilogue')
        self.partial_node, self.shuffle_node, self.sync_node, self.tree_node = partial, shuffle, sync, tree
        self.shared_nodes = {id(node) for node in shared_nodes}
        # All participating threads are in range: the base guard cannot split a
        # block, so it cannot let any thread skip a subsequent rendezvous.
        super().setup_tile(tile)
        lane, warp = self.reg('r'), self.reg('r')
        co_reg = self.ints[self.co_name].operand
        self.put(f'and.b32 {lane}, {co_reg}, 31;')
        self.put(f'shr.u32 {warp}, {co_reg}, 5;')
        self.ints['lane'] = self.integer_symbol(lane, 0, 31)
        self.ints['warp'] = self.integer_symbol(warp, 0, self.warps - 1)
        self.definitions.update(('lane', 'warp'))

    def prove_output_coverage(self):
        # setup_tile proves one scalar Write per output, indexed by its exact
        # row and strided column coordinates with a complete one-CTA-per-row launch.
        # Shared partial stores are handled by separate synchronization checks.
        return None

    def shared_address(self, name, indices):
        offset = self.flat_index(name, indices, 1)
        byte_offset, address = self.reg('r'), self.reg('r')
        self.put(f'mul.lo.u32 {byte_offset}, {offset.operand}, 4;')
        self.put(f'add.u32 {address}, {self.shared[name]}, {byte_offset};')
        return address

    def write(self, statement):
        if statement.output in self.shared:
            require(f32(statement.value_dtype) and statement.atomic is False and statement.swizzle == 'NONE',
                    'shared partial must be an ordinary f32 Write')
            values = sequence(statement.values, 'shared values')
            require(len(values) == 1, 'shared partial stores must be scalar')
            address = self.shared_address(statement.output, statement.index)
            self.put(f'st.shared.f32 [{address}], {self.value(values[0])};')
        else:
            super().write(statement)

    def combine(self, node, peers):
        outer = self.values, self.definitions
        self.values, self.definitions = dict(self.values), set()
        for name, register in zip(node.state_b, peers):
            require(name not in self.values and name not in self.ints, 'peer binding shadows an existing value')
            self.bind(name, register)
        for statement in node.combine_states:
            self.emit_assign(statement, update=statement.name in node.state)
        self.values, self.definitions = outer

    def warp_shuffle(self, node):
        for delta in (16, 8, 4, 2, 1):
            peers = []
            # Snapshot all peer states before any combiner can update a state.
            for name in node.state:
                bits, shuffled, peer = self.reg('r'), self.reg('r'), self.reg('f')
                self.put(f'mov.b32 {bits}, {self.value(name)};')
                self.put(f'shfl.sync.bfly.b32 {shuffled}, {bits}, {delta}, 31, 0xffffffff;')
                self.put(f'mov.b32 {peer}, {shuffled};')
                peers.append(peer)
            self.combine(node, peers)

    def shared_tree(self, node):
        half = self.warps // 2
        while half:
            label = f'COOP_TREE_DONE_{half}'
            predicate = self.reg('p')
            self.put(f'setp.ne.u32 {predicate}, {self.ints["lane"].operand}, 0;')
            self.put(f'@{predicate} bra {label};')
            self.put(f'setp.ge.u32 {predicate}, {self.ints["warp"].operand}, {half};')
            self.put(f'@{predicate} bra {label};')
            addresses, peers = [], []
            for buffer, state in zip(node.bufs, node.state):
                offset, address, other = self.reg('r'), self.reg('r'), self.reg('r')
                self.put(f'mul.lo.u32 {offset}, {self.ints["warp"].operand}, 4;')
                self.put(f'add.u32 {address}, {self.shared[buffer]}, {offset};')
                self.put(f'add.u32 {other}, {address}, {half * 4};')
                peer = self.reg('f')
                self.put(f'ld.shared.f32 {self.value(state)}, [{address}];')
                self.put(f'ld.shared.f32 {peer}, [{other}];')
                addresses.append(address); peers.append(peer)
            self.combine(node, peers)
            for address, state in zip(addresses, node.state):
                self.put(f'st.shared.f32 [{address}], {self.value(state)};')
            self.lines.append(label + ':')
            self.put('bar.sync 0;')
            half //= 2
        for buffer, state in zip(node.bufs, node.state):
            self.put(f'ld.shared.f32 {self.value(state)}, [{self.shared[buffer]}];')

    def statement(self, statement):
        if isinstance(statement, WarpShuffle):
            require(statement is self.shuffle_node, 'unexpected cooperative shuffle')
            self.warp_shuffle(statement)
        elif isinstance(statement, Smem):
            require(id(statement) in self.shared_nodes, 'unexpected shared declaration')
            symbol = f'coop_shared_{len(self.shared)}'
            self.extra_declarations.append(f'    .shared .align 4 .b8 {symbol}[{self.warps * 4}];')
            register = self.reg('r')
            self.put(f'mov.u32 {register}, {symbol};')
            self.shared[statement.name] = register
            self.dimensions[statement.name] = (self.warps,)
        elif isinstance(statement, Cond):
            require(statement is self.partial_node, 'unexpected cooperative condition')
            predicate = self.predicate(statement.cond)
            self.put(f'@!{predicate} bra COOP_PARTIALS_DONE;')
            for child in statement.body:
                self.statement(child)
            self.lines.append('COOP_PARTIALS_DONE:')
        elif isinstance(statement, Sync):
            require(statement is self.sync_node, 'unexpected cooperative synchronization')
            self.put('bar.sync 0;')
        elif isinstance(statement, TreeHalve):
            require(statement is self.tree_node, 'unexpected cooperative tree')
            self.shared_tree(statement)
        else:
            super().statement(statement)
