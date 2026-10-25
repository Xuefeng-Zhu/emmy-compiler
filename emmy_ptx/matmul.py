"""Direct lowering of static shared-memory scalar contractions.

The IR supplies the loops, copy stages and arithmetic. No problem identifier or
matrix dimensions are embedded here. Asynchronous copies preserve the issued
commit/wait order; this is not a proof of arbitrary shared-memory race freedom.
"""
from dataclasses import replace
from itertools import product
from math import gcd, prod

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Write

from .loader import require
from .structured import StructuredEmitter, f32, sequence, valid_name


# Exact bounded domain reasoning improves conservative interval checks for ring
# indices and clamped prefetches. Larger domains retain the base interval proof.
DOMAIN_LIMIT = 262144


def children(expression):
    if isinstance(expression, BinaryExpr):
        return expression.left, expression.right
    if isinstance(expression, TernaryExpr):
        return expression.cond, expression.if_true, expression.if_false
    return ()


def variables(expression):
    if isinstance(expression, Var):
        return {expression.name}
    return set().union(*(variables(child) for child in children(expression)))


def evaluate_index(expression, environment):
    """Our integer expression evaluator, used only for finite-domain proofs."""
    if isinstance(expression, Var):
        return environment[expression.name]
    if isinstance(expression, Literal):
        require(expression.dtype == 'int' and type(expression.value) is int,
                'shared-memory index literals must be integers')
        return expression.value
    if isinstance(expression, TernaryExpr):
        selected = expression.if_true if evaluate_index(expression.cond, environment) else expression.if_false
        return evaluate_index(selected, environment)
    require(isinstance(expression, BinaryExpr), 'unsupported shared-memory index expression')
    a, b = evaluate_index(expression.left, environment), evaluate_index(expression.right, environment)
    operations = {'+': lambda: a + b, '-': lambda: a - b, '*': lambda: a * b,
                  '/': lambda: a // b, '//': lambda: a // b, '%': lambda: a % b,
                  '<': lambda: a < b, '<=': lambda: a <= b, '>': lambda: a > b,
                  '>=': lambda: a >= b, '==': lambda: a == b, '!=': lambda: a != b}
    require(expression.op in operations, 'unsupported shared-memory index operator')
    if expression.op in ('/', '//', '%'):
        require(a >= 0 and b > 0, 'index division requires nonnegative dividend and positive divisor')
    return operations[expression.op]()


class MatmulEmitter(StructuredEmitter):
    """Extend structured f32 loops with shared storage, async copies and atomics."""

    def __init__(self, kernel):
        super().__init__(kernel)
        # The base may reference Kernel.dimensions directly. Never mutate it.
        self.dimensions = dict(self.dimensions)
        self.shared = {}
        self.shared_pointers = {}
        self.shared_declared = set()
        self.index_domains = {}
        self.domain_cache = {}
        self.uniform_axes = set()
        self.uniform_control = True
        self.async_groups = 0
        self.uncommitted_copies = False
        self.async_needs_wait = False
        self.shared_ready = False
        self.pending_destinations = []
        self.copy_scope = ()
        self.next_copy_scope = 0

    def possible_values(self, expression):
        key = repr(expression)
        if key in self.domain_cache:
            return self.domain_cache[key]
        names = sorted(variables(expression))
        if any(name not in self.index_domains for name in names):
            return None
        domains = [self.index_domains[name] for name in names]
        if prod(len(domain) for domain in domains) > DOMAIN_LIMIT:
            return None
        values = frozenset(evaluate_index(expression, dict(zip(names, values)))
                           for values in product(*domains))
        self.domain_cache[key] = values
        return values

    def index(self, expression):
        result = super().index(expression)
        values = self.possible_values(expression)
        if values:
            result = replace(result, lower=min(values), upper=max(values))
        return result

    def setup_tile(self, tile):
        super().setup_tile(tile)
        require(tile.block_threads is not None, 'shared-memory schedules require an explicit cooperative Tile')
        require(self.n == self.kernel.grid[0] * self.kernel.block[0],
                'shared-memory schedules require an exact launch of complete CTAs')
        extent_list = [axis.extent.as_static() for axis in tile.axes]
        for i, axis in enumerate(tile.axes):
            extent = extent_list[i]
            self.index_domains[axis.name] = range(extent)
            stride = prod(extent_list[i + 1:])
            if extent == 1 or (stride >= self.kernel.block[0] and stride % self.kernel.block[0] == 0):
                self.uniform_axes.add(axis.name)
        require(tile.raster_group is None,
                'shared-memory grouped raster scheduling is not yet validated')
        declarations = [stmt for stmt in sequence(tile.body, 'Tile body') if isinstance(stmt, Smem)]
        require(declarations, 'shared-memory lowering requires Smem declarations')
        cursor, pool_alignment = 0, 4
        for stmt in declarations:
            name = valid_name(stmt.name)
            require(name not in self.shared and name not in self.dimensions and name not in self.ints,
                    'duplicate or colliding shared buffer')
            extents = sequence(stmt.extents, 'shared extents')
            require(extents and all(type(extent) is int and 0 < extent < 2**31 for extent in extents),
                    'shared extents must be positive static integers')
            require(stmt.dtype in ('float', 'f32'), 'only f32 shared buffers supported')
            require(type(stmt.align) is int and 0 <= stmt.align <= 128
                    and (stmt.align == 0 or stmt.align & (stmt.align - 1) == 0),
                    'shared alignment must be zero or a power of two up to 128')
            alignment = max(4, stmt.align)
            cursor = (cursor + alignment - 1) // alignment * alignment
            size = prod(extents) * 4
            require(size < 2**31 and cursor + size <= 99 * 1024,
                    'shared allocation exceeds supported sm_89 per-block storage')
            self.shared[name] = (stmt, cursor, size, alignment)
            self.dimensions[name] = extents
            cursor += size
            pool_alignment = max(pool_alignment, alignment)
        dynamic_bytes = getattr(self.kernel, 'smem_bytes', 0)
        require(type(dynamic_bytes) is int and dynamic_bytes >= 0, 'invalid dynamic shared-memory size')
        if dynamic_bytes:
            require(dynamic_bytes == cursor, 'dynamic shared-memory size does not match packed Smem declarations')
            self.module_declarations.append(f'.extern .shared .align {pool_alignment} .b8 shared_pool[];')
            pool = self.reg('r')
            self.put(f'mov.u32 {pool}, shared_pool;')
        else:
            require(cursor <= 48 * 1024, 'shared allocation above 48 KiB requires dynamic shared metadata')
        for i, (name, (_, offset, size, alignment)) in enumerate(self.shared.items()):
            pointer = self.reg('r')
            if dynamic_bytes:
                self.put(f'add.u32 {pointer}, {pool}, {offset};')
            else:
                symbol = f'shared_{i}'
                self.extra_declarations.append(f'    .shared .align {alignment} .b8 {symbol}[{size}];')
                self.put(f'mov.u32 {pointer}, {symbol};')
            self.shared_pointers[name] = pointer

    def shared_address(self, buffer, indices, width):
        require(buffer in self.shared_declared, f'shared buffer used before declaration: {buffer}')
        offset = self.flat_index(buffer, indices, width)
        byte_offset, address = self.reg('r'), self.reg('r')
        self.put(f'mul.lo.u32 {byte_offset}, {offset.operand}, 4;')
        self.put(f'add.u32 {address}, {self.shared_pointers[buffer]}, {byte_offset};')
        return address

    def guaranteed_multiple(self, expression):
        """Return a divisor of every possible index value; zero means always zero."""
        if isinstance(expression, Literal):
            require(expression.dtype == 'int' and type(expression.value) is int, 'index literal must be integer')
            return abs(expression.value)
        if isinstance(expression, Var):
            domain = self.index_domains.get(expression.name)
            if domain is None:
                return 1
            if isinstance(domain, range):
                return gcd(domain.start, domain.step) if len(domain) > 1 else abs(domain.start)
            result = 0
            for value in domain:
                result = gcd(result, value)
            return result
        if isinstance(expression, TernaryExpr):
            return gcd(self.guaranteed_multiple(expression.if_true), self.guaranteed_multiple(expression.if_false))
        if isinstance(expression, BinaryExpr):
            left, right = self.guaranteed_multiple(expression.left), self.guaranteed_multiple(expression.right)
            if expression.op in ('+', '-'):
                return gcd(left, right)
            if expression.op == '*':
                return left * right
            if isinstance(expression.right, Literal) and type(expression.right.value) is int and expression.right.value > 0:
                divisor = expression.right.value
                if expression.op == '%':
                    return gcd(left, divisor)
                if expression.op in ('/', '//') and left % divisor == 0:
                    return left // divisor
        return 1

    def aligned(self, buffer, indices, nbytes):
        dimensions = self.dimensions[buffer]
        require(len(indices) == len(dimensions), 'copy index rank mismatch')
        multiple = 0
        for i, expression in enumerate(indices):
            multiple = gcd(multiple, self.guaranteed_multiple(expression) * prod(dimensions[i + 1:]))
        require(multiple * 4 % nbytes == 0, f'{buffer}: cannot prove {nbytes}-byte copy alignment')
        if buffer in self.shared:
            require(self.shared[buffer][3] >= nbytes, f'{buffer}: declared shared alignment is too small for copy')

    def check_pending_destination(self, buffer, indices, width):
        """Reject proven same-thread overlap within an uncommitted copy group.

        Compare effective flattened destinations, including lane coordinates.
        Only copies in the same lexical loop scope are compared: the general
        cross-iteration, cross-thread and inter-group race proof is not implied.
        """
        offset = Literal(0, 'int')
        for index, extent in zip(indices, self.dimensions[buffer]):
            offset = BinaryExpr('+', BinaryExpr('*', offset, Literal(extent, 'int')), index)
        for old_buffer, old_offset, old_width, old_scope in self.pending_destinations:
            if old_buffer != buffer or old_scope != self.copy_scope:
                continue
            differences = (frozenset({0}) if offset == old_offset else
                           self.possible_values(BinaryExpr('-', offset, old_offset)))
            require(not differences or not all(-width < value < old_width for value in differences),
                    'overlapping async copy destinations within one uncommitted group')
        self.pending_destinations.append((buffer, offset, width, self.copy_scope))

    def copy(self, statement):
        require(isinstance(statement.smem, str) and isinstance(statement.src, str), 'async buffers must be names')
        require(statement.smem in self.shared and statement.src in self.kernel.inputs,
                'async copy requires a declared shared destination and global input source')
        require(type(statement.nbytes) is int and statement.nbytes in (4, 8, 16),
                'async copy size must be 4, 8 or 16 bytes')
        require(statement.swizzle == 'NONE', 'swizzled async copies are unsupported')
        destination = sequence(statement.smem_index, 'async shared indices')
        source = sequence(statement.src_index, 'async source indices')
        if statement.lane_index is not None:
            lane = sequence(statement.lane_index, 'async lane indices')
            require(len(destination) == len(lane) == 2 and type(statement.lane_rows) is int and statement.lane_rows > 0,
                    'async lane stripe requires two coordinates and a positive row count')
            lane_row, lane_col = self.index(lane[0]), self.index(lane[1])
            require(0 <= lane_row.lower <= lane_row.upper < statement.lane_rows
                    and 0 <= lane_col.lower <= lane_col.upper < self.dimensions[statement.smem][-1],
                    'async lane coordinate exceeds its copy stripe')
            destination = tuple(BinaryExpr('+', base, offset) for base, offset in zip(destination, lane))
        else:
            require(statement.lane_rows == 0, 'async lane_rows requires lane_index')
        self.aligned(statement.smem, destination, statement.nbytes)
        self.aligned(statement.src, source, statement.nbytes)
        dst = self.shared_address(statement.smem, destination, statement.nbytes // 4)
        src = self.address(statement.src, source, statement.nbytes // 4)
        self.check_pending_destination(statement.smem, destination, statement.nbytes // 4)
        cache = 'cg' if statement.nbytes == 16 else 'ca'
        self.put(f'cp.async.{cache}.shared.global [{dst}], [{src}], {statement.nbytes};')
        self.uncommitted_copies = True
        self.async_needs_wait = True
        self.shared_ready = False

    def load(self, statement):
        if statement.input not in self.shared:
            return super().load(statement)
        require(self.shared_ready, 'shared Load requires a completed-copy CTA barrier')
        require(f32(statement.dtype, inherited=True), 'shared Load must be f32 or inherit its buffer dtype')
        names = sequence(statement.names, 'shared Load names')
        address = self.shared_address(statement.input, statement.index, len(names))
        for i, name in enumerate(names):
            destination = self.reg('f')
            self.bind(name, destination)
            self.put(f'ld.shared.f32 {destination}, [{address}+{4 * i}];')

    def write(self, statement):
        if statement.output in self.shared:
            require(f32(statement.value_dtype, inherited=True) and statement.atomic is False and statement.swizzle == 'NONE',
                    'shared Write requires plain f32 values')
            operands = [self.value(name) for name in sequence(statement.values, 'shared Write values')]
            address = self.shared_address(statement.output, statement.index, len(operands))
            for i, operand in enumerate(operands):
                self.put(f'st.shared.f32 [{address}+{4 * i}], {operand};')
            self.shared_ready = False
            return
        if statement.atomic is False:
            return super().write(statement)
        require(statement.atomic is True and statement.output in self.kernel.outputs,
                'atomic Write must target a global output')
        require(f32(statement.value_dtype) and statement.swizzle == 'NONE', 'atomic Write supports plain f32 only')
        require(statement.output in getattr(self.kernel, 'zero_outputs', frozenset()),
                'atomic-add output requires the host zero_outputs contract')
        operands = [self.value(name) for name in sequence(statement.values, 'atomic Write values')]
        address = self.address(statement.output, statement.index, len(operands))
        for i, operand in enumerate(operands):
            self.put(f'red.global.add.f32 [{address}+{4 * i}], {operand};')
        self.written.add(statement.output)

    def loop(self, statement):
        require(isinstance(statement.axis, Axis) and statement.axis.extent.is_static,
                'shared-memory loops require a static axis')
        valid_name(statement.axis.name)
        require(statement.end is None if isinstance(statement, StridedLoop) else True,
                'shared-memory loops currently require their declared extent as the end')
        axis = statement.axis.name
        extent = statement.axis.extent.as_static()
        require(type(extent) is int and 0 < extent < 2**31, 'shared-memory loop extent must be positive')
        if isinstance(statement, StridedLoop):
            starts = self.possible_values(statement.start)
            step_expr = Literal(statement.step, 'int') if type(statement.step) is int else statement.step
            steps = self.possible_values(step_expr)
            require(steps is not None and len(steps) == 1 and next(iter(steps)) > 0,
                    'shared-memory loops require a positive constant step')
            step = next(iter(steps))
            require(starts is not None and all(start >= 0 for start in starts),
                    'shared-memory loop start domain is unsupported')
            count = sum(max(0, (extent - start + step - 1) // step) for start in starts)
            if count <= DOMAIN_LIMIT:
                domain = frozenset(value for start in starts for value in range(start, extent, step))
            else:
                domain = range(extent)
            uniform = variables(statement.start) <= self.uniform_axes and variables(step_expr) <= self.uniform_axes
        else:
            domain = range(extent)
            uniform = True
        require(domain, 'shared-memory loop has no iterations for any thread')
        previous = self.index_domains.get(axis)
        old_control, old_uniform = self.uniform_control, set(self.uniform_axes)
        groups_before = self.async_groups
        previous_scope = self.copy_scope
        self.next_copy_scope += 1
        self.copy_scope += (self.next_copy_scope,)
        self.index_domains[axis] = domain
        self.domain_cache.clear()
        self.uniform_control = self.uniform_control and uniform
        if uniform:
            self.uniform_axes.add(axis)
        try:
            super().loop(statement)
            require(self.async_groups <= groups_before,
                    'async loop may accumulate unbounded committed groups; add a wait inside the loop')
        finally:
            self.copy_scope = previous_scope
            if previous is None:
                self.index_domains.pop(axis, None)
            else:
                self.index_domains[axis] = previous
            self.domain_cache.clear()
            self.uniform_control, self.uniform_axes = old_control, old_uniform

    def prove_output_coverage(self):
        kinds = {name: set() for name in self.kernel.outputs}
        def visit(body):
            for statement in body:
                if isinstance(statement, Write) and statement.output in kinds:
                    kinds[statement.output].add(statement.atomic)
                for field in ('body', 'else_body'):
                    nested = getattr(statement, field, ())
                    if nested:
                        visit(nested)
        visit(self.kernel.body)
        for name, counts in self.output_write_counts(allow_atomic=True).items():
            require(len(kinds[name]) == 1, 'mixed atomic and ordinary output writes are unsupported')
            if kinds[name] == {True}:
                require(name in getattr(self.kernel, 'zero_outputs', frozenset()),
                        'atomic output coverage requires host zero_outputs')
                require(counts and counts[0] > 0 and all(count == counts[0] for count in counts),
                        'atomic output writes must cover every element with equal contribution counts')
            else:
                require(all(count == 1 for count in counts),
                        'ordinary output writes must cover every element exactly once')
        require(not self.uncommitted_copies, 'async copies must be committed before kernel completion')
        # A pipelined IR may leave redundant prefetch groups outstanding after
        # its last consumed stage. Drain them before return; outputs are unchanged.
        if self.async_groups:
            self.put('cp.async.wait_group 0;')
            self.async_groups = 0

    def statement(self, statement):
        if isinstance(statement, Smem):
            require(statement.name in self.shared and self.shared[statement.name][0] == statement,
                    'shared declarations must be immediate children of the Tile')
            require(statement.name not in self.shared_declared, 'duplicate shared declaration')
            self.shared_declared.add(statement.name)
        elif isinstance(statement, CpAsyncCopy):
            self.copy(statement)
        elif isinstance(statement, CpAsyncCommit):
            require(self.uniform_control, 'async group commits require CTA-uniform control')
            self.put('cp.async.commit_group;')
            self.async_groups += 1
            require(self.async_groups <= 8, 'too many outstanding async copy groups')
            self.uncommitted_copies = False
            self.pending_destinations.clear()
        elif isinstance(statement, CpAsyncWait):
            require(self.uniform_control, 'async waits require CTA-uniform control')
            require(type(statement.group) is int and 0 <= statement.group <= 7, 'async wait group must be an integer from 0 to 7')
            require(not self.uncommitted_copies, 'async copies must be committed before wait_group')
            self.put(f'cp.async.wait_group {statement.group};')
            self.async_groups = min(self.async_groups, statement.group)
            self.async_needs_wait = False
        elif isinstance(statement, Sync):
            require(self.uniform_control, 'CTA barrier inside divergent loop control is unsupported')
            require(type(statement.barrier_id) is int and statement.barrier_id == 0
                    and statement.count is None and statement.warp is False,
                    'only full-CTA barrier zero is supported')
            require(not self.uncommitted_copies, 'uncommitted async copies before shared-memory barrier')
            require(not self.async_needs_wait, 'async copies require wait_group before a shared-memory barrier')
            self.put('bar.sync 0;')
            self.shared_ready = True
        else:
            return super().statement(statement)
