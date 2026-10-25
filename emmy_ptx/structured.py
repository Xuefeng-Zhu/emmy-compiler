"""Direct PTX lowering for static scalar/vector statements and structured loops.

The schedule, accumulator lifetimes and index expressions come from typed Emmy
IR. This module does not invoke Emmy's renderer. Shared/cooperative statements
are deliberately extension points; the base emitter rejects them.
"""
from dataclasses import dataclass
from array import array
from itertools import product
from math import isnan, prod
import struct

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write

from .errors import CompileError
from .loader import require
from .ptx import Emitter


def sequence(value, description):
    require(isinstance(value, (tuple, list)), description + ' must be a sequence')
    return tuple(value)


def valid_name(value):
    require(isinstance(value, str) and bool(value), 'binding name must be a nonempty string')
    return value


def f32(dtype, inherited=False):
    return (inherited and dtype is None) or getattr(dtype, 'name', None) == 'f32'


def float_literal(value):
    require(type(value) in (int, float), 'f32 literal must be numeric')
    try:
        require(not isnan(value), 'NaN literals are unsupported')
        return '0f' + struct.pack('>f', value).hex()
    except (OverflowError, struct.error) as exc:
        raise CompileError('literal exceeds f32 range') from exc


@dataclass(frozen=True)
class IntValue:
    operand: str
    lower: int
    upper: int
    # Affine provenance cancels correlated terms before bounds are checked.
    terms: tuple[tuple[str, int], ...] | None = None
    constant: int = 0


class StructuredEmitter(Emitter):
    def __init__(self, kernel):
        super().__init__(kernel)
        dimensions = getattr(kernel, 'dimensions', None) or {
            name: (() if name in kernel.scalar_outputs else (size,))
            for name, size in kernel.shapes.items()
        }
        self.dimensions = {name: tuple(shape) for name, shape in dimensions.items()}
        self.ints = {}
        self.symbol_bounds = {}
        self.symbol_steps = {}
        self.definitions = set()
        self.mutable = set()
        self.accum_context = {}
        self.extra_declarations = []
        self.module_declarations = []
        self.label_count = 0
        self.symbol_count = 0

    def bind(self, name, value, shadow_state=False):
        valid_name(name)
        require(name not in self.definitions and name not in self.ints,
                f'duplicate or colliding SSA definition: {name}')
        require(shadow_state or name not in self.mutable,
                f'ordinary SSA definition cannot replace carried state: {name}')
        self.definitions.add(name)
        self.values[name] = value

    def integer_constant(self, value):
        require(type(value) is int and -(2**31) <= value < 2**31,
                'integer constant exceeds supported signed 32-bit range')
        return IntValue(str(value), value, value, (), value)

    def integer_symbol(self, register, lower, upper, step=1):
        require(0 <= lower <= upper < 2**31, 'invalid integer coordinate range')
        key = f'axis_{self.symbol_count}'
        self.symbol_count += 1
        self.symbol_bounds[key] = (lower, upper)
        self.symbol_steps[key] = (lower, step)
        return IntValue(register, lower, upper, ((key, 1),))

    def binary_integer(self, op, left, right):
        require(op in ('+', '-', '*', '/', '//', '%'), f'unsupported integer operation: {op}')
        terms, constant = None, 0
        if op in ('+', '-'):
            sign = 1 if op == '+' else -1
            low = left.lower + (right.lower if sign == 1 else -right.upper)
            high = left.upper + (right.upper if sign == 1 else -right.lower)
            if left.terms is not None and right.terms is not None:
                result = dict(left.terms)
                for key, value in right.terms:
                    result[key] = result.get(key, 0) + sign * value
                terms = tuple(sorted((k, v) for k, v in result.items() if v))
                constant = left.constant + sign * right.constant
        elif op == '*':
            products = [a * b for a in (left.lower, left.upper) for b in (right.lower, right.upper)]
            low, high = min(products), max(products)
            if left.lower == left.upper and right.terms is not None:
                terms = tuple((k, v * left.lower) for k, v in right.terms if v * left.lower)
                constant = right.constant * left.lower
            elif right.lower == right.upper and left.terms is not None:
                terms = tuple((k, v * right.lower) for k, v in left.terms if v * right.lower)
                constant = left.constant * right.lower
        else:
            require(left.lower >= 0 and right.lower > 0,
                    'integer division/remainder needs a nonnegative dividend and positive divisor')
            if op in ('/', '//'):
                low, high = left.lower // right.upper, left.upper // right.lower
            else:
                low, high = 0, min(left.upper, right.upper - 1)
            if left.lower == left.upper and right.lower == right.upper:
                constant = left.lower // right.lower if op in ('/', '//') else left.lower % right.lower
                low = high = constant
                terms = ()
        if terms is not None:
            low = high = constant
            for key, coefficient in terms:
                lo, hi = self.symbol_bounds[key]
                low += coefficient * (lo if coefficient >= 0 else hi)
                high += coefficient * (hi if coefficient >= 0 else lo)
        require(-(2**31) <= low <= high < 2**31, 'integer arithmetic may overflow signed 32-bit range')
        result = self.reg('r')
        instruction = {'+': 'add', '-': 'sub', '*': 'mul.lo', '/': 'div', '//': 'div', '%': 'rem'}[op]
        self.put(f'{instruction}.s32 {result}, {left.operand}, {right.operand};')
        return IntValue(result, low, high, terms, constant)

    def predicate(self, expression):
        require(isinstance(expression, BinaryExpr) and expression.op in ('<', '<=', '>', '>=', '==', '!='),
                'only integer comparison predicates are supported')
        left, right = self.index(expression.left), self.index(expression.right)
        condition = {'<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge', '==': 'eq', '!=': 'ne'}[expression.op]
        result = self.reg('p')
        self.put(f'setp.{condition}.s32 {result}, {left.operand}, {right.operand};')
        return result

    def refined_bounds(self, expression, truth):
        """Narrow one affine coordinate under an integer comparison, if possible.

        This proves clamps on stepped loop coordinates without enumerating shape
        values. A condition we cannot prove leaves the interval unchanged.
        """
        bounds = dict(self.symbol_bounds)
        if expression.op not in ('<', '<=', '>', '>='):
            return bounds
        left, right = self.index(expression.left), self.index(expression.right)
        op = expression.op
        if left.lower == left.upper and right.terms is not None:
            left, right = right, left
            op = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}[op]
        if left.terms is None or len(left.terms) != 1 or right.lower != right.upper:
            return bounds
        key, coefficient = left.terms[0]
        limit = right.lower - left.constant
        if coefficient < 0:
            coefficient, limit = -coefficient, -limit
            op = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}[op]
        if not truth:
            op = {'<': '>=', '<=': '>', '>': '<=', '>=': '<'}[op]
        low, high = bounds[key]
        if op == '<':
            high = min(high, (limit - 1) // coefficient)
        elif op == '<=':
            high = min(high, limit // coefficient)
        elif op == '>':
            low = max(low, -(-(limit + 1) // coefficient))
        else:
            low = max(low, -(-limit // coefficient))
        origin, step = self.symbol_steps[key]
        low = origin + (-(-(low - origin) // step)) * step
        high = origin + ((high - origin) // step) * step
        if low > high:
            return None
        bounds[key] = (low, high)
        return bounds

    def index(self, expression):
        if isinstance(expression, Var):
            valid_name(expression.name)
            require(expression.name in self.ints, f'undefined integer variable: {expression.name}')
            value = self.ints[expression.name]
            if value.terms is not None:
                low = high = value.constant
                for key, coefficient in value.terms:
                    lo, hi = self.symbol_bounds[key]
                    low += coefficient * (lo if coefficient >= 0 else hi)
                    high += coefficient * (hi if coefficient >= 0 else lo)
                return IntValue(value.operand, low, high, value.terms, value.constant)
            return value
        if isinstance(expression, Literal):
            require(expression.dtype == 'int', 'index literals must have integer dtype')
            return self.integer_constant(expression.value)
        if isinstance(expression, BinaryExpr):
            return self.binary_integer(expression.op, self.index(expression.left), self.index(expression.right))
        if isinstance(expression, TernaryExpr):
            pred = self.predicate(expression.cond)
            original = self.symbol_bounds
            true_bounds = self.refined_bounds(expression.cond, True)
            false_bounds = self.refined_bounds(expression.cond, False)
            self.symbol_bounds = true_bounds if true_bounds is not None else original
            left = self.index(expression.if_true) if true_bounds is not None else None
            self.symbol_bounds = false_bounds if false_bounds is not None else original
            right = self.index(expression.if_false) if false_bounds is not None else None
            self.symbol_bounds = original
            if left is None:
                return right
            if right is None:
                return left
            result = self.reg('r')
            self.put(f'selp.s32 {result}, {left.operand}, {right.operand}, {pred};')
            return IntValue(result, min(left.lower, right.lower), max(left.upper, right.upper))
        raise CompileError(f'unsupported integer expression: {type(expression).__name__}')

    def flat_index(self, buffer, indices, width):
        """Return a bounded row-major element offset, for global or shared users."""
        require(isinstance(buffer, str) and buffer in self.dimensions,
                f'unknown buffer: {buffer}')
        indices = sequence(indices, 'memory indices')
        dimensions = self.dimensions[buffer]
        require(len(indices) == len(dimensions), f'{buffer}: index rank does not match buffer rank')
        require(type(width) is int and width in (1, 2, 4), 'only scalar, pair and four-element accesses supported')
        require(bool(dimensions) or width == 1, 'scalar buffers cannot have vector accesses')
        offset = self.integer_constant(0)
        for i, (expression, extent) in enumerate(zip(indices, dimensions)):
            index = self.index(expression)
            extra = width if i == len(dimensions) - 1 else 1
            require(0 <= index.lower and index.upper + extra <= extent,
                    f'out-of-bounds access to {buffer} dimension {i}')
            offset = self.binary_integer('+', self.binary_integer('*', offset, self.integer_constant(extent)), index)
        return offset

    def address(self, buffer, indices, width):
        require(isinstance(buffer, str) and buffer in self.pointers, f'unknown global buffer: {buffer}')
        offset = self.flat_index(buffer, indices, width)
        byte_offset, address = self.reg('rd'), self.reg('rd')
        self.put(f'mul.wide.u32 {byte_offset}, {offset.operand}, 4;')
        self.put(f'add.u64 {address}, {self.pointers[buffer]}, {byte_offset};')
        return address

    def emit_math(self, operation, operands, dst=None):
        arity = {'add': 2, 'subtract': 2, 'multiply': 2, 'divide': 2,
                 'maximum': 2, 'minimum': 2, 'reciprocal': 1,
                 'sqrt': 1, 'exp': 1, 'copy': 1, 'relu': 1}
        require(operation in arity, f'unsupported elementwise operation: {operation}')
        require(len(operands) == arity[operation], f'{operation}: wrong operand count')
        dst = dst or self.reg('f')
        if operation == 'exp':
            scaled = self.reg('f')
            # Natural exp via exp2; approximation remains subject to GPU validation.
            self.put(f'mul.rn.f32 {scaled}, {operands[0]}, 0f3fb8aa3b;')
            self.put(f'ex2.approx.f32 {dst}, {scaled};')
        elif operation == 'reciprocal':
            self.put(f'div.rn.f32 {dst}, 0f3f800000, {operands[0]};')
        elif operation == 'relu':
            self.put(f'max.NaN.f32 {dst}, {operands[0]}, 0f00000000;')
        else:
            instruction = {'add': 'add.rn', 'subtract': 'sub.rn', 'multiply': 'mul.rn',
                           'divide': 'div.rn', 'maximum': 'max.NaN', 'minimum': 'min.NaN',
                           'sqrt': 'sqrt.rn', 'copy': 'mov'}[operation]
            self.put(f'{instruction}.f32 {dst}, ' + ', '.join(operands) + ';')
        return dst

    def emit_assign(self, statement, update=False):
        require(isinstance(statement.op, ElementwiseImpl), 'Assign operator must be an ElementwiseImpl')
        require(f32(statement.dtype, inherited=True), 'Assign: only f32 supported')
        arguments = sequence(statement.args, 'Assign arguments')
        operands = [self.value(argument) for argument in arguments]
        if update:
            destination = self.value(statement.name)
            return self.emit_math(statement.op.name, operands, destination)
        destination = self.emit_math(statement.op.name, operands)
        self.bind(statement.name, destination)
        return destination

    def load(self, statement):
        require(f32(statement.dtype), 'Load: only f32 supported')
        require(isinstance(statement.input, str) and statement.input in self.kernel.inputs, 'Load: only input buffers supported')
        names = sequence(statement.names, 'Load names')
        address = self.address(statement.input, statement.index, len(names))
        for i, name in enumerate(names):
            destination = self.reg('f')
            self.bind(name, destination)
            self.put(f'ld.global.f32 {destination}, [{address}+{4 * i}];')

    def write(self, statement):
        require(f32(statement.value_dtype), 'Write: only f32 supported')
        require(statement.atomic is False and statement.swizzle == 'NONE', 'atomic or swizzled Write unsupported')
        require(isinstance(statement.output, str) and statement.output in self.kernel.outputs, 'Write: only output buffers supported')
        operands = [self.value(name) for name in sequence(statement.values, 'Write values')]
        address = self.address(statement.output, statement.index, len(operands))
        for i, operand in enumerate(operands):
            self.put(f'st.global.f32 [{address}+{4 * i}], {operand};')
        self.written.add(statement.output)

    def check_accumulator(self, statement, axis):
        valid_name(statement.name)
        require(f32(statement.dtype), 'Accum: only f32 supported')
        require(isinstance(statement.op, ElementwiseImpl) and statement.op.name in ('add', 'multiply', 'maximum', 'minimum'),
                'unsupported accumulator operation')
        require(sequence(statement.axes, 'Accum axes') == (axis,), 'Accum must name exactly its enclosing reduction axis')
        require(statement.base is None or (isinstance(statement.base, str) and bool(statement.base)),
                'Accum base must be an SSA name or None')

    def loop(self, statement):
        axis = statement.axis
        require(isinstance(axis, Axis) and axis.window is None and axis.extent.is_static,
                'loops require a static unwindowed axis')
        name = valid_name(axis.name)
        extent = axis.extent.as_static()
        require(type(extent) is int and 0 < extent < 2**31, 'loop extent must be a positive signed 32-bit integer')
        require(name not in self.ints and name not in self.values, 'loop axis aliases an existing binding')
        require(type(statement.seed) is bool and type(statement.unroll) is bool, 'loop seed/unroll must be booleans')
        body = sequence(statement.body, 'loop body')
        if isinstance(statement, StridedLoop):
            start = self.index(statement.start)
            step = self.index(statement.step) if not isinstance(statement.step, int) else self.integer_constant(statement.step)
            end = self.integer_constant(extent) if statement.end is None else self.index(statement.end)
        else:
            start, step, end = self.integer_constant(0), self.integer_constant(1), self.integer_constant(extent)
        require(start.lower >= 0 and 0 < step.lower == step.upper, 'loop start must be nonnegative and stride must be a positive constant')
        require(0 <= end.lower <= end.upper <= extent, 'loop end must remain inside its axis extent')
        require(extent - 1 + step.upper < 2**31, 'loop induction increment may overflow')
        accumulators = {}
        for child in body:
            if isinstance(child, Accum):
                self.check_accumulator(child, name)
                require(child.name not in accumulators or accumulators[child.name].op.name == child.op.name,
                        'one accumulator cannot mix reduction operators')
                accumulators[child.name] = child
        for accumulator in accumulators.values():
            if statement.seed:
                register = self.reg('f')
                self.bind(accumulator.name, register, shadow_state=True)
                require(accumulator.op.identity is not None, 'accumulator operation has no identity')
                self.put(f'mov.f32 {register}, {float_literal(accumulator.op.identity)};')
                self.mutable.add(accumulator.name)
            else:
                require(accumulator.name in self.mutable and accumulator.name in self.values,
                        'seed=False requires a previously initialized accumulator')
        outer = (self.values, self.ints, self.definitions, self.mutable, self.accum_context)
        self.values, self.ints = dict(self.values), dict(self.ints)
        self.definitions, self.mutable = set(), set(self.mutable)
        self.accum_context = accumulators
        induction = self.reg('r')
        lower, upper, stride = 0, extent - 1, 1
        if start.lower == start.upper and start.lower < end.upper:
            lower, stride = start.lower, step.upper
            upper = lower + ((end.upper - 1 - lower) // stride) * stride
        self.ints[name] = self.integer_symbol(induction, lower, upper, stride)
        self.definitions.add(name)
        if isinstance(statement, StridedLoop) and statement.end is not None:
            require(name + '_end' not in self.ints and name + '_end' not in self.values, 'loop end binding aliases existing name')
            self.ints[name + '_end'] = end
        label = self.label_count
        self.label_count += 1
        self.put(f'mov.s32 {induction}, {start.operand};')
        self.lines.append(f'LOOP_{label}:')
        predicate = self.reg('p')
        self.put(f'setp.ge.s32 {predicate}, {induction}, {end.operand};')
        self.put(f'@{predicate} bra LOOP_END_{label};')
        for child in body:
            self.statement(child)
        self.put(f'add.s32 {induction}, {induction}, {step.operand};')
        self.put(f'bra LOOP_{label};')
        self.lines.append(f'LOOP_END_{label}:')
        self.values, self.ints, self.definitions, self.mutable, self.accum_context = outer

    def statement(self, statement):
        if isinstance(statement, Load):
            self.load(statement)
        elif isinstance(statement, Write):
            self.write(statement)
        elif isinstance(statement, Assign):
            self.emit_assign(statement)
        elif isinstance(statement, (Loop, StridedLoop)):
            self.loop(statement)
        elif isinstance(statement, Accum):
            valid_name(statement.name)
            require(statement.name in self.accum_context, 'Accum must be in the immediate body of its reduction loop')
            self.check_accumulator(statement, next(iter(self.accum_context[statement.name].axes)))
            operands = [self.value(statement.base if statement.base is not None else statement.name), self.value(statement.value)]
            self.emit_math(statement.op.name, operands, self.value(statement.name))
        elif isinstance(statement, Init):
            require(f32(statement.dtype), 'Init: only f32 supported')
            destination = self.reg('f')
            self.bind(statement.name, destination, shadow_state=True)
            self.put(f'mov.f32 {destination}, {float_literal(statement.identity)};')
            self.mutable.add(statement.name)
        elif isinstance(statement, Let):
            if f32(statement.dtype) and isinstance(statement.value, Literal) and statement.value.dtype in ('float', 'f32'):
                destination = self.reg('f')
                self.bind(statement.name, destination)
                self.put(f'mov.f32 {destination}, {float_literal(statement.value.value)};')
            elif statement.dtype is None or getattr(statement.dtype, 'name', None) == 'i32':
                name = valid_name(statement.name)
                require(name not in self.definitions and name not in self.ints and name not in self.values, 'integer Let aliases another binding')
                self.ints[name] = self.index(statement.value)
                self.definitions.add(name)
            else:
                raise CompileError('Let supports f32 literals or bounded integer expressions')
        else:
            raise CompileError(f'unsupported structured statement: {type(statement).__name__}')

    def setup_tile(self, tile):
        axes = sequence(tile.axes, 'Tile axes')
        require(axes, 'Tile must have at least one axis')
        require(tile.aux_threads == 0 and type(tile.aux_threads) is int, 'auxiliary Tile threads unsupported')
        require(tile.block_threads is None or (type(tile.block_threads) is int and tile.block_threads == self.kernel.block[0]),
                'Tile.block_threads must match the prescribed block')
        names, extents = [], []
        for axis in axes:
            require(isinstance(axis, Axis) and axis.extent.is_static, 'Tile axes must be static')
            names.append(valid_name(axis.name))
            extent = axis.extent.as_static()
            require(type(extent) is int and 0 < extent < 2**31, 'Tile extent must be a positive signed 32-bit integer')
            extents.append(extent)
            if axis.window is not None:
                window = axis.window
                require(isinstance(window.parent, Axis) and window.parent.extent.is_static
                        and window.base is None and window.bound is None and window.partition is False,
                        'Tile supports only unbounded static window annotations')
        require(len(set(names)) == len(names), 'duplicate Tile axis names')
        require(not set(names) & set(self.kernel.arg_order), 'Tile axis aliases a parameter')
        self.n = prod(extents)
        require(0 < self.n < 2**31, 'Tile product exceeds supported signed 32-bit range')
        k = self.kernel
        require(k.grid[1:] == (1, 1) and k.block[1:] == (1, 1) and k.grid[0] * k.block[0] >= self.n,
                'prescribed 1-D launch does not cover Tile')
        tid, cta, block, self.gid = [self.reg('r') for _ in range(4)]
        self.put(f'mov.u32 {tid}, %tid.x;')
        self.put(f'mov.u32 {cta}, %ctaid.x;')
        self.put(f'mov.u32 {block}, %ntid.x;')
        self.put(f'mad.lo.u32 {self.gid}, {cta}, {block}, {tid};')
        predicate = self.reg('p')
        self.put(f'setp.ge.u32 {predicate}, {self.gid}, {self.n};')
        self.put(f'@{predicate} bra DONE;')
        strides = [prod(extents[i + 1:]) for i in range(len(extents))]
        raster = None
        require(tile.raster_orient in ('m', 'n'), 'unknown raster orientation')
        if tile.raster_axes is not None:
            raster_names = sequence(tile.raster_axes, 'raster axes')
            require(len(raster_names) == 2 and all(name in names for name in raster_names), 'raster axes must identify two Tile axes')
            mi, ni = (names.index(name) for name in raster_names)
            require(ni == mi + 1, 'raster axes must be adjacent and ordered')
            if tile.raster_group is not None:
                require(type(tile.raster_group) is int and tile.raster_group > 0, 'raster group must be a positive integer')
                grouped, other = (mi, ni) if tile.raster_orient == 'm' else (ni, mi)
                group = min(tile.raster_group, extents[grouped])
                if group > 1:
                    raster = grouped, other, group, mi, ni
        else:
            require(tile.raster_group is None, 'raster group requires raster axes')
        decoded = {}
        if raster:
            grouped, other, group, mi, ni = raster
            ea, eb = extents[grouped], extents[other]
            sub, stripe, group_id, base, available, size, within, group_offset, other_offset = [self.reg('r') for _ in range(9)]
            self.put(f'div.u32 {sub}, {self.gid}, {strides[ni]};')
            self.put(f'rem.u32 {sub}, {sub}, {ea * eb};')
            self.put(f'mov.u32 {stripe}, {group * eb};')
            self.put(f'div.u32 {group_id}, {sub}, {stripe};')
            self.put(f'mul.lo.u32 {base}, {group_id}, {group};')
            self.put(f'sub.u32 {available}, {ea}, {base};')
            self.put(f'min.u32 {size}, {group}, {available};')
            self.put(f'rem.u32 {within}, {sub}, {stripe};')
            self.put(f'rem.u32 {group_offset}, {within}, {size};')
            self.put(f'add.u32 {group_offset}, {base}, {group_offset};')
            self.put(f'div.u32 {other_offset}, {within}, {size};')
            decoded[grouped], decoded[other] = group_offset, other_offset
        for i, (name, extent) in enumerate(zip(names, extents)):
            register = decoded.get(i)
            if register is None:
                register = self.reg('r')
                self.put(f'div.u32 {register}, {self.gid}, {strides[i]};')
                self.put(f'rem.u32 {register}, {register}, {extent};')
            self.ints[name] = self.integer_symbol(register, 0, extent - 1)
            self.definitions.add(name)

    def output_write_counts(self, allow_atomic=False):
        """Prove finite write domains without evaluating any floating-point work.

        Only loops containing output writes need traversal. Attention's nested
        dot/softmax computations therefore add no proof cost. The bound limits
        compile-time resource use for irregular schedules we cannot yet prove
        algebraically; exceeding it fails explicitly rather than skipping proof.
        Rasterization is a bijection on these logical Tile coordinates.
        """
        require(self.n <= 1_000_000 and sum(self.kernel.shapes[n] for n in self.kernel.outputs) <= 1_000_000,
                'output coverage proof exceeds the supported one-million-cell domain')
        counts = {name: array('I', [0]) * self.kernel.shapes[name] for name in self.kernel.outputs}
        budget = 4_000_000

        def integer(expression, environment):
            if isinstance(expression, Var):
                require(expression.name in environment, f'coverage proof: unknown index {expression.name}')
                return environment[expression.name]
            if isinstance(expression, Literal):
                require(expression.dtype == 'int' and type(expression.value) is int, 'coverage proof needs integer literals')
                return expression.value
            if isinstance(expression, TernaryExpr):
                return integer(expression.if_true if integer(expression.cond, environment) else expression.if_false, environment)
            if isinstance(expression, BinaryExpr):
                left, right = integer(expression.left, environment), integer(expression.right, environment)
                operations = {'+': lambda: left + right, '-': lambda: left - right,
                              '*': lambda: left * right, '<': lambda: left < right,
                              '<=': lambda: left <= right, '>': lambda: left > right,
                              '>=': lambda: left >= right, '==': lambda: left == right,
                              '!=': lambda: left != right}
                if expression.op in ('/', '//', '%'):
                    require(left >= 0 and right > 0, 'coverage proof division has invalid operands')
                    return left % right if expression.op == '%' else left // right
                require(expression.op in operations, 'unsupported coverage index operation')
                return operations[expression.op]()
            raise CompileError('coverage proof requires pure integer index expressions')

        def contains_output(body):
            for statement in body:
                if isinstance(statement, Write) and statement.output in counts:
                    return True
                for field in ('body', 'else_body'):
                    nested = getattr(statement, field, ())
                    if nested and contains_output(nested):
                        return True
            return False

        def visit(body, environment):
            nonlocal budget
            for statement in body:
                budget -= 1
                require(budget >= 0, 'output coverage proof exceeds the supported traversal budget')
                if isinstance(statement, Write) and statement.output in counts:
                    require(allow_atomic or statement.atomic is False, 'ordinary output proof does not support atomic writes')
                    offset = 0
                    for expression, extent in zip(statement.index, self.dimensions[statement.output]):
                        offset = offset * extent + integer(expression, environment)
                    values = counts[statement.output]
                    require(0 <= offset and offset + len(statement.values) <= len(values), 'coverage proof found an out-of-range write')
                    for index in range(offset, offset + len(statement.values)):
                        values[index] += 1
                elif isinstance(statement, (Loop, StridedLoop)):
                    if not contains_output(statement.body):
                        continue
                    start, step, end = 0, 1, statement.axis.extent.as_static()
                    if isinstance(statement, StridedLoop):
                        start = integer(statement.start, environment)
                        step = statement.step if type(statement.step) is int else integer(statement.step, environment)
                        if statement.end is not None:
                            end = integer(statement.end, environment)
                    for coordinate in range(start, end, step):
                        inner = dict(environment)
                        inner[statement.axis.name] = coordinate
                        if isinstance(statement, StridedLoop) and statement.end is not None:
                            inner[statement.axis.name + '_end'] = end
                        visit(statement.body, inner)
                elif isinstance(statement, Let) and not f32(statement.dtype):
                    environment[statement.name] = integer(statement.value, environment)
                elif any(contains_output(getattr(statement, field, ())) for field in ('body', 'else_body')):
                    raise CompileError('output coverage proof does not support this conditional/nested write topology')

        tile = self.kernel.body[0]
        axes = [(axis.name, axis.extent.as_static()) for axis in tile.axes]
        for coordinates in product(*(range(extent) for _, extent in axes)):
            visit(tile.body, dict(zip((name for name, _ in axes), coordinates)))
        return counts

    def prove_output_coverage(self):
        for name, counts in self.output_write_counts().items():
            require(all(count == 1 for count in counts), f'{name}: output must be written exactly once (gap or overlapping writers)')

    def run(self):
        k = self.kernel
        body = sequence(k.body, 'kernel body')
        require(len(body) == 1 and isinstance(body[0], Tile), 'expected one top-level Tile')
        for name in k.arg_order:
            parameter, pointer = self.reg('rd'), self.reg('rd')
            self.put(f'ld.param.u64 {parameter}, [param_{name}];')
            self.put(f'cvta.to.global.u64 {pointer}, {parameter};')
            self.pointers[name] = pointer
        self.setup_tile(body[0])
        for statement in sequence(body[0].body, 'Tile body'):
            self.statement(statement)
        require(self.written == k.outputs, 'every output must have a Write')
        self.prove_output_coverage()
        self.lines.extend(['DONE:', '    ret;'])
        parameters = ',\n'.join(f'    .param .u64 param_{name}' for name in k.arg_order)
        registers = '\n'.join(f'    .reg .{dtype} %{kind}<{self.counts[kind]}>;'
                              for kind, dtype in [('p', 'pred'), ('r', 'b32'), ('rd', 'b64'), ('f', 'f32')]
                              if self.counts[kind])
        declarations = '\n'.join(self.extra_declarations)
        module_prefix = '\n'.join(self.module_declarations) + '\n\n' if self.module_declarations else ''
        return (f'.version 8.0\n.target sm_89\n.address_size 64\n\n{module_prefix}.visible .entry {k.name}(\n{parameters}\n)\n'
                f'.reqntid {k.block[0]}, 1, 1\n{{\n{registers}\n{declarations}\n' + '\n'.join(self.lines) + '\n}\n')
