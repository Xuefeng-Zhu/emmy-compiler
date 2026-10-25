"""Our small, direct IR-to-PTX lowering. No Emmy backend or external compiler."""
from collections import defaultdict

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write

from .errors import CompileError
from .loader import Kernel, require


class Emitter:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.lines = []
        self.counts = defaultdict(int)
        self.values = {}
        self.pointers = {}
        self.written = set()

    def reg(self, kind):
        result = f'%{kind}{self.counts[kind]}'
        self.counts[kind] += 1
        return result

    def put(self, text):
        self.lines.append('    ' + text)

    def bind(self, name, value):
        require(name not in self.values, f'duplicate SSA definition: {name}')
        self.values[name] = value

    def value(self, name):
        require(name in self.values, f'undefined SSA value: {name}')
        return self.values[name]

    def index(self, expr):
        """Emit an index and track an affine form (scale * axis + offset).

        Nonnegative affine bounds prove 32-bit arithmetic and every access safe.
        Reject non-affine multiplication instead of guessing its bounds.
        """
        if isinstance(expr, Var):
            require(expr.name == self.axis, f'unsupported index variable: {expr.name}')
            return self.gid, 1, 0
        if isinstance(expr, Literal):
            require(type(expr.value) is int and 0 <= expr.value < 2**31 and expr.dtype == 'int', 'index literal must be a nonnegative 31-bit integer')
            return str(expr.value), 0, expr.value
        if isinstance(expr, BinaryExpr):
            require(expr.op in ('+', '*'), f'unsupported index operator: {expr.op}')
            left, ls, lo = self.index(expr.left)
            right, rs, ro = self.index(expr.right)
            if expr.op == '+':
                scale, offset = ls + rs, lo + ro
                instr = 'add.u32'
            else:
                require(not (ls and rs), 'non-affine index multiplication is unsupported')
                scale, offset = ls * ro + rs * lo, lo * ro
                instr = 'mul.lo.u32'
            require(scale * (self.n - 1) + offset < 2**31, 'index arithmetic exceeds supported 31-bit range')
            dst = self.reg('r')
            self.put(f'{instr} {dst}, {left}, {right};')
            return dst, scale, offset
        raise CompileError(f'unsupported index expression: {type(expr).__name__}')

    def address(self, buffer, indices, width):
        require(buffer in self.pointers, f'unknown buffer: {buffer}')
        require(len(indices) == 1, 'only one-dimensional indices supported')
        require(width in (1, 2, 4), 'only scalar, pair or four-element memory operations supported')
        idx, scale, offset = self.index(indices[0])
        require(scale * (self.n - 1) + offset + width <= self.kernel.shapes[buffer], f'out-of-bounds access to {buffer}')
        byte_offset, address = self.reg('rd'), self.reg('rd')
        self.put(f'mul.wide.u32 {byte_offset}, {idx}, 4;')
        self.put(f'add.u64 {address}, {self.pointers[buffer]}, {byte_offset};')
        return address, scale, offset

    def statement(self, stmt):
        if isinstance(stmt, Load):
            require(getattr(stmt.dtype, 'name', None) == 'f32', 'Load: only f32 supported')
            require(stmt.input in self.kernel.inputs, 'Load: only input buffers supported')
            addr, _, _ = self.address(stmt.input, stmt.index, len(stmt.names))
            for i, name in enumerate(stmt.names):
                dst = self.reg('f')
                self.bind(name, dst)
                self.put(f'ld.global.f32 {dst}, [{addr}+{i * 4}];')
        elif isinstance(stmt, Assign):
            require(getattr(stmt.dtype, 'name', None) == 'f32', 'Assign: only f32 supported')
            require(stmt.op.name == 'add' and len(stmt.args) == 2, f'unsupported elementwise operation: {stmt.op.name}')
            a, b = (self.value(arg) for arg in stmt.args)
            dst = self.reg('f')
            self.bind(stmt.name, dst)
            self.put(f'add.rn.f32 {dst}, {a}, {b};')
        elif isinstance(stmt, Write):
            require(getattr(stmt.value_dtype, 'name', None) == 'f32', 'Write: only f32 supported')
            require(not stmt.atomic and stmt.swizzle == 'NONE', 'atomic or swizzled Write unsupported')
            require(stmt.output in self.kernel.outputs, 'Write: only output buffers supported')
            require(stmt.output not in self.written, 'multiple Writes per output unsupported')
            values = [self.value(name) for name in stmt.values]
            addr, scale, offset = self.address(stmt.output, stmt.index, len(values))
            # This bounded slice proves full output coverage without races.
            require(scale == len(values) and offset == 0 and self.n * scale == self.kernel.shapes[stmt.output], 'Write must cover its output once as contiguous per-thread cells')
            for i, value in enumerate(values):
                self.put(f'st.global.f32 [{addr}+{i * 4}], {value};')
            self.written.add(stmt.output)
        else:
            raise CompileError(f'unsupported statement: {type(stmt).__name__}')

    def run(self):
        k = self.kernel
        require(len(k.body) == 1 and isinstance(k.body[0], Tile), 'expected exactly one top-level Tile')
        tile = k.body[0]
        require(len(tile.axes) == 1, 'only one-dimensional Tile supported')
        require(tile.block_threads is None and tile.aux_threads == 0 and tile.raster_axes is None and tile.raster_group is None and tile.raster_orient == 'm', 'cooperative or rasterized Tile unsupported')
        axis = tile.axes[0]
        require(axis.window is None and axis.extent.is_static, 'windowed or dynamic Tile unsupported')
        self.axis, self.n = axis.name, axis.extent.as_static()
        require(0 < self.n < 2**31, 'Tile extent must be a positive 31-bit integer')
        require(k.grid[0] * k.block[0] >= self.n, 'launch does not cover Tile extent')
        for name in k.arg_order:
            param, ptr = self.reg('rd'), self.reg('rd')
            self.put(f'ld.param.u64 {param}, [param_{name}];')
            self.put(f'cvta.to.global.u64 {ptr}, {param};')
            self.pointers[name] = ptr
        tid, cta, block, self.gid = [self.reg('r') for _ in range(4)]
        self.put(f'mov.u32 {tid}, %tid.x;')
        self.put(f'mov.u32 {cta}, %ctaid.x;')
        self.put(f'mov.u32 {block}, %ntid.x;')
        self.put(f'mad.lo.u32 {self.gid}, {cta}, {block}, {tid};')
        pred = self.reg('p')
        self.put(f'setp.ge.u32 {pred}, {self.gid}, {self.n};')
        self.put(f'@{pred} bra DONE;')
        for stmt in tile.body:
            self.statement(stmt)
        require(self.written == k.outputs, 'every output must have one Write')
        self.lines.extend(['DONE:', '    ret;'])
        params = ',\n'.join(f'    .param .u64 param_{name}' for name in k.arg_order)
        decl = '\n'.join(f'    .reg .{dtype} %{kind}<{self.counts[kind]}>;' for kind, dtype in [('p', 'pred'), ('r', 'b32'), ('rd', 'b64'), ('f', 'f32')] if self.counts[kind])
        return f'.version 8.0\n.target sm_89\n.address_size 64\n\n.visible .entry {k.name}(\n{params}\n)\n{{\n{decl}\n' + '\n'.join(self.lines) + '\n}\n'


def emit(kernel: Kernel) -> str:
    return Emitter(kernel).run()
