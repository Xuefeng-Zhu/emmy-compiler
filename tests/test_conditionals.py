"""Guarded IR regression tests; execute PTX independently of the input IR."""
from dataclasses import replace
import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write

from emmy_ptx import CompileError
from ptx_model import input_buffers
from structured_model import run_structured_ptx
from test_structured import (add, attention_reference, axis, changed_attention,
                             compile_fixture, synthetic)


def guarded_attention(rows, keys, depth, columns):
    ir, meta = changed_attention(rows, keys, depth, columns)
    graph = Graph.from_dict(ir)
    tile = next(node.op.body[0] for node in graph.nodes.values()
                if type(node.op).__name__ == 'KernelOp')
    # Pad both output axes to force actual masked rows and columns, including
    # block/thread affine column expressions from the official attention IR.
    axes = tuple(replace(a, extent=Dim(rows + 2 if a.name == 'a0_b'
                                     else columns + 3 if a.name == 'a5_u'
                                     else a.extent.as_static())) for a in tile.axes)
    column = BinaryExpr('+', BinaryExpr('*', Var('a5_b'), Literal(32, 'int')), Var('a5_u'))
    guarded = Cond(BinaryExpr('<', Var('a0_b'), Literal(rows, 'int')), (
        Cond(BinaryExpr('<', column, Literal(columns, 'int')), tile.body),))
    tile = replace(tile, axes=axes, body=(guarded,))
    node = next(n for n in ir['nodes'].values() if n['op'] == 'KernelOp')
    node['op_fields']['body'] = [repr(tile)]
    count = np.prod([a.extent.as_static() for a in axes])
    meta['kernels'][0]['grid'] = [(int(count) + 31) // 32 + 1, 1, 1]
    return ir, meta


class ConditionalTests(unittest.TestCase):
    def test_guarded_attention_256_keys_and_rectangular_shapes(self):
        for shape in ((2, 256, 4, 3), (5, 11, 7, 6)):
            with self.subTest(shape=shape):
                ir, meta = guarded_attention(*shape)
                launch = meta['kernels'][0]
                launch['arg_order'].reverse()
                launch['args'].reverse()
                ptx = compile_fixture(ir, meta, through_dispatch=True)
                self.assertIn('ELSE_', ptx)
                result = run_structured_ptx(ptx, launch)
                np.testing.assert_allclose(result.outputs['matmul_1'],
                                           attention_reference(input_buffers(launch)),
                                           rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
                self.assertTrue(np.all(result.store_counts['matmul_1'] == 1))

    def test_guarded_attention_256_square_compiles(self):
        ir, meta = guarded_attention(256, 256, 32, 32)
        self.assertIn('setp.lt.s32', compile_fixture(ir, meta, through_dispatch=True))

    def test_compound_rectangle_guard_and_equality_guard(self):
        guard = BinaryExpr('&&', BinaryExpr('<', Var('row'), Literal(3, 'int')),
                           BinaryExpr('<', Var('column'), Literal(5, 'int')))
        indices = (Var('row'), Var('column'))
        branch = Cond(guard, (Load('value', 'x', indices, dtype=F32),
                              Write('result', indices, 'value', value_dtype=F32)))
        outside = BinaryExpr('||', BinaryExpr('>=', Var('row'), Literal(3, 'int')),
                             BinaryExpr('>=', Var('column'), Literal(5, 'int')))
        for selected in (branch, Cond(outside, (), branch.body)):
            ir, meta = synthetic(Tile((axis('row', 5), axis('column', 7)), (selected,)),
                                 {'x': (3, 5)}, (3, 5), grid=2)
            result = run_structured_ptx(compile_fixture(ir, meta, through_dispatch=True), meta['kernels'][0])
            np.testing.assert_array_equal(result.outputs['result'], input_buffers(meta['kernels'][0])['x'])
        for op in ('==', '!='):
            selected = (Load('value', 'x', (Var('i'),), dtype=F32),
                        Write('result', (Var('i'),), 'value', value_dtype=F32))
            branch = Cond(BinaryExpr(op, Var('i'), Literal(0, 'int')),
                          selected if op == '==' else (), selected if op == '!=' else ())
            ir, meta = synthetic(Tile((axis('i', 8),), (branch,)), {'x': (1,)}, (1,))
            result = run_structured_ptx(compile_fixture(ir, meta, through_dispatch=True), meta['kernels'][0])
            np.testing.assert_array_equal(result.outputs['result'], input_buffers(meta['kernels'][0])['x'])

    def test_if_else_has_separate_ssa_scopes_and_one_writer(self):
        index = Var('i')
        branch = Cond(BinaryExpr('<', index, Literal(3, 'int')), (
            Load('value', 'x', (index,), dtype=F32),
            Write('result', (index,), 'value', value_dtype=F32)), (
            Load('value', 'y', (index,), dtype=F32),
            Write('result', (index,), 'value', value_dtype=F32)))
        ir, meta = synthetic(Tile((axis('i', 7),), (branch,)), {'x': (7,), 'y': (7,)}, (7,))
        launch = meta['kernels'][0]
        inputs = input_buffers(launch)
        expected = np.where(np.arange(7) < 3, inputs['x'], inputs['y'])
        result = run_structured_ptx(compile_fixture(ir, meta, through_dispatch=True), launch)
        np.testing.assert_array_equal(result.outputs['result'], expected)

    def test_branch_local_ssa_cannot_escape_or_replace_outer_binding(self):
        for prefix in ((), (Load('value', 'x', (Var('i'),), dtype=F32),)):
            branch = Cond(BinaryExpr('<', Var('i'), Literal(2, 'int')), (
                Load('value', 'x', (Var('i'),), dtype=F32),))
            body = prefix + (branch, Write('result', (Var('i'),), 'value', value_dtype=F32))
            ir, meta = synthetic(Tile((axis('i', 3),), body), {'x': (3,)}, (3,))
            if not prefix:
                with self.assertRaisesRegex(CompileError, 'undefined SSA'):
                    compile_fixture(ir, meta, through_dispatch=True)
            else:
                # Shadowing is confined to the branch; the outer SSA survives.
                result = run_structured_ptx(compile_fixture(ir, meta, through_dispatch=True), meta['kernels'][0])
                np.testing.assert_array_equal(result.outputs['result'], input_buffers(meta['kernels'][0])['x'])

    def test_conditional_accumulator_is_seeded_by_its_enclosing_loop(self):
        loop = Loop(axis('k', 7), (Cond(BinaryExpr('<', Var('k'), Literal(4, 'int')), (
            Load('value', 'x', (Var('row'), Var('k')), dtype=F32), add('sum', 'value', 'k'))),))
        tile = Tile((axis('row', 2),), (loop, Write('result', (Var('row'),), 'sum', value_dtype=F32)))
        ir, meta = synthetic(tile, {'x': (2, 7)}, (2,))
        launch = meta['kernels'][0]
        result = run_structured_ptx(compile_fixture(ir, meta, through_dispatch=True), launch)
        expected = input_buffers(launch)['x'][:, :4].sum(axis=1, dtype=np.float32)
        np.testing.assert_allclose(result.outputs['result'], expected, rtol=1e-6, atol=1e-6)

    def test_guard_is_required_for_out_of_bounds_tail(self):
        load = Load('value', 'x', (Var('i'),), dtype=F32)
        write = Write('result', (Var('i'),), 'value', value_dtype=F32)
        for op in ('<', '>='):
            branch = Cond(BinaryExpr(op, Var('i'), Literal(5, 'int')), (load, write))
            ir, meta = synthetic(Tile((axis('i', 8),), (branch,)), {'x': (5,)}, (5,))
            if op == '<':
                run_structured_ptx(compile_fixture(ir, meta, through_dispatch=True), meta['kernels'][0])
            else:
                with self.assertRaisesRegex(CompileError, 'out-of-bounds'):
                    compile_fixture(ir, meta, through_dispatch=True)

    def test_conditional_coverage_rejects_gaps_and_overlaps(self):
        load = Load('value', 'x', (Var('i'),), dtype=F32)
        write = Write('result', (Var('i'),), 'value', value_dtype=F32)
        branch = Cond(BinaryExpr('<', Var('i'), Literal(3, 'int')), (write,))
        for body in ((load, branch), (load, branch, write)):
            ir, meta = synthetic(Tile((axis('i', 5),), body), {'x': (5,)}, (5,))
            with self.assertRaisesRegex(CompileError, 'gap or overlapping'):
                compile_fixture(ir, meta, through_dispatch=True)


if __name__ == '__main__':
    unittest.main()
