"""Additional bounded CPU-model checks for the generalized compiler.

These execute emitted PTX operand flow, never assemble PTX or run a GPU. Existing
official fixture tolerances are preserved and every numeric comparison rejects
NaNs. Synthetic activation checks use the existing activation-test tolerances.
"""
from dataclasses import replace
import math
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
from reduction_model import run_reduction_ptx
from structured_model import run_structured_ptx
from test_conditionals import guarded_attention
from test_numeric import compile_fixture, scalar_fixture
from test_reduction import get_tile
from test_reduction_composed import change_loop, change_reducer
from test_reduction_extended import row_fixture
from test_structured import add, attention_reference, axis, synthetic


def padded_attention(shape, *, inclusive_depth=False):
    """Pad the genuine attention key/depth loops as well as its output axes."""
    rows, keys, depth, columns = shape
    raw, meta = guarded_attention(*shape)
    graph = Graph.from_dict(raw)
    tile = next(node.op.body[0] for node in graph.nodes.values()
                if type(node.op).__name__ == 'KernelOp')

    def rewrite(node):
        if isinstance(node, Cond):
            return replace(node, body=tuple(rewrite(s) for s in node.body),
                           else_body=tuple(rewrite(s) for s in node.else_body))
        if isinstance(node, Loop):
            extent = keys if node.axis.name == 'a1' else depth
            operation = '<=' if inclusive_depth and node.axis.name == 'a2' else '<'
            guard = BinaryExpr(operation, Var(node.axis.name), Literal(extent, 'int'))
            return replace(node, axis=replace(node.axis, extent=Dim(extent + 3)),
                           body=(Cond(guard, tuple(rewrite(s) for s in node.body)),))
        return node

    tile = replace(tile, body=tuple(rewrite(s) for s in tile.body))
    kernel = next(node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    kernel['op_fields']['body'] = [repr(tile)]
    return raw, meta


def add_runtime_scalar_inputs(raw, meta, names):
    """Use actual one-element pointer inputs rather than compile-time literals."""
    kernel_name, = raw['outputs']
    launch = meta['kernels'][0]
    for name in names:
        raw['inputs'].append(name)
        raw['nodes'][kernel_name]['inputs'].append(name)
        raw['nodes'][name] = {'op': 'InputOp', 'op_fields': {}, 'inputs': [],
                              'outputs': [{'name': name, 'shape': [1], 'dtype': 'f32'}]}
        launch['arg_order'].insert(0, name)
        launch['args'].insert(0, {'name': name, 'shape': [1], 'dtype': 'f32', 'seed': 99})


class ExtraGeneralizationTests(unittest.TestCase):
    def test_activation_polynomial_boundaries_and_random_finite_values(self):
        rng = np.random.default_rng(91073)
        boundary = np.float32(0.125)
        edge = np.array([np.nextafter(boundary, np.float32(0)), boundary,
                         np.nextafter(boundary, np.float32(1)), np.float32(1e-30),
                         np.float32(1e20), np.finfo(np.float32).max], np.float32)
        values = np.concatenate((rng.uniform(-12, 12, 4097).astype(np.float32), -edge, edge))
        wide = values.astype(np.float64)
        references = {
            'tanh': np.tanh(wide),
            'erf': np.array([math.erf(x) for x in wide]),
            'gelu': 0.5 * wide * (1 + np.array([math.erf(x / math.sqrt(2)) for x in wide])),
            'gelu_tanh': 0.5 * wide * (1 + np.tanh(np.sqrt(2 / np.pi) * (wide + 0.044715 * wide ** 3))),
        }
        tolerances = {'tanh': (2e-6, 2e-7), 'erf': (2e-6, 6e-7),
                      'gelu': (2e-5, 1e-6), 'gelu_tanh': (2e-5, 1e-6)}
        for operation, expected in references.items():
            with self.subTest(operation=operation):
                raw, meta = scalar_fixture(values.size, [(operation, None)])
                actual = run_structured_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x0': values}).outputs['add']
                rtol, atol = tolerances[operation]
                np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)

    def test_rsqrt_positive_float32_range_preserves_separate_rounding(self):
        rng = np.random.default_rng(87329)
        values = np.power(10.0, rng.uniform(-44, 38, 4097)).astype(np.float32)
        values = np.concatenate((values, [np.nextafter(np.float32(0), np.float32(1)),
                                         np.finfo(np.float32).tiny, np.finfo(np.float32).max]))
        raw, meta = scalar_fixture(values.size, [('rsqrt', None)])
        ptx = compile_fixture(raw, meta)
        self.assertIn('sqrt.rn.f32', ptx)
        self.assertIn('div.rn.f32', ptx)
        result = run_structured_ptx(ptx, meta['kernels'][0], {'x0': values})
        expected = np.divide(np.float32(1), np.sqrt(values, dtype=np.float32), dtype=np.float32)
        np.testing.assert_allclose(result.outputs['add'], expected, rtol=0, atol=0, equal_nan=False)

    def test_masked_activation_chain_on_random_odd_matrices(self):
        rng = np.random.default_rng(61097)
        shapes = [(int(rng.integers(1, 5)) * 2 + 1, int(rng.integers(9, 35)) * 2 + 1) for _ in range(3)]
        for rows, columns in shapes:
            with self.subTest(shape=(rows, columns)):
                index = (Var('row'), Var('column'))
                guard = BinaryExpr('&&', BinaryExpr('<', Var('row'), Literal(rows, 'int')),
                                   BinaryExpr('<', Var('column'), Literal(columns, 'int')))
                body = (Load('x_value', 'x', index, dtype=F32), Load('bias_value', 'bias', index, dtype=F32),
                        Assign('activated', ElementwiseImpl('silu'), ('x_value',), F32),
                        Assign('shifted', ElementwiseImpl('add'), ('activated', 'bias_value'), F32),
                        Assign('bounded', ElementwiseImpl('tanh'), ('shifted',), F32),
                        Write('result', index, 'bounded', value_dtype=F32))
                tile = Tile((axis('row', rows + 2), axis('column', columns + 3)), (Cond(guard, body),))
                raw, meta = synthetic(tile, {'x': (rows, columns), 'bias': (rows, columns)}, (rows, columns),
                                      grid=((rows + 2) * (columns + 3) + 31) // 32 + 1)
                values = rng.uniform(-6, 6, (rows, columns)).astype(np.float32)
                bias = rng.uniform(-0.5, 0.5, (rows, columns)).astype(np.float32)
                result = run_structured_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values, 'bias': bias})
                expected = np.tanh(np.add(values / (1 + np.exp(-values)), bias, dtype=np.float32))
                np.testing.assert_allclose(result.outputs['result'], expected, rtol=2e-6, atol=2e-7, equal_nan=False)
                self.assertTrue(np.all(result.store_counts['result'] == 1))

    def test_transformed_row_reductions_read_runtime_scalar_buffers(self):
        rng = np.random.default_rng(8193)
        for rows, columns, threads in ((3, 35, 32), (5, 131, 64), (1, 263, 128)):
            for reducer in ('add', 'maximum'):
                with self.subTest(shape=(rows, columns), threads=threads, reducer=reducer):
                    raw, meta = row_fixture(rows, columns, threads)
                    add_runtime_scalar_inputs(raw, meta, ('scale', 'offset'))
                    change_reducer(raw, reducer)
                    load, accum = get_tile(raw).body[0].body
                    statements = (load, Load('s', 'scale', (Literal(0, 'int'),), dtype=F32),
                        Load('b', 'offset', (Literal(0, 'int'),), dtype=F32),
                        Assign('scaled', ElementwiseImpl('multiply'), (load.names[0], 's'), F32),
                        Assign('shifted', ElementwiseImpl('add'), ('scaled', 'b'), F32),
                        Assign('absolute', ElementwiseImpl('abs'), ('shifted',), F32),
                        replace(accum, value='absolute'))
                    change_loop(raw, statements)
                    ptx = compile_fixture(raw, meta)
                    values = rng.normal(size=(rows, columns)).astype(np.float32)
                    for scale, offset in ((-0.75, 0.125), (1.5, -2.0)):
                        overrides = {'x': values, 'scale': [scale], 'offset': [offset]}
                        result = run_reduction_ptx(ptx, meta['kernels'][0], overrides)
                        transformed = np.abs(np.add(np.multiply(values, np.float32(scale), dtype=np.float32),
                                                    np.float32(offset), dtype=np.float32))
                        expected = (transformed.astype(np.float64).sum(axis=1) if reducer == 'add'
                                    else transformed.max(axis=1)).astype(np.float32)
                        np.testing.assert_allclose(result.outputs['sum_1'], expected,
                                                   rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
                        self.assertTrue(np.all(result.load_counts['x'] == 1))
                        self.assertEqual(int(result.load_counts['scale'][0]), rows * columns)
                        self.assertEqual(int(result.load_counts['offset'][0]), rows * columns)

    def test_erf_and_gelu_compose_inside_odd_row_reductions(self):
        rng = np.random.default_rng(99127)
        for operation, reducer in (('gelu', 'add'), ('erf', 'maximum')):
            raw, meta = row_fixture(3, 137, threads=64)
            change_reducer(raw, reducer)
            load, accum = get_tile(raw).body[0].body
            transformed = Assign('transformed', ElementwiseImpl(operation), load.names, F32)
            change_loop(raw, (load, transformed, replace(accum, value='transformed')))
            values = rng.uniform(-4, 4, (3, 137)).astype(np.float32)
            values[:, :4] = np.array([-0.125, -0.0, 0.0, 0.125], np.float32)
            mathematical_erf = np.array([math.erf(float(x) / (math.sqrt(2) if operation == 'gelu' else 1))
                                         for x in values.flat]).reshape(values.shape)
            reference = 0.5 * values.astype(np.float64) * (1 + mathematical_erf) if operation == 'gelu' else mathematical_erf
            expected = (reference.sum(axis=1) if reducer == 'add' else reference.max(axis=1)).astype(np.float32)
            result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
            np.testing.assert_allclose(result.outputs['sum_1'], expected,
                                       rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)

    def test_conditional_reduction_empty_prefix_and_if_else_state(self):
        values = np.random.default_rng(2718).integers(-4, 5, (5, 7)).astype(np.float32)
        for use_else in (False, True):
            with self.subTest(use_else=use_else):
                load = Load('value', 'x', (Var('row'), Var('k')), dtype=F32)
                branch = (load, add('total', 'value', 'k'))
                other = ((load, Assign('negative', ElementwiseImpl('negative'), ('value',), F32),
                          add('total', 'negative', 'k')) if use_else else ())
                loop = Loop(axis('k', 7), (Cond(BinaryExpr('<', Var('k'), Var('row')), branch, other),))
                tile = Tile((axis('row', 5),), (loop, Write('result', (Var('row'),), 'total', value_dtype=F32)))
                raw, meta = synthetic(tile, {'x': (5, 7)}, (5,))
                result = run_structured_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                selected = np.arange(7)[None, :] < np.arange(5)[:, None]
                expected = np.where(selected, values, -values if use_else else np.float32(0)).sum(axis=1, dtype=np.float32)
                np.testing.assert_allclose(result.outputs['result'], expected, rtol=0, atol=0, equal_nan=False)
                if not use_else: self.assertEqual(result.outputs['result'][0], 0)

    def test_attention_guards_padded_key_depth_and_output_domains(self):
        rng = np.random.default_rng(33281)
        for shape in ((1, 1, 1, 1), (3, 11, 5, 7), (5, 3, 13, 9)):
            with self.subTest(shape=shape):
                raw, meta = padded_attention(shape)
                launch = meta['kernels'][0]
                launch['args'].reverse()
                launch['arg_order'].reverse()
                inputs = input_buffers(launch)
                for name in ('q', 'kt', 'v'):
                    inputs[name] = rng.normal(size=inputs[name].shape).astype(np.float32)
                result = run_structured_ptx(compile_fixture(raw, meta), launch, inputs)
                np.testing.assert_allclose(result.outputs['matmul_1'], attention_reference(inputs),
                                           rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
                self.assertTrue(np.all(result.store_counts['matmul_1'] == 1))

    def test_padded_attention_rejects_inclusive_out_of_bounds_depth_guard(self):
        raw, meta = padded_attention((3, 5, 7, 4), inclusive_depth=True)
        with self.assertRaisesRegex(CompileError, 'out-of-bounds'):
            compile_fixture(raw, meta)


if __name__ == '__main__':
    unittest.main()
