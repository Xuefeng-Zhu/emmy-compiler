"""Changed operators/compositions through the cooperative reduction schedule."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.leaves import Assign, Let

from emmy_ptx import CompileError
from reduction_model import run_reduction_ptx
from test_reduction import compile_fixture, get_tile, put_tile, sum_fixture
from test_reduction_extended import row_fixture


def change_reducer(raw, operation):
    tile = get_tile(raw)
    body = list(tile.body)
    loop = body[0]
    body[0] = replace(loop, body=(*loop.body[:-1], replace(loop.body[-1], op=ElementwiseImpl(operation))))
    for index in (1, 5):
        combine, = body[index].combine_states
        body[index] = replace(body[index], combine_states=(replace(combine, op=ElementwiseImpl(operation)),))
    put_tile(raw, replace(tile, body=tuple(body)))


def change_loop(raw, body):
    tile = get_tile(raw)
    statements = list(tile.body)
    statements[0] = replace(statements[0], body=tuple(body))
    put_tile(raw, replace(tile, body=tuple(statements)))


class ComposedReductionTests(unittest.TestCase):
    def test_amax_all_negative_tails_and_idle_lanes(self):
        for operation in ('maximum', 'amax'):
            for size, threads in ((1, 128), (31, 32), (33, 64), (129, 128), (257, 256)):
                with self.subTest(operation=operation, size=size, threads=threads):
                    raw, meta = sum_fixture(size, threads)
                    change_reducer(raw, operation)
                    values = -(np.arange(size, dtype=np.float32) + 1)
                    result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                    np.testing.assert_array_equal(result.outputs['sum_1'], [-1])
                    self.assertTrue(np.all(result.load_counts['x'] == 1))

    def test_row_max_new_rectangular_shapes(self):
        for rows, columns, threads in ((3, 17, 64), (5, 257, 128), (2, 67, 32)):
            with self.subTest(rows=rows, columns=columns, threads=threads):
                raw, meta = row_fixture(rows, columns, threads)
                change_reducer(raw, 'maximum')
                values = -(np.arange(rows * columns, dtype=np.float32).reshape(rows, columns) + 1)
                values[:, -1] = -np.arange(1, rows + 1, dtype=np.float32) / 2
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                np.testing.assert_array_equal(result.outputs['sum_1'], values.max(axis=1))
                self.assertTrue(np.all(result.load_counts['x'] == 1))

    def test_sum_of_squares_scalar_and_row_reductions(self):
        for row_shape in (None, (3, 67)):
            with self.subTest(row_shape=row_shape):
                raw, meta = sum_fixture(259) if row_shape is None else row_fixture(*row_shape, threads=64)
                load, accum = get_tile(raw).body[0].body
                square = Assign('squared', ElementwiseImpl('multiply'), (load.names[0], load.names[0]), accum.dtype)
                change_loop(raw, (load, square, replace(accum, value='squared')))
                shape = (259,) if row_shape is None else row_shape
                values = ((np.arange(np.prod(shape)) % 19) - 9).astype(np.float32).reshape(shape)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                expected = np.sum(values * values, axis=-1, dtype=np.float32).reshape(-1)
                np.testing.assert_array_equal(result.outputs['sum_1'], expected)
                self.assertTrue(np.all(result.load_counts['x'] == 1))

    def test_two_inputs_and_multiple_scalar_operations(self):
        raw, meta = sum_fixture(73, 64)
        raw['inputs'].append('y')
        raw['nodes']['y'] = deepcopy(raw['nodes']['x'])
        raw['nodes']['y']['outputs'][0]['name'] = 'y'
        raw['nodes']['sum_1']['inputs'].append('y')
        launch = meta['kernels'][0]
        launch['arg_order'].insert(0, 'y')
        descriptor = deepcopy(next(arg for arg in launch['args'] if arg['name'] == 'x'))
        descriptor['name'] = 'y'
        launch['args'].insert(0, descriptor)
        load, accum = get_tile(raw).body[0].body
        second = replace(load, names=('second',), input='y')
        product = Assign('product', ElementwiseImpl('multiply'), (load.names[0], 'second'), accum.dtype)
        square = Assign('squared', ElementwiseImpl('multiply'), ('product', 'product'), accum.dtype)
        change_loop(raw, (load, second, product, square, replace(accum, value='squared')))
        x = (np.arange(73) % 7 - 3).astype(np.float32)
        y = (np.arange(73) % 5 - 2).astype(np.float32)
        result = run_reduction_ptx(compile_fixture(raw, meta), launch, {'x': x, 'y': y})
        np.testing.assert_array_equal(result.outputs['sum_1'], [np.sum((x * y) ** 2, dtype=np.float32)])
        self.assertTrue(all(np.all(count == 1) for count in result.load_counts.values()))

    def test_shared_scalar_operations_compose_with_reductions(self):
        for operation, reducer in (('absolute', 'maximum'), ('exp', 'add'), ('sigmoid', 'add'), ('tanh', 'add')):
            with self.subTest(operation=operation, reducer=reducer):
                raw, meta = sum_fixture(73, 64)
                change_reducer(raw, reducer)
                load, accum = get_tile(raw).body[0].body
                transformed = Assign('transformed', ElementwiseImpl(operation), load.names, accum.dtype)
                change_loop(raw, (load, transformed, replace(accum, value='transformed')))
                values = np.linspace(-3, 4, 73, dtype=np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                transformed_values = ElementwiseImpl(operation)(values).astype(np.float32)
                expected = np.max(transformed_values) if reducer == 'maximum' else np.sum(transformed_values, dtype=np.float32)
                np.testing.assert_allclose(result.outputs['sum_1'], [expected], rtol=2e-6, atol=2e-6)

    def test_loop_literal_and_minimum_or_product_identities(self):
        for operation, values, expected in (
            ('minimum', np.array([9, 4, 7], dtype=np.float32), 4),
            ('multiply', np.array([2, 3, 4], dtype=np.float32), 24),
        ):
            with self.subTest(operation=operation):
                raw, meta = sum_fixture(3, 128)
                change_reducer(raw, operation)
                load, accum = get_tile(raw).body[0].body
                one = Let('unit', Literal(1.0, 'float'), accum.dtype)
                transformed = Assign('transformed', ElementwiseImpl('multiply'), (load.names[0], 'unit'), accum.dtype)
                change_loop(raw, (one, load, transformed, replace(accum, value='transformed')))
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                np.testing.assert_array_equal(result.outputs['sum_1'], [expected])

    def test_selecting_reductions_preserve_ir_identity(self):
        for operation in ('maximum', 'minimum'):
            with self.subTest(operation=operation):
                raw, meta = sum_fixture(3, 64)
                change_reducer(raw, operation)
                identity = np.float32(ElementwiseImpl(operation).identity)
                values = np.full(3, identity * np.float32(10), dtype=np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                np.testing.assert_array_equal(result.outputs['sum_1'], [identity])

    def test_composed_loop_rejects_invalid_ssa_and_multiple_accumulators(self):
        for case in ('undefined operand', 'duplicate name', 'multiple accumulators', 'changed combine'):
            with self.subTest(case=case):
                raw, meta = sum_fixture(33)
                load, accum = get_tile(raw).body[0].body
                if case == 'undefined operand':
                    changed = Assign('square', ElementwiseImpl('multiply'), ('missing', load.names[0]), accum.dtype)
                    change_loop(raw, (load, changed, replace(accum, value='square')))
                elif case == 'duplicate name':
                    changed = Assign(load.names[0], ElementwiseImpl('multiply'), (load.names[0], load.names[0]), accum.dtype)
                    change_loop(raw, (load, changed, accum))
                elif case == 'multiple accumulators':
                    change_loop(raw, (load, accum, replace(accum, name='other')))
                else:
                    change_reducer(raw, 'maximum')
                    tile = get_tile(raw)
                    body = list(tile.body)
                    combine, = body[5].combine_states
                    body[5] = replace(body[5], combine_states=(replace(combine, op=ElementwiseImpl('add')),))
                    put_tile(raw, replace(tile, body=tuple(body)))
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)


if __name__ == '__main__':
    unittest.main()
