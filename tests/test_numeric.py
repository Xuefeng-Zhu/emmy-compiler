"""CPU-model numeric/address checks; these never assemble PTX or execute a GPU.

The model checks emitted operand flow against independent NumPy/official data.
It cannot prove PTX legality, NVIDIA instruction behavior or hardware performance.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy_ptx import compile_file

from ptx_model import input_buffers, run_ptx

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ('p00_scalar_add', 'p01_vector_add', 'p02_scale_shift', 'p03_relu_mul')


def assert_official_array_matches(actual, expected, *, rtol, atol):
    """Keep submit.py's NaN policy and the problem's unchanged tolerances."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)


def fixture(problem):
    folder = ROOT / '.reference/problems' / problem
    paths = sorted(folder.glob('*.ir.json'))
    if len(paths) != 1:
        raise AssertionError(f'Run tools/fetch_reference.py to prepare {problem}')
    return json.loads(paths[0].read_text()), json.loads((folder / 'meta.json').read_text())


def compile_fixture(ir, meta):
    with TemporaryDirectory(prefix='emmy-numeric-') as tmp:
        path = Path(tmp) / 'kernel.ir.json'
        path.write_text(json.dumps(ir))
        path.with_name('meta.json').write_text(json.dumps(meta))
        return compile_file(path)


def scalar_fixture(size, operations):
    """Use genuine Emmy objects to derive a generic width-1 pointwise program."""
    ir, meta = fixture('p00_scalar_add')
    tile = Graph.from_dict(ir).nodes['add'].op.body[0]
    axis = tile.axes[0].name
    load0 = replace(tile.body[0], names=('left',), index=(Var(axis),))
    load1 = replace(tile.body[1], names=('right',), index=(Var(axis),))
    statements = [load0, load1]
    value = 'left'
    for i, (op, rhs) in enumerate(operations):
        name = f'result_{i}'
        args = (value,) if rhs is None else (value, rhs)
        statements.append(replace(tile.body[2], name=name, op=ElementwiseImpl(op), args=args))
        value = name
    statements.append(replace(tile.body[-1], index=(Var(axis),), values=(value,)))
    tile = replace(tile, axes=(replace(tile.axes[0], extent=Dim(size)),), body=tuple(statements))
    ir['nodes']['add']['op_fields']['body'] = [repr(tile)]
    for node in ir['nodes'].values():
        for output in node['outputs']:
            output['shape'] = [size]
    launch = meta['kernels'][0]
    for arg in launch['args']:
        arg['shape'] = [size]
    launch['grid'] = [(size + launch['block'][0] - 1) // launch['block'][0], 1, 1]
    return ir, meta


class NumericModelTests(unittest.TestCase):
    def test_four_official_problems_match_expected_in_cpu_model(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                ir, meta = fixture(problem)
                result = run_ptx(compile_fixture(ir, meta), meta['kernels'][0])
                expected_path = ROOT / '.reference/problems' / problem / 'expected.npz'
                with np.load(expected_path, allow_pickle=False) as expected:
                    self.assertEqual(set(result.outputs), set(expected.files))
                    for name in expected.files:
                        assert_official_array_matches(result.outputs[name], expected[name], rtol=meta['rtol'], atol=meta['atol'])

    def test_official_comparator_nan_and_signed_zero_policy(self):
        cases = (
            ('nan/nan', [np.nan], [np.nan], False),
            ('nan/zero', [np.nan], [0.0], False),
            ('zero/nan', [0.0], [np.nan], False),
            ('signed zeros', [-0.0, 0.0], [0.0, -0.0], True),
        )
        for label, actual, expected, accepts in cases:
            with self.subTest(case=label):
                actual = np.asarray(actual, dtype=np.float32)
                expected = np.asarray(expected, dtype=np.float32)
                if accepts:
                    assert_official_array_matches(actual, expected, rtol=0, atol=0)
                else:
                    with self.assertRaises(AssertionError):
                        assert_official_array_matches(actual, expected, rtol=0, atol=0)

    def test_odd_lengths_and_block_boundaries_in_cpu_model(self):
        for size in (1, 3, 255, 256, 257, 1023, 1025):
            with self.subTest(size=size):
                ir, meta = scalar_fixture(size, [('add', 'right')])
                launch = meta['kernels'][0]
                inputs = input_buffers(launch)
                result = run_ptx(compile_fixture(ir, meta), launch)
                expected = np.add(inputs['x0'], inputs['x1'], dtype=np.float32)
                np.testing.assert_array_equal(result.outputs['add'], expected)
                self.assertTrue(np.all(result.store_counts['add'] == 1))

    def test_parameter_reordering_preserves_computation(self):
        ir, meta = scalar_fixture(257, [('multiply', 'right')])
        launch = meta['kernels'][0]
        launch['arg_order'] = ['add', 'x1', 'x0']
        descriptors = {arg['name']: arg for arg in launch['args']}
        launch['args'] = [descriptors[name] for name in launch['arg_order']]
        inputs = input_buffers(launch)
        result = run_ptx(compile_fixture(ir, meta), launch)
        np.testing.assert_array_equal(result.outputs['add'], np.multiply(inputs['x0'], inputs['x1'], dtype=np.float32))

    def test_constant_values_are_read_at_runtime_for_varied_shapes(self):
        base_ir, base_meta = fixture('p02_scale_shift')
        base_tile = Graph.from_dict(base_ir).nodes['add'].op.body[0]
        for size in (4, 12, 1028):
            prior_ptx = None
            for factor, shift in ((np.float32(-3.25), np.float32(0.125)), (np.float32(0), np.float32(-2.5))):
                with self.subTest(size=size, factor=factor, shift=shift):
                    ir, meta = copy.deepcopy(base_ir), copy.deepcopy(base_meta)
                    tile = replace(base_tile, axes=(replace(base_tile.axes[0], extent=Dim(size // 4)),))
                    ir['nodes']['add']['op_fields']['body'] = [repr(tile)]
                    for name in ('x', 'add'):
                        ir['nodes'][name]['outputs'][0]['shape'] = [size]
                    for name, value in (('mul_c1', factor), ('add_c1', shift)):
                        ir['nodes'][name]['op_fields']['value'] = float(value)
                    launch = meta['kernels'][0]
                    for arg in launch['args']:
                        if arg['name'] in ('x', 'add'):
                            arg['shape'] = [size]
                        if arg['name'] == 'mul_c1':
                            arg['data'] = [float(factor)]
                        if arg['name'] == 'add_c1':
                            arg['data'] = [float(shift)]
                    launch['grid'] = [((size // 4) + 255) // 256, 1, 1]
                    ptx = compile_fixture(ir, meta)
                    if prior_ptx is not None:
                        self.assertEqual(ptx, prior_ptx, 'runtime constants must not be embedded in PTX')
                    prior_ptx = ptx
                    inputs = input_buffers(launch)
                    expected = np.add(np.multiply(inputs['x'], factor, dtype=np.float32), shift, dtype=np.float32)
                    np.testing.assert_array_equal(run_ptx(ptx, launch).outputs['add'], expected)

    def test_constant_buffer_can_have_multiple_elements(self):
        ir, meta = scalar_fixture(9, [('multiply', 'right')])
        ir['inputs'].remove('x1')
        ir['nodes']['x1']['op'] = 'ConstantOp'
        ir['nodes']['x1']['op_fields'] = {'name': 'x1'}
        launch = meta['kernels'][0]
        descriptor = next(arg for arg in launch['args'] if arg['name'] == 'x1')
        descriptor.pop('seed')
        descriptor['data'] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        inputs = input_buffers(launch)
        result = run_ptx(compile_fixture(ir, meta), launch)
        np.testing.assert_array_equal(result.outputs['add'], np.multiply(inputs['x0'], inputs['x1'], dtype=np.float32))

    def test_relu_nan_zero_infinity_and_subnormal_semantics(self):
        # PTX max treats +0 > -0; NumPy's host zero-sign choice is platform
        # dependent. Compare host values, then independently check PTX bits.
        # The challenge CUDA fmaxf reference still differs on NaN.
        tiny = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
        values = np.array([-np.inf, -1, -tiny, -0.0, 0.0, tiny, 1, np.inf, np.nan], dtype=np.float32)
        ir, meta = scalar_fixture(len(values), [('relu', None)])
        ptx = compile_fixture(ir, meta)
        self.assertIn('max.NaN.f32', ptx)
        self.assertNotIn('.ftz', ptx)
        actual = run_ptx(ptx, meta['kernels'][0], {'x0': values}).outputs['add']
        expected = ElementwiseImpl('relu')(values)
        np.testing.assert_array_equal(actual, expected)
        ptx_bits = np.array([0, 0, 0, 0, 0, 1, 0x3f800000, 0x7f800000], dtype=np.uint32)
        np.testing.assert_array_equal(actual[:-1].view(np.uint32), ptx_bits)
        self.assertTrue(np.isnan(actual[-1]))
        self.assertFalse(np.signbit(actual[3]))
        self.assertEqual(actual[5], tiny)

    def test_relu_multiply_matches_expected_generation_expression(self):
        # Official tools/expressions_open.py p03: np.maximum(x, 0) * y.
        # This checks operation semantics, not runner acceptance of NaN outputs.
        tiny = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
        x = np.array([np.nan, np.nan, -np.inf, -1, -0.0, 0.0, tiny, np.inf, np.inf], dtype=np.float32)
        y = np.array([1, -2, 1, 1, 1, -2, 1, 1, 0], dtype=np.float32)
        ir, meta = scalar_fixture(len(x), [('relu', None), ('multiply', 'right')])
        actual = run_ptx(compile_fixture(ir, meta), meta['kernels'][0], {'x0': x, 'x1': y}).outputs['add']
        with np.errstate(invalid='ignore'):
            expected = np.maximum(x, 0) * y
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        non_nan = ~np.isnan(expected)
        # The official comparison treats signed zeros as equal. Keep exact
        # value equality here and specify the PTX zero signs independently.
        np.testing.assert_array_equal(actual[non_nan], expected[non_nan])
        ptx_bits = np.array([0, 0, 0, 0x80000000, 1, 0x7f800000], dtype=np.uint32)
        np.testing.assert_array_equal(actual[non_nan].view(np.uint32), ptx_bits)

    def test_multiply_rounding_and_special_values(self):
        tiny = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
        left = np.array([tiny, -tiny, -0.0, 0.0, np.inf, -np.inf, np.nan, np.finfo(np.float32).max], dtype=np.float32)
        right = np.array([1, 1, 1, -1, 0, 2, 3, 2], dtype=np.float32)
        ir, meta = scalar_fixture(len(left), [('multiply', 'right')])
        ptx = compile_fixture(ir, meta)
        self.assertIn('mul.rn.f32', ptx)
        self.assertNotIn('.ftz', ptx)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            expected = np.multiply(left, right, dtype=np.float32)
        actual = run_ptx(ptx, meta['kernels'][0], {'x0': left, 'x1': right}).outputs['add']
        np.testing.assert_array_equal(actual, expected)
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))

    def test_multiply_add_remains_separately_rounded(self):
        # a*b + a with b near -1 can distinguish fused and separate rounding.
        a = np.float32(1 + 2**-23)
        b = np.float32(-1 + 2**-24)
        ir, meta = scalar_fixture(1, [('multiply', 'right'), ('add', 'left')])
        ptx = compile_fixture(ir, meta)
        self.assertIn('mul.rn.f32', ptx)
        self.assertIn('add.rn.f32', ptx)
        self.assertNotIn('fma.', ptx)
        separate = np.add(np.multiply(a, b, dtype=np.float32), a, dtype=np.float32)
        fused = np.float32(np.float64(a) * np.float64(b) + np.float64(a))
        self.assertNotEqual(separate, fused, 'test inputs must detect accidental FMA')
        actual = run_ptx(ptx, meta['kernels'][0], {'x0': np.array([a]), 'x1': np.array([b])}).outputs['add']
        self.assertEqual(actual[0], separate)

    def test_model_rejects_unknown_instruction(self):
        ir, meta = scalar_fixture(3, [('add', 'right')])
        ptx = compile_fixture(ir, meta).replace('add.rn.f32', 'div.approx.f32')
        with self.assertRaisesRegex(AssertionError, 'unsupported modeled instruction'):
            run_ptx(ptx, meta['kernels'][0])


if __name__ == '__main__':
    unittest.main()
