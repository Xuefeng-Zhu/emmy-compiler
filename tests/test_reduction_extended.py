"""Local mean/row-sum checks; model results do not establish GPU correctness."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy_ptx import CompileError
from emmy_ptx.loader import load_kernel

from reduction_model import run_reduction_ptx
from test_reduction import fixture, get_tile, put_tile, compile_fixture

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ('r03_mean', 'r04_row_sum')


def resize_schedule(tile, size, threads):
    body = list(tile.body)
    body[0] = replace(body[0], axis=replace(body[0].axis, extent=Dim(size)),
                      step=Literal(threads, 'int'))
    body[2] = replace(body[2], extents=(threads // 32,))
    body[5] = replace(body[5], length=threads // 32)
    axes = (*tile.axes[:-1], replace(tile.axes[-1], extent=Dim(threads)))
    return replace(tile, axes=axes, block_threads=threads, body=tuple(body))


def mean_fixture(size, divisor=None, threads=128):
    raw, meta = fixture('r03_mean')
    divisor = float(size) if divisor is None else float(divisor)
    put_tile(raw, resize_schedule(get_tile(raw), size, threads))
    raw['nodes']['x']['outputs'][0]['shape'] = [size]
    raw['nodes']['mean_keepdim_count']['op_fields']['value'] = divisor
    launch = meta['kernels'][0]
    launch['block'] = [threads, 1, 1]
    for arg in launch['args']:
        if arg['name'] == 'x':
            arg['shape'] = [size]
        elif arg['name'] == 'mean_keepdim_count':
            arg['data'] = [divisor]
    return raw, meta


def row_fixture(rows, columns, threads=128):
    raw, meta = fixture('r04_row_sum')
    tile = resize_schedule(get_tile(raw), columns, threads)
    tile = replace(tile, axes=(replace(tile.axes[0], extent=Dim(rows)), tile.axes[1]))
    put_tile(raw, tile)
    raw['nodes']['x']['outputs'][0]['shape'] = [rows, columns]
    raw['nodes']['sum_1']['outputs'][0]['shape'] = [rows]
    launch = meta['kernels'][0]
    launch['grid'] = [rows, 1, 1]
    launch['block'] = [threads, 1, 1]
    for arg in launch['args']:
        arg['shape'] = [rows, columns] if arg['name'] == 'x' else [rows]
    return raw, meta


def change_final(raw, body):
    tile = get_tile(raw)
    statements = list(tile.body)
    statements[-1] = replace(statements[-1], body=tuple(body))
    put_tile(raw, replace(tile, body=tuple(statements)))


def renamed(value, names):
    if isinstance(value, dict):
        return {names.get(key, key): renamed(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [renamed(item, names) for item in value]
    if isinstance(value, str):
        if value in names:
            return names[value]
        for old, new in names.items():
            value = value.replace(repr(old), repr(new))
    return value


class ExtendedReductionTests(unittest.TestCase):
    def test_official_mean_and_row_sum_match_expected_in_cpu_model(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                raw, meta = fixture(problem)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0])
                with np.load(ROOT / '.reference/problems' / problem / 'expected.npz', allow_pickle=False) as expected:
                    self.assertEqual(set(result.outputs), set(expected.files))
                    for name in expected.files:
                        actual = result.outputs[name].reshape(expected[name].shape)
                        np.testing.assert_allclose(actual, expected[name], rtol=meta['rtol'],
                                                   atol=meta['atol'], equal_nan=False)
                        self.assertTrue(np.all(result.store_counts[name] == 1))
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                rows = meta['kernels'][0]['grid'][0]
                self.assertEqual(result.barrier_count, 3 * rows)

    def test_loader_keeps_rank_dimensions_and_flat_allocation_counts(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem), TemporaryDirectory(prefix='emmy-dimensions-') as tmp:
                raw, meta = fixture(problem)
                path = Path(tmp) / 'kernel.ir.json'
                path.write_text(json.dumps(raw))
                path.with_name('meta.json').write_text(json.dumps(meta))
                kernel = load_kernel(path)
                if problem == 'r03_mean':
                    self.assertEqual(kernel.scalar_outputs, frozenset({'mean'}))
                    self.assertEqual(kernel.dimensions['mean'], ())
                    self.assertEqual(kernel.shapes['mean'], 1)
                    self.assertEqual(kernel.dimensions['mean_keepdim_count'], (1,))
                else:
                    self.assertEqual(kernel.dimensions['x'], (64, 512))
                    self.assertEqual(kernel.shapes['x'], 64 * 512)
                    self.assertEqual(kernel.dimensions['sum_1'], (64,))
                    self.assertEqual(kernel.scalar_outputs, frozenset())

    def test_mean_emits_runtime_pointer_load_and_rounded_division(self):
        raw, meta = mean_fixture(129, divisor=7)
        ptx = compile_fixture(raw, meta)
        self.assertIn('.param .u64 param_mean_keepdim_count', ptx)
        self.assertEqual(ptx.count('div.rn.f32'), 1)
        self.assertNotIn('div.approx', ptx)
        self.assertNotIn('rcp.', ptx)
        self.assertNotIn('.ftz', ptx)
        self.assertEqual(ptx.count('st.global.f32'), 1)
        self.assertEqual(ptx.count('bar.sync'), 3)

    def test_mean_runtime_divisor_is_independent_of_input_length(self):
        size = 33
        values = np.ones(size, dtype=np.float32)
        baseline_ptx = None
        for divisor in (1, 3, 7, -2.5):
            with self.subTest(divisor=divisor):
                raw, meta = mean_fixture(size, divisor=divisor, threads=64)
                ptx = compile_fixture(raw, meta)
                if baseline_ptx is not None:
                    self.assertEqual(ptx, baseline_ptx, 'Runtime divisor must not enter generated PTX')
                baseline_ptx = ptx
                result = run_reduction_ptx(ptx, meta['kernels'][0], {'x': values})
                expected = np.divide(np.float32(size), np.float32(divisor), dtype=np.float32)
                self.assertEqual(result.outputs['mean'][0], expected)
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                np.testing.assert_array_equal(result.load_counts['mean_keepdim_count'], [1])
                np.testing.assert_array_equal(result.store_counts['mean'], [1])

    def test_mean_tails_and_idle_lanes(self):
        for size in (1, 3, 31, 32, 33, 127, 128, 129, 1025):
            with self.subTest(size=size):
                raw, meta = mean_fixture(size)
                values = np.ones(size, dtype=np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                self.assertEqual(result.outputs['mean'][0], np.float32(1))
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                np.testing.assert_array_equal(result.store_counts['mean'], [1])

    def test_row_lengths_row_offsets_and_idle_lanes(self):
        for rows, columns in ((1, 1), (2, 3), (3, 31), (2, 32), (5, 33),
                              (3, 127), (2, 128), (3, 129), (5, 513)):
            with self.subTest(rows=rows, columns=columns):
                raw, meta = row_fixture(rows, columns)
                values = (np.arange(rows)[:, None] * 5 - 7 + np.arange(columns)[None, :] % 7).astype(np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                expected = np.sum(values, axis=1, dtype=np.float64).astype(np.float32)
                np.testing.assert_array_equal(result.outputs['sum_1'], expected)
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                self.assertTrue(np.all(result.store_counts['sum_1'] == 1))
                self.assertEqual(result.barrier_count, rows * 3)

    def test_row_sum_and_mean_support_changed_block_sizes(self):
        for threads in (32, 64, 128, 256, 512, 1024):
            with self.subTest(threads=threads):
                rows, columns = 3, 2 * threads + 3
                raw, meta = row_fixture(rows, columns, threads)
                values = (np.arange(rows)[:, None] + np.arange(columns)[None, :] % 5 - 2).astype(np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                np.testing.assert_array_equal(result.outputs['sum_1'], np.sum(values, axis=1, dtype=np.float64).astype(np.float32))
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                self.assertTrue(np.all(result.store_counts['sum_1'] == 1))
                self.assertEqual(result.barrier_count, rows * (threads // 32).bit_length())
                raw, meta = mean_fixture(columns, threads=threads)
                ones = np.ones(columns, dtype=np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': ones})
                self.assertEqual(result.outputs['mean'][0], np.float32(1))

    def test_changed_names_and_reordered_parameters_for_both_topologies(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                if problem == 'r03_mean':
                    raw, meta = mean_fixture(33, divisor=3, threads=64)
                    names = {'x': 'data', 'mean': 'answer', 'mean_keepdim_count': 'divisor',
                             'k_mean_ffcca3': 'renamed_mean', 'a0': 'index', 'a0_co': 'worker',
                             'in1': 'denominator', 'v0': 'quotient'}
                    values = np.ones(33, dtype=np.float32)
                    expected = np.array([11], dtype=np.float32)
                else:
                    raw, meta = row_fixture(3, 33, threads=64)
                    names = {'x': 'data', 'sum_1': 'answer', 'k_sum_1_reduce': 'renamed_rows',
                             'a0': 'row', 'a1': 'column', 'a1_co': 'worker'}
                    values = np.arange(99, dtype=np.float32).reshape(3, 33)
                    expected = np.sum(values, axis=1, dtype=np.float64).astype(np.float32)
                names.update({'in0': 'element', 'acc0': 'total', 'acc0__o': 'peer', 'acc0_smem': 'scratch'})
                raw, meta = renamed(raw, names), renamed(meta, names)
                launch = meta['kernels'][0]
                launch['arg_order'].reverse()
                launch['args'].reverse()
                result = run_reduction_ptx(compile_fixture(raw, meta), launch, {'data': values})
                np.testing.assert_array_equal(result.outputs['answer'], expected)
                self.assertTrue(np.all(result.load_counts['data'] == 1))

    def test_unsupported_mean_epilogues_fail_clearly(self):
        for case in ('multiply', 'reversed operands', 'unbound operand', 'non-f32 division',
                     'out-of-range divisor index', 'wrong result', 'missing division', 'unguarded write'):
            with self.subTest(case=case):
                raw, meta = mean_fixture(33)
                tile = get_tile(raw)
                final = tile.body[-1]
                body = list(final.body)
                if case == 'multiply':
                    body[1] = replace(body[1], op=ElementwiseImpl('multiply'))
                elif case == 'reversed operands':
                    body[1] = replace(body[1], args=tuple(reversed(body[1].args)))
                elif case == 'unbound operand':
                    body[1] = replace(body[1], args=('missing', body[1].args[1]))
                elif case == 'non-f32 division':
                    body[1] = replace(body[1], dtype=None)
                elif case == 'out-of-range divisor index':
                    body[0] = replace(body[0], index=(Literal(1, 'int'),))
                elif case == 'wrong result':
                    body[2] = replace(body[2], values=('acc0',))
                elif case == 'missing division':
                    body.pop(1)
                else:
                    statements = (*tile.body[:-1], replace(final, cond=replace(final.cond, right=Literal(1, 'int'))))
                    put_tile(raw, replace(tile, body=statements))
                if case != 'unguarded write':
                    change_final(raw, body)
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_mean_divisor_descriptor_and_shape_mismatches_fail_clearly(self):
        for case in ('two elements', 'scalar argument', 'rank-two divisor', 'wrong output shape'):
            with self.subTest(case=case):
                raw, meta = mean_fixture(33)
                args = {arg['name']: arg for arg in meta['kernels'][0]['args']}
                if case == 'two elements':
                    raw['nodes']['mean_keepdim_count']['outputs'][0]['shape'] = [2]
                    args['mean_keepdim_count'].update(shape=[2], data=[33.0, 33.0])
                elif case == 'scalar argument':
                    args['mean_keepdim_count']['scalar'] = True
                elif case == 'rank-two divisor':
                    raw['nodes']['mean_keepdim_count']['outputs'][0]['shape'] = [1, 1]
                    args['mean_keepdim_count']['shape'] = [1, 1]
                else:
                    raw['nodes']['mean']['outputs'][0]['shape'] = [2]
                    args['mean']['shape'] = [2]
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_unsupported_row_shapes_and_grid_mismatches_fail_clearly(self):
        for case in ('empty rows', 'empty columns', 'rank-three input', 'scalar output',
                     'rank-two output', 'wrong output length', 'wrong row extent', 'grid too small', 'grid too large'):
            with self.subTest(case=case):
                raw, meta = row_fixture(3, 33)
                args = {arg['name']: arg for arg in meta['kernels'][0]['args']}
                if case == 'empty rows':
                    raw, meta = row_fixture(0, 33)
                elif case == 'empty columns':
                    raw, meta = row_fixture(3, 0)
                elif case == 'rank-three input':
                    raw['nodes']['x']['outputs'][0]['shape'] = [1, 3, 33]
                    args['x']['shape'] = [1, 3, 33]
                elif case == 'scalar output':
                    raw['nodes']['sum_1']['outputs'][0]['shape'] = []
                    args['sum_1']['shape'] = [1]
                elif case == 'rank-two output':
                    raw['nodes']['sum_1']['outputs'][0]['shape'] = [3, 1]
                    args['sum_1']['shape'] = [3, 1]
                elif case == 'wrong output length':
                    raw['nodes']['sum_1']['outputs'][0]['shape'] = [4]
                    args['sum_1']['shape'] = [4]
                elif case == 'wrong row extent':
                    tile = get_tile(raw)
                    put_tile(raw, replace(tile, axes=(replace(tile.axes[0], extent=Dim(2)), tile.axes[1])))
                elif case == 'grid too small':
                    meta['kernels'][0]['grid'] = [2, 1, 1]
                else:
                    meta['kernels'][0]['grid'] = [4, 1, 1]
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_transposed_or_mismatched_row_indices_fail_clearly(self):
        for case in ('transposed input', 'fixed row input', 'loop axis output', 'scalar output index', 'row leader condition'):
            with self.subTest(case=case):
                raw, meta = row_fixture(3, 33)
                tile = get_tile(raw)
                body = list(tile.body)
                loop, final = body[0], body[-1]
                if case == 'transposed input':
                    load = replace(loop.body[0], index=tuple(reversed(loop.body[0].index)))
                    body[0] = replace(loop, body=(load, loop.body[1]))
                elif case == 'fixed row input':
                    load = replace(loop.body[0], index=(Literal(0, 'int'), loop.body[0].index[1]))
                    body[0] = replace(loop, body=(load, loop.body[1]))
                elif case == 'loop axis output':
                    body[-1] = replace(final, body=(replace(final.body[0], index=(Var(loop.axis.name),)),))
                elif case == 'scalar output index':
                    body[-1] = replace(final, body=(replace(final.body[0], index=()),))
                else:
                    body[-1] = replace(final, cond=replace(final.cond, left=Var(tile.axes[0].name)))
                put_tile(raw, replace(tile, body=tuple(body)))
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_cli_success_and_bad_epilogue_failure_are_clean(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                source, = (ROOT / '.reference/problems' / problem).glob('*.ir.json')
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(source)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                self.assertTrue(result.stdout.startswith('.version '))
        raw, meta = mean_fixture(33)
        body = list(get_tile(raw).body[-1].body)
        body[1] = replace(body[1], op=ElementwiseImpl('multiply'))
        change_final(raw, body)
        with TemporaryDirectory(prefix='emmy-mean-cli-') as tmp:
            path = Path(tmp) / 'kernel.ir.json'
            path.write_text(json.dumps(raw))
            path.with_name('meta.json').write_text(json.dumps(meta))
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(path)],
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        self.assertTrue(result.stderr.strip())
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
