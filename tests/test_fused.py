"""Fused cooperative IR checks using a CPU PTX model, never GPU execution."""
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
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign
from emmy_ptx import CompileError

from reduction_model import run_reduction_ptx
from test_reduction import fixture, get_tile, put_tile, compile_fixture
from test_reduction_extended import renamed

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ('f01_rmsnorm', 'f02_row_softmax', 'f03_l2norm')
OUTPUTS = {'f01_rmsnorm': 'mul_1', 'f02_row_softmax': 'softmax', 'f03_l2norm': 'div'}


def fused_fixture(problem, rows=2, columns=33, threads=128, count=None, epsilon=1e-5):
    """Resize the actual typed official IR, preserving both strided passes."""
    raw, meta = fixture(problem)
    tile = get_tile(raw)
    body = []
    for stmt in tile.body:
        if isinstance(stmt, StridedLoop):
            stmt = replace(stmt, axis=replace(stmt.axis, extent=Dim(columns)),
                           step=Literal(threads, 'int'))
        elif isinstance(stmt, Smem):
            stmt = replace(stmt, extents=(threads // 32,))
        elif isinstance(stmt, TreeHalve):
            stmt = replace(stmt, length=threads // 32)
        body.append(stmt)
    axes = (replace(tile.axes[0], extent=Dim(rows)),
            replace(tile.axes[1], extent=Dim(threads)))
    put_tile(raw, replace(tile, axes=axes, body=tuple(body), block_threads=threads))
    shapes = {'x': [rows, columns], OUTPUTS[problem]: [rows, columns]}
    if problem == 'f01_rmsnorm':
        shapes['w'] = [columns]
        constants = {'mean_count': float(columns if count is None else count), 'add_c1': float(epsilon)}
        for name, value in constants.items():
            raw['nodes'][name]['op_fields']['value'] = value
    else:
        constants = {}
    for node in raw['nodes'].values():
        for output in node['outputs']:
            if output['name'] in shapes:
                output['shape'] = shapes[output['name']]
    launch = meta['kernels'][0]
    launch['grid'], launch['block'] = [rows, 1, 1], [threads, 1, 1]
    for arg in launch['args']:
        if arg['name'] in shapes:
            arg['shape'] = shapes[arg['name']]
        if arg['name'] in constants:
            arg['data'] = [constants[arg['name']]]
    return raw, meta


def values_for(rows, columns):
    # Distinct rows and mixed signs make wrong row strides/broadcasts observable.
    row = np.arange(rows, dtype=np.float32)[:, None]
    column = np.arange(columns, dtype=np.float32)[None, :]
    return ((column % 11 - 5) * np.float32(0.25) + row * np.float32(0.375) +
            np.float32(0.125)).astype(np.float32)


def expected_values(problem, values, weights=None, count=None, epsilon=1e-5):
    """Independent NumPy expressions from official metadata, not PTX instructions."""
    if problem == 'f02_row_softmax':
        exp = np.exp(values - np.max(values, axis=1, keepdims=True), dtype=np.float32)
        return exp / np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
    squared_sum = np.sum(values * values, axis=1, keepdims=True, dtype=np.float32)
    if problem == 'f01_rmsnorm':
        count = values.shape[1] if count is None else count
        return values / np.sqrt(squared_sum / np.float32(count) + np.float32(epsilon)) * weights
    return values / np.sqrt(squared_sum)


def change_statement(raw, cls, update):
    tile = get_tile(raw)
    body = list(tile.body)
    index = next(i for i, stmt in enumerate(body) if isinstance(stmt, cls))
    body[index] = update(body[index])
    put_tile(raw, replace(tile, body=tuple(body)))


def change_epilogue(raw, update):
    tile = get_tile(raw)
    body = list(tile.body)
    body[-1] = update(body[-1])
    put_tile(raw, replace(tile, body=tuple(body)))


class FusedTests(unittest.TestCase):
    def check_values(self, problem, raw, meta, values, weights=None, count=None, epsilon=1e-5):
        inputs = {'x': values}
        if weights is not None:
            inputs['w'] = weights
        result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], inputs)
        expected = expected_values(problem, values, weights, count, epsilon)
        actual = result.outputs[OUTPUTS[problem]].reshape(values.shape)
        np.testing.assert_allclose(actual, expected, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
        self.assertTrue(np.all(result.store_counts[OUTPUTS[problem]] == 1))
        self.assertTrue(np.all(result.load_counts['x'] == 2))
        if weights is not None:
            self.assertTrue(np.all(result.load_counts['w'] == values.shape[0]))
        threads = meta['kernels'][0]['block'][0]
        self.assertEqual(result.barrier_count, values.shape[0] * (threads // 32).bit_length())
        return result

    def test_official_fixtures_match_original_expected_arrays_in_cpu_model(self):
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
                self.assertTrue(np.all(result.load_counts['x'] == 2))

    def test_epilogue_loops_cover_tails_and_idle_lanes(self):
        for problem in PROBLEMS:
            for rows, columns in ((1, 1), (2, 3), (1, 31), (2, 32), (3, 33),
                                  (2, 127), (1, 128), (2, 129), (3, 257)):
                with self.subTest(problem=problem, rows=rows, columns=columns):
                    raw, meta = fused_fixture(problem, rows, columns)
                    weights = np.linspace(-1, 2, columns, dtype=np.float32) if problem == 'f01_rmsnorm' else None
                    self.check_values(problem, raw, meta, values_for(rows, columns), weights)

    def test_changed_block_sizes(self):
        for problem in PROBLEMS:
            for threads in (32, 64, 128, 256, 512, 1024):
                with self.subTest(problem=problem, threads=threads):
                    columns = 33
                    raw, meta = fused_fixture(problem, 2, columns, threads)
                    weights = np.linspace(0.5, 1.5, columns, dtype=np.float32) if problem == 'f01_rmsnorm' else None
                    self.check_values(problem, raw, meta, values_for(2, columns), weights)

    def test_rms_runtime_count_and_epsilon_are_not_folded(self):
        values = values_for(2, 33)
        weights = np.linspace(-2, 2, 33, dtype=np.float32)
        baseline = None
        outputs = []
        for count, epsilon in ((33, 1e-5), (7, 1e-5), (7, 0.75)):
            with self.subTest(count=count, epsilon=epsilon):
                raw, meta = fused_fixture('f01_rmsnorm', 2, 33, 64, count, epsilon)
                ptx = compile_fixture(raw, meta)
                if baseline is not None:
                    self.assertEqual(ptx, baseline, 'Runtime constant data must not change emitted PTX')
                baseline = ptx
                result = self.check_values('f01_rmsnorm', raw, meta, values, weights, count, epsilon)
                self.assertGreater(result.load_counts['mean_count'][0], 0)
                self.assertGreater(result.load_counts['add_c1'][0], 0)
                outputs.append(result.outputs['mul_1'])
        self.assertFalse(np.array_equal(outputs[0], outputs[1]))
        self.assertFalse(np.array_equal(outputs[1], outputs[2]))

    def test_zero_rms_input_and_softmax_constant_rows(self):
        raw, meta = fused_fixture('f01_rmsnorm', 3, 33, 128, epsilon=0.5)
        self.check_values('f01_rmsnorm', raw, meta, np.zeros((3, 33), dtype=np.float32),
                          np.ones(33, dtype=np.float32), epsilon=0.5)
        raw, meta = fused_fixture('f02_row_softmax', 3, 33, 128)
        values = np.array([-70, 0, 70], dtype=np.float32)[:, None] * np.ones((1, 33), dtype=np.float32)
        result = self.check_values('f02_row_softmax', raw, meta, values)
        np.testing.assert_allclose(result.outputs['softmax'], np.float32(1 / 33),
                                   rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)

    def test_softmax_multistate_shuffle_and_tree_snapshots(self):
        raw, meta = fused_fixture('f02_row_softmax', 3, 257, 128)
        columns = np.arange(257)
        values = np.vstack(((columns % 32 - 16) * 0.25,
                            ((columns // 32) % 2 * 24 - 12) + columns % 7,
                            np.where(columns == 256, 40, -40))).astype(np.float32)
        original = self.check_values('f02_row_softmax', raw, meta, values)
        ptx = compile_fixture(raw, meta)
        self.assertEqual(ptx.count('shfl.sync.bfly.b32'), 10)
        # Reorder both declared states and their peer/buffer pairing. Their
        # dependent combine expression still reads the same pre-combine states.
        change_statement(raw, WarpShuffle, lambda stmt: replace(
            stmt, state=stmt.state[::-1], state_b=stmt.state_b[::-1]))
        change_statement(raw, TreeHalve, lambda stmt: replace(
            stmt, state=stmt.state[::-1], state_b=stmt.state_b[::-1], bufs=stmt.bufs[::-1]))
        reordered = self.check_values('f02_row_softmax', raw, meta, values)
        np.testing.assert_array_equal(original.outputs['softmax'], reordered.outputs['softmax'])

    def test_renamed_buffers_axes_states_and_reordered_arguments(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                raw, meta = fused_fixture(problem, 2, 33, 64)
                names = {'x': 'samples', OUTPUTS[problem]: 'normalized',
                         'a0': 'row', 'a1_co': 'worker', 'a1': 'scan', 'a2': 'write_col',
                         'a3': 'write_col', 'acc0': 'state_max_or_sum', 'acc1': 'state_total',
                         'acc0__o': 'peer_max_or_sum', 'acc1__o': 'peer_total',
                         'acc0_smem': 'partial_first', 'acc1_smem': 'partial_second',
                         meta['kernels'][0]['entry']: 'renamed_fused_kernel'}
                if problem == 'f01_rmsnorm':
                    names.update({'w': 'weights', 'mean_count': 'count_data', 'add_c1': 'epsilon_data'})
                raw, meta = renamed(raw, names), renamed(meta, names)
                launch = meta['kernels'][0]
                launch['arg_order'].reverse()
                launch['args'].reverse()
                values = values_for(2, 33)
                weights = np.linspace(-1, 2, 33, dtype=np.float32) if problem == 'f01_rmsnorm' else None
                inputs = {'samples': values}
                if weights is not None:
                    inputs['weights'] = weights
                result = run_reduction_ptx(compile_fixture(raw, meta), launch, inputs)
                np.testing.assert_allclose(result.outputs['normalized'].reshape(values.shape),
                                           expected_values(problem, values, weights),
                                           rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
                self.assertTrue(np.all(result.store_counts['normalized'] == 1))
                self.assertTrue(np.all(result.load_counts['samples'] == 2))

    def test_emitted_numeric_operations_preserve_separate_rounding(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                raw, meta = fused_fixture(problem)
                ptx = compile_fixture(raw, meta)
                self.assertIn('mul.rn.f32', ptx)
                self.assertNotIn('fma.', ptx)
                self.assertNotIn('mad.f32', ptx)
                self.assertNotIn('.ftz', ptx)
                self.assertIn('div.rn.f32', ptx)
                if problem == 'f02_row_softmax':
                    self.assertIn('ex2.approx.f32', ptx)
                    self.assertIn('max.', ptx)
                else:
                    self.assertIn('sqrt.rn.f32', ptx)

    def test_unsupported_or_malformed_reduction_contracts_fail_clearly(self):
        mutations = {
            'non_string_accumulator': lambda raw: change_statement(raw, StridedLoop, lambda stmt: replace(stmt, body=tuple(replace(node, name=['acc0']) if isinstance(node, Accum) else node for node in stmt.body))),
            'duplicate_state': lambda raw: change_statement(raw, WarpShuffle, lambda stmt: replace(stmt, state=('acc0', 'acc0'))),
            'peer_alias': lambda raw: change_statement(raw, WarpShuffle, lambda stmt: replace(stmt, state_b=stmt.state)),
            'missing_shared_buffer': lambda raw: change_statement(raw, TreeHalve, lambda stmt: replace(stmt, bufs=('absent', stmt.bufs[1]))),
            'duplicate_tree_buffer': lambda raw: change_statement(raw, TreeHalve, lambda stmt: replace(stmt, bufs=(stmt.bufs[0], stmt.bufs[0]))),
            'counted_barrier': lambda raw: change_statement(raw, Sync, lambda stmt: replace(stmt, count=32)),
            'warp_only_barrier': lambda raw: change_statement(raw, Sync, lambda stmt: replace(stmt, warp=True)),
            'wrong_tree_length': lambda raw: change_statement(raw, TreeHalve, lambda stmt: replace(stmt, length=3)),
            'wrong_shared_size': lambda raw: change_statement(raw, Smem, lambda stmt: replace(stmt, extents=(3,))),
            'noninteger_shared_size': lambda raw: change_statement(raw, Smem, lambda stmt: replace(stmt, extents=(4.0,))),
            'non_full_warp': lambda raw: change_statement(raw, WarpShuffle, lambda stmt: replace(stmt, length=16)),
            'bad_epilogue_step': lambda raw: change_epilogue(raw, lambda stmt: replace(stmt, step=Literal(256, 'int'))),
            'custom_epilogue_end': lambda raw: change_epilogue(raw, lambda stmt: replace(stmt, end=Literal(1, 'int'))),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                raw, meta = fused_fixture('f02_row_softmax')
                mutate(raw)
                with self.assertRaisesRegex(CompileError, '.+'):
                    compile_fixture(raw, meta)

    def test_bad_epilogue_indices_and_operations_fail_clearly(self):
        def mutate_body(raw, update):
            change_epilogue(raw, lambda stmt: replace(stmt, body=update(list(stmt.body))))

        def wrong_row(body):
            body[-1] = replace(body[-1], index=(Literal(0, 'int'), body[-1].index[1]))
            return tuple(body)

        def transposed_load(body):
            body[0] = replace(body[0], index=body[0].index[::-1])
            return tuple(body)

        def bad_op(body):
            index = next(i for i, stmt in enumerate(body) if isinstance(stmt, Assign))
            body[index] = replace(body[index], op=ElementwiseImpl('sin'), args=(body[index].args[0],))
            return tuple(body)

        def undefined_value(body):
            body[-1] = replace(body[-1], values=('undefined',))
            return tuple(body)

        def missing_op(body):
            index = next(i for i, stmt in enumerate(body) if isinstance(stmt, Assign))
            body[index] = replace(body[index], op=None)
            return tuple(body)

        def missing_args(body):
            index = next(i for i, stmt in enumerate(body) if isinstance(stmt, Assign))
            body[index] = replace(body[index], args=None)
            return tuple(body)

        for label, update in (('fixed_row', wrong_row), ('transposed_input', transposed_load),
                              ('unsupported_op', bad_op), ('undefined_value', undefined_value),
                              ('missing_op', missing_op), ('missing_args', missing_args)):
            with self.subTest(label=label):
                raw, meta = fused_fixture('f03_l2norm')
                mutate_body(raw, update)
                with self.assertRaisesRegex(CompileError, '.+'):
                    compile_fixture(raw, meta)

    def test_shape_launch_and_output_coverage_mismatches_fail_clearly(self):
        for label in ('short_grid', 'wrong_block', 'wrong_output_shape', 'zero_columns'):
            with self.subTest(label=label):
                raw, meta = fused_fixture('f03_l2norm')
                if label == 'short_grid':
                    meta['kernels'][0]['grid'][0] = 1
                elif label == 'wrong_block':
                    meta['kernels'][0]['block'][0] = 64
                elif label == 'wrong_output_shape':
                    raw['nodes']['div']['outputs'][0]['shape'] = [2, 34]
                    next(arg for arg in meta['kernels'][0]['args'] if arg.get('output'))['shape'] = [2, 34]
                else:
                    raw, meta = fused_fixture('f03_l2norm', columns=0)
                with self.assertRaisesRegex(CompileError, '.+'):
                    compile_fixture(raw, meta)

    def test_cli_generates_ptx_and_rejects_malformed_combiner_without_traceback(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                source, = (ROOT / '.reference/problems' / problem).glob('*.ir.json')
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(source)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                self.assertIn('.visible .entry', result.stdout)
        raw, meta = fused_fixture('f02_row_softmax')
        change_statement(raw, WarpShuffle, lambda stmt: replace(stmt, combine_states=None))
        with TemporaryDirectory(prefix='emmy-fused-cli-') as tmp:
            source = Path(tmp) / 'kernel.ir.json'
            source.write_text(json.dumps(raw))
            source.with_name('meta.json').write_text(json.dumps(meta))
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(source)],
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        self.assertNotIn('Traceback', result.stderr)
        self.assertTrue(result.stderr)


if __name__ == '__main__':
    unittest.main()
