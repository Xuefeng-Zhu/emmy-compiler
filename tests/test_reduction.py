"""Local single-block sum checks; no PTX assembly, GPU or runner requests."""
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy_ptx import CompileError, compile_file
from emmy_ptx.loader import load_kernel

from reduction_model import run_reduction_ptx

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ('r01_array_sum', 'r02_big_sum')


def fixture(problem='r01_array_sum'):
    folder = ROOT / '.reference/problems' / problem
    source, = folder.glob('*.ir.json')
    return json.loads(source.read_text()), json.loads((folder / 'meta.json').read_text())


def get_tile(raw):
    node, = (node for node in Graph.from_dict(raw).nodes.values()
             if type(node.op).__name__ == 'KernelOp')
    return node.op.body[0]


def put_tile(raw, tile):
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    node['op_fields']['body'] = [repr(tile)]


def compile_fixture(raw, meta):
    with TemporaryDirectory(prefix='emmy-reduction-') as tmp:
        source = Path(tmp) / 'kernel.ir.json'
        source.write_text(json.dumps(raw))
        source.with_name('meta.json').write_text(json.dumps(meta))
        return compile_file(source)


def sum_fixture(size, threads=128):
    """Vary the real typed schedule and metadata together, not the public name."""
    raw, meta = fixture()
    tile = get_tile(raw)
    body = list(tile.body)
    loop = body[0]
    body[0] = replace(loop, axis=replace(loop.axis, extent=Dim(size)),
                      step=Literal(threads, 'int'))
    body[2] = replace(body[2], extents=(threads // 32,))
    body[5] = replace(body[5], length=threads // 32)
    tile = replace(tile, axes=(replace(tile.axes[0], extent=Dim(threads)),),
                   block_threads=threads, body=tuple(body))
    put_tile(raw, tile)
    raw['nodes']['x']['outputs'][0]['shape'] = [size]
    launch = meta['kernels'][0]
    launch['block'] = [threads, 1, 1]
    next(arg for arg in launch['args'] if arg['name'] == 'x')['shape'] = [size]
    return raw, meta


def change_body(raw, index, **fields):
    tile = get_tile(raw)
    body = list(tile.body)
    body[index] = replace(body[index], **fields)
    put_tile(raw, replace(tile, body=tuple(body)))


class ReductionTests(unittest.TestCase):
    def test_official_scalar_ir_and_pointer_abi(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem), TemporaryDirectory(prefix='emmy-reduction-') as tmp:
                raw, meta = fixture(problem)
                tile = get_tile(raw)
                self.assertIsInstance(tile, Tile)
                self.assertIsInstance(tile.body[0], StridedLoop)
                self.assertIsNone(tile.body[1].combine_states[0].dtype)
                source = Path(tmp) / 'kernel.ir.json'
                source.write_text(json.dumps(raw))
                source.with_name('meta.json').write_text(json.dumps(meta))
                kernel = load_kernel(source)
                self.assertEqual(kernel.scalar_outputs, frozenset({'sum_1'}))
                self.assertEqual(kernel.shapes['sum_1'], 1)
                self.assertEqual(raw['nodes']['sum_1']['outputs'][0]['shape'], [])
                self.assertEqual(meta['kernels'][0]['args'][1]['shape'], [1])
                ptx = compile_file(source)
                self.assertEqual(ptx.count('.param .u64'), 2)
                self.assertIn('.param .u64 param_sum_1', ptx)
                self.assertNotIn('.param .f32', ptx)

    def test_emission_has_strided_loop_full_warp_shuffle_and_barriers(self):
        raw, meta = sum_fixture(129)
        ptx = compile_fixture(raw, meta)
        self.assertIn('.target sm_89', ptx)
        self.assertIn('setp.ge.u32', ptx)
        self.assertIn('ld.global.f32', ptx)
        self.assertIn('add.rn.f32', ptx)
        self.assertIn('.shared', ptx)
        self.assertIn('st.shared.f32', ptx)
        self.assertIn('ld.shared.f32', ptx)
        self.assertEqual(ptx.count('shfl.sync.bfly.b32'), 5)
        self.assertEqual(ptx.count('0xffffffff'), 5)
        self.assertEqual(ptx.count('bar.sync'), 3)
        self.assertEqual(ptx.count('st.global.f32'), 1)
        self.assertNotIn('.ftz', ptx)
        self.assertNotIn('atom.', ptx)
        self.assertEqual(ptx.count('ret;'), 1)
        self.assertGreater(ptx.index('ret;'), ptx.rindex('bar.sync'))
        self.assertEqual(ptx, compile_fixture(raw, meta))

    def test_official_cases_match_expected_in_cooperative_cpu_model(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                raw, meta = fixture(problem)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0])
                with np.load(ROOT / '.reference/problems' / problem / 'expected.npz', allow_pickle=False) as expected:
                    reference = expected['sum_1']
                    self.assertEqual(reference.shape, ())
                    actual = result.outputs['sum_1'].reshape(reference.shape)
                    np.testing.assert_allclose(actual, reference, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
                np.testing.assert_array_equal(result.store_counts['sum_1'], [1])
                self.assertTrue(np.all(result.load_counts['x'] == 1))

    def test_tail_lengths_and_idle_lanes_use_neutral_zero(self):
        for size in (1, 3, 31, 32, 33, 127, 128, 129, 1025):
            with self.subTest(size=size):
                raw, meta = sum_fixture(size)
                values = np.ones(size, dtype=np.float32)
                result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
                self.assertEqual(result.outputs['sum_1'][0], np.float32(size))
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                np.testing.assert_array_equal(result.store_counts['sum_1'], [1])

    def test_supported_block_sizes_are_derived_from_the_schedule(self):
        for threads in (32, 64, 128, 256, 512, 1024):
            with self.subTest(threads=threads):
                size = 2 * threads + 3
                raw, meta = sum_fixture(size, threads)
                values = (np.arange(size) % 7 - 3).astype(np.float32)
                ptx = compile_fixture(raw, meta)
                result = run_reduction_ptx(ptx, meta['kernels'][0], {'x': values})
                self.assertEqual(result.outputs['sum_1'][0], np.float32(np.sum(values, dtype=np.float64)))
                self.assertEqual(ptx.count('bar.sync'), 1 + (threads // 32).bit_length() - 1)
                self.assertIn(f'.reqntid {threads}, 1, 1', ptx)
                self.assertTrue(np.all(result.load_counts['x'] == 1))
                np.testing.assert_array_equal(result.store_counts['sum_1'], [1])

    def test_all_zero_input_retains_additive_identity(self):
        raw, meta = sum_fixture(3, 64)
        values = np.array([-0.0, 0.0, -0.0], dtype=np.float32)
        result = run_reduction_ptx(compile_fixture(raw, meta), meta['kernels'][0], {'x': values})
        self.assertEqual(result.outputs['sum_1'].view(np.uint32)[0], 0)
        self.assertTrue(np.all(result.load_counts['x'] == 1))

    def test_buffer_entry_axis_ssa_names_and_argument_order_are_not_hardcoded(self):
        raw, meta = sum_fixture(33, 64)
        names = {'x': 'values', 'sum_1': 'result', 'k_sum_1_reduce': 'renamed_sum',
                 'a0_co': 'worker', 'a0': 'item_index', 'in0': 'loaded_value',
                 'acc0': 'total', 'acc0__o': 'peer', 'acc0_smem': 'partials'}

        def rename(value):
            if isinstance(value, dict):
                return {names.get(key, key): rename(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename(item) for item in value]
            if isinstance(value, str):
                if value in names:
                    return names[value]
                for old, new in names.items():
                    value = value.replace(repr(old), repr(new))
            return value

        raw, meta = rename(raw), rename(meta)
        launch = meta['kernels'][0]
        launch['arg_order'].reverse()
        launch['args'].reverse()
        values = np.arange(33, dtype=np.float32) - 16
        ptx = compile_fixture(raw, meta)
        self.assertIn('.entry renamed_sum(', ptx)
        self.assertLess(ptx.index('.param .u64 param_result'), ptx.index('.param .u64 param_values'))
        result = run_reduction_ptx(ptx, launch, {'values': values})
        self.assertEqual(result.outputs['result'][0], np.float32(0))
        self.assertTrue(np.all(result.load_counts['values'] == 1))

    def test_explicit_f32_combine_dtype_is_also_supported(self):
        raw, meta = fixture()
        tile = get_tile(raw)
        body = list(tile.body)
        dtype = body[0].body[1].dtype
        for index in (1, 5):
            combine = body[index].combine_states[0]
            body[index] = replace(body[index], combine_states=(replace(combine, dtype=dtype),))
        put_tile(raw, replace(tile, body=tuple(body)))
        self.assertIn('shfl.sync.bfly.b32', compile_fixture(raw, meta))

    def test_unsupported_shapes_and_scalar_descriptors_fail_clearly(self):
        for case in ('empty input', 'two-dimensional input', 'rank-one output', 'rank-two output', 'by-value output', 'empty meta shape', 'f16 input'):
            with self.subTest(case=case):
                raw, meta = fixture()
                if case == 'empty input':
                    raw, meta = sum_fixture(0)
                elif case == 'two-dimensional input':
                    raw['nodes']['x']['outputs'][0]['shape'] = [32, 32]
                    meta['kernels'][0]['args'][0]['shape'] = [32, 32]
                elif case == 'rank-one output':
                    raw['nodes']['sum_1']['outputs'][0]['shape'] = [1]
                elif case == 'rank-two output':
                    raw['nodes']['sum_1']['outputs'][0]['shape'] = [1, 1]
                elif case == 'by-value output':
                    meta['kernels'][0]['args'][1]['scalar'] = True
                elif case == 'empty meta shape':
                    meta['kernels'][0]['args'][1]['shape'] = []
                else:
                    raw['nodes']['x']['outputs'][0]['dtype'] = 'f16'
                    meta['kernels'][0]['args'][0]['dtype'] = 'f16'
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_unsupported_launch_and_cooperative_mapping_fail_clearly(self):
        for case in ('two blocks', 'block mismatch', 'partial warp', 'three warps', 'extent mismatch', 'aux threads'):
            with self.subTest(case=case):
                raw, meta = fixture()
                tile = get_tile(raw)
                if case == 'two blocks':
                    meta['kernels'][0]['grid'] = [2, 1, 1]
                elif case == 'block mismatch':
                    put_tile(raw, replace(tile, block_threads=64))
                elif case == 'partial warp':
                    raw, meta = sum_fixture(1024, 16)
                elif case == 'three warps':
                    raw, meta = sum_fixture(1024, 96)
                elif case == 'extent mismatch':
                    put_tile(raw, replace(tile, axes=(replace(tile.axes[0], extent=Dim(64)),)))
                else:
                    put_tile(raw, replace(tile, aux_threads=32))
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_unsupported_loop_semantics_fail_clearly(self):
        for fields in ({'step': Literal(64, 'int')}, {'seed': False}, {'unroll': True},
                       {'start': Literal(1, 'int')}, {'end': Literal(1000, 'int')}):
            with self.subTest(fields=fields):
                raw, meta = fixture()
                change_body(raw, 0, **fields)
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)
        raw, meta = fixture()
        loop = get_tile(raw).body[0]
        change_body(raw, 0, axis=replace(loop.axis, extent=Dim(1023)))
        with self.assertRaises(CompileError):
            compile_fixture(raw, meta)

    def test_non_sum_accumulators_and_invalid_loads_are_rejected(self):
        for fields in ({'op': ElementwiseImpl('maximum')}, {'base': 'unbound'},
                       {'axes': ('wrong_axis',)}, {'dtype': None}):
            with self.subTest(fields=fields):
                raw, meta = fixture()
                loop = get_tile(raw).body[0]
                change_body(raw, 0, body=(loop.body[0], replace(loop.body[1], **fields)))
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)
        raw, meta = fixture()
        loop = get_tile(raw).body[0]
        change_body(raw, 0, body=(replace(loop.body[0], index=(Literal(0, 'int'),)), loop.body[1]))
        with self.assertRaises(CompileError):
            compile_fixture(raw, meta)

    def test_bad_warp_tree_and_shared_memory_contracts_are_rejected(self):
        mutations = ((1, {'length': 16}), (2, {'extents': (5,)}), (2, {'dtype': 'double'}),
                     (4, {'count': 64}), (4, {'warp': True}), (5, {'length': 3}),
                     (5, {'tid_var': 'lane'}), (5, {'barrier_count': 64}))
        for index, fields in mutations:
            with self.subTest(index=index, fields=fields):
                raw, meta = fixture()
                change_body(raw, index, **fields)
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_non_sum_shuffle_combines_are_rejected(self):
        for index in (1, 5):
            with self.subTest(node=index):
                raw, meta = fixture()
                statement = get_tile(raw).body[index]
                combine = replace(statement.combine_states[0], op=ElementwiseImpl('multiply'))
                change_body(raw, index, combine_states=(combine,))
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)

    def test_nonleader_writes_and_atomic_result_are_rejected(self):
        for index in (3, 6):
            with self.subTest(conditional=index):
                raw, meta = fixture()
                cond = get_tile(raw).body[index].cond
                change_body(raw, index, cond=replace(cond, right=Literal(1, 'int')))
                with self.assertRaises(CompileError):
                    compile_fixture(raw, meta)
        raw, meta = fixture()
        condition = get_tile(raw).body[6]
        change_body(raw, 6, body=(replace(condition.body[0], atomic=True),))
        with self.assertRaises(CompileError):
            compile_fixture(raw, meta)

    def test_cli_official_success_and_unsupported_failure_are_clean(self):
        source = ROOT / '.reference/problems/r01_array_sum/k_sum_1_reduce.ir.json'
        success = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(source)],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(success.stderr, '')
        self.assertTrue(success.stdout.startswith('.version '))
        raw, meta = fixture()
        change_body(raw, 0, seed=False)
        with TemporaryDirectory(prefix='emmy-reduction-cli-') as tmp:
            source = Path(tmp) / 'kernel.ir.json'
            source.write_text(json.dumps(raw))
            source.with_name('meta.json').write_text(json.dumps(meta))
            failure = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(source)],
                                     capture_output=True, text=True, timeout=30)
        self.assertEqual(failure.returncode, 1)
        self.assertEqual(failure.stdout, '')
        self.assertTrue(failure.stderr.strip())
        self.assertNotIn('Traceback', failure.stderr)

    def test_cpu_model_rejects_missing_barrier_and_uninitialized_shared_state(self):
        raw, meta = sum_fixture(129)
        ptx = compile_fixture(raw, meta)
        missing_barrier, count = re.subn(r'bar\.sync[^;]*;', '', ptx, count=1)
        self.assertEqual(count, 1)
        with self.assertRaises(AssertionError):
            run_reduction_ptx(missing_barrier, meta['kernels'][0])
        no_initial_store, count = re.subn(r'st\.shared\.f32[^;]*;', '', ptx, count=1)
        self.assertEqual(count, 1)
        with self.assertRaises(AssertionError):
            run_reduction_ptx(no_initial_store, meta['kernels'][0])


    def test_cpu_model_rejects_incomplete_participation_races_and_unknown_instructions(self):
        raw, meta = sum_fixture(129)
        ptx = compile_fixture(raw, meta)
        tid = re.search(r'mov.u32 (%r\d+), %tid.x;', ptx).group(1)
        pred = re.search(r'setp.ge.u32 (%p\d+),', ptx).group(1)
        early_return = f'    setp.ne.u32 {pred}, {tid}, 0;\n    @{pred} ret;\n'
        mutations = {
            'incomplete warp': ('SUM_LOOP_DONE:\n', 'SUM_LOOP_DONE:\n' + early_return),
            'incomplete block': ('PARTIALS_DONE:\n', 'PARTIALS_DONE:\n' + early_return),
            'shared write race': (f'    @{pred} bra PARTIALS_DONE;\n', ''),
            'unknown instruction': ('add.rn.f32', 'unsupported.rn.f32'),
        }
        for case, (before, after) in mutations.items():
            with self.subTest(case=case):
                self.assertIn(before, ptx)
                with self.assertRaises(AssertionError):
                    run_reduction_ptx(ptx.replace(before, after, 1), meta['kernels'][0])


if __name__ == '__main__':
    unittest.main()
