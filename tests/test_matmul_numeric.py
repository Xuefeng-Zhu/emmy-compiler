"""Execute emitted scalar-matmul PTX in a CPU model, never on a GPU.

Official expected arrays/tolerances remain authoritative. Additional input and
malformed-PTX probes check that the model observes actual pointers, control flow,
async completion, synchronization and split-K output accumulation.
"""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from emmy_ptx import compile_file
from matmul_model import run_matmul_ptx

ROOT = Path(__file__).resolve().parents[1]
CASES = ('m01_matmul_512', 'm02_matmul_rect')


def fixture(case):
    folder = ROOT / '.reference/problems' / case
    source, = folder.glob('*.ir.json')
    return source, json.loads(source.read_text()), json.loads((folder / 'meta.json').read_text())


def emit(raw, meta):
    with TemporaryDirectory(prefix='emmy-matmul-numeric-') as tmp:
        source = Path(tmp) / 'kernel.ir.json'
        source.write_text(json.dumps(raw))
        source.with_name('meta.json').write_text(json.dumps(meta))
        return compile_file(source)


class MatmulNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.programs = {}
        for case in CASES:
            source, _, meta = fixture(case)
            cls.programs[case] = (compile_file(source), meta)

    def test_full_official_matmuls_match_expected_arrays_in_ptx_model(self):
        for case in CASES:
            with self.subTest(case=case):
                ptx, meta = self.programs[case]
                result = run_matmul_ptx(ptx, meta['kernels'][0])
                with np.load(ROOT / '.reference/problems' / case / 'expected.npz', allow_pickle=False) as expected:
                    self.assertEqual(set(expected.files), set(result.outputs))
                    for name in expected.files:
                        np.testing.assert_allclose(result.outputs[name], expected[name],
                                                   rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
                if case == 'm01_matmul_512':
                    self.assertTrue(np.all(result.store_counts['matmul'] == 1))
                    self.assertEqual(result.barrier_count, 16)
                    self.assertEqual(result.committed_groups, 10)
                    self.assertEqual(result.completed_groups, 10)
                else:
                    self.assertTrue(np.all(result.store_counts['matmul'] == 32))
                    self.assertEqual(result.barrier_count, 2)
                    self.assertEqual(result.committed_groups, 1)
                    self.assertEqual(result.completed_groups, 1)
                self.assertGreater(result.async_copy_bytes, 0)

    def test_split_k_accumulates_changed_runtime_inputs(self):
        ptx, meta = self.programs['m02_matmul_rect']
        launch = meta['kernels'][0]
        left = np.ones((128, 256), np.float32)
        columns = (np.arange(64) % 3 - 1).astype(np.float32)
        right = np.broadcast_to(columns, (256, 64)).copy()
        result = run_matmul_ptx(ptx, launch, {'x0': left, 'x1': right})
        expected = np.broadcast_to(columns * np.float32(256), (128, 64))
        np.testing.assert_array_equal(result.outputs['matmul'], expected)
        self.assertTrue(np.all(result.store_counts['matmul'] == 32))

    def test_parameter_reordering_and_renaming_preserve_numerical_result(self):
        _, raw, meta = fixture('m02_matmul_rect')
        for old, new in [('x0', 'left_values'), ('x1', 'right_values'), ('matmul', 'product_values')]:
            raw = json.loads(json.dumps(raw).replace(old, new))
            meta = json.loads(json.dumps(meta).replace(old, new))
        launch = meta['kernels'][0]
        launch['arg_order'].reverse()
        launch['args'].reverse()
        result = run_matmul_ptx(emit(raw, meta), launch)
        with np.load(ROOT / '.reference/problems/m02_matmul_rect/expected.npz', allow_pickle=False) as expected:
            np.testing.assert_allclose(result.outputs['product_values'], expected['matmul'],
                                       rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)

    def test_model_rejects_missing_async_wait_or_commit(self):
        ptx, meta = self.programs['m02_matmul_rect']
        for deleted in ('cp.async.wait_group 0;', 'cp.async.commit_group;'):
            with self.subTest(deleted=deleted):
                self.assertIn(deleted, ptx)
                broken = ptx.replace(deleted, '', 1)
                with self.assertRaisesRegex(AssertionError, 'async completion|initialization'):
                    run_matmul_ptx(broken, meta['kernels'][0])

    def test_model_rejects_missing_cross_thread_visibility_barrier(self):
        ptx, meta = self.programs['m02_matmul_rect']
        broken = ptx.replace('bar.sync 0;', '', 1)
        with self.assertRaisesRegex(AssertionError, 'CTA synchronization'):
            run_matmul_ptx(broken, meta['kernels'][0])

    def test_model_rejects_ring_reuse_without_read_completion_barrier(self):
        ptx, meta = self.programs['m01_matmul_512']
        first = ptx.index('bar.sync 0;')
        second = ptx.index('bar.sync 0;', first + 1)
        broken = ptx[:second] + ptx[second:].replace('bar.sync 0;', '', 1)
        with self.assertRaisesRegex(AssertionError, 'readers synchronize'):
            run_matmul_ptx(broken, meta['kernels'][0])

    def test_model_rejects_return_with_unfinished_async_copies(self):
        # A small explicit PTX program isolates the drain check from missing-read
        # diagnostics; the official m01 test above checks all ten groups finish.
        launch = {'grid': [1, 1, 1], 'block': [1, 1, 1], 'smem_bytes': 0,
                  'args': [{'name': 'source', 'shape': [4], 'dtype': 'f32', 'data': [1, 2, 3, 4]},
                           {'name': 'result', 'shape': [1], 'dtype': 'f32', 'output': True}]}
        ptx = """.version 8.0
.target sm_89
.address_size 64
.visible .entry pending(.param .u64 a, .param .u64 b) {
.reg .b64 %rd<2>;
.reg .b32 %r<1>;
.reg .f32 %f<1>;
.shared .align 16 .b8 scratch[16];
ld.param.u64 %rd0, [a];
ld.param.u64 %rd1, [b];
mov.u32 %r0, scratch;
cp.async.cg.shared.global [%r0], [%rd0], 16;
cp.async.commit_group;
mov.f32 %f0, 0f3f800000;
st.global.f32 [%rd1], %f0;
ret;
}
"""
        with self.assertRaisesRegex(AssertionError, 'unfinished async copies'):
            run_matmul_ptx(ptx, launch)
        drained = ptx.replace('ret;', 'cp.async.wait_group 0;\nret;')
        result = run_matmul_ptx(drained, launch)
        self.assertEqual(result.completed_groups, 1)
        np.testing.assert_array_equal(result.outputs['result'], [1])

    def test_model_rejects_insufficient_dynamic_pool_and_missing_zero_contract(self):
        ptx, meta = self.programs['m01_matmul_512']
        launch = copy.deepcopy(meta['kernels'][0])
        launch['smem_bytes'] -= 16
        with self.assertRaisesRegex(AssertionError, 'shared access'):
            run_matmul_ptx(ptx, launch)
        ptx, meta = self.programs['m02_matmul_rect']
        launch = copy.deepcopy(meta['kernels'][0])
        launch['zero_outputs'] = []
        with self.assertRaisesRegex(AssertionError, 'zero_outputs'):
            run_matmul_ptx(ptx, launch)


if __name__ == '__main__':
    unittest.main()
