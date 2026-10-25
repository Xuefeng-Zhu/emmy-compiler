"""New pointwise syntax/contract coverage; these checks do not execute a GPU."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from emmy_ptx import CompileError, compile_file

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / '.reference/problems'


class PointwiseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix='emmy-pointwise-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'kernel.ir.json'

    def fixture(self, problem):
        source, = (REFERENCE / problem).glob('*.ir.json')
        self.raw = json.loads(source.read_text())
        self.meta = json.loads(source.with_name('meta.json').read_text())
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.raw))
        self.path.with_name('meta.json').write_text(json.dumps(self.meta))

    def test_p02_uses_runtime_constants_and_separate_rounding(self):
        self.fixture('p02_scale_shift')
        ptx = compile_file(self.path)
        self.assertIn('param_mul_c1', ptx)
        self.assertIn('param_add_c1', ptx)
        self.assertEqual(ptx.count('mul.rn.f32'), 4)
        self.assertEqual(ptx.count('add.rn.f32'), 4)
        self.assertNotIn('fma.', ptx)
        self.assertNotIn('.ftz.', ptx)
        self.assertEqual(ptx.count('st.global.f32'), 4)
        for name, value in [('mul_c1', -3.5), ('add_c1', 0.125)]:
            self.raw['nodes'][name]['op_fields']['value'] = value
            next(a for a in self.meta['kernels'][0]['args'] if a['name'] == name)['data'] = [value]
        self.save()
        self.assertEqual(compile_file(self.path), ptx, 'Runtime values must not be baked into PTX')

    def test_p03_relu_is_unary_and_nan_aware(self):
        self.fixture('p03_relu_mul')
        ptx = compile_file(self.path)
        self.assertEqual(ptx.count('max.NaN.f32'), 4)
        self.assertEqual(ptx.count('mul.rn.f32'), 4)
        self.assertNotIn('add.rn.f32', ptx)
        self.assertNotIn('.ftz.', ptx)

    def test_all_pointwise_ptx_matches_previously_gpu_verified_artifacts(self):
        for problem in ('p00_scalar_add', 'p01_vector_add', 'p02_scale_shift', 'p03_relu_mul'):
            with self.subTest(problem=problem):
                self.fixture(problem)
                date = '2026-10-24' if problem in ('p00_scalar_add', 'p01_vector_add') else '2026-10-25'
                receipt = json.loads((ROOT / 'docs/validation' / date / f'{problem}.json').read_text())
                self.assertEqual(hashlib.sha256(compile_file(self.path).encode()).hexdigest(), receipt['ptx_sha256'])

    def test_constant_names_and_parameter_order_are_not_hardcoded(self):
        self.fixture('p02_scale_shift')
        for old, new in [('mul_c1', 'scale_parameter'), ('add_c1', 'bias_parameter')]:
            self.raw = json.loads(json.dumps(self.raw).replace(old, new))
            self.meta = json.loads(json.dumps(self.meta).replace(old, new))
        self.meta['kernels'][0]['arg_order'].reverse()
        self.meta['kernels'][0]['args'].reverse()
        self.save()
        ptx = compile_file(self.path)
        order = self.meta['kernels'][0]['arg_order']
        positions = [ptx.index('.param .u64 param_' + name) for name in order]
        self.assertEqual(positions, sorted(positions))

    def test_malformed_constant_descriptors(self):
        self.fixture('p02_scale_shift')
        original = copy.deepcopy(self.meta)
        for change in ({'data': []}, {'data': [True]}, {'data': ['2']}, {'data': None}, {'seed': 0}, {'scalar': True}, {'shape': [2]}, {'dtype': 'f16'}):
            with self.subTest(change=change):
                self.meta = copy.deepcopy(original)
                self.meta['kernels'][0]['args'][0].update(change)
                self.save()
                with self.assertRaises(CompileError):
                    compile_file(self.path)
        self.meta = copy.deepcopy(original)
        del self.meta['kernels'][0]['args'][0]['data']
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_missing_constant_kernel_input_is_rejected(self):
        self.fixture('p02_scale_shift')
        self.raw['nodes']['add']['inputs'].remove('mul_c1')
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_constant_is_not_a_graph_activation_input(self):
        self.fixture('p02_scale_shift')
        self.raw['inputs'].append('mul_c1')
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_wrong_relu_arity_and_unsupported_operation(self):
        self.fixture('p03_relu_mul')
        original = copy.deepcopy(self.raw)
        for old, new in (("args=('in0__u0',)", "args=('in0__u0', 'in0__u0')"),
                         ("ElementwiseImpl('relu')", "ElementwiseImpl('exp')")):
            with self.subTest(replacement=new):
                self.raw = copy.deepcopy(original)
                body = self.raw['nodes']['mul']['op_fields']['body']
                self.assertIn(old, body[0])
                body[0] = body[0].replace(old, new)
                self.save()
                with self.assertRaises(CompileError):
                    compile_file(self.path)
