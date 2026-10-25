"""Regression checks for metadata ABI agreement and malformed Assign fields."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from emmy_ptx import CompileError, compile_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.reference/problems/p00_scalar_add'


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((SOURCE / 'k_add_pointwise.ir.json').read_text())
        self.meta = json.loads((SOURCE / 'meta.json').read_text())
        self.tmp = TemporaryDirectory(prefix='emmy-validation-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'kernel.ir.json'

    def check_failure(self, raw=None, meta=None):
        self.path.write_text(json.dumps(self.raw if raw is None else raw))
        self.path.with_name('meta.json').write_text(json.dumps(self.meta if meta is None else meta))
        with self.assertRaises(CompileError):
            compile_file(self.path)
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'compile.py'), str(self.path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        self.assertTrue(result.stderr.strip())
        self.assertNotIn('Traceback', result.stderr)

    def test_conflicting_argument_orders(self):
        for key in ('args', 'arg_order'):
            with self.subTest(field=key):
                meta = copy.deepcopy(self.meta)
                meta['kernels'][0][key].reverse()
                self.check_failure(meta=meta)

    def test_by_value_scalar_is_rejected(self):
        self.meta['kernels'][0]['args'][0].update(scalar=True, value=1.0)
        self.check_failure()

    def test_buffer_output_roles(self):
        for index, value in ((0, True), (2, False), (2, 'true'), (2, 1)):
            with self.subTest(index=index, value=value):
                meta = copy.deepcopy(self.meta)
                meta['kernels'][0]['args'][index]['output'] = value
                self.check_failure(meta=meta)
        del self.meta['kernels'][0]['args'][2]['output']
        self.check_failure()

    def test_malformed_assign_operator_and_operands(self):
        for old, new in (("op=ElementwiseImpl('add')", 'op=None'),
                         ("args=('in0__u0', 'in1__u0')", 'args=None'),
                         ("args=('in0__u0', 'in1__u0')", "args='in0__u0'"),
                         ("args=('in0__u0', 'in1__u0')", "args=([], 'in1__u0')"),
                         ("args=('in0__u0', 'in1__u0')", "args=('in0__u0',)"),
                         ("Assign(name='v0__u0'", 'Assign(name=[]')):
            with self.subTest(replacement=new):
                raw = copy.deepcopy(self.raw)
                body = raw['nodes']['add']['op_fields']['body']
                self.assertIn(old, body[0])
                body[0] = body[0].replace(old, new)
                self.check_failure(raw=raw)

    def test_intermediate_descriptors_are_rejected(self):
        self.meta['kernels'][0]['args'][0]['intermediate'] = True
        self.check_failure()

    def test_boolean_shape_is_rejected(self):
        self.meta['kernels'][0]['args'][0]['shape'] = [True]
        self.check_failure()
