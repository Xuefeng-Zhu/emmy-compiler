"""Temporary-file, mocked-transport tests. These cannot contact the runner."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('synthetic_runner', PROJECT / 'tools/run_synthetic.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SyntheticRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='emmy-synthetic-mock-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.case = self.base / 'case'
        self.case.mkdir()
        self.attempt = self.base / 'attempt'
        self.manifest = self.base / 'manifest.json'
        self.write_case()

    def write_case(self, shape=(5,), scalar=False):
        self.expected = np.arange(np.prod(shape), dtype=np.float32).reshape(() if scalar else shape)
        self.meta = {'rtol': 0, 'atol': 0, 'kernels': [{
            'entry': 'k_synthetic', 'arg_order': ['x', 'result'],
            'args': [{'name': 'x', 'shape': list(shape), 'dtype': 'f32', 'seed': 17},
                     {'name': 'result', 'shape': list(shape), 'dtype': 'f32', 'output': True}],
            'grid': [1, 1, 1], 'block': [256, 1, 1], 'smem_bytes': 0, 'zero_outputs': []}]}
        (self.case / 'meta.json').write_text(json.dumps(self.meta))
        (self.case / 'k_synthetic.ir.json').write_text('{}')
        np.savez(self.case / 'expected.npz', result=self.expected)
        self.pin()

    def pin(self):
        self.manifest.write_text(json.dumps({'files': {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.case.iterdir()}}))

    def run_case(self, mode=None, mutation=None):
        calls = []
        owner = self
        ptx = '.version 8.0\n.target sm_89\n.address_size 64\n.visible .entry k_synthetic() {ret;}\n'
        output = {'dtype': 'f32', 'shape': self.meta['kernels'][0]['args'][-1]['shape'],
                  'values': self.expected.reshape(-1).tolist(), 'truncated': False}
        outputs = {'result': output}
        if mutation:
            mutation(outputs)

        class FakeSession:
            trust_env = True
            closed = False
            cookies = SimpleNamespace(clear=lambda: None)

            def post(session, url, *, json, timeout, allow_redirects):
                owner.assertFalse(session.trust_env)
                owner.assertFalse(allow_redirects)
                owner.assertEqual(timeout, (10, 300))
                calls.append((url, copy.deepcopy(json)))
                report = __import__('json').loads((owner.attempt / 'report.json').read_text())
                owner.assertEqual(report['requests'][-1]['state'], 'request_started')
                owner.assertTrue(report['uncertain_request'])
                if url == runner.SERVER + '/compile':
                    owner.assertEqual(json, {'ptx': ptx})
                    data = {'ok': True, 'cubin_id': 'mock-cubin', 'entry_points': ['k_synthetic'],
                            'registers': {'k_synthetic': 9}}
                    if mode == 'compile_error':
                        data = {'ok': False, 'stage': 'ptxas', 'stderr': 'mock assembly error'}
                elif url == runner.SERVER + '/run':
                    k = owner.meta['kernels'][0]
                    owner.assertEqual(json, {'cubin_id': 'mock-cubin', 'entry': 'k_synthetic',
                        'grid': k['grid'], 'block': k['block'], 'smem_bytes': k['smem_bytes'],
                        'zero_outputs': k['zero_outputs'], 'args': k['args'],
                        'warmup': 0, 'iters': 1, 'max_return': max(4096, owner.expected.size)})
                    if mode == 'timeout':
                        raise TimeoutError('mock timeout')
                    data = {'ok': True, 'outputs': outputs, 'median_us': 1.5, 'iters': 1,
                            'gpu': 'Mock GPU'}
                    if mode == 'busy':
                        data = {'ok': False, 'stage': 'busy', 'stderr': 'mock busy'}
                else:
                    raise AssertionError('Network destination is forbidden')
                return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)

            def close(session):
                session.closed = True

        session = FakeSession()
        def forbidden(*args, **kwargs):
            raise AssertionError('Unconstrained transport is forbidden')
        fake_requests = SimpleNamespace(Session=lambda: session, post=forbidden,
                                        ConnectionError=ConnectionError, Timeout=TimeoutError)
        generated = SimpleNamespace(returncode=2 if mode == 'local_compile_error' else 0,
                                    stdout='' if mode == 'local_compile_error' else ptx,
                                    stderr='mock compiler diagnostic\n' if mode == 'local_compile_error' else '')
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {'requests': fake_requests}), \
                patch.object(runner.subprocess, 'run', return_value=generated), \
                patch.object(sys, 'argv', ['run_synthetic.py', str(self.case), '--project', str(PROJECT),
                    '--manifest', str(self.manifest), '--attempt', str(self.attempt)]), \
                patch('time.sleep'), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = runner.main()
        report = json.loads((self.attempt / 'report.json').read_text())
        expected_calls = [] if mode == 'local_compile_error' else ['compile'] + (
            [] if mode == 'compile_error' else ['run'] * (3 if mode == 'busy' else 1))
        self.assertEqual([url.rsplit('/', 1)[1] for url, _ in calls], expected_calls)
        self.assertEqual(session.closed, mode != 'local_compile_error')
        return code, stdout.getvalue(), stderr.getvalue(), report

    def test_success_exact_tolerance_and_full_output_hash(self):
        code, stdout, stderr, report = self.run_case()
        self.assertEqual(code, 0)
        self.assertIn('PASS  result:', stdout)
        self.assertFalse(stderr)
        self.assertEqual(report['comparison'], {'rtol': 0, 'atol': 0, 'equal_nan': False})
        receipt = report['requests'][-1]
        output = self.attempt / receipt['output_file']
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), receipt['output_file_sha256'])
        np.testing.assert_array_equal(np.load(output, allow_pickle=False), self.expected)
        self.assertFalse(report['uncertain_request'])

    def test_scalar_and_matrix_outputs(self):
        for shape, scalar in (((1,), True), ((65, 65), False)):
            with self.subTest(shape=shape):
                self.write_case(shape, scalar)
                self.attempt = self.base / ('attempt-' + str(scalar))
                code, _, _, report = self.run_case()
                self.assertEqual(code, 0)
                self.assertEqual(report['output_shape'], list(shape))
                self.assertEqual(report['expected_shape'], [] if scalar else list(shape))

    def test_malformed_outputs_fail(self):
        mutations = [lambda o: o.clear(), lambda o: o['result'].update(truncated=True),
                     lambda o: o['result'].update(shape=[1]), lambda o: o['result']['values'].pop(),
                     lambda o: o.update(extra=o['result']),
                     lambda o: o['result'].update(values=[o['result']['values']])]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.attempt = self.base / f'bad-{index}'
                code, _, stderr, report = self.run_case(mutation=mutation)
                self.assertEqual(code, 1)
                self.assertIn('malformed', stderr)
                self.assertFalse(report['passed'])

    def test_nan_and_finite_mismatch_fail_without_relaxing_tolerances(self):
        for index, value in enumerate((float('nan'), 0.00001)):
            with self.subTest(value=value):
                self.attempt = self.base / f'mismatch-{index}'
                def change(outputs):
                    outputs['result']['values'][0] = value
                code, stdout, _, report = self.run_case(mutation=change)
                self.assertEqual(code, 1)
                self.assertIn('FAIL  result:', stdout)
                self.assertFalse(report['requests'][-1]['comparison_passed'])
                self.assertTrue((self.attempt / report['requests'][-1]['output_file']).exists())

    def test_busy_limit_timeout_and_assembly_failure(self):
        for mode in ('busy', 'timeout', 'compile_error'):
            with self.subTest(mode=mode):
                self.attempt = self.base / mode
                code, _, _, report = self.run_case(mode=mode)
                self.assertEqual(code, 1)
                self.assertEqual(report['uncertain_request'], mode == 'timeout')
                self.assertFalse(report['passed'])
                if mode == 'compile_error':
                    self.assertEqual(report['requests'][-1]['stderr'], 'mock assembly error')

    def test_local_compiler_failure_has_no_transport(self):
        code, _, stderr, report = self.run_case(mode='local_compile_error')
        self.assertEqual(code, 1)
        self.assertEqual(stderr, 'mock compiler diagnostic\n')
        self.assertEqual(report['local_generation_exit_code'], 2)
        self.assertEqual(report['requests'], [])

    def test_hash_mismatch_and_attempt_reuse_reject(self):
        with self.assertRaises(FileExistsError):
            self.attempt.mkdir()
            self.run_case()
        (self.case / 'k_synthetic.ir.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            runner.load_case(self.case, self.manifest)

    def test_invalid_data_or_limits_reject_before_transport(self):
        mutations = [lambda k: k['args'][0].update(seed=-1),
                     lambda k: k['args'][0].update(data=[0] * 5),
                     lambda k: k['args'][0].update(shape=[2 ** 26]),
                     lambda k: k['args'][-1].update(shape=[2 ** 22 + 1]),
                     lambda k: k.update(arg_order=['result', 'x']),
                     lambda k: k.update(smem_bytes=99 * 1024 + 1)]
        for mutation in mutations:
            self.write_case()
            mutation(self.meta['kernels'][0])
            (self.case / 'meta.json').write_text(json.dumps(self.meta))
            self.pin()
            with self.assertRaises(ValueError):
                runner.load_case(self.case, self.manifest)


if __name__ == '__main__':
    unittest.main()
