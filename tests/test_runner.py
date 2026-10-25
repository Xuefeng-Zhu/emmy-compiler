"""Mock-only runner transport checks; no HTTP, GPU work or receipt writes."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RunnerTransportTests(unittest.TestCase):
    def run_mock_case(self, problem, mutation=None):
        folder = ROOT / '.reference/problems' / problem
        meta = json.loads((folder / 'meta.json').read_text())
        kernel, = meta['kernels']
        output_arg, = [arg for arg in kernel['args'] if arg.get('output')]
        output_name = output_arg['name']
        with np.load(folder / 'expected.npz', allow_pickle=False) as expected:
            output = {'shape': output_arg['shape'], 'dtype': 'f32',
                      'values': expected[output_name].tolist(), 'truncated': False}
        outputs = {output_name: output}
        if mutation is not None:
            mutation(outputs, output_name)
        calls, saved = [], {}
        cubin_id = 'a' * 64
        owner = self

        class FakeSession:
            def __init__(self):
                self.trust_env = True
                self.cookies = SimpleNamespace(clear=lambda: None)
                self.closed = False

            def post(self, url, *, json, timeout, allow_redirects):
                owner.assertFalse(self.trust_env)
                owner.assertFalse(allow_redirects)
                calls.append((url, copy.deepcopy(json)))
                if url == 'http://runner.riftstack.ai:48701/compile':
                    owner.assertEqual(set(json), {'ptx'})
                    owner.assertIn(f'.visible .entry {kernel["entry"]}(', json['ptx'])
                    owner.assertTrue(json['ptx'].isascii())
                    data = {'ok': True, 'cubin_id': cubin_id,
                            'entry_points': [kernel['entry']], 'registers': {kernel['entry']: 1}}
                elif url == 'http://runner.riftstack.ai:48701/run':
                    owner.assertEqual(json, {
                        'cubin_id': cubin_id, 'entry': kernel['entry'],
                        'grid': kernel['grid'], 'block': kernel['block'],
                        'smem_bytes': kernel['smem_bytes'], 'zero_outputs': kernel['zero_outputs'],
                        'args': kernel['args'], 'warmup': 0, 'iters': 1, 'max_return': 4096,
                    })
                    data = {'ok': True, 'outputs': outputs, 'median_us': 1.0, 'iters': 1}
                else:
                    raise AssertionError('Unexpected mocked destination')
                return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)

            def close(self):
                self.closed = True

        session = FakeSession()

        def forbidden_post(*args, **kwargs):
            raise AssertionError('Unconstrained transport is forbidden in this test')

        fake_requests = SimpleNamespace(Session=lambda: session, post=forbidden_post,
                                        ConnectionError=ConnectionError, Timeout=TimeoutError)
        spec = importlib.util.spec_from_file_location('_emmy_runner_under_test', ROOT / 'tools/run_official.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        def capture_write(path, data, *args, **kwargs):
            owner.assertEqual(path.parent, ROOT / 'build/runner')
            saved[path.name] = data
            return len(data)

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {'requests': fake_requests}), \
                patch.dict(os.environ, {'PYTHONDONTWRITEBYTECODE': '1'}), \
                patch.object(sys, 'argv', ['run_official.py', problem]), \
                patch.object(Path, 'write_text', autospec=True, side_effect=capture_write), \
                patch.object(Path, 'mkdir', autospec=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = runner.main()
        self.assertTrue(session.closed)
        self.assertEqual([url.rsplit('/', 1)[1] for url, _ in calls], ['compile', 'run'])
        report = json.loads(saved[problem + '.json'])
        return code, stdout.getvalue(), stderr.getvalue(), report

    def test_mocked_p02_and_p03_use_exact_public_payload_and_output_names(self):
        for problem, output_name in (('p02_scale_shift', 'add'), ('p03_relu_mul', 'mul')):
            with self.subTest(problem=problem):
                code, stdout, stderr, report = self.run_mock_case(problem)
                self.assertEqual(code, 0)
                self.assertEqual(stderr, '')
                self.assertIn(f'PASS  {output_name}:', stdout)
                self.assertTrue(report['passed'])
                self.assertEqual(report['exit_code'], 0)
                self.assertEqual(report['local_generation_exit_code'], 0)
                self.assertEqual(report['expected_output'], output_name)
                self.assertIs(report['comparison']['equal_nan'], False)
                self.assertTrue(report['requests'][-1]['comparison_passed'])
                self.assertEqual(report['requests'][-1]['max_abs_error'], 0.0)

    def test_mocked_missing_or_malformed_outputs_are_rejected(self):
        mutations = (
            ('missing', lambda outputs, name: outputs.clear()),
            ('wrong dtype', lambda outputs, name: outputs[name].update(dtype='f16')),
            ('wrong shape', lambda outputs, name: outputs[name].update(shape=[1])),
            ('truncated', lambda outputs, name: outputs[name].update(truncated=True)),
        )
        for label, mutation in mutations:
            with self.subTest(case=label):
                code, stdout, stderr, report = self.run_mock_case('p03_relu_mul', mutation)
                self.assertEqual(code, 1)
                self.assertNotIn('PASS  mul:', stdout)
                self.assertIn('malformed expected output', stderr)
                self.assertFalse(report['passed'])

    def test_mocked_nan_output_cannot_pass_finite_official_expectation(self):
        def inject_nan(outputs, name):
            outputs[name]['values'][0] = float('nan')
        code, stdout, stderr, report = self.run_mock_case('p03_relu_mul', inject_nan)
        self.assertEqual(code, 1)
        self.assertNotIn('PASS  mul:', stdout)
        self.assertFalse(report['passed'])
        self.assertFalse(report['requests'][-1]['comparison_passed'])
        self.assertIsNone(report['requests'][-1]['max_abs_error'])


if __name__ == '__main__':
    unittest.main()
