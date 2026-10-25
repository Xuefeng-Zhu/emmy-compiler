"""Mock-only runner transport checks; no HTTP, GPU work or receipt writes."""
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RunnerTransportTests(unittest.TestCase):
    def run_mock_case(self, problem, mutation=None, mode=None):
        folder = ROOT / '.reference/problems' / problem
        meta = json.loads((folder / 'meta.json').read_text())
        kernel, = meta['kernels']
        output_arg, = [arg for arg in kernel['args'] if arg.get('output')]
        output_name = output_arg['name']
        with np.load(folder / 'expected.npz', allow_pickle=False) as expected:
            output = {'shape': output_arg['shape'], 'dtype': 'f32',
                      'values': expected[output_name].reshape(-1).tolist(), 'truncated': False}
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
                started = __import__('json').loads(saved[problem + '.json'])['requests'][-1]
                owner.assertEqual(started['state'], 'request_started')
                if url == 'http://runner.riftstack.ai:48701/compile':
                    owner.assertEqual(set(json), {'ptx'})
                    owner.assertIn(f'.visible .entry {kernel["entry"]}(', json['ptx'])
                    owner.assertTrue(json['ptx'].isascii())
                    data = {'ok': True, 'cubin_id': cubin_id,
                            'entry_points': [kernel['entry']], 'registers': {kernel['entry']: 1}}
                    if mode == 'compile_error':
                        data = {'ok': False, 'stage': 'ptxas', 'stderr': 'mock ptxas failure'}
                elif url == 'http://runner.riftstack.ai:48701/run':
                    owner.assertEqual(json, {
                        'cubin_id': cubin_id, 'entry': kernel['entry'],
                        'grid': kernel['grid'], 'block': kernel['block'],
                        'smem_bytes': kernel['smem_bytes'], 'zero_outputs': kernel['zero_outputs'],
                        'args': kernel['args'], 'warmup': 0, 'iters': 1,
                        'max_return': max(4096, math.prod(output_arg['shape'])),
                    })
                    if mode == 'timeout':
                        raise TimeoutError('mock connection timeout')
                    data = {'ok': True, 'outputs': outputs, 'median_us': 1.0, 'iters': 1, 'gpu': 'Mock GPU'}
                    if mode == 'busy':
                        data = {'ok': False, 'stage': 'busy', 'stderr': 'mock busy'}
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

        compile_patch = (patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(
            returncode=2, stdout='', stderr='mock compiler diagnostic\n'))
            if mode == 'local_compile_error' else contextlib.nullcontext())
        stdout, stderr = io.StringIO(), io.StringIO()
        with compile_patch, patch.dict(sys.modules, {'requests': fake_requests}), \
                patch.dict(os.environ, {'PYTHONDONTWRITEBYTECODE': '1'}), \
                patch.object(sys, 'argv', ['run_official.py', problem]), \
                patch.object(Path, 'write_text', autospec=True, side_effect=capture_write), \
                patch.object(Path, 'mkdir', autospec=True), \
                patch('time.sleep'), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = runner.main()
        self.assertEqual(session.closed, mode != 'local_compile_error')
        if mode == 'local_compile_error':
            expected_paths = []
        else:
            expected_paths = ['compile'] + (['run'] * 3 if mode == 'busy' else [] if mode == 'compile_error' else ['run'])
        self.assertEqual([url.rsplit('/', 1)[1] for url, _ in calls], expected_paths)
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

    def test_mocked_reduction_scalar_and_vector_outputs(self):
        for problem in ('r01_array_sum', 'r02_big_sum', 'r03_mean', 'r04_row_sum'):
            with self.subTest(problem=problem):
                code, stdout, stderr, report = self.run_mock_case(problem)
                self.assertEqual(code, 0, stderr)
                self.assertTrue(report['passed'])
                self.assertEqual(report['max_return'], 4096)
                self.assertEqual(report['expected_shape'], [64] if problem == 'r04_row_sum' else [])
                self.assertEqual(report['output_shape'], [64] if problem == 'r04_row_sum' else [1])

    def test_mocked_matrix_requests_every_output_value(self):
        code, stdout, stderr, report = self.run_mock_case('m01_matmul_512')
        self.assertEqual(code, 0, stderr)
        self.assertTrue(report['passed'])
        self.assertEqual(report['max_return'], 262144)
        self.assertEqual(report['output_count'], 262144)
        self.assertEqual(report['output_shape'], [512, 512])
        self.assertEqual(report['requests'][-1]['output_count'], 262144)

    def test_mocked_missing_or_malformed_outputs_are_rejected(self):
        mutations = (
            ('missing', lambda outputs, name: outputs.clear()),
            ('wrong dtype', lambda outputs, name: outputs[name].update(dtype='f16')),
            ('wrong shape', lambda outputs, name: outputs[name].update(shape=[1])),
            ('truncated', lambda outputs, name: outputs[name].update(truncated=True)),
            ('wrong count', lambda outputs, name: outputs[name]['values'].pop()),
            ('extra output', lambda outputs, name: outputs.update(unexpected=copy.deepcopy(outputs[name]))),
            ('nested values', lambda outputs, name: outputs[name].update(values=[outputs[name]['values']])),
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


    def test_mocked_busy_retries_are_bounded(self):
        code, stdout, stderr, report = self.run_mock_case('r01_array_sum', mode='busy')
        self.assertEqual(code, 1)
        self.assertIn('Bounded request limit', stderr)
        self.assertFalse(report['passed'])
        self.assertFalse(report['uncertain_request'])
        self.assertEqual(len(report['requests']), 4)
        self.assertTrue(all(item['stage'] == 'busy' for item in report['requests'][1:]))

    def test_mocked_timeout_preserves_uncertain_request(self):
        code, stdout, stderr, report = self.run_mock_case('r01_array_sum', mode='timeout')
        self.assertEqual(code, 1)
        self.assertFalse(report['passed'])
        self.assertTrue(report['uncertain_request'])
        self.assertEqual(report['requests'][-1]['state'], 'request_started')

    def test_mocked_assembly_failure_preserves_stage_and_error(self):
        code, stdout, stderr, report = self.run_mock_case('r01_array_sum', mode='compile_error')
        self.assertEqual(code, 1)
        self.assertIn('mock ptxas failure', stdout)
        self.assertFalse(report['passed'])
        self.assertFalse(report['uncertain_request'])
        self.assertEqual(report['requests'][0]['stage'], 'ptxas')
        self.assertEqual(report['requests'][0]['stderr'], 'mock ptxas failure')


    def test_mocked_local_compile_failure_preserves_stderr_without_requests(self):
        code, stdout, stderr, report = self.run_mock_case('r01_array_sum', mode='local_compile_error')
        self.assertEqual(code, 1)
        self.assertEqual(stdout, '')
        self.assertEqual(stderr, 'mock compiler diagnostic\n')
        self.assertFalse(report['passed'])
        self.assertFalse(report['uncertain_request'])
        self.assertEqual(report['stage'], 'local_generation')
        self.assertEqual(report['local_generation_exit_code'], 2)
        self.assertEqual(report['exit_code'], 2)
        self.assertEqual(report['requests'], [])


if __name__ == '__main__':
    unittest.main()
