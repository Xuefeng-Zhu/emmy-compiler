"""Ensure CI cannot go green after empty discovery, failures or invalid CLI output."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('emmy_check_cli', ROOT / 'tools/check_cli.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class TestRunnerContract(unittest.TestCase):
    def run_fixture(self, body=None, decorator=''):
        with TemporaryDirectory(prefix='emmy-runner-test-') as directory:
            if body is not None:
                source = ('import unittest\nclass Fixture(unittest.TestCase):\n'
                          f'    {decorator}\n    def test_case(self):\n        {body}\n')
                (Path(directory) / 'test_fixture.py').write_text(source)
            return subprocess.run([sys.executable, '-B', str(ROOT / 'tools/run_tests.py'),
                                   '--test-directory', directory], cwd=directory,
                                  capture_output=True, text=True, timeout=15)

    def test_success_exits_zero(self):
        result = self.run_fixture('self.assertEqual(2 + 2, 4)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Ran 1 test', result.stderr)

    def test_assertion_failure_exits_nonzero(self):
        result = self.run_fixture("self.fail('deliberate CI failure probe')")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAILED (failures=1)', result.stderr)

    def test_exception_exits_nonzero(self):
        result = self.run_fixture("raise RuntimeError('deliberate CI error probe')")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAILED (errors=1)', result.stderr)

    def test_zero_tests_exits_nonzero(self):
        result = self.run_fixture()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('test discovery found zero tests', result.stderr)

    def test_skipped_tests_exit_nonzero(self):
        result = self.run_fixture('pass', "@unittest.skip('deliberate skip probe')")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('every discovered test to run and pass', result.stderr)

    def test_expected_failure_exits_nonzero(self):
        result = self.run_fixture('self.fail()', '@unittest.expectedFailure')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('every discovered test to run and pass', result.stderr)

    def test_import_error_exits_nonzero(self):
        with TemporaryDirectory(prefix='emmy-runner-import-') as directory:
            (Path(directory) / 'test_broken.py').write_text("raise ImportError('deliberate probe')\n")
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/run_tests.py'),
                                     '--test-directory', directory], capture_output=True,
                                    text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAILED (errors=1)', result.stderr)


class CLIOutputContract(unittest.TestCase):
    PTX = b'.version 8.0\n.target sm_89\n.address_size 64\n\n.visible .entry fixture()\n{\n    ret;\n}\n'

    def result(self, stdout=None, stderr=b'', returncode=0):
        return SimpleNamespace(stdout=self.PTX if stdout is None else stdout,
                               stderr=stderr, returncode=returncode)

    def test_clean_ptx_returns_hash(self):
        import hashlib
        self.assertEqual(CLI.validate_success(self.result(), 'fixture'),
                         hashlib.sha256(self.PTX).hexdigest())

    def test_success_rejects_output_contamination_or_wrong_entry(self):
        cases = [self.result(stdout=b'log\n' + self.PTX),
                 self.result(stdout=self.PTX + b'log\n'),
                 self.result(stdout=self.PTX + b'\xff'),
                 self.result(stderr=b'warning\n'),
                 self.result(returncode=1),
                 self.result(stdout=self.PTX.replace(b'fixture()', b'wrong()')),
                 self.result(stdout=self.PTX + self.PTX)]
        for result in cases:
            with self.subTest(result=result), self.assertRaises(ValueError):
                CLI.validate_success(result, 'fixture')

    def test_failure_requires_nonzero_stderr_and_no_partial_ptx(self):
        CLI.validate_failure(self.result(stdout=b'', stderr=b'compile.py: invalid IR\n', returncode=1))
        for result in (self.result(returncode=1, stderr=b'error'),
                       self.result(stdout=b'', stderr=b'error'),
                       self.result(stdout=b'', returncode=1),
                       self.result(stdout=b'', stderr=b'Traceback: bad', returncode=1)):
            with self.subTest(result=result), self.assertRaises(ValueError):
                CLI.validate_failure(result)
