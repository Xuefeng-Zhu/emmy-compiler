"""Offline integrity checks for reference acquisition; no upstream downloads."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('emmy_reference_fetch', ROOT / 'tools/fetch_reference.py')
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class ReferenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix='emmy-reference-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b'{"public": "synthetic fixture for this test"}\n'
        self.manifest = {
            'commit': 'e' * 40,
            'files': [{'path': 'problems/synthetic/meta.json',
                       'sha256': hashlib.sha256(self.payload).hexdigest()}],
        }
        (self.root / 'docs').mkdir()
        self.save_manifest()
        self.path = self.root / '.reference' / self.manifest['files'][0]['path']

    def save_manifest(self):
        (self.root / 'docs/reference-manifest.json').write_text(json.dumps(self.manifest))

    def cache(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)

    def test_valid_cache_is_verified_without_network(self):
        self.cache(self.payload)
        for verify_only in (False, True):
            with self.subTest(verify_only=verify_only), patch.object(fetch.urllib.request, 'urlopen') as request:
                self.assertEqual(fetch.prepare_reference(self.root, verify_only=verify_only),
                                 (1, self.manifest['commit']))
                request.assert_not_called()
                self.assertEqual(self.path.read_bytes(), self.payload)

    def test_download_uses_fixed_commit_and_checks_hash(self):
        with patch.object(fetch.urllib.request, 'urlopen', return_value=io.BytesIO(self.payload)) as request, \
                contextlib.redirect_stdout(io.StringIO()):
            fetch.prepare_reference(self.root)
        url = request.call_args.args[0].full_url
        self.assertEqual(url, 'https://raw.githubusercontent.com/cloudrift-ai/emmy-hackathon-2026/'
                         + self.manifest['commit'] + '/problems/synthetic/meta.json')
        self.assertEqual(request.call_args.kwargs, {'timeout': 30})
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_corrupt_cache_is_replaced_only_by_verified_bytes(self):
        self.cache(b'corrupt cache')
        with patch.object(fetch.urllib.request, 'urlopen', return_value=io.BytesIO(self.payload)), \
                contextlib.redirect_stdout(io.StringIO()):
            fetch.prepare_reference(self.root)
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_bad_download_never_creates_or_overwrites_cached_file(self):
        for existing in (None, b'preserve existing corrupt bytes'):
            with self.subTest(existing=existing):
                if existing is not None:
                    self.cache(existing)
                with patch.object(fetch.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong download')):
                    with self.assertRaisesRegex(RuntimeError, 'reference hash mismatch'):
                        fetch.prepare_reference(self.root)
                if existing is None:
                    self.assertFalse((self.root / '.reference').exists())
                else:
                    self.assertEqual(self.path.read_bytes(), existing)

    def test_verify_only_rejects_missing_and_corrupt_cache_without_network(self):
        for existing in (None, b'corrupt cache'):
            with self.subTest(existing=existing):
                if existing is not None:
                    self.cache(existing)
                with patch.object(fetch.urllib.request, 'urlopen') as request:
                    with self.assertRaisesRegex(RuntimeError, 'missing or corrupt reference'):
                        fetch.prepare_reference(self.root, verify_only=True)
                    request.assert_not_called()
                if existing is None:
                    self.assertFalse((self.root / '.reference').exists())
                else:
                    self.assertEqual(self.path.read_bytes(), existing)

    def test_cli_integrity_failure_returns_nonzero_and_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(fetch, 'ROOT', self.root), patch.object(fetch.urllib.request, 'urlopen') as request, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = fetch.main(['--verify-only'])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), '')
        self.assertIn('missing or corrupt reference', stderr.getvalue())
        request.assert_not_called()

    def test_manifest_validation_precedes_any_download_or_cache_write(self):
        original = copy.deepcopy(self.manifest)
        invalid_entries = [
            {'path': name, 'sha256': original['files'][0]['sha256']}
            for name in ('../outside', '/absolute', 'nested/../../outside', './noncanonical',
                         'nested//duplicate-slash', 'windows\\path', '.', '', 1)
        ]
        invalid_entries += [{'path': 'safe/path', 'sha256': 'not-a-sha256'}, None]
        for entry in invalid_entries:
            with self.subTest(entry=entry):
                self.manifest = copy.deepcopy(original)
                # A valid earlier entry must not be fetched before the bad entry is found.
                self.manifest['files'].append(entry)
                self.save_manifest()
                with patch.object(fetch.urllib.request, 'urlopen') as request:
                    with self.assertRaises(ValueError):
                        fetch.prepare_reference(self.root)
                    request.assert_not_called()
                self.assertFalse((self.root / '.reference').exists())

    def test_manifest_rejects_duplicate_paths_and_moving_refs(self):
        original = copy.deepcopy(self.manifest)
        invalid_manifests = [
            {**original, 'files': original['files'] * 2},
            {**original, 'commit': 'main'},
            {**original, 'files': []},
            [],
        ]
        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest):
                self.manifest = manifest
                self.save_manifest()
                with patch.object(fetch.urllib.request, 'urlopen') as request:
                    with self.assertRaises(ValueError):
                        fetch.prepare_reference(self.root)
                    request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
