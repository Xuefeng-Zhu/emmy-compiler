"""Capture complete public runner responses around the existing bounded wrapper."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

import requests

root = Path('/Users/frank/mygit/emmy-compiler')
case = sys.argv[1]
destination = root / 'build/validation-0e16a83/official' / case
destination.mkdir(parents=True, exist_ok=False)
original = requests.Session.post
records = []


def capture(self, url, **kwargs):
    response = original(self, url, **kwargs)
    raw = response.content
    name = f'response-{len(records) + 1}.json'
    (destination / name).write_bytes(raw)
    payload = json.dumps(kwargs['json'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    records.append({'path': url.removeprefix('http://runner.riftstack.ai:48701'),
                    'request_sha256': hashlib.sha256(payload).hexdigest(),
                    'response_file': name, 'response_sha256': hashlib.sha256(raw).hexdigest(),
                    'http_status': response.status_code})
    (destination / 'transport.json').write_text(json.dumps(records, indent=2) + '\n')
    return response


requests.Session.post = capture
stdout, stderr = io.StringIO(), io.StringIO()
exit_code = 1
try:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        module = runpy.run_path(str(root / 'tools/run_official.py'))
        exit_code = module['main']()
finally:
    requests.Session.post = original
    (destination / 'wrapper.stdout.txt').write_text(stdout.getvalue())
    (destination / 'wrapper.stderr.txt').write_text(stderr.getvalue())
    for path in (root / 'build/runner').glob(case + '.*'):
        (destination / path.name).write_bytes(path.read_bytes())
    (destination / 'wrapper-exit.json').write_text(json.dumps({'exit_code': exit_code}) + '\n')
sys.stdout.write(stdout.getvalue())
sys.stderr.write(stderr.getvalue())
raise SystemExit(exit_code)
