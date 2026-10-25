#!/usr/bin/env python3
"""Run one explicitly authorized case through the unchanged, pinned official client.

This command uploads generated PTX and official launch descriptions over HTTP.
It does not grant standing permission to run jobs; obtain user authorization first.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SERVER = 'http://runner.riftstack.ai:48701'
CLIENT_SHA256 = 'b3c2fac2e05e33a8d21a24c4ba7d2dfc23c9b71911646437910e0d6fd0e66dbf'


def main():
    import numpy as np
    import requests

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('problem', choices=['p00_scalar_add', 'p01_vector_add'])
    problem = ap.parse_args().problem
    client = ROOT / '.reference/submit.py'
    if hashlib.sha256(client.read_bytes()).hexdigest() != CLIENT_SHA256:
        raise RuntimeError('Official client hash differs from reviewed source')
    manifest = json.loads((ROOT / 'docs/reference-manifest.json').read_text())
    folder = ROOT / '.reference/problems' / problem
    for item in manifest['files']:
        if item['path'].startswith(f'problems/{problem}/'):
            if hashlib.sha256((ROOT / '.reference' / item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError('Official fixture hash mismatch')
    meta = json.loads((folder / 'meta.json').read_text())
    kernel, = meta['kernels']
    with np.load(folder / 'expected.npz', allow_pickle=False) as expected:
        if expected.files != ['add'] or expected['add'].shape != tuple(kernel['args'][-1]['shape']):
            raise RuntimeError('Unexpected expected-output contract')
    generated = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(folder / (kernel['entry'] + '.ir.json'))], capture_output=True, text=True, check=True)
    ptx = generated.stdout
    ptx.encode('ascii')
    report = {'problem': problem, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'client_sha256': CLIENT_SHA256, 'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(), 'requests': []}
    session = requests.Session()
    session.trust_env = False  # No inherited proxy, .netrc credentials or environment auth.
    compiled = None
    counts = {'/compile': 0, '/run': 0}

    def approved_post(url, *, json, timeout):
        nonlocal compiled
        if url not in (SERVER + '/compile', SERVER + '/run'):
            raise RuntimeError('Unexpected outbound destination')
        path = url.removeprefix(SERVER)
        counts[path] += 1
        if counts[path] > (1 if path == '/compile' else 3):
            raise RuntimeError('Bounded request limit reached; stopped')
        if path == '/compile':
            wanted = {'ptx': ptx}
        else:
            if not compiled:
                raise RuntimeError('No successful assembly receipt')
            wanted = {'cubin_id': compiled, 'entry': kernel['entry'], 'grid': kernel['grid'],
                      'block': kernel['block'], 'smem_bytes': kernel['smem_bytes'],
                      'zero_outputs': kernel['zero_outputs'], 'args': kernel['args'],
                      'warmup': 0, 'iters': 1, 'max_return': 4096}
        if json != wanted:
            raise RuntimeError('Outbound payload differs from approved case')
        session.cookies.clear()
        receipt = {'path': path, 'state': 'request_started'}
        report['requests'].append(receipt)
        response = session.post(url, json=json, timeout=timeout, allow_redirects=False)
        receipt['http_status'] = response.status_code
        if 300 <= response.status_code < 400:
            raise RuntimeError('Unexpected redirect; stopped')
        response.raise_for_status()
        data = response.json()
        receipt.update(state='response_received', ok=data.get('ok'), stage=data.get('stage'))
        if path == '/compile' and data.get('ok'):
            compiled = data['cubin_id']
            receipt.update(cubin_id=compiled, entry_points=data['entry_points'], registers=data.get('registers'))
        if path == '/run' and data.get('ok'):
            output = data.get('outputs', {}).get('add')
            if (not output or output.get('truncated') or output.get('dtype') != 'f32'
                    or output.get('shape') != kernel['args'][-1]['shape']
                    or len(output.get('values', [])) != kernel['args'][-1]['shape'][0]):
                raise RuntimeError('Missing, truncated or malformed expected output')
            receipt.update(median_us=data.get('median_us'), iters=data.get('iters'),
                           output_shape=output['shape'], output_count=len(output['values']))
        return response

    # Only the transport is constrained. Official compilation/comparison logic is unchanged.
    original_post, original_argv = requests.post, sys.argv
    requests.post = approved_post
    sys.argv = [str(client), problem, '--compiler', str(ROOT / 'compile.py'), '--server', SERVER]
    stdout, stderr, code = io.StringIO(), io.StringIO(), 1
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                module = runpy.run_path(str(client))
                code = module['main']()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if isinstance(exc.code, str):
                    print(exc.code, file=sys.stderr)
            except Exception as exc:
                print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    finally:
        requests.post, sys.argv = original_post, original_argv
        session.close()
    passed = code == 0 and 'PASS  add:' in stdout.getvalue() and counts['/run'] > 0
    report.update(exit_code=code, passed=passed, finished_at_utc=datetime.now(timezone.utc).isoformat())
    dest = ROOT / 'build/runner'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f'{problem}.stdout.txt').write_text(stdout.getvalue())
    (dest / f'{problem}.stderr.txt').write_text(stderr.getvalue())
    (dest / f'{problem}.json').write_text(json.dumps(report, indent=2) + '\n')
    (dest / f'{problem}.ptx').write_text(ptx)
    sys.stdout.write(stdout.getvalue())
    sys.stderr.write(stderr.getvalue())
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
