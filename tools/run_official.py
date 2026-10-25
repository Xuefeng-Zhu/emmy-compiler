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
import math
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SERVER = 'http://runner.riftstack.ai:48701'
CLIENT_SHA256 = 'b3c2fac2e05e33a8d21a24c4ba7d2dfc23c9b71911646437910e0d6fd0e66dbf'
PROBLEMS = (
    'p00_scalar_add', 'p01_vector_add', 'p02_scale_shift', 'p03_relu_mul',
    'r01_array_sum', 'r02_big_sum', 'r03_mean', 'r04_row_sum',
    'm01_matmul_512', 'm02_matmul_rect',
    'f01_rmsnorm', 'f02_row_softmax', 'f03_l2norm',
    'a01_attention', 'a02_attention_128',
)


def main():
    import numpy as np
    import requests

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('problem', choices=PROBLEMS)
    problem = ap.parse_args().problem
    client = ROOT / '.reference/submit.py'
    if hashlib.sha256(client.read_bytes()).hexdigest() != CLIENT_SHA256:
        raise RuntimeError('Official client hash differs from reviewed source')
    manifest = json.loads((ROOT / 'docs/reference-manifest.json').read_text())
    folder = ROOT / '.reference/problems' / problem
    case_files = [item for item in manifest['files'] if item['path'].startswith(f'problems/{problem}/')]
    if len(case_files) != 5:
        raise RuntimeError('Expected five pinned official problem files')
    for item in case_files:
        if hashlib.sha256((ROOT / '.reference' / item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Official fixture hash mismatch')
    meta = json.loads((folder / 'meta.json').read_text())
    kernel, = meta['kernels']
    output_desc, = [arg for arg in kernel['args'] if arg.get('output')]
    output_name = output_desc['name']
    output_shape = tuple(output_desc['shape'])
    output_count = math.prod(output_shape)
    max_return = max(4096, output_count)
    with np.load(folder / 'expected.npz', allow_pickle=False) as expected:
        if expected.files != [output_name]:
            raise RuntimeError('Unexpected expected-output contract')
        reference = expected[output_name].copy()
        # The official packager normalizes rank-0 buffer descriptors to [1].
        scalar_buffer = reference.ndim == 0 and output_shape == (1,) and output_count == 1
        if (reference.dtype != np.float32 or reference.size != output_count
                or not (reference.shape == output_shape or scalar_buffer)):
            raise RuntimeError('Unexpected expected-output contract')
    generated = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(folder / (kernel['entry'] + '.ir.json'))], capture_output=True, text=True)
    ptx = generated.stdout
    report = {'problem': problem, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'client_sha256': CLIENT_SHA256, 'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
              'local_generation_exit_code': generated.returncode, 'expected_output': output_name,
              'comparison': {'rtol': meta['rtol'], 'atol': meta['atol'], 'equal_nan': False},
              'expected_shape': list(reference.shape), 'output_shape': list(output_shape),
              'output_count': output_count, 'max_return': max_return, 'requests': []}
    dest = ROOT / 'build/runner'
    dest.mkdir(parents=True, exist_ok=True)

    def save_report():
        (dest / f'{problem}.json').write_text(json.dumps(report, indent=2) + '\n')

    if generated.returncode != 0:
        report.update(stage='local_generation', exit_code=generated.returncode, passed=False,
                      uncertain_request=False, finished_at_utc=datetime.now(timezone.utc).isoformat())
        (dest / f'{problem}.stdout.txt').write_text(generated.stdout)
        (dest / f'{problem}.stderr.txt').write_text(generated.stderr)
        save_report()
        sys.stdout.write(generated.stdout)
        sys.stderr.write(generated.stderr)
        return 1

    ptx.encode('ascii')
    session = requests.Session()
    session.trust_env = False  # No inherited proxy, .netrc credentials or environment auth.
    compiled = None
    comparison_ok = False
    counts = {'/compile': 0, '/run': 0}

    def approved_post(url, *, json, timeout):
        nonlocal compiled, comparison_ok
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
                      'warmup': 0, 'iters': 1, 'max_return': max_return}
        if json != wanted:
            raise RuntimeError('Outbound payload differs from approved case')
        session.cookies.clear()
        receipt = {'path': path, 'state': 'request_started'}
        report['requests'].append(receipt)
        save_report()  # Persist an uncertain request before any network side effect.
        response = session.post(url, json=json, timeout=timeout, allow_redirects=False)
        receipt['http_status'] = response.status_code
        save_report()
        if 300 <= response.status_code < 400:
            raise RuntimeError('Unexpected redirect; stopped')
        response.raise_for_status()
        data = response.json()
        receipt.update(state='response_received', ok=data.get('ok'), stage=data.get('stage'))
        if isinstance(data.get('stderr'), str):
            receipt['stderr'] = data['stderr']
        for key in ('gpu', 'device', 'arch'):
            if isinstance(data.get(key), (str, int, float)):
                receipt[key] = data[key]
        save_report()
        if path == '/compile' and data.get('ok'):
            compiled = data['cubin_id']
            receipt.update(cubin_id=compiled, entry_points=data['entry_points'], registers=data.get('registers'))
        if path == '/run' and data.get('ok'):
            outputs = data.get('outputs')
            output = outputs.get(output_name) if isinstance(outputs, dict) else None
            if (not isinstance(outputs, dict) or set(outputs) != {output_name}
                    or not isinstance(output, dict) or output.get('truncated')
                    or output.get('dtype') != 'f32' or output.get('shape') != list(output_shape)
                    or not isinstance(output.get('values'), list)
                    or len(output['values']) != output_count
                    or not all(type(value) in (int, float) for value in output['values'])):
                raise RuntimeError('Missing, truncated or malformed expected output')
            # Runner values are flat even when the descriptor/reference is 2-D.
            actual = np.asarray(output['values'], dtype=np.float32).reshape(reference.shape)
            comparison_ok = bool(np.allclose(actual, reference, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False))
            with np.errstate(invalid='ignore', over='ignore'):
                error = float(np.max(np.abs(actual - reference)))
            receipt.update(median_us=data.get('median_us'), iters=data.get('iters'),
                           output_name=output_name, output_shape=output['shape'], output_count=output_count,
                           comparison_passed=comparison_ok, max_abs_error=error if np.isfinite(error) else None)
        save_report()
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
    passed = code == 0 and f'PASS  {output_name}:' in stdout.getvalue() and comparison_ok and counts['/run'] > 0
    report.update(exit_code=code, passed=passed, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  uncertain_request=any(item['state'] == 'request_started' for item in report['requests']))
    (dest / f'{problem}.stdout.txt').write_text(stdout.getvalue())
    (dest / f'{problem}.stderr.txt').write_text(stderr.getvalue())
    save_report()
    (dest / f'{problem}.ptx').write_text(ptx)
    sys.stdout.write(stdout.getvalue())
    sys.stderr.write(stderr.getvalue())
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
