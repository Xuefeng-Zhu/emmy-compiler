#!/usr/bin/env python3
"""Run one explicitly approved synthetic case through the pinned official client.

Only generated PTX and exact, hash-pinned launch descriptions leave this process.
The attempt directory must be new. This is a correctness check, not a benchmark.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import runpy
import subprocess
import sys

SERVER = 'http://runner.riftstack.ai:48701'
CLIENT_SHA256 = 'b3c2fac2e05e33a8d21a24c4ba7d2dfc23c9b71911646437910e0d6fd0e66dbf'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_case(folder, manifest_path):
    """Validate a single f32-buffer kernel and return its immutable file bytes."""
    import numpy as np

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or len(files) != 3:
        raise ValueError('Manifest must contain exactly three filename-to-SHA256 entries')
    captured = {}
    for name, expected_hash in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Manifest paths must be simple filenames')
        data = (folder / name).read_bytes()
        if digest(data) != expected_hash:
            raise ValueError(f'Manifest hash mismatch: {name}')
        captured[name] = data
    if not {'meta.json', 'expected.npz'} <= captured.keys():
        raise ValueError('Missing metadata or expected array in manifest')
    meta = json.loads(captured['meta.json'])
    for key in ('rtol', 'atol'):
        if type(meta.get(key)) not in (int, float) or not math.isfinite(meta[key]) or meta[key] < 0:
            raise ValueError('Tolerances must be finite and nonnegative')
    if not isinstance(meta.get('kernels'), list) or len(meta['kernels']) != 1:
        raise ValueError('Exactly one kernel is supported')
    kernel = meta['kernels'][0]
    entry = kernel.get('entry')
    if not isinstance(entry, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', entry):
        raise ValueError('Invalid entry point')
    if set(captured) != {'meta.json', 'expected.npz', entry + '.ir.json'}:
        raise ValueError('Manifest IR filename must match the entry point')
    for key in ('grid', 'block'):
        dims = kernel.get(key)
        if not isinstance(dims, list) or len(dims) != 3 or any(type(v) is not int or v <= 0 for v in dims):
            raise ValueError('Launch dimensions must contain three positive integers')
    if math.prod(kernel['block']) > 1024:
        raise ValueError('More than 1024 threads per block')
    if type(kernel.get('smem_bytes')) is not int or not 0 <= kernel['smem_bytes'] <= 99 * 1024:
        raise ValueError('Shared memory exceeds the runner contract')
    args = kernel.get('args')
    if not isinstance(args, list) or not args:
        raise ValueError('Missing arguments')
    names, outputs, count = [], [], 0
    for arg in args:
        if not isinstance(arg, dict) or set(arg) - {'name', 'dtype', 'shape', 'seed', 'data', 'output'}:
            raise ValueError('Only f32 buffers with seed, data, or output are supported')
        name, shape = arg.get('name'), arg.get('shape')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('Invalid argument name')
        if name in names or arg.get('dtype') != 'f32':
            raise ValueError('Arguments must have unique names and f32 dtype')
        if not isinstance(shape, list) or not shape or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Buffer dimensions must be positive; scalar buffers use [1]')
        size = math.prod(shape)
        count += size
        if count > 2 ** 26:
            raise ValueError('Runner total-buffer limit exceeded')
        forms = [key for key in ('seed', 'data', 'output') if key in arg]
        if len(forms) != 1:
            raise ValueError('Each argument needs exactly one seed/data/output form')
        if 'output' in arg:
            if arg['output'] is not True:
                raise ValueError('Output marker must be true')
            outputs.append(arg)
        elif 'seed' in arg:
            if type(arg['seed']) is not int or not 0 <= arg['seed'] < 2 ** 64:
                raise ValueError('Seed must be a nonnegative integer below 2^64')
        else:
            if (not isinstance(arg['data'], list) or len(arg['data']) != size
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in arg['data'])):
                raise ValueError('Data must be a flat finite numeric list matching the buffer size')
            with np.errstate(over='ignore'):
                if not np.all(np.isfinite(np.asarray(arg['data'], dtype=np.float32))):
                    raise ValueError('Input data overflows f32')
        names.append(name)
    if kernel.get('arg_order') != names or len(outputs) != 1:
        raise ValueError('Argument order must match the signature; exactly one output is supported')
    zeros = kernel.get('zero_outputs')
    if not isinstance(zeros, list) or len(set(zeros)) != len(zeros) or any(n != outputs[0]['name'] for n in zeros):
        raise ValueError('zero_outputs must contain only the output name')
    output = outputs[0]
    size = math.prod(output['shape'])
    if size > 2 ** 22:
        raise ValueError('Runner max_return limit exceeded')
    with np.load(io.BytesIO(captured['expected.npz']), allow_pickle=False) as arrays:
        if arrays.files != [output['name']]:
            raise ValueError('Expected array names differ from the output')
        reference = arrays[output['name']].copy()
    scalar = reference.ndim == 0 and output['shape'] == [1]
    if (reference.dtype != np.float32 or reference.size != size
            or not (list(reference.shape) == output['shape'] or scalar)):
        raise ValueError('Expected array must have the exact f32 output shape/count')
    return meta, kernel, output, reference, captured, files


def main():
    import numpy as np
    import requests

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case', type=Path)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--attempt', type=Path, required=True)
    options = ap.parse_args()
    project, folder, dest = options.project.resolve(), options.case.resolve(), options.attempt.resolve()
    client = project / '.reference/submit.py'
    if digest(client.read_bytes()) != CLIENT_SHA256:
        raise ValueError('Official client differs from the reviewed source')
    meta, kernel, output, reference, captured, hashes = load_case(folder, options.manifest)
    dest.mkdir(parents=True, exist_ok=False)
    frozen = dest / 'case'
    frozen.mkdir()
    for name, data in captured.items():
        (frozen / name).write_bytes(data)
    report = {'case': folder.name, 'started_at_utc': utc(), 'client_sha256': CLIENT_SHA256,
              'input_artifact_sha256': hashes, 'requests': [], 'passed': False,
              'expected_output': output['name'], 'expected_shape': list(reference.shape),
              'output_shape': output['shape'], 'output_count': reference.size,
              'comparison': {'rtol': meta['rtol'], 'atol': meta['atol'], 'equal_nan': False},
              'compiler_source_sha256': {str(p.relative_to(project)): digest(p.read_bytes())
                  for p in [project / 'compile.py', *sorted((project / 'emmy_ptx').glob('*.py'))]}}

    def save():
        report['uncertain_request'] = any(r['state'] == 'request_started' for r in report['requests'])
        (dest / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    generated = subprocess.run([sys.executable, str(project / 'compile.py'), str(frozen / (kernel['entry'] + '.ir.json'))], capture_output=True, text=True)
    ptx = generated.stdout
    report.update(local_generation_exit_code=generated.returncode, ptx_sha256=digest(ptx.encode()))
    (dest / 'generation.stdout.txt').write_text(ptx)
    (dest / 'generation.stderr.txt').write_text(generated.stderr)
    if generated.returncode or not ptx.isascii() or len(ptx) > 2_000_000:
        report.update(stage='local_generation', exit_code=generated.returncode or 1, finished_at_utc=utc())
        save()
        sys.stderr.write(generated.stderr or 'PTX must be ASCII and at most 2,000,000 characters\n')
        return 1
    (dest / 'kernel.ptx').write_text(ptx)
    session = requests.Session()
    session.trust_env = False
    compiled, comparison_ok = None, False
    counts = {'/compile': 0, '/run': 0}
    max_return = max(4096, reference.size)

    def approved_post(url, *, json, timeout):
        nonlocal compiled, comparison_ok
        if url not in (SERVER + '/compile', SERVER + '/run'):
            raise RuntimeError('Unexpected outbound destination')
        path = url.removeprefix(SERVER)
        counts[path] += 1
        if counts[path] > (1 if path == '/compile' else 3):
            raise RuntimeError('Bounded request limit reached')
        if path == '/compile':
            wanted = {'ptx': ptx}
        else:
            if not compiled:
                raise RuntimeError('Missing successful assembly receipt')
            wanted = {'cubin_id': compiled, 'entry': kernel['entry'], 'grid': kernel['grid'],
                      'block': kernel['block'], 'smem_bytes': kernel['smem_bytes'],
                      'zero_outputs': kernel['zero_outputs'], 'args': kernel['args'],
                      'warmup': 0, 'iters': 1, 'max_return': max_return}
        if json != wanted:
            raise RuntimeError('Outbound payload differs from the pinned case')
        session.cookies.clear()
        receipt = {'path': path, 'state': 'request_started', 'started_at_utc': utc(),
                   'request_sha256': digest(__import__('json').dumps(json, sort_keys=True, allow_nan=False).encode())}
        report['requests'].append(receipt)
        save()
        response = session.post(url, json=json, timeout=timeout, allow_redirects=False)
        receipt['http_status'] = response.status_code
        save()
        if 300 <= response.status_code < 400:
            raise RuntimeError('Unexpected redirect')
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError('Malformed response object')
        receipt.update(state='response_received', ok=data.get('ok'), stage=data.get('stage'), finished_at_utc=utc())
        for key in ('stderr', 'gpu', 'device', 'arch'):
            if isinstance(data.get(key), (str, int, float)):
                receipt[key] = data[key]
        save()
        if path == '/compile' and data.get('ok'):
            compiled = data['cubin_id']
            receipt.update(cubin_id=compiled, entry_points=data['entry_points'], registers=data.get('registers'))
        if path == '/run' and data.get('ok'):
            outputs = data.get('outputs')
            actual_output = outputs.get(output['name']) if isinstance(outputs, dict) else None
            if (not isinstance(outputs, dict) or set(outputs) != {output['name']}
                    or not isinstance(actual_output, dict) or actual_output.get('truncated')
                    or actual_output.get('dtype') != 'f32' or actual_output.get('shape') != output['shape']
                    or not isinstance(actual_output.get('values'), list)
                    or len(actual_output['values']) != reference.size
                    or any(type(v) not in (int, float) for v in actual_output['values'])):
                raise RuntimeError('Missing, truncated or malformed expected output')
            actual = np.asarray(actual_output['values'], dtype=np.float32).reshape(reference.shape)
            array_path = dest / f'run-{counts[path]:02d}-output.npy'
            np.save(array_path, actual, allow_pickle=False)
            comparison_ok = bool(np.allclose(actual, reference, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False))
            with np.errstate(invalid='ignore', over='ignore'):
                error = float(np.max(np.abs(actual - reference)))
            receipt.update(output_file=array_path.name, output_file_sha256=digest(array_path.read_bytes()),
                           output_values_sha256=digest(actual.astype('<f4', copy=False).tobytes()),
                           output_name=output['name'], output_shape=output['shape'], output_count=reference.size,
                           comparison_passed=comparison_ok, max_abs_error=error if math.isfinite(error) else None,
                           median_us=data.get('median_us'), iters=data.get('iters'))
        save()
        return response

    original_post, original_argv = requests.post, sys.argv
    requests.post = approved_post
    sys.argv = [str(client), str(frozen), '--compiler', str(project / 'compile.py'), '--server', SERVER]
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
    passed = code == 0 and f'PASS  {output["name"]}:' in stdout.getvalue() and comparison_ok and counts['/run'] > 0
    report.update(exit_code=code, passed=passed, finished_at_utc=utc())
    (dest / 'stdout.txt').write_text(stdout.getvalue())
    (dest / 'stderr.txt').write_text(stderr.getvalue())
    save()
    sys.stdout.write(stdout.getvalue())
    sys.stderr.write(stderr.getvalue())
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
