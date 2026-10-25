#!/usr/bin/env python3
"""CPU-only CLI smoke checks for every manifest-listed public IR fixture.

This checks the command contract and emitted text; it does not assemble PTX or
execute GPU kernels. Run fetch_reference.py --verify-only before this command.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def validate_success(result, entry):
    if result.returncode != 0:
        raise ValueError(f'compiler exited {result.returncode}')
    if result.stderr:
        raise ValueError('successful compiler wrote to stderr')
    try:
        ptx = result.stdout.decode('ascii')
    except UnicodeDecodeError as exc:
        raise ValueError('stdout is not ASCII PTX') from exc
    if not ptx.startswith('.version 8.0\n.target sm_89\n.address_size 64\n'):
        raise ValueError('stdout lacks the expected PTX header')
    entries = re.findall(r'\.visible\s+\.entry\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(', ptx)
    if entries != [entry]:
        raise ValueError('stdout has an unexpected PTX entry')
    if not ptx.endswith('}\n'):
        raise ValueError('stdout has an incomplete module or trailing diagnostic')
    return hashlib.sha256(result.stdout).hexdigest()


def validate_failure(result):
    if result.returncode == 0 or result.stdout or not result.stderr.strip():
        raise ValueError('invalid input must fail with stderr and empty stdout')
    if b'Traceback' in result.stderr:
        raise ValueError('invalid input produced an uncaught traceback')


def main():
    manifest = json.loads((ROOT / 'docs/reference-manifest.json').read_text())
    inputs = sorted(item['path'] for item in manifest['files']
                    if item['path'].startswith('problems/')
                    and item['path'].endswith('.ir.json'))
    if not inputs:
        raise ValueError('manifest contains no IR fixtures')
    command = [sys.executable, '-B', str(ROOT / 'compile.py')]
    with TemporaryDirectory(prefix='emmy-cli-smoke-') as directory:
        for relative in inputs:
            source = ROOT / '.reference' / relative
            kernel, = json.loads((source.parent / 'meta.json').read_text())['kernels']
            result = subprocess.run([*command, str(source)], cwd=directory,
                                    capture_output=True, timeout=60)
            digest = validate_success(result, kernel['entry'])
            print(f'PASS {source.parent.name} PTX SHA-256 {digest}', flush=True)
        for args in ([], [str(Path(directory) / 'missing.ir.json')]):
            result = subprocess.run([*command, *args], cwd=directory,
                                    capture_output=True, timeout=60)
            validate_failure(result)
    print(f'PASS: {len(inputs)} CLI emissions and 2 invalid-input checks; CPU/text only.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f'check_cli.py: {exc}', file=sys.stderr)
        raise SystemExit(1)
