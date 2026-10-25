#!/usr/bin/env python3
"""Official CLI: python compile.py path/to/kernel.ir.json > out.ptx."""
import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description='Compile trusted Emmy Kernel IR to PTX (initial add slice).')
    parser.add_argument('ir', type=Path, help='Kernel IR JSON; meta.json must be in the same directory')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        print('compile.py: Python 3.12 or newer is required', file=sys.stderr)
        return 1
    try:
        from emmy_ptx import CompileError, compile_file
        result = compile_file(args.ir)
    except ImportError as exc:
        print(f'compile.py: dependency import failed: {exc}; install requirements.lock in a working Python 3.12+ venv', file=sys.stderr)
        return 1
    except (CompileError, OSError) as exc:
        print(f'compile.py: {exc}', file=sys.stderr)
        return 1
    sys.stdout.write(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
