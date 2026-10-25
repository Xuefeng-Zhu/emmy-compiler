#!/usr/bin/env python3
"""Run the local suite, rejecting empty discovery and skipped/expected failures."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-directory', type=Path, default=ROOT / 'tests',
                        help='Override discovery directory for runner regression tests')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(args.test_directory.resolve()))
    if suite.countTestCases() == 0:
        print('ERROR: test discovery found zero tests', file=sys.stderr)
        return 1
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped or result.expectedFailures:
        print('ERROR: CI requires every discovered test to run and pass', file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
