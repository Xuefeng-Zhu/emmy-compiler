#!/usr/bin/env python3
"""Fetch immutable public challenge data only; never execute repository scripts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(root):
    manifest = json.loads((root / 'docs/reference-manifest.json').read_text())
    if (not isinstance(manifest, dict)
            or not isinstance(manifest.get('commit'), str)
            or not re.fullmatch(r'[0-9a-f]{40}', manifest['commit'])
            or not isinstance(manifest.get('files'), list)
            or not manifest['files']):
        raise ValueError('invalid reference manifest: expected a fixed commit and nonempty file list')
    seen = set()
    for item in manifest['files']:
        if not isinstance(item, dict):
            raise ValueError('invalid reference manifest entry')
        name, digest = item.get('path'), item.get('sha256')
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute()
                or '..' in PurePosixPath(name).parts
                or str(PurePosixPath(name)) != name or name == '.'
                or name in seen):
            raise ValueError(f'invalid or duplicate reference path: {name!r}')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError(f'invalid reference SHA-256: {name}')
        seen.add(name)
    return manifest


def prepare_reference(root, *, verify_only=False):
    """Check every cached byte; fetch missing or corrupt files only when allowed."""
    manifest = read_manifest(root)
    for item in manifest['files']:
        path = root / '.reference' / item['path']
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            continue
        if verify_only:
            raise RuntimeError(f'missing or corrupt reference: {item["path"]}; run tools/fetch_reference.py')
        url = f"https://raw.githubusercontent.com/cloudrift-ai/emmy-hackathon-2026/{manifest['commit']}/{item['path']}"
        request = urllib.request.Request(url, headers={'User-Agent': 'emmy-compiler-reference-fetch'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError(f'reference hash mismatch: {item["path"]}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f'fetched {item["path"]}')
    return len(manifest['files']), manifest['commit']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true',
                        help='verify all cached hashes without downloading or writing any files')
    args = parser.parse_args(argv)
    try:
        count, commit = prepare_reference(ROOT, verify_only=args.verify_only)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f'fetch_reference.py: {exc}', file=sys.stderr)
        return 1
    print(f'Verified {count} reference files at {commit}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
