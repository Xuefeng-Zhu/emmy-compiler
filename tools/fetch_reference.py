#!/usr/bin/env python3
"""Fetch immutable public challenge data only; never execute repository scripts."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'docs/reference-manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / '.reference' / item['path']
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            continue
        url = f"https://raw.githubusercontent.com/cloudrift-ai/emmy-hackathon-2026/{manifest['commit']}/{item['path']}"
        request = urllib.request.Request(url, headers={'User-Agent': 'emmy-compiler-reference-fetch'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError(f'reference hash mismatch: {item["path"]}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f'fetched {item["path"]}')
    print(f'Verified {len(manifest["files"])} reference files at {manifest["commit"]}.')


if __name__ == '__main__':
    main()
