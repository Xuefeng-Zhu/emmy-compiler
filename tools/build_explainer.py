#!/usr/bin/env python3
"""Build the offline learning page from original examples and saved synthetic GPU data.

No network, GPU request, upstream script execution or official reference arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + '\n'


def build(root, directory):
    import numpy as np
    sys.path.insert(0, str(root))
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F32
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.kernel.ir import Tile
    from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
    from emmy_ptx import compile_file

    # Original educational program, constructed from typed nodes; no fixture copy.
    size, block = 10, 8
    body = (Load('a', 'left', (Var('i'),), dtype=F32),
            Load('b', 'right', (Var('i'),), dtype=F32),
            Assign('c', ElementwiseImpl('add'), ('a', 'b'), F32),
            Write('result', (Var('i'),), 'c', value_dtype=F32))
    tile = Tile((Axis('i', Dim(size)),), body)
    nodes = {name: {'op': 'InputOp', 'op_fields': {}, 'inputs': [],
                    'outputs': [{'name': name, 'shape': [size], 'dtype': 'f32'}]}
             for name in ('left', 'right')}
    nodes['result'] = {'op': 'KernelOp', 'op_fields': {'name': 'learn_add',
        'body': [repr(tile)], 'zero_delegated': [], 'serial': []},
        'inputs': ['left', 'right'], 'outputs': [{'name': 'result', 'shape': [size], 'dtype': 'f32'}]}
    ir = {'inputs': ['left', 'right'], 'outputs': ['result'], 'nodes': nodes}
    left, right = list(range(size)), [2 * i + 1 for i in range(size)]
    args = [{'name': name, 'shape': [size], 'dtype': 'f32', 'data': data}
            for name, data in [('left', left), ('right', right)]]
    args.append({'name': 'result', 'shape': [size], 'dtype': 'f32', 'output': True})
    meta = {'rtol': 0, 'atol': 0, 'kernels': [{'node': 'result', 'entry': 'learn_add',
        'arg_order': ['left', 'right', 'result'], 'args': args, 'grid': [2, 1, 1],
        'block': [block, 1, 1], 'smem_bytes': 0, 'zero_outputs': []}]}
    with TemporaryDirectory(prefix='emmy-learning-') as temporary:
        path = Path(temporary) / 'learn_add.ir.json'
        path.write_text(json_text(ir))
        path.with_name('meta.json').write_text(json_text(meta))
        ptx = compile_file(path)
    source_files = ['compile.py'] + [str(p.relative_to(root)) for p in sorted((root / 'emmy_ptx').glob('*.py'))]
    sources = {name: digest((root / name).read_bytes()) for name in source_files}
    evidence = {}
    arrays, receipts = [], []
    for stage in ('2026-10-25-regression', '2026-10-25-exp-fix'):
        folder = root / 'docs/validation' / stage / 'synthetic/s15_exp_wide_4097'
        receipt = json.loads((folder / 'report.json').read_text())
        expected_path, output_path = folder / 'expected.npz', folder / 'run-01-output.npy'
        assert digest(expected_path.read_bytes()) == receipt['input_artifact_sha256']['expected.npz']
        run, = [request for request in receipt['requests'] if request['path'] == '/run']
        assert digest(output_path.read_bytes()) == run['output_file_sha256']
        assert digest((folder / 'kernel.ptx').read_bytes()) == receipt['ptx_sha256']
        with np.load(expected_path, allow_pickle=False) as archive:
            expected = archive['add'].copy()
        actual = np.load(output_path, allow_pickle=False)
        assert expected.shape == actual.shape == (4097,)
        assert expected.dtype == actual.dtype == np.float32
        arrays.append((expected, actual)); receipts.append(receipt)
        for name in ('expected.npz', 'run-01-output.npy', 'report.json', 'kernel.ptx'):
            relative = str((folder / name).relative_to(root))
            evidence[relative] = digest((folder / name).read_bytes())
    expected, before = arrays[0]
    expected_after, after = arrays[1]
    assert np.array_equal(expected, expected_after)
    x = np.linspace(-80, 80, 4097, dtype=np.float32)
    # Verify reconstruction against the saved high-precision synthetic oracle.
    assert np.array_equal(np.exp(x.astype(np.float64)).astype(np.float32), expected)
    for receipt in receipts:
        assert receipt['comparison'] == {'rtol': 2e-6, 'atol': 2e-7, 'equal_nan': False}
    failures = [int(np.count_nonzero(~np.isclose(a, expected, rtol=2e-6, atol=2e-7, equal_nan=False)))
                for a in (before, after)]
    assert failures == [251, 0]
    data = {'sample': {'ir': ir, 'meta': meta, 'ptx': ptx, 'left': left, 'right': right,
            'result': [a + b for a, b in zip(left, right)], 'size': size, 'block': block,
            'ptx_sha256': digest(ptx.encode())},
        'exp': {'rows': np.column_stack((x, expected, before, after)).tolist(),
            'columns': ['x', 'expected', 'before', 'after'], 'rtol': 2e-6, 'atol': 2e-7,
            'failures': failures, 'device': 'NVIDIA GeForce RTX 5090',
            'before_ptx_sha256': receipts[0]['ptx_sha256'], 'after_ptx_sha256': receipts[1]['ptx_sha256'],
            'max_relative': [float(np.max(np.abs(a.astype(np.float64) - expected) / expected)) for a in (before, after)]},
        'source_sha256': sources, 'evidence_sha256': evidence}
    template = (directory / 'template.html').read_text()
    assert template.count('__EMMY_DATA__') == 1
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    files = {'index.html': template.replace('__EMMY_DATA__', encoded),
             'sample/learn_add.ir.json': json_text(ir), 'sample/meta.json': json_text(meta),
             'sample/learn_add.ptx': ptx}
    provenance = {'description': 'Original educational sample plus archived synthetic exp GPU measurements. No official reference arrays.',
        'source_sha256': sources, 'evidence_sha256': evidence,
        'sample_ptx_sha256': digest(ptx.encode()), 'exp_samples': 4097, 'exp_failures': failures,
        'generated_sha256': {name: digest(text.encode()) for name, text in files.items()}}
    files['provenance.json'] = json_text(provenance)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    directory = args.output or root / 'docs/explainer'
    files = build(root, directory)
    if args.check:
        for name, text in files.items():
            path = directory / name
            if not path.is_file() or path.read_text() != text:
                raise SystemExit(f'Stale explainer artifact: {path}; run tools/build_explainer.py')
    else:
        for name, text in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print(f'{"Verified" if args.check else "Built"} {len(files)} explainer artifacts; 4097 real synthetic GPU samples, no network/GPU work.')


if __name__ == '__main__':
    main()
