"""Generate deterministic synthetic exp stress fixtures; no GPU or network.

The attention case is newly constructed from a public typed schedule. Its name
does not identify the fixture behind an earlier single-run timing. Oracles use
float64 NumPy expressions rounded to f32 and never read compiler/model outputs.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 when running this generator.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'tests')]

import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.ir.stmt.leaves import Load
from emmy_ptx import compile_file
from ptx_model import input_buffers
from test_fused import fused_fixture
from test_numeric import scalar_fixture
from test_structured import changed_attention
from tools.run_synthetic import load_case


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rename(raw, meta, entry):
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    node['op_fields']['name'] = meta['kernels'][0]['entry'] = entry
    meta.update(id=entry, graph_inputs=list(raw['inputs']), graph_outputs=list(raw['outputs']))
    meta.pop('code', None)


def explicit_inputs(meta, buffers):
    """Freeze actual f32 values so all compiler variants receive identical data."""
    for argument in meta['kernels'][0]['args']:
        if not argument.get('output'):
            values = np.asarray(buffers[argument['name']], np.float32)
            assert list(values.shape) == argument['shape']
            assert np.isfinite(values).all()
            argument.pop('seed', None)
            argument['data'] = values.reshape(-1).tolist()


def save(root, case, raw, meta, expected, provenance):
    folder = root / case
    folder.mkdir(parents=True, exist_ok=False)
    rename(raw, meta, case)
    launch, = meta['kernels']
    output, = (argument for argument in launch['args'] if argument.get('output'))
    expected = np.asarray(expected, np.float32)
    assert list(expected.shape) == output['shape']
    assert np.isfinite(expected).all()
    assert np.isfinite(np.sum(expected, dtype=np.float32))
    ir_name = case + '.ir.json'
    write_json(folder / ir_name, raw)
    write_json(folder / 'meta.json', meta)
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, expected, allow_pickle=False)
    information = zipfile.ZipInfo(output['name'] + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(folder / 'expected.npz', 'w') as archive:
        archive.writestr(information, buffer.getvalue())
    hashes = {name: digest((folder / name).read_bytes())
              for name in (ir_name, 'meta.json', 'expected.npz')}
    write_json(folder / 'manifest.json', {'files': hashes})
    load_case(folder, folder / 'manifest.json')
    ptx = compile_file(folder / ir_name)
    (folder / 'generated.ptx').write_text(ptx)
    result = {
        'case': case, 'synthetic': True, 'prior_single_run_fixture_identity_established': False,
        'core_sha256': hashes, 'output_shape': output['shape'], 'output_name': output['name'],
        'rtol': meta['rtol'], 'atol': meta['atol'], 'equal_nan': False,
        'grid': launch['grid'], 'block': launch['block'], 'smem_bytes': launch['smem_bytes'],
        'ptx_sha256': digest(ptx.encode('ascii')), 'provenance': provenance,
        'expected_values_sha256': digest(expected.astype('<f4', copy=False).tobytes()),
        'input_values_sha256': {
            argument['name']: digest(np.asarray(argument['data'], dtype='<f4').tobytes())
            for argument in launch['args'] if not argument.get('output')},
        'cpu_model_executed': False, 'gpu_executed': False, 'ptx_assembled': False,
    }
    write_json(folder / 'provenance.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    root = options.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = []

    raw, meta = changed_attention(256, 256, 32, 32)
    inputs = input_buffers(meta['kernels'][0])
    explicit_inputs(meta, inputs)
    scores = ((inputs['q'].astype(np.float64) @ inputs['kt'].astype(np.float64))
              / float(inputs['div_c1'][0]))
    exponential = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exponential / exponential.sum(axis=-1, keepdims=True)
    expected = (weights @ inputs['v'].astype(np.float64)).astype(np.float32)
    summary.append(save(root, 's_a_attn_256', raw, meta, expected, {
        'helper': 'tests.test_structured.changed_attention(256,256,32,32)',
        'dimensions': {'query_rows': 256, 'keys': 256, 'depth': 32, 'output_columns': 32},
        'base_schedule': 'Pinned public a02_attention_128 typed IR; inherited seeds and runtime f32 scale frozen to explicit data.',
        'oracle': 'float64 Q@KT / actual f32 scale; stable float64 softmax @ float64 V, rounded f32',
        'tolerance': 'Unchanged original a02 metadata values.',
        'identity_limit': 'New synthetic fixture; not asserted identical to a previous s_a_attn_256 timing fixture.',
    }))

    raw, meta = fused_fixture('f02_row_softmax', rows=7, columns=1025, threads=128)
    inputs = input_buffers(meta['kernels'][0])
    # Widen score magnitude and retain a non-divisible tail, with exact values
    # frozen before either the oracle or any compiler variant is evaluated.
    inputs['x'] *= np.float32(9)
    explicit_inputs(meta, inputs)
    values = inputs['x'].astype(np.float64)
    exponential = np.exp(values - values.max(axis=-1, keepdims=True))
    expected = (exponential / exponential.sum(axis=-1, keepdims=True)).astype(np.float32)
    summary.append(save(root, 's_f_softmax_7x1025', raw, meta, expected, {
        'helper': 'tests.test_fused.fused_fixture(f02_row_softmax,7,1025,128)',
        'base_schedule': 'Pinned public f02_row_softmax typed IR; inherited x seed, multiplied by f32 9 and frozen to explicit data.',
        'oracle': 'Stable float64 softmax, rounded f32',
        'tolerance': 'Unchanged original f02 metadata values.',
        'purpose': 'Wide score magnitudes, multiple row reductions and non-divisible 1025-column tail.',
    }))

    size = 262145
    values = np.linspace(-20, 20, size, dtype=np.float32)
    raw, meta = scalar_fixture(size, [('exp', None)])
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    tile = Graph.from_dict(raw).nodes['add'].op.body[0]
    tile = replace(tile, body=tuple(statement for statement in tile.body
                                   if not (isinstance(statement, Load) and statement.input == 'x1')))
    node['op_fields']['body'] = [repr(tile)]
    node['inputs'].remove('x1')
    raw['inputs'].remove('x1')
    del raw['nodes']['x1']
    launch, = meta['kernels']
    launch['args'] = [argument for argument in launch['args'] if argument['name'] != 'x1']
    launch['arg_order'].remove('x1')
    explicit_inputs(meta, {'x0': values})
    meta['rtol'], meta['atol'] = 2e-6, 2e-7
    summary.append(save(root, 's_exp_262145', raw, meta,
                        np.exp(values.astype(np.float64)).astype(np.float32), {
        'helper': 'tests.test_numeric.scalar_fixture(262145,[(exp,None)]) with unused x1 removed',
        'input': 'np.linspace(-20,20,262145,dtype=float32)',
        'oracle': 'float64 exp, rounded f32',
        'tolerance': 'Existing local exp activation threshold 2e-6/2e-7; not organizer-defined.',
        'purpose': 'Large direct-exp workload with a partial final CTA and finite normal outputs.',
    }))

    write_json(root / 'stress-summary.json', {
        'generator_sha256': digest(Path(__file__).read_bytes()),
        'numpy_version': np.__version__, 'case_count': len(summary), 'cases': summary,
        'cpu_model_executed': False, 'gpu_executed': False, 'ptx_assembled': False,
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
