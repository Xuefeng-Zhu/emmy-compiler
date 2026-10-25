"""Create deterministic synthetic Emmy fixtures and execute CPU PTX models only.

No network, renderer, external compiler, upstream generation script or GPU call.
Expected arrays use independent NumPy/math formulas, never the compiler/model.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
REPO = options.repo.resolve()
ROOT = options.output.resolve()
ROOT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(REPO), str(REPO / 'tests')]

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy_ptx import compile_file
from ptx_model import input_buffers
from reduction_model import run_reduction_ptx
from structured_model import run_structured_ptx
from test_conditionals import guarded_attention
from test_numeric import scalar_fixture
from test_reduction import get_tile, sum_fixture
from test_reduction_composed import change_loop, change_reducer
from test_structured import axis, synthetic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def expected_npz(path, output, value):
    """Use a fixed ZIP timestamp so identical fixtures have identical hashes."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value, np.float32), allow_pickle=False)
    information = zipfile.ZipInfo(output + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(information, buffer.getvalue())


def rename_entry(raw, meta, entry):
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    node['op_fields']['name'] = entry
    meta['kernels'][0]['entry'] = entry


def input_data(meta, name, values):
    descriptor, = (arg for arg in meta['kernels'][0]['args'] if arg['name'] == name)
    values = np.asarray(values, np.float32)
    assert tuple(descriptor['shape']) == values.shape
    assert np.all(np.isfinite(values))
    descriptor.pop('seed', None)
    descriptor['data'] = values.reshape(-1).tolist()


def witnesses():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    positive = [tiny, np.float32(2**-24), np.float32(1e-8), np.float32(1e-6)]
    for center in (np.float32(0.125), np.float32(0.1767766922712326),
                   np.float32(0.15649288892745972)):
        positive.extend([np.nextafter(center, np.float32(0)), center,
                         np.nextafter(center, np.float32(np.inf))])
    positive.extend([np.float32(1), np.float32(8)])
    result = [np.float32(-0.0), np.float32(0.0)]
    for value in positive:
        result.extend([-value, value])
    return np.asarray(result, np.float32)


def attention_oracle(buffers, scale_name):
    # Match the public high-precision oracle but use the actual f32 runtime scale.
    scores = (buffers['q'].astype(np.float64) @ buffers['kt'].astype(np.float64)) / float(buffers[scale_name][0])
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
    return (weights @ buffers['v'].astype(np.float64)).astype(np.float32)


def unary_oracle(operation, values):
    if operation == 'exp':
        return np.exp(values)
    if operation == 'tanh':
        return np.tanh(values)
    if operation == 'sigmoid':
        return 1 / (1 + np.exp(-values))
    if operation == 'silu':
        return values / (1 + np.exp(-values))
    if operation == 'erf':
        return np.asarray([math.erf(float(x)) for x in values], np.float64)
    if operation == 'gelu':
        error = np.asarray([math.erf(float(x) / math.sqrt(2)) for x in values], np.float64)
        return 0.5 * values.astype(np.float64) * (1 + error)
    if operation == 'gelu_tanh':
        return 0.5 * values * (1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * values ** 3)))
    raise AssertionError(operation)


def activation_ir(operation, values, entry):
    raw, meta = scalar_fixture(len(values), [(operation, None)])
    # Remove the helper's unused second input/load; this is still typed Emmy IR.
    tile = get_tile(raw)
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    tile = replace(tile, body=tuple(statement for statement in tile.body
                                   if not (isinstance(statement, Load) and statement.input == 'x1')))
    node['op_fields']['body'] = [repr(tile)]
    node['inputs'].remove('x1')
    raw['inputs'].remove('x1')
    del raw['nodes']['x1']
    launch = meta['kernels'][0]
    launch['arg_order'].remove('x1')
    launch['args'] = [arg for arg in launch['args'] if arg['name'] != 'x1']
    meta['graph_inputs'] = ['x0']
    rename_entry(raw, meta, entry)
    input_data(meta, 'x0', values)
    return raw, meta


SUMMARY = []
ALL_HASHES = {}


def save_case(case_id, raw, meta, expected, *, model, provenance, diagnostic=False):
    folder = ROOT / case_id
    folder.mkdir(exist_ok=True)
    meta['id'] = case_id
    # Inherited public description may describe a different expression/shape.
    meta.pop('code', None)
    launch, = meta['kernels']
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    output, = (argument for argument in launch['args'] if argument.get('output'))
    assert launch['entry'] == node['op_fields']['name']
    assert launch['arg_order'] == [argument['name'] for argument in launch['args']]
    for argument in launch['args']:
        assert argument['dtype'] == 'f32'
        assert sum(key in argument for key in ('seed', 'data', 'output')) == 1
        if 'data' in argument:
            assert len(argument['data']) == math.prod(argument['shape'])
            assert np.all(np.isfinite(np.asarray(argument['data'], np.float32)))
    meta['graph_inputs'] = list(raw['inputs'])
    meta['graph_outputs'] = list(raw['outputs'])
    entry_file = launch['entry'] + '.ir.json'
    source = folder / entry_file
    write_json(source, raw)
    write_json(folder / 'meta.json', meta)
    expected = np.asarray(expected, np.float32)
    assert expected.size == math.prod(output['shape']) and np.all(np.isfinite(expected))
    expected_npz(folder / 'expected.npz', output['name'], expected)
    manifest = {'files': {name: sha(folder / name) for name in (entry_file, 'meta.json', 'expected.npz')}}
    write_json(folder / 'manifest.json', manifest)
    ALL_HASHES.update({case_id + '/' + name: digest for name, digest in manifest['files'].items()})
    record = {'case': case_id, 'entry': launch['entry'], 'output': output['name'],
              'expected_shape': list(expected.shape), 'rtol': meta['rtol'], 'atol': meta['atol'],
              'equal_nan': False, 'grid': launch['grid'], 'block': launch['block'],
              'model': model, 'diagnostic': diagnostic, 'provenance': provenance,
              'gpu_executed': False, 'ptx_assembled': False}
    started = time.perf_counter()
    try:
        Graph.from_dict(raw)
        ptx = compile_file(source)
        (folder / 'generated.ptx').write_text(ptx)
        record['ptx_sha256'] = hashlib.sha256(ptx.encode()).hexdigest()
        result = (run_reduction_ptx if model == 'cooperative' else run_structured_ptx)(ptx, launch)
        assert set(result.outputs) == {output['name']}
        actual = np.asarray(result.outputs[output['name']], np.float32).reshape(expected.shape)
        expected_npz(folder / 'cpu-actual.npz', output['name'], actual)
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        close = np.isclose(actual, expected, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
        record.update({'max_abs_error': float(np.max(error)), 'mismatched_values': int(np.count_nonzero(~close)),
                       'stores_once': bool(np.all(result.store_counts[output['name']] == 1)),
                       'actual_finite': bool(np.all(np.isfinite(actual)))})
        nonzero = expected != 0
        record['max_relative_error_nonzero'] = float(np.max(error[nonzero] / np.abs(expected.astype(np.float64)[nonzero]))) if np.any(nonzero) else 0.0
        assert record['stores_once'] and record['actual_finite']
        np.testing.assert_allclose(actual, expected, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
        record['cpu_result'] = 'PASS'
    except Exception as error:
        record['cpu_result'] = 'FAIL'
        record['failure_type'] = type(error).__name__
        record['failure'] = str(error)[:2000]
    record['elapsed_seconds'] = round(time.perf_counter() - started, 6)
    write_json(folder / 'cpu-result.json', record)
    SUMMARY.append(record)
    write_json(ROOT / 'summary.json', {'compiler_commit': COMMIT, 'complete': False, 'cases': SUMMARY})
    write_json(ROOT / 'manifest.json', {'files': ALL_HASHES})
    print(json.dumps({'case': case_id, 'cpu_result': record['cpu_result'],
                      'mismatched_values': record.get('mismatched_values'),
                      'max_abs_error': record.get('max_abs_error')}, sort_keys=True), flush=True)


COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()

for number, shape in enumerate(((7, 13), (17, 19)), 1):
    rows, columns = shape
    indices = (Var('row'), Var('column'))
    tile = Tile((axis('row', rows), axis('column', columns)), (
        Load('lhs', 'left_matrix', indices, dtype=F32),
        Load('rhs', 'right_matrix', indices, dtype=F32),
        Assign('sum', ElementwiseImpl('add'), ('lhs', 'rhs'), F32),
        Write('matrix_sum', indices, 'sum', value_dtype=F32)))
    case = f's{number:02}_add_{rows}x{columns}'
    raw, meta = synthetic(tile, {'left_matrix': shape, 'right_matrix': shape}, shape,
                          output='matrix_sum', entry=case, grid=(rows * columns + 31) // 32)
    meta['rtol'] = meta['atol'] = 0.0
    for seed, argument in enumerate(meta['kernels'][0]['args'][:-1], 20261025 + number * 2):
        argument['seed'] = seed
    buffers = input_buffers(meta['kernels'][0])
    expected = np.add(buffers['left_matrix'], buffers['right_matrix'], dtype=np.float32)
    save_case(case, raw, meta, expected, model='structured', provenance={
        'helper': 'tests.test_structured.synthetic', 'test': 'tests/test_pointwise_extended.py::test_rank_two_pointwise_add_with_two_axes_or_flattened_axis',
        'oracle': 'np.add(lhs, rhs, dtype=float32); exact comparison',
        'launch': 'existing synthetic helper block=32; grid=ceil(product(Tile.extents)/32)',
        'inputs': 'NumPy default_rng(seed).standard_normal(shape).astype(float32), official seed contract'})

boundary_values = witnesses()
random_seed = 202610251
values = np.concatenate((boundary_values, np.random.default_rng(random_seed).uniform(-8, 8, 193 - len(boundary_values)).astype(np.float32)))
thresholds = {'exp': (2e-6, 2e-7), 'tanh': (2e-6, 2e-7), 'sigmoid': (2e-6, 2e-7), 'silu': (2e-6, 2e-7),
              'erf': (2e-6, 6e-7), 'gelu': (2e-5, 1e-6), 'gelu_tanh': (2e-5, 1e-6)}
for number, operation in enumerate(thresholds, 3):
    case = f's{number:02}_{operation}_193'
    raw, meta = activation_ir(operation, values, case)
    meta['rtol'], meta['atol'] = thresholds[operation]
    save_case(case, raw, meta, unary_oracle(operation, values), model='structured', provenance={
        'helper': 'tests.test_numeric.scalar_fixture; unused second input removed with typed IR',
        'test': 'tests/test_pointwise_extended.py', 'oracle': operation + ' independent NumPy/math formula matching existing test, finalized to float32',
        'tolerance': 'unchanged test_unary_activations_changed_sizes_and_inputs or test_erf_and_gelu_variants_against_independent_formulas',
        'random': {'generator': 'default_rng.uniform(-8,8).astype(float32)', 'seed': random_seed, 'count': 193 - len(boundary_values)},
        'witness_prefix_count': len(boundary_values), 'witness_input_f32_hex': [f'{bits:08x}' for bits in boundary_values.view(np.uint32)],
        'witness_description': 'signed zeros, signed smallest subnormal, near-zero, neighboring f32 values around 0.125 and both GELU internal transition inputs; +/-1,+/-8'})

for number, operation, size, threads in ((10, 'sum', 1031, 256), (11, 'sum_squares', 4099, 256), (12, 'max_negative', 73, 64)):
    case = f's{number:02}_{operation}_{size}_t{threads}'
    raw, meta = sum_fixture(size, threads)
    rename_entry(raw, meta, case)
    seed = 202610250 + number
    values = np.random.default_rng(seed).standard_normal(size).astype(np.float32)
    if operation == 'sum_squares':
        load, accumulator = get_tile(raw).body[0].body
        square = Assign('squared', ElementwiseImpl('multiply'), (load.names[0], load.names[0]), accumulator.dtype)
        change_loop(raw, (load, square, replace(accumulator, value='squared')))
        expected = np.sum(values.astype(np.float64) ** 2, dtype=np.float64).astype(np.float32)
        oracle = 'sum(float64(x)**2, dtype=float64).astype(float32), public squared-norm reference precision'
    elif operation == 'max_negative':
        change_reducer(raw, 'maximum')
        values = -np.random.default_rng(seed).uniform(0.25, 32, size).astype(np.float32)
        values[-1] = np.float32(-0.03125)
        expected = np.max(values.astype(np.float64)).astype(np.float32)
        oracle = 'max(float64(x)).astype(float32); final tail element is the strictly negative maximum'
    else:
        expected = values.astype(np.float64).sum().astype(np.float32)
        oracle = 'float64(x).sum().astype(float32), same as public r01 reference'
    input_data(meta, 'x', values)
    assert meta['rtol'] == meta['atol'] == 1e-4
    save_case(case, raw, meta, expected, model='cooperative', provenance={
        'helpers': ['tests.test_reduction.sum_fixture', 'tests.test_reduction_composed.change_loop/change_reducer'],
        'base': 'pinned official r01_array_sum typed IR and unchanged 1e-4/1e-4 tolerance',
        'input_seed_used_before_explicit_data': seed, 'oracle': oracle,
        'launch': 'one CTA; Tile extent/block_threads, loop stride, warp scratch and tree length all changed together by sum_fixture'})

raw, meta = guarded_attention(5, 11, 7, 6)
case = 's13_guarded_attention_5x11x7x6'
rename_entry(raw, meta, case)
save_case(case, raw, meta, attention_oracle(input_buffers(meta['kernels'][0]), 'div_c1'), model='structured', provenance={
    'helper': 'tests.test_conditionals.guarded_attention(5,11,7,6)',
    'oracle': 'float64 Q@KT / actual f32 runtime scale; stable float64 softmax @ float64 V; output float32',
    'launch': 'unchanged helper launch including padded axes and extra guard CTA',
    'tolerance': 'unchanged original a02 metadata: 1e-3/1e-4'})

saved = REPO / 'docs/validation/2026-10-24-generalization/generated-attention/greedy/5x9x7x11'
raw = json.loads((saved / 'result.ir.json').read_text())
meta = json.loads((saved / 'meta.json').read_text())
save_case('s14_generated_greedy_attention_5x9x7x11', raw, meta,
          attention_oracle(input_buffers(meta['kernels'][0]), 'scale'), model='structured', provenance={
    'source_ir': str((saved / 'result.ir.json').relative_to(REPO)), 'source_ir_sha256': sha(saved / 'result.ir.json'),
    'source_meta_sha256': sha(saved / 'meta.json'), 'source_entry_preserved': True,
    'generation': 'previously saved fresh high-level Graph -> official KERNEL_PASSES, greedy; no regeneration script executed here',
    'oracle': 'float64 Q@KT / actual f32 runtime scale; stable float64 softmax @ float64 V; output float32',
    'launch_and_tolerance': 'saved metadata unchanged; IR filename changed to exact entry.ir.json'})

case = 's15_exp_wide_4097'
values = np.linspace(-80, 80, 4097, dtype=np.float32)
raw, meta = activation_ir('exp', values, case)
meta['rtol'], meta['atol'] = thresholds['exp']
save_case(case, raw, meta, np.exp(values.astype(np.float64)).astype(np.float32), model='structured', diagnostic=True, provenance={
    'purpose': 'preserve a known test-discovered wide-range exp accuracy limitation; no tolerance relaxation or compiler change',
    'input': 'np.linspace(-80,80,4097,dtype=float32)', 'oracle': 'np.exp(input.astype(float64)).astype(float32)',
    'tolerance': 'unchanged activation exp threshold 2e-6/2e-7',
    'runner_note': 'one output has 4097 elements, so a subsequent authorized runner must request max_return>=4097'})

assert len(SUMMARY) == 15
compiler_hashes = {str(path.relative_to(REPO)): sha(path) for path in sorted((REPO / 'emmy_ptx').glob('*.py'))}
summary = {'compiler_commit': COMMIT, 'compiler_files': compiler_hashes, 'complete': True,
           'case_count': len(SUMMARY), 'pass_count': sum(x['cpu_result'] == 'PASS' for x in SUMMARY),
           'fail_count': sum(x['cpu_result'] == 'FAIL' for x in SUMMARY),
           'equal_nan': False, 'gpu_executed': False, 'ptx_assembled': False, 'cases': SUMMARY}
write_json(ROOT / 'summary.json', summary)
write_json(ROOT / 'manifest.json', {'files': ALL_HASHES})
print('COMPLETE', summary['pass_count'], 'PASS', summary['fail_count'], 'FAIL', flush=True)
raise SystemExit(1 if summary['fail_count'] else 0)
