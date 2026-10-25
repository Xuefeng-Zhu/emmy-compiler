"""Materialize and CPU-model one finite shared-exp boundary fixture.

No network request, GPU call or third-party generator is executed here.
"""
from dataclasses import replace
import argparse
import sys
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project), str(project / 'tests')]

import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.ir.stmt.leaves import Load
from emmy_ptx import compile_file
from structured_model import run_structured_ptx
from test_numeric import scalar_fixture
from tools.run_synthetic import load_case
from test_exp_emitted import f32_neighbors, exp_reference


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


folder = options.output.resolve()
folder.mkdir(parents=True, exist_ok=False)
smallest = np.nextafter(np.float32(0), np.float32(1))
thresholds = [math.log(float(np.finfo(np.float32).tiny)), math.log(float(smallest)),
              math.log(float(smallest) / 2)]
values = np.concatenate((
    np.array([-0.0, 0.0, -smallest, smallest, -np.finfo(np.float32).tiny,
              np.finfo(np.float32).tiny, -1e-30, 1e-30, 44.609375, 86.0, 87.0], np.float32),
    *(f32_neighbors(x, 32) for x in thresholds),
    *(f32_neighbors((n + 0.5) * math.log(2), 1) for n in range(-150, 121)),
))
assert np.isfinite(values).all() and values.max() <= 87
expected = exp_reference(values)
assert np.isfinite(expected).all()
# The existing runner may compute f32 output summaries. Avoid a diagnostic
# sum overflow as well as avoiding all nonfinite inputs/outputs themselves.
assert np.isfinite(np.sum(expected, dtype=np.float32))

raw, meta = scalar_fixture(len(values), [('exp', None)])
kernel, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
tile = Graph.from_dict(raw).nodes['add'].op.body[0]
tile = replace(tile, body=tuple(statement for statement in tile.body
                               if not (isinstance(statement, Load) and statement.input == 'x1')))
kernel['op_fields']['body'] = [repr(tile)]
kernel['inputs'].remove('x1')
raw['inputs'].remove('x1')
del raw['nodes']['x1']
launch, = meta['kernels']
launch['arg_order'].remove('x1')
launch['args'] = [arg for arg in launch['args'] if arg['name'] != 'x1']
entry = 's16_exp_boundary_finite'
kernel['op_fields']['name'] = launch['entry'] = entry
descriptor, = (arg for arg in launch['args'] if arg['name'] == 'x0')
descriptor.pop('seed', None)
descriptor['data'] = values.tolist()
meta.update(id=entry, rtol=2e-6, atol=2e-7, graph_inputs=list(raw['inputs']), graph_outputs=list(raw['outputs']))
meta.pop('code', None)
output, = (arg['name'] for arg in launch['args'] if arg.get('output'))
ir_path = folder / (entry + '.ir.json')
json_write(ir_path, raw)
json_write(folder / 'meta.json', meta)
buffer = io.BytesIO()
np.lib.format.write_array(buffer, expected, allow_pickle=False)
information = zipfile.ZipInfo(output + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
information.compress_type = zipfile.ZIP_STORED
with zipfile.ZipFile(folder / 'expected.npz', 'w') as archive:
    archive.writestr(information, buffer.getvalue())
manifest = {'files': {name: sha(folder / name) for name in (ir_path.name, 'meta.json', 'expected.npz')}}
json_write(folder / 'manifest.json', manifest)
load_case(folder, folder / 'manifest.json')
ptx = compile_file(ir_path)
(folder / 'generated.ptx').write_text(ptx)
result = run_structured_ptx(ptx, launch)
actual = result.outputs[output]
np.testing.assert_allclose(actual, expected, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
np.testing.assert_array_equal(actual == 0, expected == 0)
assert np.isfinite(actual).all() and np.all(result.store_counts[output] == 1)
np.save(folder / 'cpu-actual.npy', actual, allow_pickle=False)
error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
nonzero = expected != 0
record = {'case': entry, 'count': values.size, 'entry': entry, 'output': output,
          'grid': launch['grid'], 'block': launch['block'], 'rtol': meta['rtol'], 'atol': meta['atol'],
          'equal_nan': False, 'cpu_result': 'PASS', 'mismatched_values': 0,
          'input_min': float(values.min()), 'input_max': float(values.max()),
          'actual_nonfinite': int(np.count_nonzero(~np.isfinite(actual))),
          'expected_nonfinite': int(np.count_nonzero(~np.isfinite(expected))),
          'zero_classification_matches': bool(np.array_equal(actual == 0, expected == 0)),
          'expected_sum_f32_finite': bool(np.isfinite(np.sum(expected, dtype=np.float32))),
          'max_abs_error': float(error.max()),
          'max_relative_error_nonzero': float((error[nonzero] / expected[nonzero].astype(np.float64)).max()),
          'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
          'gpu_executed': False, 'ptx_assembled': False,
          'provenance': 'Typed scalar_fixture, finite neighboring f32 boundary inputs; independent float64 exp rounded to f32.'}
json_write(folder / 'cpu-result.json', record)
print(json.dumps(record, sort_keys=True))
