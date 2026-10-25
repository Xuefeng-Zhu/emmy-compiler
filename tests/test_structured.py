"""CPU-only tests of emitted structured PTX, including official attention data.

The model follows PTX operand flow, not the IR or a reconstructed formula.
NumPy models exp2 mathematically; NVIDIA approximation/assembly need GPU tests.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write

from emmy_ptx import CompileError, compile_file
from emmy_ptx.loader import load_kernel
from emmy_ptx.structured import StructuredEmitter
from ptx_model import input_buffers
from structured_model import run_structured_ptx

ROOT = Path(__file__).resolve().parents[1]


def official_fixture(problem):
    folder = ROOT / '.reference/problems' / problem
    paths = sorted(folder.glob('*.ir.json'))
    if len(paths) != 1:
        raise AssertionError('prepare official fixture: ' + problem)
    return json.loads(paths[0].read_text()), json.loads((folder / 'meta.json').read_text())


def compile_fixture(ir, meta, *, through_dispatch=False):
    with TemporaryDirectory(prefix='emmy-structured-test-') as tmp:
        path = Path(tmp) / 'arbitrary_program_name.ir.json'
        path.write_text(json.dumps(ir))
        path.with_name('meta.json').write_text(json.dumps(meta))
        if through_dispatch:
            return compile_file(path)
        return StructuredEmitter(load_kernel(path)).run()


def synthetic(tile, inputs, output_shape, *, output='result', entry='independent_program', grid=1, block=32):
    """Serialize actual typed statements into the official Graph JSON contract."""
    nodes = {name: {'op': 'InputOp', 'op_fields': {}, 'inputs': [],
                    'outputs': [{'name': name, 'shape': list(shape), 'dtype': 'f32'}]}
             for name, shape in inputs.items()}
    nodes[output] = {'op': 'KernelOp',
                     'op_fields': {'name': entry, 'body': [repr(tile)], 'zero_delegated': [], 'serial': []},
                     'inputs': list(inputs),
                     'outputs': [{'name': output, 'shape': list(output_shape), 'dtype': 'f32'}]}
    ir = {'inputs': list(inputs), 'outputs': [output], 'nodes': nodes}
    args = [{'name': name, 'shape': list(shape), 'dtype': 'f32', 'seed': seed}
            for seed, (name, shape) in enumerate(inputs.items())]
    args.append({'name': output, 'shape': list(output_shape or (1,)), 'dtype': 'f32', 'output': True})
    meta = {'rtol': 1e-6, 'atol': 1e-6, 'kernels': [{'node': output, 'entry': entry,
             'arg_order': [a['name'] for a in args], 'args': args,
             'grid': [grid, 1, 1], 'block': [block, 1, 1], 'smem_bytes': 0, 'zero_outputs': []}]}
    return ir, meta


def axis(name, extent):
    return Axis(name, Dim(extent))


def add(name, value, loop_axis, *, base=None, operation='add'):
    return Accum(name, value, ElementwiseImpl(operation), F32, (loop_axis,), base)


def changed_attention(rows=3, keys=5, depth=7, columns=4):
    """Reshape the genuine a02 typed schedule, preserving its operation graph."""
    ir, meta = official_fixture('a02_attention_128')
    kernel_node = next(n for n in ir['nodes'].values() if n['op'] == 'KernelOp')
    typed = Graph.from_dict(copy.deepcopy(ir))
    tile = next(n.op.body[0] for n in typed.nodes.values() if type(n.op).__name__ == 'KernelOp')

    def resize(statement):
        if isinstance(statement, Loop):
            extent = keys if statement.axis.name == 'a1' else depth
            return replace(statement, axis=replace(statement.axis, extent=Dim(extent)),
                           body=tuple(resize(s) for s in statement.body))
        return statement

    tile = replace(tile, axes=tuple(replace(a, extent=Dim({'a0_b': rows, 'a5_u': columns}.get(a.name, a.extent.as_static())))
                                    for a in tile.axes), body=tuple(resize(s) for s in tile.body))
    kernel_node['op_fields']['body'] = [repr(tile)]
    kernel_node['op_fields']['name'] = 'reshaped_and_renamed_attention'
    shapes = {'div_c1': [1], 'q': [rows, depth], 'kt': [depth, keys], 'v': [keys, columns],
              'matmul_1': [rows, columns]}
    for node in ir['nodes'].values():
        for tensor in node['outputs']:
            tensor['shape'] = shapes[tensor['name']]
    launch = meta['kernels'][0]
    launch['entry'] = kernel_node['op_fields']['name']
    launch['grid'] = [(rows * columns + 31) // 32 + 1, 1, 1]
    for arg in launch['args']:
        arg['shape'] = shapes[arg['name']]
    return ir, meta


def attention_reference(inputs):
    """Independent float64 formula in official expressions_open.py, then f32."""
    scores = (inputs['q'].astype(np.float64) @ inputs['kt'].astype(np.float64)) / float(inputs['div_c1'][0])
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
    return (weights @ inputs['v'].astype(np.float64)).astype(np.float32)


class StructuredTests(unittest.TestCase):
    def assert_outputs(self, result, expected, meta):
        for name, value in expected.items():
            np.testing.assert_allclose(result.outputs[name], value, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False)
            self.assertTrue(np.all(result.store_counts[name] == 1))

    def test_official_attention_fixtures_against_unchanged_expected_arrays(self):
        for problem in ('a01_attention', 'a02_attention_128'):
            with self.subTest(problem=problem):
                ir, meta = official_fixture(problem)
                ptx = compile_fixture(ir, meta, through_dispatch=True)
                result = run_structured_ptx(ptx, meta['kernels'][0])
                with np.load(ROOT / '.reference/problems' / problem / 'expected.npz', allow_pickle=False) as expected:
                    self.assertEqual(set(result.outputs), set(expected.files))
                    self.assert_outputs(result, {name: expected[name] for name in expected.files}, meta)

    def test_reshaped_attention_new_seeds_entry_and_parameter_order(self):
        for shape in ((1, 1, 1, 1), (3, 5, 7, 4), (5, 3, 2, 7)):
            with self.subTest(shape=shape):
                ir, meta = changed_attention(*shape)
                launch = meta['kernels'][0]
                for arg in launch['args']:
                    if 'seed' in arg: arg['seed'] += 73
                launch['args'] = list(reversed(launch['args']))
                launch['arg_order'] = [arg['name'] for arg in launch['args']]
                ptx = compile_fixture(ir, meta, through_dispatch=True)
                self.assertIn('.entry reshaped_and_renamed_attention', ptx)
                inputs = input_buffers(launch)
                result = run_structured_ptx(ptx, launch)
                self.assert_outputs(result, {'matmul_1': attention_reference(inputs)}, meta)

    def test_attention_runtime_scale_and_numerically_stable_inputs(self):
        ir, meta = changed_attention(3, 5, 7, 4)
        launch = meta['kernels'][0]
        ptx = compile_fixture(ir, meta)
        for scale, magnitude in ((2.25, 0.0), (0.125, 8.0), (19.0, 1.0)):
            with self.subTest(scale=scale, magnitude=magnitude):
                inputs = input_buffers(launch)
                inputs['div_c1'][:] = scale
                inputs['q'] *= magnitude
                inputs['kt'] *= magnitude
                result = run_structured_ptx(ptx, launch, inputs)
                self.assert_outputs(result, {'matmul_1': attention_reference(inputs)}, meta)

    def test_nested_accumulator_resets_inner_and_preserves_outer(self):
        inner = Loop(axis('k', 4), (
            Load('element', 'x', (Var('row'), BinaryExpr('+', BinaryExpr('*', Var('j'), Literal(4, 'int')), Var('k'))), dtype=F32),
            add('dot', 'element', 'k')))
        outer = Loop(axis('j', 5), (inner, Assign('square', ElementwiseImpl('multiply'), ('dot', 'dot'), F32),
                                    add('total', 'square', 'j')))
        tile = Tile((axis('row', 3),), (outer, Write('result', (Var('row'),), 'total', value_dtype=F32)))
        ir, meta = synthetic(tile, {'x': (3, 20)}, (3,))
        inputs = input_buffers(meta['kernels'][0])
        sums = inputs['x'].reshape(3, 5, 4).sum(axis=2, dtype=np.float32)
        expected = np.multiply(sums, sums, dtype=np.float32).sum(axis=1, dtype=np.float32)
        self.assert_outputs(run_structured_ptx(compile_fixture(ir, meta), meta['kernels'][0]), {'result': expected}, meta)

    def test_accum_base_uses_updated_prior_state(self):
        body = (Let('two', Literal(2.0), F32), Loop(axis('k', 3), (
            Load('element', 'x', (Var('row'), Var('k')), dtype=F32),
            Assign('doubled', ElementwiseImpl('multiply'), ('state', 'two'), F32),
            add('state', 'element', 'k', base='doubled'))),
            Write('result', (Var('row'),), 'state', value_dtype=F32))
        ir, meta = synthetic(Tile((axis('row', 2),), body), {'x': (2, 3)}, (2,))
        result = run_structured_ptx(compile_fixture(ir, meta), meta['kernels'][0], {'x': [[1, 2, 3], [-1, -2, -3]]})
        self.assert_outputs(result, {'result': np.array([11, -11], np.float32)}, meta)

    def test_seed_false_preserves_explicit_initial_value(self):
        loop = Loop(axis('k', 3), (Load('element', 'x', (Var('row'), Var('k')), dtype=F32), add('state', 'element', 'k')), seed=False)
        tile = Tile((axis('row', 2),), (Init('state', 5.0, dtype=F32), loop,
                                      Write('result', (Var('row'),), 'state', value_dtype=F32)))
        ir, meta = synthetic(tile, {'x': (2, 3)}, (2,))
        result = run_structured_ptx(compile_fixture(ir, meta), meta['kernels'][0], {'x': [[1, 2, 3], [-1, -2, -3]]})
        self.assert_outputs(result, {'result': np.array([11, -1], np.float32)}, meta)

    def test_maximum_reduction_all_negative_does_not_seed_zero(self):
        loop = Loop(axis('k', 3), (Load('element', 'x', (Var('row'), Var('k')), dtype=F32),
                                  add('maximum', 'element', 'k', operation='maximum')))
        tile = Tile((axis('row', 2),), (loop, Write('result', (Var('row'),), 'maximum', value_dtype=F32)))
        ir, meta = synthetic(tile, {'x': (2, 3)}, (2,))
        result = run_structured_ptx(compile_fixture(ir, meta), meta['kernels'][0], {'x': [[-3, -7, -2], [-10, -4, -9]]})
        self.assert_outputs(result, {'result': np.array([-2, -4], np.float32)}, meta)

    def test_ragged_raster_group_is_bijective_with_expected_thread_mapping(self):
        for orientation in ('m', 'n'):
            for group in (1, 2, 8):
                with self.subTest(orientation=orientation, group=group):
                    tile = Tile((axis('row', 3), axis('column', 2)), (
                        Load('element', 'x', (Var('row'), Var('column')), dtype=F32),
                        Write('result', (Var('row'), Var('column')), 'element', value_dtype=F32)),
                        raster_axes=('row', 'column'), raster_group=group, raster_orient=orientation)
                    ir, meta = synthetic(tile, {'x': (3, 2)}, (3, 2))
                    result = run_structured_ptx(compile_fixture(ir, meta), meta['kernels'][0])
                    self.assert_outputs(result, {'result': input_buffers(meta['kernels'][0])['x']}, meta)
                    expected = np.empty((3, 2), np.int64)
                    coordinates = []
                    # Emmy disables raster remapping when the effective group
                    # is one, including orientation='n'; native order remains.
                    if group == 1:
                        coordinates = [(r, c) for r in range(3) for c in range(2)]
                    elif orientation == 'm':
                        for start in range(0, 3, group):
                            coordinates.extend((r, c) for c in range(2) for r in range(start, min(start + group, 3)))
                    else:
                        for start in range(0, 2, group):
                            coordinates.extend((r, c) for r in range(3) for c in range(start, min(start + group, 2)))
                    for thread, coordinate in enumerate(coordinates): expected[coordinate] = thread
                    np.testing.assert_array_equal(result.writer_threads['result'].reshape(3, 2), expected)

    def test_vector_load_cannot_cross_a_row_boundary(self):
        tile = Tile((axis('row', 1),), (
            Load(names=('left', 'right'), input='x', index=(Literal(0, 'int'), Literal(2, 'int')), dtype=F32),
            Write('result', (Var('row'),), 'left', value_dtype=F32)))
        ir, meta = synthetic(tile, {'x': (2, 3)}, (1,))
        with self.assertRaisesRegex(CompileError, '(?i)(bound|dimension)'):
            compile_fixture(ir, meta)

    def test_output_coverage_and_thread_write_ownership_are_required(self):
        cases = ((1, (2,), Var('row')), (2, (1,), Literal(0, 'int')))
        for extent, output_shape, index in cases:
            with self.subTest(extent=extent, output_shape=output_shape):
                tile = Tile((axis('row', extent),), (
                    Load('element', 'x', (Var('row'),), dtype=F32),
                    Write('result', (index,), 'element', value_dtype=F32)))
                ir, meta = synthetic(tile, {'x': (2,)}, output_shape)
                with self.assertRaises(CompileError): compile_fixture(ir, meta)

    def test_loop_local_ssa_does_not_escape_and_unseeded_accum_requires_init(self):
        load = Load('element', 'x', (Var('row'), Var('k')), dtype=F32)
        bad_bodies = (
            (Loop(axis('k', 3), (load,)), Write('result', (Var('row'),), 'element', value_dtype=F32)),
            (Loop(axis('k', 3), (load, add('state', 'element', 'k')), seed=False),
             Write('result', (Var('row'),), 'state', value_dtype=F32)),
        )
        for body in bad_bodies:
            with self.subTest(body=repr(body)):
                ir, meta = synthetic(Tile((axis('row', 1),), body), {'x': (1, 3)}, (1,))
                with self.assertRaises(CompileError): compile_fixture(ir, meta)

    def test_malformed_assign_and_unimplemented_operation_fail_clearly(self):
        for operation, args in ((None, ('element',)), (ElementwiseImpl('exp'), None), (ElementwiseImpl('log'), ('element',))):
            with self.subTest(operation=operation, args=args):
                tile = Tile((axis('row', 1),), (Load('element', 'x', (Var('row'),), dtype=F32),
                    Assign('value', operation, args, F32), Write('result', (Var('row'),), 'value', value_dtype=F32)))
                ir, meta = synthetic(tile, {'x': (1,)}, (1,))
                with self.assertRaises(CompileError): compile_fixture(ir, meta)

    def test_model_rejects_unknown_instructions_and_missing_output_writes(self):
        tile = Tile((axis('row', 1),), (Load('element', 'x', (Var('row'),), dtype=F32),
                                      Write('result', (Var('row'),), 'element', value_dtype=F32)))
        ir, meta = synthetic(tile, {'x': (1,)}, (1,))
        ptx = compile_fixture(ir, meta)
        with self.assertRaisesRegex(AssertionError, 'unsupported modeled instruction'):
            run_structured_ptx(ptx.replace('ld.global.f32', 'unsupported.f32'), meta['kernels'][0])
        without_store = '\n'.join(line for line in ptx.splitlines() if 'st.global.f32' not in line)
        with self.assertRaisesRegex(AssertionError, 'written exactly once'):
            run_structured_ptx(without_store, meta['kernels'][0])


if __name__ == '__main__':
    unittest.main()
