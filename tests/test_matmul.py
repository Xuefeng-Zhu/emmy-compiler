"""Static and validation tests for shared-memory contraction PTX; no GPU calls."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Init, Load, Write

from emmy_ptx.errors import CompileError
from emmy_ptx.loader import load_kernel
from emmy_ptx.matmul import MatmulEmitter

ROOT = Path(__file__).resolve().parents[1]


def fixture(problem):
    path, = (ROOT / '.reference/problems' / problem).glob('*.ir.json')
    return json.loads(path.read_text()), json.loads(path.with_name('meta.json').read_text())


def tile_from(raw):
    node, = (node for node in Graph.from_dict(raw).nodes.values() if type(node.op).__name__ == 'KernelOp')
    return node.op.body[0]


def with_tile(raw, tile):
    node, = (node for node in raw['nodes'].values() if node['op'] == 'KernelOp')
    node['op_fields']['body'] = [repr(tile)]


def emit(raw, meta):
    with TemporaryDirectory(prefix='emmy-matmul-') as tmp:
        source = Path(tmp) / 'kernel.ir.json'
        source.write_text(json.dumps(raw))
        source.with_name('meta.json').write_text(json.dumps(meta))
        return MatmulEmitter(load_kernel(source)).run()


def change_first(tile, kind, update):
    """Replace one typed statement without matching an upstream node name."""
    found = False
    def visit(body):
        nonlocal found
        result = []
        for stmt in body:
            if not found and isinstance(stmt, kind):
                stmt, found = update(stmt), True
            elif not found and isinstance(stmt, (Loop, StridedLoop)):
                stmt = replace(stmt, body=tuple(visit(stmt.body)))
            result.append(stmt)
        return tuple(result)
    result = replace(tile, body=visit(tile.body))
    if not found:
        raise AssertionError(f'fixture lacks {kind.__name__}')
    return result


class MatmulTests(unittest.TestCase):
    def test_official_pipeline_uses_one_dynamic_pool_and_ir_group_order(self):
        raw, meta = fixture('m01_matmul_512')
        ptx = emit(raw, meta)
        self.assertIn('.extern .shared .align 16 .b8 shared_pool[];', ptx)
        self.assertLess(ptx.index('.extern .shared'), ptx.index('.visible .entry'))
        self.assertNotIn('.extern', ptx.split('{', 1)[1])
        self.assertEqual(ptx.count('.shared '), 1)
        self.assertIn(', 24576;', ptx)
        self.assertEqual(meta['kernels'][0]['smem_bytes'], 73728)
        self.assertEqual(ptx.count('cp.async.cg.shared.global'), 9)
        self.assertEqual(ptx.count('cp.async.commit_group;'), 3)
        self.assertEqual(ptx.count('cp.async.wait_group 2;'), 1)
        self.assertEqual(ptx.count('cp.async.wait_group 0;'), 1)
        self.assertLess(ptx.rindex('cp.async.wait_group 0;'), ptx.rindex('ret;'))
        self.assertEqual(ptx.count('bar.sync 0;'), 2)
        self.assertEqual(ptx.count('st.global.f32'), 4)
        self.assertEqual(ptx.count('mul.rn.f32'), 4)
        self.assertEqual(ptx.count('add.rn.f32'), 4)
        self.assertEqual(ptx.count('mov.f32'), 4)
        self.assertNotIn('fma.', ptx)
        self.assertNotIn('    red.global', ptx)
        self.assertIn('.reqntid 512, 1, 1', ptx)
        self.assertEqual(ptx, emit(raw, meta))

    def test_official_split_k_uses_static_shared_and_zeroed_atomic_output(self):
        raw, meta = fixture('m02_matmul_rect')
        ptx = emit(raw, meta)
        self.assertIn('.shared .align 16 .b8 shared_0[512];', ptx)
        self.assertIn('.shared .align 16 .b8 shared_1[1024];', ptx)
        self.assertNotIn('.extern .shared', ptx)
        self.assertEqual(ptx.count('cp.async.cg.shared.global'), 2)
        self.assertEqual(ptx.count('cp.async.wait_group 0;'), 1)
        self.assertEqual(ptx.count('red.global.add.f32'), 2)
        self.assertNotIn('st.global.f32', ptx)
        self.assertIn('.reqntid 256, 1, 1', ptx)
        meta['kernels'][0]['zero_outputs'] = []
        with self.assertRaisesRegex(CompileError, 'zero_outputs'):
            emit(raw, meta)

    def test_entry_buffer_axis_and_ssa_names_do_not_select_lowering(self):
        raw, meta = fixture('m02_matmul_rect')
        names = {'x0': 'left_matrix', 'x1': 'right_matrix', 'matmul': 'result',
                 'k_matmul': 'renamed_contract', '_a_smem': 'left_stage',
                 '_b_smem': 'right_stage', '_a2_ks': 'split_index',
                 '_ks': 'stage_index', '_ki': 'reduced_index',
                 'a0_b': 'row_tile', 'a1_b': 'col_tile',
                 'a0_u': 'local_row', 'a1_u': 'local_col',
                 'acc0__c0_0': 'first_total', 'acc0__c1_0': 'second_total'}
        def rename(value):
            if isinstance(value, dict):
                return {names.get(key, key): rename(child) for key, child in value.items()}
            if isinstance(value, list):
                return [rename(child) for child in value]
            if isinstance(value, str):
                if value in names:
                    return names[value]
                for old, new in names.items():
                    value = value.replace(repr(old), repr(new))
            return value
        raw, meta = rename(raw), rename(meta)
        meta['kernels'][0]['arg_order'].reverse()
        meta['kernels'][0]['args'].reverse()
        ptx = emit(raw, meta)
        self.assertIn('.entry renamed_contract(', ptx)
        self.assertLess(ptx.index('param_result'), ptx.index('param_left_matrix'))
        self.assertEqual(ptx.count('red.global.add.f32'), 2)

    def test_static_rectangular_shapes_are_derived_from_ir(self):
        for split, row_tiles, col_tiles in ((1, 1, 1), (3, 2, 1), (5, 1, 3)):
            with self.subTest(split=split, row_tiles=row_tiles, col_tiles=col_tiles):
                raw, meta = fixture('m02_matmul_rect')
                tile = tile_from(raw)
                from emmy.compiler.dim import Dim
                axes = tuple(replace(axis, extent=Dim(extent)) for axis, extent in
                             zip(tile.axes, (split, row_tiles, col_tiles, 8, 32)))
                with_tile(raw, replace(tile, axes=axes))
                shapes = {'x0': [row_tiles * 16, split * 8],
                          'x1': [split * 8, col_tiles * 32],
                          'matmul': [row_tiles * 16, col_tiles * 32]}
                for node in raw['nodes'].values():
                    for tensor in node['outputs']:
                        tensor['shape'] = shapes[tensor['name']]
                for arg in meta['kernels'][0]['args']:
                    arg['shape'] = shapes[arg['name']]
                meta['kernels'][0]['grid'][0] = split * row_tiles * col_tiles
                self.assertEqual(emit(raw, meta).count('red.global.add.f32'), 2)

    def test_missing_wait_is_rejected_before_shared_barrier(self):
        raw, meta = fixture('m02_matmul_rect')
        tile = tile_from(raw)
        outer = next(stmt for stmt in tile.body if isinstance(stmt, StridedLoop))
        changed_outer = replace(outer, body=tuple(stmt for stmt in outer.body if not isinstance(stmt, CpAsyncWait)))
        with_tile(raw, replace(tile, body=tuple(changed_outer if stmt is outer else stmt for stmt in tile.body)))
        with self.assertRaisesRegex(CompileError, 'wait_group'):
            emit(raw, meta)

    def test_global_write_holes_duplicates_and_unequal_atomic_contributions_fail(self):
        for problem in ('m01_matmul_512', 'm02_matmul_rect'):
            for change in ('remove', 'duplicate', 'constant row'):
                with self.subTest(problem=problem, change=change):
                    raw, meta = fixture(problem)
                    tile = tile_from(raw)
                    body = list(tile.body)
                    index = next(i for i, stmt in enumerate(body) if isinstance(stmt, Write))
                    if change == 'remove':
                        body.pop(index)
                    elif change == 'duplicate':
                        body.append(body[index])
                    else:
                        body[index] = replace(body[index], index=(Literal(0, 'int'), body[index].index[1]))
                    with_tile(raw, replace(tile, body=tuple(body)))
                    with self.assertRaisesRegex(CompileError, 'cover'):
                        emit(raw, meta)

    def test_shared_reads_require_a_post_wait_barrier(self):
        raw, meta = fixture('m02_matmul_rect')
        tile = tile_from(raw)
        outer = next(stmt for stmt in tile.body if isinstance(stmt, StridedLoop))
        removed = False
        body = []
        for stmt in outer.body:
            if not removed and isinstance(stmt, Sync):
                removed = True
            else:
                body.append(stmt)
        changed_outer = replace(outer, body=tuple(body))
        with_tile(raw, replace(tile, body=tuple(changed_outer if stmt is outer else stmt for stmt in tile.body)))
        with self.assertRaisesRegex(CompileError, 'barrier'):
            emit(raw, meta)

    def test_dynamic_allocation_metadata_must_match_layout(self):
        for size in (0, 73724, 73732):
            with self.subTest(size=size):
                raw, meta = fixture('m01_matmul_512')
                meta['kernels'][0]['smem_bytes'] = size
                with self.assertRaises(CompileError):
                    emit(raw, meta)

    def test_copy_sizes_alignment_and_swizzles_are_validated(self):
        changes = ({'nbytes': 3}, {'nbytes': True}, {'swizzle': 'B32'},
                   {'lane_rows': 0}, {'src': 'unknown'}, {'smem': 'unknown'})
        for change in changes:
            with self.subTest(change=change):
                raw, meta = fixture('m01_matmul_512')
                with_tile(raw, change_first(tile_from(raw), CpAsyncCopy, lambda stmt: replace(stmt, **change)))
                with self.assertRaises(CompileError):
                    emit(raw, meta)
        raw, meta = fixture('m01_matmul_512')
        with_tile(raw, change_first(tile_from(raw), CpAsyncCopy,
                                   lambda stmt: replace(stmt, src_index=(stmt.src_index[0], BinaryExpr('+', stmt.src_index[1], Literal(1, 'int'))))))
        with self.assertRaisesRegex(CompileError, 'alignment'):
            emit(raw, meta)

    def test_proven_overlapping_pending_copy_destinations_are_rejected(self):
        for variant in ('identical', 'equivalent index', 'partial overlap', 'different source'):
            with self.subTest(variant=variant):
                raw, meta = fixture('m02_matmul_rect')
                tile = tile_from(raw)
                outer = next(stmt for stmt in tile.body if isinstance(stmt, StridedLoop))
                fill = next(stmt for stmt in outer.body if isinstance(stmt, StridedLoop))
                original, = fill.body
                duplicate = original
                if variant == 'equivalent index':
                    duplicate = replace(original, smem_index=tuple(BinaryExpr('+', index, Literal(0, 'int')) for index in original.smem_index))
                elif variant == 'partial overlap':
                    duplicate = replace(original, nbytes=4,
                                        smem_index=(original.smem_index[0], BinaryExpr('+', original.smem_index[1], Literal(1, 'int'))))
                elif variant == 'different source':
                    duplicate = replace(original, src_index=(original.src_index[0], Literal(0, 'int')))
                changed_fill = replace(fill, body=(original, duplicate))
                changed_outer = replace(outer, body=tuple(changed_fill if stmt is fill else stmt for stmt in outer.body))
                with_tile(raw, replace(tile, body=tuple(changed_outer if stmt is outer else stmt for stmt in tile.body)))
                with self.assertRaisesRegex(CompileError, 'overlapping async copy destinations'):
                    emit(raw, meta)

    def test_copy_overlap_check_includes_lane_coordinates_and_resets_after_commit(self):
        raw, meta = fixture('m01_matmul_512')
        tile = tile_from(raw)
        body = list(tile.body)
        index = next(i for i, stmt in enumerate(body) if isinstance(stmt, CpAsyncCopy))
        original = body[index]
        # The effective destination also includes the separately stored lane terms.
        equivalent = replace(original, lane_index=tuple(BinaryExpr('+', value, Literal(0, 'int')) for value in original.lane_index))
        body.insert(index + 1, equivalent)
        with_tile(raw, replace(tile, body=tuple(body)))
        with self.assertRaisesRegex(CompileError, 'overlapping async copy destinations'):
            emit(raw, meta)
        # Commit+wait explicitly completes the first copy before repeating it.
        raw, meta = fixture('m01_matmul_512')
        tile = tile_from(raw)
        body = list(tile.body)
        body[index + 1:index + 1] = [CpAsyncCommit(), CpAsyncWait(group=0), original]
        with_tile(raw, replace(tile, body=tuple(body)))
        self.assertEqual(emit(raw, meta).count('cp.async.cg.shared.global'), 10)

    def test_smaller_legal_copy_sizes_use_ca(self):
        for nbytes in (4, 8):
            with self.subTest(nbytes=nbytes):
                raw, meta = fixture('m01_matmul_512')
                with_tile(raw, change_first(tile_from(raw), CpAsyncCopy,
                                           lambda stmt: replace(stmt, nbytes=nbytes)))
                ptx = emit(raw, meta)
                self.assertIn('cp.async.ca.shared.global', ptx)
                self.assertIn(f'], {nbytes};', ptx)
                # This is an instruction-format test: shortening one copy does
                # not preserve the complete matrix kernel's initialized data.

    def test_shared_shape_dtype_and_alignment_are_validated(self):
        changes = ({'dtype': 'half'}, {'extents': (0, 64)}, {'align': 3}, {'align': 4}, {'extents': (1, 64)})
        for change in changes:
            with self.subTest(change=change):
                raw, meta = fixture('m02_matmul_rect')
                with_tile(raw, change_first(tile_from(raw), Smem, lambda stmt: replace(stmt, **change)))
                with self.assertRaises(CompileError):
                    emit(raw, meta)

    def test_wait_barrier_and_launch_contracts_are_validated(self):
        for group in (-1, 8, True):
            with self.subTest(group=group):
                raw, meta = fixture('m01_matmul_512')
                with_tile(raw, change_first(tile_from(raw), CpAsyncWait, lambda stmt: replace(stmt, group=group)))
                with self.assertRaises(CompileError):
                    emit(raw, meta)
        for change in ({'warp': True}, {'count': 128}, {'barrier_id': 1}, {'barrier_id': False}):
            with self.subTest(change=change):
                raw, meta = fixture('m02_matmul_rect')
                with_tile(raw, change_first(tile_from(raw), Sync, lambda stmt: replace(stmt, **change)))
                with self.assertRaises(CompileError):
                    emit(raw, meta)
        raw, meta = fixture('m02_matmul_rect')
        meta['kernels'][0]['grid'][0] += 1
        with self.assertRaisesRegex(CompileError, 'complete CTAs'):
            emit(raw, meta)

    def test_accumulator_init_and_inherited_shared_dtype_are_respected(self):
        raw, meta = fixture('m02_matmul_rect')
        tile = tile_from(raw)
        self.assertEqual(sum(isinstance(stmt, Init) for stmt in tile.body), 2)
        with_tile(raw, replace(tile, body=tuple(stmt for stmt in tile.body if not isinstance(stmt, Init))))
        with self.assertRaisesRegex(CompileError, 'previously initialized'):
            emit(raw, meta)
        raw, meta = fixture('m02_matmul_rect')
        with_tile(raw, change_first(tile_from(raw), Load, lambda stmt: replace(stmt, dtype='f16')))
        with self.assertRaises(CompileError):
            emit(raw, meta)

    def test_shared_use_before_declaration_and_divergent_barrier_rejected(self):
        raw, meta = fixture('m01_matmul_512')
        tile = tile_from(raw)
        body = list(tile.body)
        first_copy = next(i for i, stmt in enumerate(body) if isinstance(stmt, CpAsyncCopy))
        first_shared = next(i for i, stmt in enumerate(body) if isinstance(stmt, Smem))
        body[first_copy], body[first_shared] = body[first_shared], body[first_copy]
        with_tile(raw, replace(tile, body=tuple(body)))
        with self.assertRaisesRegex(CompileError, 'before declaration'):
            emit(raw, meta)
        raw, meta = fixture('m02_matmul_rect')
        tile = tile_from(raw)
        outer = next(stmt for stmt in tile.body if isinstance(stmt, StridedLoop))
        fill = next(stmt for stmt in outer.body if isinstance(stmt, StridedLoop))
        changed_fill = replace(fill, body=tuple(fill.body) + (Sync(),))
        changed_outer = replace(outer, body=tuple(changed_fill if stmt is fill else stmt for stmt in outer.body))
        with_tile(raw, replace(tile, body=tuple(changed_outer if stmt is outer else stmt for stmt in tile.body)))
        with self.assertRaisesRegex(CompileError, 'divergent'):
            emit(raw, meta)


if __name__ == '__main__':
    unittest.main()
