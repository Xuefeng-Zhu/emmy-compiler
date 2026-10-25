"""Independent CPU-model checks; these are not GPU or PTX assembly tests.

Small hand-authored instruction streams exercise model safety boundaries without
reusing the compiler's reduction algorithm. Division cases describe the model's
explicit f32 arithmetic, not support for arbitrary exceptional compiler inputs.
"""
import unittest

import numpy as np

from reduction_model import run_reduction_ptx


def model_kernel(body, arguments=('values', 'result')):
    parameters = ',\n'.join(f'    .param .u64 {name}' for name in arguments)
    return f'''.version 8.0
.target sm_89
.address_size 64
.visible .entry model_probe(
{parameters}
)
.reqntid 32, 1, 1
{{
    .reg .pred %p<1>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;
    .shared .align 4 .b8 scratch[4];
{body}
}}
'''


def shared_probe():
    """Each CTA's selected writer broadcasts one input through shared memory."""
    return model_kernel('''    ld.param.u64 %rd0, [values];
    ld.param.u64 %rd1, [result];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, scratch;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    setp.ne.u32 %p0, %r0, %r1;
    @%p0 bra PUBLISHED;
    ld.global.f32 %f0, [%rd3];
    st.shared.f32 [%r2], %f0;
PUBLISHED:
    bar.sync 0;
    ld.shared.f32 %f0, [%r2];
    setp.ne.u32 %p0, %r0, 0;
    @%p0 ret;
    st.global.f32 [%rd4], %f0;
    ret;''')


def shared_launch(count=3):
    return {'grid': [count, 1, 1], 'block': [32, 1, 1], 'args': [
        {'name': 'values', 'shape': [count], 'data': list(range(1, count + 1))},
        {'name': 'result', 'shape': [count], 'output': True},
    ]}


def divide_probe():
    return model_kernel('''    ld.param.u64 %rd0, [numerator];
    ld.param.u64 %rd1, [denominator];
    ld.param.u64 %rd2, [result];
    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 ret;
    mov.u32 %r1, %ctaid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    div.rn.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
    ret;''', ('numerator', 'denominator', 'result'))


def binary_launch(left, right):
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 1:
        raise AssertionError('probe operands must have matching one-dimensional shapes')
    return {'grid': [left.size, 1, 1], 'block': [32, 1, 1], 'args': [
        {'name': 'numerator', 'shape': [left.size], 'data': left.tolist()},
        {'name': 'denominator', 'shape': [left.size], 'data': right.tolist()},
        {'name': 'result', 'shape': [left.size], 'output': True},
    ]}


class CooperativeModelTests(unittest.TestCase):
    def test_ctas_have_private_shared_state_and_aggregate_counts(self):
        launch = shared_launch()
        values = np.array([3, -7, 11], dtype=np.float32)
        result = run_reduction_ptx(shared_probe(), launch, {'values': values})
        np.testing.assert_array_equal(result.outputs['result'], values)
        np.testing.assert_array_equal(result.load_counts['values'], [1, 1, 1])
        np.testing.assert_array_equal(result.store_counts['result'], [1, 1, 1])
        self.assertEqual(result.barrier_count, 3)

    def test_previous_cta_cannot_initialize_shared_memory_for_next_cta(self):
        ptx = shared_probe().replace(
            '    setp.ne.u32 %p0, %r0, %r1;',
            '    setp.eq.u32 %p0, %r1, 1;\n'
            '    @%p0 bra PUBLISHED;\n'
            '    setp.ne.u32 %p0, %r0, %r1;', 1)
        with self.assertRaisesRegex(AssertionError, 'uninitialized shared memory'):
            run_reduction_ptx(ptx, shared_launch())

    def test_registers_do_not_leak_between_ctas(self):
        ptx = shared_probe().replace(
            '    ld.shared.f32 %f0, [%r2];',
            '    setp.ne.u32 %p0, %r1, 0;\n'
            '    @!%p0 ld.shared.f32 %f0, [%r2];', 1)
        with self.assertRaisesRegex(AssertionError, 'uninitialized modeled register'):
            run_reduction_ptx(ptx, shared_launch())

    def test_next_cta_still_requires_complete_barrier_participation(self):
        ptx = shared_probe().replace(
            'PUBLISHED:\n',
            'PUBLISHED:\n'
            '    setp.ne.u32 %p0, %r1, 1;\n'
            '    @%p0 bra WAIT;\n'
            '    setp.eq.u32 %p0, %r0, 0;\n'
            '    @%p0 ret;\n'
            'WAIT:\n', 1)
        with self.assertRaisesRegex(AssertionError, 'deadlock'):
            run_reduction_ptx(ptx, shared_launch())

    def test_ctas_cannot_alias_an_output_element(self):
        ptx = shared_probe().replace('add.u64 %rd4, %rd1, %rd2;',
                                     'mov.u64 %rd4, %rd1;', 1)
        with self.assertRaisesRegex(AssertionError, 'multiple threads or CTAs'):
            run_reduction_ptx(ptx, shared_launch())

    def test_threads_cannot_alias_an_output_element(self):
        ptx = shared_probe().replace('    @%p0 ret;\n', '', 1)
        with self.assertRaisesRegex(AssertionError, 'multiple threads or CTAs'):
            run_reduction_ptx(ptx, shared_launch(1))

    def test_each_output_requires_exactly_one_store(self):
        store = '    st.global.f32 [%rd4], %f0;\n'
        for replacement in ('', store * 2):
            with self.subTest(replacement=replacement):
                ptx = shared_probe().replace(store, replacement, 1)
                with self.assertRaisesRegex(AssertionError, 'must be written exactly once'):
                    run_reduction_ptx(ptx, shared_launch())

    def test_runtime_division_rounds_to_known_f32_bit_patterns(self):
        numerator = np.array([1.0, -0.0, 2.0 ** -126, 1.0], dtype=np.float32)
        denominator = np.array([3.0, 2.0, 2.0, -3.0], dtype=np.float32)
        launch = {'grid': [4, 1, 1], 'block': [32, 1, 1], 'args': [
            {'name': 'numerator', 'shape': [4], 'data': numerator.tolist()},
            {'name': 'denominator', 'shape': [4], 'data': denominator.tolist()},
            {'name': 'result', 'shape': [4], 'output': True},
        ]}
        ptx = divide_probe()
        result = run_reduction_ptx(ptx, launch)
        np.testing.assert_array_equal(result.outputs['result'].view(np.uint32),
                                      [0x3eaaaaab, 0x80000000, 0x00400000, 0xbeaaaaab])
        for name in ('numerator', 'denominator'):
            np.testing.assert_array_equal(result.load_counts[name], [1, 1, 1, 1])
        np.testing.assert_array_equal(result.store_counts['result'], [1, 1, 1, 1])
        self.assertEqual(result.barrier_count, 0)
        # Keep the instruction stream fixed while changing runtime operands.
        changed = run_reduction_ptx(ptx, launch, {'denominator': np.ones(4, dtype=np.float32)})
        np.testing.assert_array_equal(changed.outputs['result'].view(np.uint32),
                                      numerator.view(np.uint32))

    def test_unmodeled_division_mode_fails_instead_of_approximating(self):
        ptx = divide_probe().replace('div.rn.f32', 'div.approx.f32', 1)
        launch = {'grid': [1, 1, 1], 'block': [32, 1, 1], 'args': [
            {'name': 'numerator', 'shape': [1], 'data': [1]},
            {'name': 'denominator', 'shape': [1], 'data': [3]},
            {'name': 'result', 'shape': [1], 'output': True},
        ]}
        with self.assertRaisesRegex(AssertionError, 'unsupported modeled instruction'):
            run_reduction_ptx(ptx, launch)

    def test_subtract_and_multiply_use_f32_rounding(self):
        cases = (
            ('sub.rn.f32', [2**24, 1.5], [-1, 2], [0x4b800000, 0xbf000000]),
            ('mul.rn.f32', [1.5, -(2**-126)], [2, 0.5], [0x40400000, 0x80400000]),
        )
        for opcode, left, right, expected in cases:
            with self.subTest(opcode=opcode):
                ptx = divide_probe().replace('div.rn.f32', opcode, 1)
                actual = run_reduction_ptx(ptx, binary_launch(left, right)).outputs['result']
                np.testing.assert_array_equal(actual.view(np.uint32), expected)

    def test_nan_min_max_and_signed_zeros_follow_modeled_ptx_rules(self):
        left = [float('nan'), 2, -0.0, 0.0, -0.0]
        right = [1, float('nan'), 0.0, -0.0, -0.0]
        for opcode, expected_zero_bits in (
                ('max.NaN.f32', [0, 0, 0x80000000]),
                ('min.NaN.f32', [0x80000000, 0x80000000, 0x80000000])):
            with self.subTest(opcode=opcode):
                ptx = divide_probe().replace('div.rn.f32', opcode, 1)
                actual = run_reduction_ptx(ptx, binary_launch(left, right)).outputs['result']
                self.assertTrue(np.all(np.isnan(actual[:2])))
                np.testing.assert_array_equal(actual[2:].view(np.uint32), expected_zero_bits)

    def test_sqrt_and_mathematical_exp2_model_on_exact_values(self):
        cases = (
            ('sqrt.rn.f32', [4, 1, 0, -0.0], [0x40000000, 0x3f800000, 0, 0x80000000]),
            ('ex2.approx.f32', [-2, -1, 0, 1], [0x3e800000, 0x3f000000, 0x3f800000, 0x40000000]),
        )
        for opcode, values, expected in cases:
            with self.subTest(opcode=opcode):
                ptx = divide_probe().replace('div.rn.f32 %f2, %f0, %f1;', f'{opcode} %f2, %f0;', 1)
                actual = run_reduction_ptx(ptx, binary_launch(values, [1] * 4)).outputs['result']
                np.testing.assert_array_equal(actual.view(np.uint32), expected)

    def test_signed_integer_division_remainder_and_wrapping(self):
        cases = (
            ('mov.s32 %r2, -7;\n    div.s32 %r3, %r2, 3;', 0xfffffffe),
            ('mov.s32 %r2, -7;\n    rem.s32 %r3, %r2, 3;', 0xffffffff),
            ('sub.s32 %r2, 0, 3;\n    mul.lo.s32 %r3, %r2, -5;', 15),
            ('mov.s32 %r2, 2147483647;\n    add.s32 %r3, %r2, 1;', 0x80000000),
            ('mov.u32 %r2, 4294967295;\n    div.u32 %r3, %r2, 2;', 0x7fffffff),
            ('mov.u32 %r2, 5;\n    mad.lo.u32 %r3, %r2, 3, 2;', 17),
        )
        for instructions, expected_bits in cases:
            with self.subTest(instructions=instructions):
                ptx = divide_probe().replace('div.rn.f32 %f2, %f0, %f1;',
                                             instructions + '\n    mov.b32 %f2, %r3;', 1)
                actual = run_reduction_ptx(ptx, binary_launch([1], [1])).outputs['result']
                self.assertEqual(int(actual.view(np.uint32)[0]), expected_bits)

    def test_signed_comparisons_and_predicate_selection(self):
        instructions = '''mov.s32 %r2, -1;
    setp.lt.s32 %p0, %r2, 0;
    and.pred %p0, %p0, 1;
    not.pred %p0, %p0;
    xor.pred %p0, %p0, 1;
    selp.s32 %r3, -2, 1, %p0;
    mov.b32 %f2, %r3;'''
        ptx = divide_probe().replace('div.rn.f32 %f2, %f0, %f1;', instructions, 1)
        actual = run_reduction_ptx(ptx, binary_launch([1], [1])).outputs['result']
        self.assertEqual(int(actual.view(np.uint32)[0]), 0xfffffffe)

    def test_undefined_integer_division_is_rejected(self):
        for operation in ('div.s32 %r3, 1, 0;', 'rem.u32 %r3, 1, 0;',
                          'div.s32 %r3, -2147483648, -1;'):
            with self.subTest(operation=operation):
                ptx = divide_probe().replace('div.rn.f32 %f2, %f0, %f1;', operation, 1)
                with self.assertRaisesRegex(AssertionError, 'undefined modeled integer division'):
                    run_reduction_ptx(ptx, binary_launch([1], [1]))

    def test_invalid_or_mismatched_launch_is_rejected(self):
        for grid, block in (([0, 1, 1], [32, 1, 1]),
                            ([-1, 1, 1], [32, 1, 1]),
                            ([True, 1, 1], [32, 1, 1]),
                            ([1, 2, 1], [32, 1, 1]),
                            ([1, 1, 1], [32, 2, 1]),
                            ([1, 1, 1], [16, 1, 1]),
                            ([1, 1, 1], [64, 1, 1])):
            with self.subTest(grid=grid, block=block):
                launch = shared_launch(1)
                launch.update(grid=grid, block=block)
                with self.assertRaises(AssertionError):
                    run_reduction_ptx(shared_probe(), launch)


if __name__ == '__main__':
    unittest.main()
