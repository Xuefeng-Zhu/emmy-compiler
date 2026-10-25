"""Hand-written PTX probes for integral-valued f32 conversion modes.

These exercise independent CPU models, not a PTX assembler or NVIDIA hardware.
Expected values below are explicit IEEE-754 cases, not NumPy round/trunc output.
NaN payload and sign are deliberately unspecified; only NaN-ness is checked.
"""
import unittest

import numpy as np

from structured_model import run_structured_ptx
from reduction_model import run_reduction_ptx


MODELS = (run_structured_ptx, run_reduction_ptx)


def probe(instructions):
    """One 32-thread CTA independently transforms one f32 per thread."""
    return f'''.version 8.0
.target sm_89
.address_size 64
.visible .entry float_rounding_probe(
    .param .u64 source,
    .param .u64 output
)
.reqntid 32, 1, 1
{{
    .reg .b32 %r<1>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [source];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd5, %rd0;
    cvta.to.global.u64 %rd6, %rd1;
    mov.u32 %r0, %tid.x;
    mul.wide.u32 %rd2, %r0, 4;
    add.u64 %rd3, %rd5, %rd2;
    add.u64 %rd4, %rd6, %rd2;
    ld.global.f32 %f0, [%rd3];
    {instructions}
    st.global.f32 [%rd4], %f1;
    ret;
}}
'''


def launch(values):
    values = np.asarray(values, np.float32)
    assert values.size <= 32
    padded = np.zeros(32, np.float32)
    padded[:values.size] = values
    return {'grid': [1, 1, 1], 'block': [32, 1, 1], 'args': [
        {'name': 'source', 'shape': [32], 'data': padded.tolist()},
        {'name': 'result', 'shape': [32], 'output': True},
    ]}


class FloatIntegralConversionModelTests(unittest.TestCase):
    def check(self, opcode, values, expected):
        expected = np.asarray(expected, np.float32)
        for model in MODELS:
            with self.subTest(model=model.__name__, opcode=opcode):
                result = model(probe(f'{opcode} %f1, %f0;'), launch(values))
                actual = result.outputs['result']
                self.assertEqual(actual.dtype, np.dtype(np.float32))
                np.testing.assert_array_equal(result.store_counts['result'], np.ones(32))
                np.testing.assert_array_equal(actual[:expected.size].view(np.uint32),
                                              expected.view(np.uint32))

    def test_nearest_ties_even_with_signed_inputs(self):
        self.check('cvt.rni.f32.f32',
                   [0., -0., .25, -.25, .5, -.5, 1.5, -1.5,
                    2.5, -2.5, 3.5, -3.5, 4.5, -4.5],
                   [0., -0., 0., -0., 0., -0., 2., -2.,
                    2., -2., 4., -4., 4., -4.])

    def test_nearest_adjacent_values_around_tie(self):
        low = np.nextafter(np.float32(.5), np.float32(0))
        high = np.nextafter(np.float32(.5), np.float32(1))
        low_odd = np.nextafter(np.float32(1.5), np.float32(1))
        high_odd = np.nextafter(np.float32(1.5), np.float32(2))
        self.check('cvt.rni.f32.f32',
                   [low, high, -low, -high, low_odd, high_odd, -low_odd, -high_odd],
                   [0., 1., -0., -1., 1., 2., -1., -2.])

    def test_truncation_toward_zero_with_signed_inputs(self):
        self.check('cvt.rzi.f32.f32',
                   [0., -0., .25, -.25, .5, -.5, 1.5, -1.5,
                    2.5, -2.5, 3.999, -3.999, 4., -4.],
                   [0., -0., 0., -0., 0., -0., 1., -1.,
                    2., -2., 3., -3., 4., -4.])

    def test_subnormal_inputs_round_to_signed_zero(self):
        tiny = np.array([1], np.uint32).view(np.float32)[0]
        for opcode in ('cvt.rni.f32.f32', 'cvt.rzi.f32.f32'):
            self.check(opcode, [tiny, -tiny, 2**-126, -(2**-126)],
                       [0., -0., 0., -0.])

    def test_large_integral_f32_values_are_preserved(self):
        largest = np.finfo(np.float32).max
        values = [2**23, -(2**23), 2**24, -(2**24), largest, -largest]
        for opcode in ('cvt.rni.f32.f32', 'cvt.rzi.f32.f32'):
            self.check(opcode, values, values)
        self.check('cvt.rni.f32.f32', [8388607.5, -8388607.5],
                   [8388608., -8388608.])
        self.check('cvt.rzi.f32.f32', [8388607.5, -8388607.5],
                   [8388607., -8388607.])

    def test_infinities_and_nans_remain_floating_values(self):
        values = [np.inf, -np.inf, np.nan, -np.nan]
        for model in MODELS:
            for opcode in ('cvt.rni.f32.f32', 'cvt.rzi.f32.f32'):
                with self.subTest(model=model.__name__, opcode=opcode):
                    actual = model(probe(f'{opcode} %f1, %f0;'),
                                   launch(values)).outputs['result']
                    self.assertEqual(actual.dtype, np.dtype(np.float32))
                    np.testing.assert_array_equal(actual[:2].view(np.uint32),
                                                  [0x7f800000, 0xff800000])
                    self.assertTrue(np.all(np.isnan(actual[2:4])))

    def test_conversion_result_keeps_separate_f32_arithmetic_rounding(self):
        # 2**24 + 1 rounds back to 2**24 before the subtraction. Delaying
        # rounding until the final store would incorrectly produce 1.
        for model in MODELS:
            for opcode in ('cvt.rni.f32.f32', 'cvt.rzi.f32.f32'):
                instructions = (f'{opcode} %f1, %f0;\n'
                                '    add.rn.f32 %f2, %f1, 0f3f800000;\n'
                                '    sub.rn.f32 %f1, %f2, %f1;')
                with self.subTest(model=model.__name__, opcode=opcode):
                    actual = model(probe(instructions),
                                   launch([2**24] * 32)).outputs['result']
                    self.assertEqual(actual.dtype, np.dtype(np.float32))
                    np.testing.assert_array_equal(actual.view(np.uint32), np.zeros(32, np.uint32))

    def test_unimplemented_conversion_modifiers_still_fail(self):
        for model in MODELS:
            for opcode in ('cvt.rmi.f32.f32', 'cvt.rpi.f32.f32',
                           'cvt.rni.ftz.f32.f32', 'cvt.rzi.s32.f32'):
                with self.subTest(model=model.__name__, opcode=opcode):
                    with self.assertRaisesRegex(AssertionError, 'unsupported modeled instruction'):
                        model(probe(f'{opcode} %f1, %f0;'), launch([1.5]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
