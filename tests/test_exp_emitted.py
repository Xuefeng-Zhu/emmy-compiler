"""CPU operand-flow regressions for shared range-reduced exp lowering.

These run emitted PTX in the existing CPU subset model. They are not PTX
assembly or GPU evidence. Comparisons preserve activation rtol/atol and reject
NaNs; exceptional classification has a separate explicit test.
"""
import math
import unittest

import numpy as np

from structured_model import run_structured_ptx
from test_numeric import compile_fixture, scalar_fixture


def f32_neighbors(center, radius):
    center = np.float32(center)
    lower = upper = center
    result = [center]
    for _ in range(radius):
        lower = np.nextafter(lower, np.float32(-np.inf))
        upper = np.nextafter(upper, np.float32(np.inf))
        result.extend((lower, upper))
    return np.asarray(result, np.float32)


def exp_reference(values):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return np.exp(np.asarray(values, np.float32).astype(np.float64)).astype(np.float32)


class RangeReducedExpModelTests(unittest.TestCase):
    def emit_and_run(self, values, operation='exp'):
        values = np.asarray(values, np.float32)
        raw, meta = scalar_fixture(values.size, [(operation, None)])
        ptx = compile_fixture(raw, meta)
        result = run_structured_ptx(ptx, meta['kernels'][0], {'x0': values})
        self.assertTrue(np.all(result.store_counts['add'] == 1))
        return result.outputs['add']

    def compare_exp(self, values):
        actual = self.emit_and_run(values)
        expected = exp_reference(values)
        # Classification deserves explicit checks because atol masks tiny values.
        np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
        np.testing.assert_array_equal(actual == 0, expected == 0)
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite],
                                   rtol=2e-6, atol=2e-7, equal_nan=False)
        return actual

    def test_original_wide_failure_grid_and_specific_counterexample(self):
        self.compare_exp(np.linspace(-80, 80, 4097, dtype=np.float32))
        self.compare_exp(np.array([44.609375, -76.1328125, -103.97207641601562], np.float32))

    def test_reduction_switches_random_range_and_finite_extremes(self):
        switches = np.concatenate([f32_neighbors((n + 0.5) * math.log(2), 3)
                                   for n in range(-150, 129)])
        rng = np.random.default_rng(621304)
        random = rng.uniform(-104, 89, 10001).astype(np.float32)
        maximum = np.finfo(np.float32).max
        values = np.concatenate((switches, random, np.array([-maximum, maximum], np.float32)))
        self.compare_exp(values)

    def test_output_classification_boundaries_and_exceptional_inputs(self):
        smallest = float(np.nextafter(np.float32(0), np.float32(1)))
        boundaries = [math.log(float(np.finfo(np.float32).max)),
                      math.log(float(np.finfo(np.float32).tiny)),
                      math.log(smallest), math.log(smallest / 2)]
        self.compare_exp(np.concatenate([f32_neighbors(x, 32) for x in boundaries]))
        actual = self.emit_and_run(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], np.float32))
        self.assertTrue(np.isnan(actual[0]))
        self.assertTrue(np.isposinf(actual[1]))
        self.assertEqual(int(actual[2].view(np.uint32)), 0)
        np.testing.assert_array_equal(actual[3:].view(np.uint32), [0x3F800000, 0x3F800000])

    def test_composed_activations_keep_existing_finite_tolerances(self):
        values = np.concatenate((np.linspace(-100, 100, 2001, dtype=np.float32),
                                 np.array([-1e-30, 1e-30, -0.0, 0.0], np.float32)))
        wide = values.astype(np.float64)
        # Match Emmy's explicit f32 exp then add/divide evaluation, including
        # finite inputs for which the intermediate f32 exponential overflows.
        exponential = exp_reference(-values)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            denominator = np.add(np.float32(1), exponential, dtype=np.float32)
            references = {
                'sigmoid': np.divide(np.float32(1), denominator, dtype=np.float32),
                'silu': np.divide(values, denominator, dtype=np.float32),
                'tanh': np.tanh(wide),
                'erf': np.array([math.erf(x) for x in wide]),
            }
        for operation, expected in references.items():
            with self.subTest(operation=operation):
                actual = self.emit_and_run(values, operation)
                np.testing.assert_allclose(actual, expected, rtol=2e-6,
                                           atol=6e-7 if operation == 'erf' else 2e-7,
                                           equal_nan=False)


if __name__ == '__main__':
    unittest.main()
