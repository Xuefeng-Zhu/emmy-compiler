"""Independent operand-flow checks for shape/op generalization, CPU only."""
import math
import unittest

from test_cooperative_model import binary_launch, divide_probe

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write

from emmy_ptx import CompileError
from structured_model import run_structured_ptx
from test_numeric import compile_fixture, scalar_fixture
from test_structured import axis, synthetic


class ExtendedPointwiseTests(unittest.TestCase):
    def execute(self, operations, left, right=None):
        left = np.asarray(left, np.float32)
        ir, meta = scalar_fixture(left.size, operations)
        overrides = {'x0': left}
        if right is not None:
            overrides['x1'] = np.asarray(right, np.float32)
        ptx = compile_fixture(ir, meta)
        result = run_structured_ptx(ptx, meta['kernels'][0], overrides)
        self.assertTrue(np.all(result.store_counts['add'] == 1))
        return result.outputs['add']

    def test_unary_activations_changed_sizes_and_inputs(self):
        references = {
            'exp': np.exp, 'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'silu': lambda x: x / (1 + np.exp(-x)),
            'abs': np.abs, 'absolute': np.abs,
            'negative': np.negative,
        }
        for size in (1, 31, 257):
            values = np.linspace(-8, 8, size, dtype=np.float32)
            for op, reference in references.items():
                with self.subTest(size=size, op=op):
                    actual = self.execute([(op, None)], values)
                    np.testing.assert_allclose(actual, reference(values), rtol=2e-6, atol=2e-7)

    def test_tanh_signed_zero_infinities_and_nan(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        values = np.array([-np.inf, -tiny, -0.0, 0.0, tiny, np.inf, np.nan], np.float32)
        actual = self.execute([('tanh', None)], values)
        np.testing.assert_array_equal(actual, np.tanh(values))
        np.testing.assert_array_equal(np.signbit(actual[:-1]), np.signbit(values[:-1]))

    def test_sigmoid_and_silu_extreme_values_follow_emmy_formula(self):
        values = np.array([-np.inf, -100, -90, -0.0, 0, 90, 100, np.inf, np.nan], np.float32)
        with np.errstate(all='ignore'):
            for operation in ('sigmoid', 'silu'):
                with self.subTest(operation=operation):
                    expected = ElementwiseImpl(operation)(values)
                    actual = self.execute([(operation, None)], values)
                    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7, equal_nan=True)

    def test_exp_divide_and_abs_max_compose(self):
        values = np.linspace(-6, 6, 257, dtype=np.float32)
        divisors = np.linspace(0.5, 3, 257, dtype=np.float32)
        actual = self.execute([('exp', None), ('divide', 'right')], values, divisors)
        np.testing.assert_allclose(actual, np.exp(values) / divisors, rtol=2e-6, atol=2e-6)
        actual = self.execute([('abs', None), ('maximum', 'right')], values, divisors)
        np.testing.assert_array_equal(actual, np.maximum(np.abs(values), divisors))

    def test_silu_composes_from_sigmoid_multiply(self):
        values = np.linspace(-6, 6, 129, dtype=np.float32)
        actual = self.execute([('sigmoid', None), ('multiply', 'left')], values)
        np.testing.assert_allclose(actual, values / (1 + np.exp(-values)), rtol=2e-6, atol=2e-7)

    def test_erf_and_gelu_variants_against_independent_formulas(self):
        values = np.linspace(-8, 8, 1025, dtype=np.float32)
        erf = np.array([math.erf(float(x)) for x in values], np.float64)
        actual = self.execute([('erf', None)], values)
        np.testing.assert_allclose(actual, erf, rtol=2e-6, atol=6e-7)
        gelu = 0.5 * values.astype(np.float64) * (1 + np.array([math.erf(float(x) / math.sqrt(2)) for x in values]))
        actual = self.execute([('gelu', None)], values)
        np.testing.assert_allclose(actual, gelu, rtol=2e-5, atol=1e-6)
        gelu_tanh = 0.5 * values * (1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * values ** 3)))
        actual = self.execute([('gelu_tanh', None)], values)
        np.testing.assert_allclose(actual, gelu_tanh, rtol=2e-5, atol=1e-6)

    def test_rank_two_pointwise_add_with_two_axes_or_flattened_axis(self):
        for rows, columns in ((1, 1), (3, 7), (17, 19), (32, 33)):
            for flattened in (False, True):
                with self.subTest(shape=(rows, columns), flattened=flattened):
                    if flattened:
                        axes = (axis('cell', rows * columns),)
                        from emmy.compiler.ir.expr import BinaryExpr, Literal
                        indices = (BinaryExpr('//', Var('cell'), Literal(columns, 'int')),
                                   BinaryExpr('%', Var('cell'), Literal(columns, 'int')))
                    else:
                        axes = (axis('row', rows), axis('column', columns))
                        indices = (Var('row'), Var('column'))
                    tile = Tile(axes, (
                        Load('lhs', 'left_matrix', indices, dtype=F32),
                        Load('rhs', 'right_matrix', indices, dtype=F32),
                        Assign('sum', ElementwiseImpl('add'), ('lhs', 'rhs'), F32),
                        Write('matrix_sum', indices, 'sum', value_dtype=F32),
                    ))
                    ir, meta = synthetic(tile, {'left_matrix': (rows, columns), 'right_matrix': (rows, columns)},
                                         (rows, columns), output='matrix_sum', grid=(rows * columns + 31) // 32)
                    left = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns) / 7
                    right = np.random.default_rng(983).normal(size=(rows, columns)).astype(np.float32)
                    actual = run_structured_ptx(compile_fixture(ir, meta), meta['kernels'][0],
                                                {'left_matrix': left, 'right_matrix': right})
                    np.testing.assert_array_equal(actual.outputs['matrix_sum'], left + right)
                    self.assertTrue(np.all(actual.store_counts['matrix_sum'] == 1))

    def test_model_copysign_uses_ptx_sign_then_magnitude_order(self):
        # PTX copies the sign of its first operand into its second operand;
        # NumPy's copysign has the reverse argument convention.
        program = divide_probe().replace('div.rn.f32', 'copysign.f32')
        launch = binary_launch([-1.0, 2.0, -0.0, 0.0], [3.0, -4.0, 5.0, -6.0])
        actual = run_structured_ptx(program, launch).outputs['result']
        np.testing.assert_array_equal(actual, np.array([-3.0, 4.0, -5.0, 6.0], np.float32))

    def test_new_unary_operations_reject_incorrect_arity(self):
        for operation in ('exp', 'tanh', 'sigmoid', 'abs', 'gelu', 'silu'):
            with self.subTest(operation=operation):
                ir, meta = scalar_fixture(3, [(operation, 'right')])
                with self.assertRaisesRegex(CompileError, 'operand count'):
                    compile_fixture(ir, meta)


if __name__ == '__main__':
    unittest.main()
