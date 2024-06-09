import unittest
import sympy as sp
from dataset import *


class TestFunctionDerivativeGenerator(unittest.TestCase):

    def test_function_contains_x(self):
        generator = generate_function_derivative_pairs()
        for _ in range(10):
            func, deriv = next(generator)
            self.assertIn(x, func.free_symbols, f"Function {func} does not contain x")

    def test_derivative_not_zero(self):
        generator = generate_function_derivative_pairs()
        for _ in range(10):
            func, deriv = next(generator)
            self.assertNotEqual(deriv, 0, f"Derivative of function {func} is zero")

    def test_correct_derivative(self):
        generator = generate_function_derivative_pairs()
        for _ in range(10):
            func, deriv = next(generator)
            expected_deriv = sp.diff(func, x)
            self.assertEqual(
                deriv, expected_deriv, f"Derivative of function {func} is incorrect"
            )


if __name__ == "__main__":
    unittest.main()
