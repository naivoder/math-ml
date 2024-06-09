import unittest
import sympy as sp
from dataset import *


class TestFunctionDerivativeGenerator(unittest.TestCase):

    def test_function_contains_x(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            self.assertIn(x, func.free_symbols, f"Function {func} does not contain x")

    def test_derivative_not_zero(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            self.assertNotEqual(deriv, 0, f"Derivative of function {func} is zero")

    def test_correct_derivative(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            expected_deriv = sp.diff(func, x)
            self.assertEqual(
                deriv, expected_deriv, f"Derivative of function {func} is incorrect"
            )

    def test_no_zoo_in_function(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            self.assertNotIn("zoo", str(func), f"Function {func} contains 'zoo'")

    def test_no_zoo_in_derivative(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            self.assertNotIn("zoo", str(deriv), f"Derivative {deriv} contains 'zoo'")


if __name__ == "__main__":
    unittest.main()
