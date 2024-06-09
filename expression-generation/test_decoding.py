import unittest
import sympy as sp
from dataset import *


def prefix_to_infix(prefix_expr):
    tokens = prefix_expr.split()
    stack = []
    for token in reversed(tokens):
        if token in operators + unary_operators:
            if token in unary_operators:
                operand = stack.pop()
                stack.append(f"{token}({operand})")
            else:
                left = stack.pop()
                right = stack.pop()
                stack.append(f"({left} {token} {right})")
        else:
            stack.append(token)
    return stack[0]


class TestFunctionDerivativeGenerator(unittest.TestCase):

    def test_function_contains_x(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            func_expr = sp.sympify(prefix_to_infix(func))
            self.assertIn(
                x, func_expr.free_symbols, f"Function {func} does not contain x"
            )

    def test_derivative_not_zero(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            deriv_expr = sp.sympify(prefix_to_infix(deriv))
            self.assertNotEqual(deriv_expr, 0, f"Derivative of function {func} is zero")

    def test_correct_derivative(self):
        generator = generate_function_derivative_pairs(10)
        for func, deriv in generator:
            func_expr = sp.sympify(prefix_to_infix(func))
            deriv_expr = sp.sympify(prefix_to_infix(deriv))
            expected_deriv = sp.diff(func_expr, x)
            print(f"Function (prefix): {func}")
            print(f"Function (infix): {func_expr}")
            print(f"Derivative (prefix): {deriv}")
            print(f"Derivative (infix): {deriv_expr}")
            print(f"Expected Derivative: {expected_deriv}")
            self.assertTrue(
                sp.simplify(expected_deriv - deriv_expr) == 0,
                f"Derivative of function {func} is incorrect",
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
