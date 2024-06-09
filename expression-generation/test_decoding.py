import unittest
from decoding import are_expressions_equivalent, clean_expression


class TestDecoding(unittest.TestCase):

    def test_are_expressions_equivalent(self):
        test_cases = [
            ("2*x + 3", "3 + 2*x", True),
            ("(x + 1)*(x - 1)", "x**2 - 1", True),
            ("2*(x + 3)", "2*x + 6", True),
            ("3*x + 4*y", "4*y + 3*x", True),
            ("(x + 1)*(y - 2)", "(y - 2)*(x + 1)", True),
            ("2*x*(3 + y)", "6*x + 2*x*y", True),
            ("sin(x)*cos(y)", "cos(y)*sin(x)", True),
            ("x/y", "x/y", True),
            ("x + y", "y + x", True),
            ("x * 1/y", "x/y", True),
            ("x**2 + 2*x + 1", "(x + 1)**2", True),
            ("x**2 - 4", "(x - 2)*(x + 2)", True),
            ("sin(x)**2 + cos(x)**2", "1", True),
            ("log(x*y)", "log(x) + log(y)", True),
            ("exp(x + y)", "exp(x)*exp(y)", True),
            ("2*x + 3", "2*x + 2", False),
            ("x**2 + 2*x + 1", "x**2 + 2*x", False),
            ("sin(x)**2 + cos(x)**2", "0", False),
        ]

        for expr1, expr2, expected in test_cases:
            with self.subTest(expr1=clean_expression(expr1), expr2=expr2):
                self.assertEqual(are_expressions_equivalent(expr1, expr2), expected)


if __name__ == "__main__":
    unittest.main()
