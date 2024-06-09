import unittest
from tokenizer import *


class TestTokenizer(unittest.TestCase):

    def test_tokenize_expression(self):
        expr = "2*x + 3*sin(x) ** 2"
        expected_tokens = [
            "2",
            "*",
            "x",
            "+",
            "3",
            "*",
            "sin",
            "(",
            "x",
            ")",
            "**",
            "2",
        ]
        tokens = tokenize_expression(expr)
        self.assertEqual(tokens, expected_tokens)

    def test_detokenize_expression(self):
        tokens = ["2", "*", "x", "+", "3", "*", "sin", "(", "x", ")", "**", "2"]
        expected_expr = "2 * x + 3 * sin ( x ) ** 2"
        expr = detokenize_expression(tokens)
        self.assertEqual(expr, expected_expr)

    def test_build_vocab(self):
        vocab = build_vocab()
        expected_vocab = sorted(
            set("0123456789()+-*/").union(
                {"sin", "cos", "tan", "exp", "log", "x", "**"}
            )
        )
        self.assertEqual(vocab, expected_vocab)


if __name__ == "__main__":
    unittest.main()
