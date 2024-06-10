import sympy as sp
import random
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset
from tokenizer import tokenize_expression

x = sp.symbols("x")
operators = ["+", "-", "*", "/"]
unary_operators = ["log", "sin", "cos", "tan", "exp"]
constants = [
    sp.Rational(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)
]


class ExpressionDataset(Dataset):
    def __init__(self, vocab, n_pairs):
        self.vocab = {word: i for i, word in enumerate(vocab)}
        self.data = generate_function_derivative_pairs(n_pairs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        func, deriv = self.data[idx]
        func_tokens = tokenize_expression(func)
        deriv_tokens = tokenize_expression(deriv)

        func_indices = [self.vocab[token] for token in func_tokens]
        deriv_indices = [self.vocab[token] for token in deriv_tokens]

        return torch.tensor(func_indices), torch.tensor(deriv_indices)


def collate_fn(batch):
    src, trg = zip(*batch)
    src_lens = [len(s) for s in src]
    trg_lens = [len(t) for t in trg]
    src_padded = torch.nn.utils.rnn.pad_sequence(src, padding_value=0)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg, padding_value=0)
    return src_padded, trg_padded, src_lens, trg_lens


def apply_unary_op(expr):
    op = random.choice(unary_operators)
    return getattr(sp, op)(expr)


def generate_expression(depth, current_depth=0, include_x=True):
    if current_depth >= depth or (random.random() > 0.5 and not include_x):
        return random.choice(constants) if random.random() > 0.5 else x
    op = random.choice(operators)
    if op in ["+", "-", "*", "/"]:
        left_expr = generate_expression(depth, current_depth + 1, include_x)
        right_expr = generate_expression(depth, current_depth + 1, False)
        return sp.sympify(f"({left_expr} {op} {right_expr})")
    base_expr = generate_expression(depth, current_depth + 1, include_x)
    exponent = random.randint(2, 3)
    return sp.sympify(f"({base_expr} {op} {exponent})")


def infix_to_prefix(expr):
    def helper(node):
        if isinstance(node, sp.Symbol) or isinstance(node, sp.Number):
            return str(node)
        if isinstance(node, sp.Mul):
            return f"* {helper(node.args[0])} {helper(node.args[1])}"
        if isinstance(node, sp.Add):
            return f"+ {helper(node.args[0])} {helper(node.args[1])}"
        if isinstance(node, sp.Pow):
            return f"^ {helper(node.args[0])} {helper(node.args[1])}"
        if isinstance(node, sp.Rational):
            return f"/ {helper(node.args[0])} {helper(node.args[1])}"
        if isinstance(node, sp.sin):
            return f"sin {helper(node.args[0])}"
        if isinstance(node, sp.cos):
            return f"cos {helper(node.args[0])}"
        if isinstance(node, sp.tan):
            return f"tan {helper(node.args[0])}"
        if isinstance(node, sp.log):
            return f"log {helper(node.args[0])}"
        if isinstance(node, sp.exp):
            return f"exp {helper(node.args[0])}"
        raise ValueError(f"Unsupported node type: {type(node)}")

    return helper(expr)


def generate_pair(_):
    while True:
        depth = random.randint(1, 5)
        expr = generate_expression(depth)
        if random.random() > 0.5:
            expr = apply_unary_op(expr)
        derivative = sp.diff(expr, x)
        if "zoo" in str(expr) or "zoo" in str(derivative) or derivative == 0:
            continue
        expr_prefix = infix_to_prefix(expr)
        derivative_prefix = infix_to_prefix(derivative)
        return expr_prefix, derivative_prefix


def generate_function_derivative_pairs(n_pairs):
    with Pool() as pool:
        return pool.map(generate_pair, range(n_pairs))


if __name__ == "__main__":
    n_pairs = 10
    pairs = generate_function_derivative_pairs(n_pairs)
    for func, deriv in pairs:
        print(f"Function: {func}")
        print(f"Derivative: {deriv}\n")
