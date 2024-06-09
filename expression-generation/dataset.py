import sympy as sp
import random
from multiprocessing import Pool

x = sp.symbols("x")
operators = ["+", "-", "*", "/", "**"]
unary_operators = ["sin", "cos", "tan", "exp", "log"]
constants = [
    sp.Rational(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)
]


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


def generate_pair(_):
    while True:
        depth = random.randint(1, 5)
        expr = generate_expression(depth)
        if random.random() > 0.5:
            expr = apply_unary_op(expr)
        derivative = sp.diff(expr, x)
        if "zoo" in str(expr) or "zoo" in str(derivative) or derivative == 0:
            continue
        return expr, derivative


def generate_function_derivative_pairs(n_pairs):
    with Pool() as pool:
        return pool.map(generate_pair, range(n_pairs))


if __name__ == "__main__":
    n_pairs = 10
    pairs = generate_function_derivative_pairs(n_pairs)
    for func, deriv in pairs:
        print(f"Function: {func}")
        print(f"Derivative: {deriv}\n")
