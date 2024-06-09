import sympy as sp
import random

x = sp.symbols("x")
operators = ["+", "-", "*", "/", "**"]
unary_operators = ["sin", "cos", "tan", "exp", "log"]
constants = [
    sp.Rational(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)
]


def apply_unary_op(expr):
    op = random.choice(unary_operators)
    if op == "sin":
        return sp.sin(expr)
    elif op == "cos":
        return sp.cos(expr)
    elif op == "tan":
        return sp.tan(expr)
    elif op == "exp":
        return sp.exp(expr)
    elif op == "log":
        return sp.log(expr)


def generate_expression(depth, current_depth=0, include_x=True):
    if current_depth >= depth or (random.random() > 0.5 and not include_x):
        if random.random() > 0.5:
            return random.choice(constants)
        else:
            return x
    else:
        op = random.choice(operators)
        if op in ["+", "-", "*", "/"]:
            left_expr = generate_expression(depth, current_depth + 1, include_x)
            right_expr = generate_expression(depth, current_depth + 1, False)
            return sp.sympify(f"({left_expr} {op} {right_expr})")
        elif op == "**":
            base_expr = generate_expression(depth, current_depth + 1, include_x)
            exponent = random.randint(2, 3)
            return sp.sympify(f"({base_expr} {op} {exponent})")


def generate_function_derivative_pairs():
    while True:
        depth = random.randint(1, 5)
        expr = generate_expression(depth)
        if random.random() > 0.5:
            expr = apply_unary_op(expr)
        derivative = sp.diff(expr, x)
        if "zoo" in str(expr) or "zoo" in str(derivative):
            continue
        if derivative != 0:
            yield expr, derivative


if __name__ == "__main__":
    generator = generate_function_derivative_pairs()
    for _ in range(10):
        func, deriv = next(generator)
        print(f"Function: {func}")
        print(f"Derivative: {deriv}\n")
