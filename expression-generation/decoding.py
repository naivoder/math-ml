import sympy as sp


def clean_expression(expr):
    # # # Remove excessive spaces
    expr = expr.replace(" ", "")

    # # Handle double operators (e.g., //) that are not valid in SymPy
    expr = expr.replace("//", "/")

    return expr


def are_expressions_equivalent(expr1, expr2):
    try:
        # Clean the expressions
        expr1 = clean_expression(expr1)
        expr2 = clean_expression(expr2)

        # Parse the expressions
        parsed_expr1 = sp.sympify(expr1)
        parsed_expr2 = sp.sympify(expr2)

        # Expand logarithmic and exponential expressions
        expanded_expr1 = sp.expand_log(parsed_expr1, force=True)
        expanded_expr2 = sp.expand_log(parsed_expr2, force=True)
        expanded_expr1 = sp.expand(expanded_expr1)
        expanded_expr2 = sp.expand(expanded_expr2)

        # Simplify both expressions
        simplified_expr1 = sp.simplify(expanded_expr1)
        simplified_expr2 = sp.simplify(expanded_expr2)

        # Compare the simplified expressions
        return sp.simplify(simplified_expr1 - simplified_expr2) == 0
    except (sp.SympifyError, SyntaxError):
        return False
