import random
import fractions
import torch
from torch.utils.data import Dataset

MIN_NUMBER = 1
MAX_NUMBER = 100

OPERATORS = ["+", "-", "*", "/", "^"]
OPERATOR_TO_IDX = {op: i for i, op in enumerate(OPERATORS)}


class MathProblemDataset(Dataset):
    def __init__(self, generator_func, num_samples=10000):
        self.generator_func = generator_func
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        operands, operator_idx, solution = self.generator_func()
        return (
            torch.tensor(operands, dtype=torch.float32),
            torch.tensor(operator_idx, dtype=torch.float32),
            torch.tensor(solution, dtype=torch.float32),
        )


def generate_addition_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, MAX_NUMBER)
    problem = (a, b)
    operator_idx = OPERATOR_TO_IDX["+"]
    solution = a + b
    return problem, operator_idx, solution


def generate_subtraction_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, a)
    problem = (a, b)
    operator_idx = OPERATOR_TO_IDX["-"]
    solution = a - b
    return problem, operator_idx, solution


def generate_multiplication_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, MAX_NUMBER)
    problem = (a, b)
    operator_idx = OPERATOR_TO_IDX["*"]
    solution = a * b
    return problem, operator_idx, solution


def generate_division_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, MAX_NUMBER)
    problem = (a, b)
    operator_idx = OPERATOR_TO_IDX["/"]
    solution = a / b
    return problem, operator_idx, solution


def generate_exponent_problem():
    base = random.randint(MIN_NUMBER, 10)
    exponent = random.choice(
        [random.randint(2, 5), 1 / random.randint(2, 5)]
    )  # Include fractional exponents
    problem = (base, exponent)
    operator_idx = OPERATOR_TO_IDX["^"]
    solution = base**exponent
    return problem, operator_idx, solution


def generate_fraction_problem():
    numerator1 = random.randint(1, 20)
    denominator1 = random.randint(1, 20)
    numerator2 = random.randint(1, 20)
    denominator2 = random.randint(1, 20)
    frac1 = float(fractions.Fraction(numerator1, denominator1))
    frac2 = float(fractions.Fraction(numerator2, denominator2))
    operation = random.choice(["+", "-", "*", "/"])
    problem = (frac1, frac2)
    operator_idx = OPERATOR_TO_IDX[operation]
    if operation == "+":
        solution = frac1 + frac2
    elif operation == "-":
        solution = frac1 - frac2
    elif operation == "*":
        solution = frac1 * frac2
    elif operation == "/":
        solution = frac1 / frac2
    return problem, operator_idx, solution


def generate_random_problem():
    problem_generators = [
        generate_addition_problem,
        generate_subtraction_problem,
        generate_multiplication_problem,
        generate_division_problem,
        generate_exponent_problem,
        generate_fraction_problem,
    ]
    generator = random.choice(problem_generators)
    return generator()
