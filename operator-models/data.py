import random
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
            torch.tensor(operator_idx, dtype=torch.int64),
            torch.tensor(solution, dtype=torch.float32),
        )


def normalize(value):
    return (value - MIN_NUMBER) / (MAX_NUMBER - MIN_NUMBER)


def generate_addition_problem():
    a = random.uniform(MIN_NUMBER, MAX_NUMBER)
    b = random.uniform(MIN_NUMBER, MAX_NUMBER)
    problem = (normalize(a), normalize(b))
    operator_idx = OPERATOR_TO_IDX["+"]
    solution = a + b
    return problem, operator_idx, solution


def generate_subtraction_problem():
    a = random.uniform(MIN_NUMBER, MAX_NUMBER)
    b = random.uniform(MIN_NUMBER, a)
    problem = (normalize(a), normalize(b))
    operator_idx = OPERATOR_TO_IDX["-"]
    solution = a - b
    return problem, operator_idx, solution


def generate_multiplication_problem():
    a = random.uniform(MIN_NUMBER, MAX_NUMBER)
    b = random.uniform(MIN_NUMBER, MAX_NUMBER)
    problem = (normalize(a), normalize(b))
    operator_idx = OPERATOR_TO_IDX["*"]
    solution = a * b
    return problem, operator_idx, solution


def generate_division_problem():
    a = random.uniform(MIN_NUMBER, MAX_NUMBER)
    b = random.uniform(MIN_NUMBER, MAX_NUMBER)
    problem = (normalize(a), normalize(b))
    operator_idx = OPERATOR_TO_IDX["/"]
    solution = a / b
    return problem, operator_idx, solution


def generate_exponent_problem():
    base = random.uniform(MIN_NUMBER, 10)
    exponent = random.choice(
        [random.uniform(2, 5), 1 / random.uniform(2, 5)]
    )  # Include fractional exponents
    problem = (normalize(base), normalize(exponent))
    operator_idx = OPERATOR_TO_IDX["^"]
    solution = base**exponent
    return problem, operator_idx, solution


def generate_random_problem():
    problem_generators = [
        generate_addition_problem,
        generate_subtraction_problem,
        generate_multiplication_problem,
        generate_division_problem,
        generate_exponent_problem,
    ]
    generator = random.choice(problem_generators)
    return generator()


def collate_fn(batch):
    operands, operator_idxs, solutions = zip(*batch)
    operands = torch.stack(operands)
    operator_idxs = torch.tensor(operator_idxs)
    solutions = torch.stack(solutions)
    return operands, operator_idxs, solutions
