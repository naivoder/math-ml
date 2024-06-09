import random
import torch
from torch.utils.data import Dataset

MIN_NUMBER = 1
MAX_NUMBER = 100


class MathProblemDataset(Dataset):
    def __init__(self, generator_func, num_samples=10000):
        self.generator_func = generator_func
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        operands, solution = self.generator_func()
        return torch.tensor(operands, dtype=torch.float32), torch.tensor(
            solution, dtype=torch.float32
        )


def generate_addition_problem():
    a = random.uniform(MIN_NUMBER, MAX_NUMBER)
    b = random.uniform(MIN_NUMBER, MAX_NUMBER)
    problem = (a, b)
    solution = a + b
    return problem, solution


def generate_random_problem():
    return generate_addition_problem()


def collate_fn(batch):
    operands, solutions = zip(*batch)
    operands = torch.stack(operands)
    solutions = torch.stack(solutions)
    return operands, solutions
