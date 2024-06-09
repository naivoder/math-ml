import random
import fractions
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

MIN_NUMBER = 1
MAX_NUMBER = 100

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


class MathProblemDataset(Dataset):
    def __init__(self, generator_func, num_samples=10000):
        self.generator_func = generator_func
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        problem, solution = self.generator_func()
        return problem, solution


def collate_fn(batch):
    problems, solutions = zip(*batch)
    tokenized_problems = tokenizer(
        list(problems), padding=True, truncation=True, return_tensors="pt"
    )
    solutions = torch.tensor([float(sol) for sol in solutions], dtype=torch.float32)
    return problems, tokenized_problems, solutions


def generate_addition_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, MAX_NUMBER)
    problem = f"{a} + {b}"
    solution = a + b
    return problem, solution


def generate_subtraction_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, a)
    problem = f"{a} - {b}"
    solution = a - b
    return problem, solution


def generate_multiplication_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, MAX_NUMBER)
    problem = f"{a} * {b}"
    solution = a * b
    return problem, solution


def generate_division_problem():
    a = random.randint(MIN_NUMBER, MAX_NUMBER)
    b = random.randint(MIN_NUMBER, MAX_NUMBER)
    problem = f"{a} / {b}"
    solution = a / b
    return problem, solution


def generate_exponent_problem():
    base = random.randint(MIN_NUMBER, 10)
    exponent = random.randint(2, 5)
    problem = f"{base} ^ {exponent}"
    solution = base**exponent
    return problem, solution


def generate_root_problem():
    base = random.randint(1, 10)
    exponent = random.randint(2, 5)
    root = base**exponent
    problem = f"{root} ^ (1/{exponent})"
    solution = base
    return problem, solution


def generate_fraction_problem():
    numerator1 = random.randint(1, 20)
    denominator1 = random.randint(1, 20)
    numerator2 = random.randint(1, 20)
    denominator2 = random.randint(1, 20)
    frac1 = fractions.Fraction(numerator1, denominator1)
    frac2 = fractions.Fraction(numerator2, denominator2)
    operation = random.choice(["+", "-", "*", "/"])
    if operation == "+":
        problem = f"({frac1}) + ({frac2})"
        solution = frac1 + frac2
    elif operation == "-":
        problem = f"({frac1}) - ({frac2})"
        solution = frac1 - frac2
    elif operation == "*":
        problem = f"({frac1}) * ({frac2})"
        solution = frac1 * frac2
    elif operation == "/":
        problem = f"({frac1}) / ({frac2})"
        solution = frac1 / frac2
    return problem, solution


def generate_random_problem():
    problem_generators = [
        generate_addition_problem,
        generate_subtraction_problem,
        generate_multiplication_problem,
        generate_division_problem,
        generate_exponent_problem,
        generate_root_problem,
        generate_fraction_problem,
    ]
    generator = random.choice(problem_generators)
    return generator()


def generate_batch(batch_size):
    batch = [generate_random_problem() for _ in range(batch_size)]
    problems, solutions = zip(*batch)
    return list(problems), list(solutions)


if __name__ == "__main__":
    batch_size = 10
    problems, solutions = generate_batch(batch_size)
    for problem, solution in zip(problems, solutions):
        print(f"Problem: {problem}, Solution: {solution}")

    dataset = MathProblemDataset(generate_random_problem, num_samples=100000)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    for batch in dataloader:
        problems, solutions = batch
        print("Problems:\n", problems)
        print("Solutions:\n", solutions)
        break
