import torch
from torch.utils.data import DataLoader, Dataset
from model import MathModel
from data import generate_random_problem, collate_fn


class CombinedMathProblemDataset(Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        operands, operator_idx, solution = generate_random_problem()
        return (
            torch.tensor(operands, dtype=torch.float32),
            operator_idx,
            torch.tensor(solution, dtype=torch.float32),
        )


def load_model(operator, device):
    input_size = 2
    hidden_size = 128
    output_size = 1
    model = MathModel(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(
        torch.load(f"weights/model_{operator}_weights.pth", map_location=device)
    )
    model.eval()
    return model


def evaluate_model(models, dataloader, device):
    # criterion = torch.nn.MSELoss()        # MSE
    criterion = torch.nn.L1Loss()  # MAE
    # criterion = torch.nn.SmoothL1Loss()  # Huber
    # criterion = log_cosh_loss             # Log-CosH

    operator_losses = {operator: 0 for operator in models.keys()}
    operator_counts = {operator: 0 for operator in models.keys()}
    operator_accuracies = {operator: 0 for operator in models.keys()}
    total_loss = 0
    total_correct = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            operands, operator_idxs, solutions = batch
            operands = operands.to(device)
            solutions = solutions.to(device)

            for i in range(len(operands)):
                operator_idx = operator_idxs[i].item()
                operator = {
                    0: "add",
                    1: "subtract",
                    2: "multiply",
                    3: "divide",
                    4: "exponent",
                }[operator_idx]
                model = models[operator]
                operand = operands[i].unsqueeze(0)
                solution = solutions[i].unsqueeze(0)

                output = model(operand).squeeze(0)

                loss = criterion(output, solution)
                operator_losses[operator] += loss.item()
                operator_counts[operator] += 1
                total_loss += loss.item()
                count += 1

                if abs(output.item() - solution.item()) < 0.01:
                    operator_accuracies[operator] += 1
                    total_correct += 1

    avg_losses = {
        operator: operator_losses[operator] / operator_counts[operator]
        for operator in models.keys()
    }
    accuracies = {
        operator: f"{operator_accuracies[operator]}/{operator_counts[operator]}"
        for operator in models.keys()
    }
    combined_avg_loss = total_loss / count
    combined_accuracy = f"{total_correct}/{count}"

    return avg_losses, accuracies, combined_avg_loss, combined_accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    operator_names = ["add", "subtract", "multiply", "divide", "exponent"]
    for operator in operator_names:
        print(f"Loading model for operator: {operator}")
        models[operator] = load_model(operator, device)

    dataset = CombinedMathProblemDataset(num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    avg_losses, accuracies, combined_avg_loss, combined_accuracy = evaluate_model(
        models, dataloader, device
    )

    for operator, accuracy in accuracies.items():
        print(f"Accuracy for operator '{operator}': {accuracy}")

    print(f"\nCombined Average Loss: {combined_avg_loss:.4f}")
    print(f"Combined Accuracy: {combined_accuracy}")
