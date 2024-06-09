from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from model import MathModel
from data import (
    MathProblemDataset,
    generate_addition_problem,
    generate_subtraction_problem,
    generate_multiplication_problem,
    generate_division_problem,
    generate_exponent_problem,
    collate_fn,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau


def log_cosh_loss(output, target):
    return torch.mean(torch.log(torch.cosh(output - target)))


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epochs=1000,
    patience=50,
    model_name="model",
):
    early_stop_count = 0
    model.train()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            operands, _, solutions = batch
            operands = operands.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()
            outputs = model(operands).squeeze()
            loss = criterion(outputs, solutions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_count = 0
            torch.save(model.state_dict(), f"weights/{model_name}_weights.pth")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Stopping early due to lack of improvement")
                break


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = 2
    hidden_size = 128
    output_size = 1

    operator_functions = {
        "add": generate_addition_problem,
        "subtract": generate_subtraction_problem,
        "multiply": generate_multiplication_problem,
        "divide": generate_division_problem,
        "exponent": generate_exponent_problem,
    }

    for operator, generate_problem in operator_functions.items():
        print(f"\nTraining model for operator: {operator}")

        dataset = MathProblemDataset(generate_problem, num_samples=10000)
        dataloader = DataLoader(
            dataset, batch_size=32, collate_fn=collate_fn, shuffle=True
        )

        model = MathModel(input_size, hidden_size, output_size).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        # criterion = torch.nn.MSELoss()        # MSE
        criterion = torch.nn.L1Loss()  # MAE
        # criterion = torch.nn.SmoothL1Loss()  # Huber
        # criterion = log_cosh_loss             # Log-CosH

        train(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            model_name=f"model_{operator}",
        )
