from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from model import MathModel
from data import MathProblemDataset, generate_random_problem, collate_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, optimizer, criterion, device, epochs=1000, patience=50):
    early_stop_count = 0
    model.train()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    best_loss = float("inf")

    for epoch in range(epochs):
        dataset = MathProblemDataset(generate_random_problem, num_samples=32000)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        total_loss, count = 0, 0

        for batch in dataloader:
            count += 1
            operands, solutions = batch
            operands = operands.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()
            outputs = model(operands)
            loss = criterion(outputs.squeeze(), solutions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print one example problem, answer, and solution from the batch
            if count % 1000 == 0:
                problem_str = f"{operands[0][0].item()} + {operands[0][1].item()}"
                answer = outputs[0].item()
                solution = solutions[0].item()
                print(
                    f"Problem: {problem_str}\tAnswer: {answer:.2f}\tSolution: {solution:.2f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}], Average Loss: {avg_loss}\n")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_count = 0
            torch.save(model.state_dict(), "best_model_weights.pth")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Stopping early due to lack of improvement")
                break


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = 2  # Two operands
    hidden_size = 64
    output_size = 1

    model = MathModel(input_size, hidden_size, output_size).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    train(model, optimizer, criterion, device)
