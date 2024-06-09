from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from model import MathModel
from data import MathProblemDataset, generate_random_problem, collate_fn, tokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, optimizer, criterion, device, epochs=10000, patience=50):
    early_stop_count = 0
    model.train()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    best_loss = float("inf")

    for epoch in range(epochs):
        dataset = MathProblemDataset(generate_random_problem, num_samples=3200)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        total_loss = 0
        count = 0

        for batch in dataloader:
            count += 1
            problems, tokenized_problems, solutions = batch
            input_ids = tokenized_problems["input_ids"].to(device)
            attention_mask = tokenized_problems["attention_mask"].to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, solutions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if count % 10 == 0:
                problem_str = problems[0]
                # problem_str = tokenizer.decode(problem_str, skip_special_tokens=True)
                answer = outputs[0].item()
                solution = solutions[0].item()

                print(
                    f"Problem: {problem_str} \t\tAnswer: {answer:.4f} \tSolution: {solution:.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}\n")

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

    model = MathModel().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    train(model, optimizer, criterion, device)
