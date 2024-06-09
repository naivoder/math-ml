import os
import subprocess
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerModel
from dataset import ExpressionDataset, collate_fn


def train_model(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg, _, _) in enumerate(iterator):
        print(f" 🏋️ Training on Batch {i+1}/{len(iterator)}...", end="\r")
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg, _, _) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg)

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == "__main__":
    # Check if vocab.txt exists, and generate it if it doesn't
    if not os.path.exists("vocab.txt"):
        result = subprocess.run(["python", "tokenizer.py"], check=True)
        if result.returncode != 0:
            raise RuntimeError("tokenizer.py did not run successfully.")

    with open("vocab.txt", "r") as f:
        vocab = [line.strip() for line in f.readlines()]

    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    BATCH_SIZE = 50
    N_EPOCHS = 1000
    CLIP = 1
    LEARNING_RATE = 1e-4
    PATIENCE = 10  # Early stopping patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        INPUT_DIM,
        OUTPUT_DIM,
        D_MODEL,
        NHEAD,
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        DIM_FEEDFORWARD,
        DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_valid_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
        # Generate new dataset for each epoch
        train_dataset = ExpressionDataset(vocab, 10 * BATCH_SIZE)
        val_dataset = ExpressionDataset(vocab, 2 * BATCH_SIZE)

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
        )

        train_loss = train_model(model, train_dataloader, optimizer, criterion, CLIP)
        val_loss = evaluate_model(model, val_dataloader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "weights/model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print("Early stopping")
            break

        print(
            f"[Epoch {epoch+1:04}/{N_EPOCHS}] Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}"
        )
