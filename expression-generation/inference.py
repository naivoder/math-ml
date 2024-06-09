import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerModel
from tokenizer import detokenize_expression
from dataset import ExpressionDataset, collate_fn
import sympy as sp
import os
import subprocess
from tabulate import tabulate
import random


def evaluate_model(model, iterator, vocab, vocab_rev):
    model.eval()
    num_correct = 0
    num_total = 0
    results = []

    with torch.no_grad():
        for _, (src, trg, _, _) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg)

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)

            _, predicted = torch.max(output, 1)
            num_correct += (predicted == trg).sum().item()
            num_total += trg.size(0)

            # Collect results for a random selection of 10 samples
            for i in range(src.shape[1]):
                src_expr = detokenize_expression(
                    [vocab[idx.item()] for idx in src[:, i]]
                )
                trg_expr = detokenize_expression(
                    [vocab[idx.item()] for idx in trg.view(-1)]
                )
                pred_expr = detokenize_expression(
                    [
                        vocab[idx.item()]
                        for idx in predicted[i * trg.shape[0] : (i + 1) * trg.shape[0]]
                    ]
                )

                results.append((src_expr, pred_expr, trg_expr))

    # Select a random subset of 10 samples
    random_results = random.sample(results, 10)

    # Display the results in a table
    headers = ["Input", "Predicted Output", "Correct Output"]
    table = tabulate(random_results, headers, tablefmt="grid")
    print(table)

    return num_correct / num_total


if __name__ == "__main__":
    # Check if vocab.txt exists, and generate it if it doesn't
    if not os.path.exists("vocab.txt"):
        result = subprocess.run(["python", "tokenizer.py"], check=True)
        if result.returncode != 0:
            raise RuntimeError("tokenizer.py did not run successfully.")

    with open("vocab.txt", "r") as f:
        vocab = [line.strip() for line in f.readlines()]

    vocab_dict = {i: token for i, token in enumerate(vocab)}
    vocab_rev = {token: i for i, token in enumerate(vocab)}

    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    BATCH_SIZE = 50
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

    model.load_state_dict(torch.load("weights/model.pt"))

    dataset = ExpressionDataset(vocab, 1000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    accuracy = evaluate_model(model, dataloader, vocab_dict, vocab_rev)
    print(f"Test Accuracy: {accuracy:.4f}")
