import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerModel
from tokenizer import detokenize_expression
from dataset import ExpressionDataset, collate_fn
import sympy as sp
import os
import subprocess


def beam_search(model, src, src_mask, max_len, start_symbol, beam_width=10):
    src = src.to(model.device)
    src_mask = src_mask.to(model.device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)

    hypotheses = [(ys, 0)]
    completed_hypotheses = []

    for _ in range(max_len):
        all_hypotheses = []
        for ys, score in hypotheses:
            trg_mask = (
                model.generate_square_subsequent_mask(ys.size(0))
                .type(torch.bool)
                .to(model.device)
            )
            out = model.decode(ys, memory, trg_mask)
            out = out.transpose(0, 1)
            prob = model.fc_out(out[:, -1])
            log_prob = nn.functional.log_softmax(prob, dim=-1)
            top_k_probs, top_k_ids = log_prob.topk(beam_width)

            for i in range(beam_width):
                next_ys = torch.cat([ys, top_k_ids[:, i].unsqueeze(1)], dim=0)
                score_new = score + top_k_probs[:, i].item()
                all_hypotheses.append((next_ys, score_new))

        hypotheses = sorted(all_hypotheses, key=lambda x: x[1], reverse=True)[
            :beam_width
        ]

        completed_hypotheses += [h for h in hypotheses if h[0][-1].item() == 1]
        hypotheses = [h for h in hypotheses if h[0][-1].item() != 1]

    completed_hypotheses += hypotheses
    completed_hypotheses = sorted(
        completed_hypotheses, key=lambda x: x[1] / len(x[0]), reverse=True
    )

    return completed_hypotheses[0][0]


def evaluate_model(model, iterator, vocab):
    model.eval()
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for i, (src, trg, src_lens, trg_lens) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            src_mask = (
                model.generate_square_subsequent_mask(src.size(0))
                .type(torch.bool)
                .to(model.device)
            )
            start_symbol = vocab["<sos>"]

            predicted_indices = beam_search(
                model, src, src_mask, trg.size(0), start_symbol, beam_width=10
            )
            predicted_expressions = detokenize_expression(
                [vocab[idx.item()] for idx in predicted_indices]
            )
            target_expressions = detokenize_expression(
                [vocab[idx.item()] for idx in trg]
            )

            try:
                predicted_expr = sp.sympify(predicted_expressions)
                target_expr = sp.sympify(target_expressions)
                if sp.simplify(predicted_expr - target_expr) == 0:
                    num_correct += 1
            except Exception as e:
                print(f"Sympy error: {e}")
                print(f"Predicted expression: {predicted_expressions}")
                print(f"Target expression: {target_expressions}")

            num_total += 1

    accuracy = num_correct / num_total
    return accuracy


if __name__ == "__main__":
    # Check if vocab.txt exists, and generate it if it doesn't
    if not os.path.exists("vocab.txt"):
        result = subprocess.run(["python", "tokenizer.py"], check=True)
        if result.returncode != 0:
            raise RuntimeError("tokenizer.py did not run successfully.")

    with open("vocab.txt", "r") as f:
        vocab = [line.strip() for line in f.readlines()]

    vocab_dict = {i: token for i, token in enumerate(vocab)}

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

    accuracy = evaluate_model(model, dataloader, vocab_dict)
    print(f"Validation Accuracy: {accuracy:.4f}")
