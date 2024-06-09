import torch
import torch.nn.functional as F


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
            prob = model.generator(out[:, -1])
            log_prob = F.log_softmax(prob, dim=-1)
            top_k_probs, top_k_ids = log_prob.topk(beam_width)

            for i in range(beam_width):
                next_ys = torch.cat([ys, top_k_ids[:, i].unsqueeze(1)], dim=0)
                score_new = score + top_k_probs[:, i].item()
                all_hypotheses.append((next_ys, score_new))

        hypotheses = sorted(all_hypotheses, key=lambda x: x[1], reverse=True)[
            :beam_width
        ]

        completed_hypotheses += [h for h in hypotheses if h[0][-1].item() == end_symbol]
        hypotheses = [h for h in hypotheses if h[0][-1].item() != end_symbol]

    completed_hypotheses += hypotheses
    completed_hypotheses = sorted(
        completed_hypotheses, key=lambda x: x[1] / len(x[0]), reverse=True
    )

    return completed_hypotheses[0][0]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask
