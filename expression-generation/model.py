import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.src_tok_emb = nn.Embedding(input_dim, d_model)
        self.trg_tok_emb = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def encode(self, src, src_mask):
        src_emb = self.dropout(self.positional_encoding(self.src_tok_emb(src)))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, trg, memory, trg_mask):
        trg_emb = self.dropout(self.positional_encoding(self.trg_tok_emb(trg)))
        return self.transformer.decoder(trg_emb, memory, trg_mask)

    def forward(self, src, trg):
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(self.device)
        trg_mask = self.generate_square_subsequent_mask(trg.size(0)).to(self.device)

        memory = self.encode(src, src_mask)
        output = self.decode(trg, memory, trg_mask)

        return self.fc_out(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
