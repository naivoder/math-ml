import unittest
import torch
from model import *


class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        self.INPUT_DIM = 20
        self.OUTPUT_DIM = 20
        self.D_MODEL = 512
        self.NHEAD = 8
        self.NUM_ENCODER_LAYERS = 6
        self.NUM_DECODER_LAYERS = 6
        self.DIM_FEEDFORWARD = 2048
        self.DROPOUT = 0.1
        self.device = torch.device("cpu")

        self.model = TransformerModel(
            self.INPUT_DIM,
            self.OUTPUT_DIM,
            self.D_MODEL,
            self.NHEAD,
            self.NUM_ENCODER_LAYERS,
            self.NUM_DECODER_LAYERS,
            self.DIM_FEEDFORWARD,
            self.DROPOUT,
        ).to(self.device)

    def test_positional_encoding(self):
        pe = PositionalEncoding(self.D_MODEL, self.DROPOUT)
        x = torch.zeros(50, 10, self.D_MODEL)
        out = pe(x)
        self.assertEqual(out.shape, (50, 10, self.D_MODEL))
        self.assertFalse(
            torch.equal(x, out), "Positional encoding should alter the input tensor"
        )

    def test_transformer_forward(self):
        src = torch.randint(0, self.INPUT_DIM, (10, 32))  # (src_len, batch_size)
        trg = torch.randint(0, self.OUTPUT_DIM, (20, 32))  # (trg_len, batch_size)
        output = self.model(src, trg)
        self.assertEqual(output.shape, (20, 32, self.OUTPUT_DIM))


if __name__ == "__main__":
    unittest.main()
