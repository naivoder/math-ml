from torch.nn import Module, Linear
from transformers import GPT2Model, GPT2Tokenizer


class MathModel(Module):
    def __init__(self, model_name="gpt2"):
        super(MathModel, self).__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.regression_head = Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]  # Use the representation of the [CLS] token
        regression_output = self.regression_head(pooled_output)
        return regression_output.squeeze(-1)


if __name__ == "__main__":
    model = MathModel()
    print(model.state_dict)
