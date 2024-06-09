import re


def tokenize_expression(expr):
    # tokens = re.findall(r"\d|\w+|\*\*|[-+*/()]", str(expr))
    # tokens = re.findall(r"\*\*|[\d]+|[\w]+|[-+*/()]", str(expr))
    tokens = re.findall(r"\d+|[\w]+|\*\*|[-+*/()]", str(expr))
    digit_tokens = []
    for token in tokens:
        if re.fullmatch(r"\d+", token):
            digit_tokens.extend(list(token))
        else:
            digit_tokens.append(token)
    return digit_tokens


def detokenize_expression(tokens):
    return " ".join(tokens)


def build_vocab():
    vocab = set("0123456789()+-*/")
    vocab.update(["sin", "cos", "tan", "exp", "log", "x", "**", "<sos>", "<eos>"])
    return sorted(vocab)


if __name__ == "__main__":
    vocab = build_vocab()
    with open("vocab.txt", "w") as f:
        for token in vocab:
            f.write(f"{token}\n")
