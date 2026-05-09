import argparse
import os

import torch

from config import ARTIFACTS_DIR
from model import TextCNN
from utils import Vocab, load_json, tokenize


def parse_args():
    parser = argparse.ArgumentParser(description="Predict SMS spam with a trained TextCNN model")
    parser.add_argument("--text", type=str, required=True, help="Input SMS text")
    parser.add_argument("--artifacts_dir", type=str, default=ARTIFACTS_DIR)
    return parser.parse_args()


def load_model(artifacts_dir: str):
    checkpoint_path = os.path.join(artifacts_dir, "best_model.pt")
    vocab_path = os.path.join(artifacts_dir, "vocab.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = Vocab.from_dict(load_json(vocab_path))

    model = TextCNN(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        num_classes=checkpoint["num_classes"],
        num_filters=checkpoint["num_filters"],
        filter_sizes=checkpoint["filter_sizes"],
        dropout=checkpoint["dropout"],
        pad_idx=checkpoint["pad_idx"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, checkpoint["max_len"]


def predict(text: str, artifacts_dir: str):
    model, vocab, max_len = load_model(artifacts_dir)
    tokens = tokenize(text)
    ids = [vocab.lookup_token(token) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    label = "spam" if pred == 1 else "ham"
    spam_prob = float(probs[1].item())
    return label, spam_prob


def main():
    args = parse_args()
    label, spam_prob = predict(args.text, args.artifacts_dir)
    print(f"输入文本: {args.text}")
    print(f"预测标签: {label}")
    print(f"垃圾短信概率: {spam_prob:.4f}")


if __name__ == "__main__":
    main()
