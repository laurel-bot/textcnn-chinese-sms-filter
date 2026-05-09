import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    ARTIFACTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_FILTER_SIZES,
    DEFAULT_LR,
    DEFAULT_MAX_LEN,
    DEFAULT_MAX_VOCAB_SIZE,
    DEFAULT_MIN_FREQ,
    DEFAULT_NUM_FILTERS,
    DEFAULT_RANDOM_SEED,
)
from dataset import SMSDataset
from model import TextCNN
from preprocess import load_dataset, tokenize_texts
from utils import build_vocab, classification_metrics, ensure_dir, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train TextCNN for Chinese SMS spam detection")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset file. Supports CSV with text/label columns, or tab-separated .txt/.tsv like mudou_spam.",
    )
    parser.add_argument("--artifacts_dir", type=str, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--num_filters", type=int, default=DEFAULT_NUM_FILTERS)
    parser.add_argument("--filter_sizes", type=str, default=",".join(map(str, DEFAULT_FILTER_SIZES)))
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--max_vocab_size", type=int, default=DEFAULT_MAX_VOCAB_SIZE)
    parser.add_argument("--min_freq", type=int, default=DEFAULT_MIN_FREQ)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.artifacts_dir)

    filter_sizes = [int(x) for x in args.filter_sizes.split(",") if x.strip()]

    df = load_dataset(args.data_path)
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["label"],
    )

    train_tokens = tokenize_texts(train_df["text"].tolist(), train_df["tokens"].tolist())
    val_tokens = tokenize_texts(val_df["text"].tolist(), val_df["tokens"].tolist())

    vocab = build_vocab(
        train_tokens,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
    )

    train_dataset = SMSDataset(train_tokens, train_df["label"].tolist(), vocab, args.max_len)
    val_dataset = SMSDataset(val_tokens, val_df["label"].tolist(), vocab, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=2,
        num_filters=args.num_filters,
        filter_sizes=filter_sizes,
        dropout=args.dropout,
        pad_idx=0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['f1']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": args.embed_dim,
                "num_classes": 2,
                "num_filters": args.num_filters,
                "filter_sizes": filter_sizes,
                "dropout": args.dropout,
                "pad_idx": 0,
                "max_len": args.max_len,
            }
            torch.save(checkpoint, os.path.join(args.artifacts_dir, "best_model.pt"))
            save_json(vocab.to_dict(), os.path.join(args.artifacts_dir, "vocab.json"))
            save_json({"ham": 0, "spam": 1}, os.path.join(args.artifacts_dir, "label_mapping.json"))
            save_json(vars(args), os.path.join(args.artifacts_dir, "train_config.json"))
            save_json(val_metrics, os.path.join(args.artifacts_dir, "metrics.json"))

    save_json(history, os.path.join(args.artifacts_dir, "history.json"))
    print(f"Training complete. Best val F1: {best_f1:.4f}")
    print(f"Artifacts saved to: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
