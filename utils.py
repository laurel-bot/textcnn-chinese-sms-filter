import json
import os
import random
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import jieba
import numpy as np
import torch

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]


class Vocab:
    def __init__(self, token_to_idx: Dict[str, int]):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def lookup_token(self, token: str) -> int:
        return self.token_to_idx.get(token, UNK_IDX)

    def lookup_index(self, idx: int) -> str:
        return self.idx_to_token.get(idx, UNK_TOKEN)

    def to_dict(self) -> Dict[str, int]:
        return self.token_to_idx

    @classmethod
    def from_dict(cls, token_to_idx: Dict[str, int]) -> "Vocab":
        normalized = {str(k): int(v) for k, v in token_to_idx.items()}
        return cls(normalized)


def build_vocab(
    tokenized_texts: Sequence[Sequence[str]],
    max_vocab_size: int = 20000,
    min_freq: int = 1,
) -> Vocab:
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    token_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    sorted_tokens = sorted(
        [item for item in counter.items() if item[1] >= min_freq],
        key=lambda x: (-x[1], x[0]),
    )

    for token, _ in sorted_tokens:
        if token in token_to_idx:
            continue
        if len(token_to_idx) >= max_vocab_size:
            break
        token_to_idx[token] = len(token_to_idx)

    return Vocab(token_to_idx)


def encode_tokens(tokens: Sequence[str], vocab: Vocab, max_len: int) -> List[int]:
    ids = [vocab.lookup_token(token) for token in tokens]
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "1": 1,
        "ham": 0,
        "spam": 1,
        "normal": 0,
        "junk": 1,
        "正常": 0,
        "普通": 0,
        "非垃圾": 0,
        "垃圾": 1,
        "垃圾短信": 1,
    }
    if text in mapping:
        return mapping[text]
    raise ValueError(f"Unsupported label value: {value}")


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
