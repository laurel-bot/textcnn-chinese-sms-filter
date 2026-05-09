from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from utils import Vocab, encode_tokens


class SMSDataset(Dataset):
    def __init__(
        self,
        tokenized_texts: Sequence[Sequence[str]],
        labels: Sequence[int],
        vocab: Vocab,
        max_len: int,
    ):
        self.labels = list(labels)
        self.features: List[List[int]] = [
            encode_tokens(tokens, vocab, max_len) for tokens in tokenized_texts
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
