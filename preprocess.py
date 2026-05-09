from typing import List, Sequence

import pandas as pd

from utils import clean_text, normalize_label, tokenize


def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["label"].map(normalize_label)
    return df


def tokenize_texts(texts: Sequence[str]) -> List[List[str]]:
    return [tokenize(text) for text in texts]
