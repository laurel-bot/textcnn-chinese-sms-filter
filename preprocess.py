from typing import List, Optional, Sequence

import pandas as pd

from utils import clean_text, normalize_label, tokenize


def _parse_pretokenized_tokens(value) -> Optional[List[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    tokens = [tok.strip() for tok in text.split() if tok.strip()]
    return tokens or None


def _load_tab_txt_dataset(data_path: str) -> pd.DataFrame:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid tab-separated record at line {line_no}: expected at least 2 fields, got {len(parts)}"
                )

            label = parts[0]
            text = parts[1]
            segmented = parts[2] if len(parts) >= 3 else ""
            records.append(
                {
                    "label": label,
                    "text": text,
                    "tokens": _parse_pretokenized_tokens(segmented),
                }
            )

    if not records:
        raise ValueError(f"No valid samples found in dataset: {data_path}")

    return pd.DataFrame(records)


def load_dataset(data_path: str) -> pd.DataFrame:
    normalized_path = data_path.lower()

    if normalized_path.endswith(".txt") or normalized_path.endswith(".tsv"):
        df = _load_tab_txt_dataset(data_path)
    else:
        df = pd.read_csv(data_path)
        required_columns = {"text", "label"}
        if required_columns.issubset(df.columns):
            df = df.copy()
            if "tokens" not in df.columns:
                df["tokens"] = None
        elif df.shape[1] >= 2:
            renamed_columns = list(df.columns)
            renamed_columns[0] = "text"
            renamed_columns[1] = "label"
            df.columns = renamed_columns
            if "tokens" not in df.columns:
                df["tokens"] = None
            df = df[["text", "label", "tokens"]].copy()
        else:
            raise ValueError(
                "Dataset must either contain 'text' and 'label' columns, or be a tab-separated .txt/.tsv file "
                "with label, text, and optional tokenized text fields."
            )

    df = df[["text", "label", "tokens"]].copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["label"].map(normalize_label)
    df["tokens"] = df["tokens"].map(_parse_pretokenized_tokens)
    return df


def tokenize_texts(texts: Sequence[str], pretokenized_texts: Optional[Sequence[Optional[Sequence[str]]]] = None) -> List[List[str]]:
    if pretokenized_texts is None:
        return [tokenize(text) for text in texts]

    tokenized = []
    for text, tokens in zip(texts, pretokenized_texts):
        if tokens:
            tokenized.append([str(token).strip() for token in tokens if str(token).strip()])
        else:
            tokenized.append(tokenize(text))
    return tokenized
