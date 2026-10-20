import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# EmoContext emotion mapping (4 classes)
EMOTION_TO_LABEL = {
    "others": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
}
LABEL_TO_EMOTION = {value: key for key, value in EMOTION_TO_LABEL.items()}

# MELD emotion mapping (7 classes)
MELD_LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "disgust": 6,
}
MELD_ID_TO_LABEL = {v: k for k, v in MELD_LABEL_MAP.items()}

DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_MAX_LENGTH = 128


def get_data_dir() -> Path:
    """Find the Data directory relative to the current file or repository root."""
    candidate1 = Path(__file__).resolve().parent.parent / "Data"
    if candidate1.exists():
        return candidate1
    candidate2 = Path("Data").resolve()
    if candidate2.exists():
        return candidate2
    return candidate1


class EmoContextDataset(Dataset):
    """PyTorch Dataset for 3-turn conversational dialogues (EmoContext)."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx].clone().detach(),
            "attention_mask": self.encodings["attention_mask"][idx].clone().detach(),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx].clone().detach()
        return item


def _read_emocontext_split(split_name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    base_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    file_path = base_dir / f"{split_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    df = pd.read_csv(file_path, sep="\t").dropna().reset_index(drop=True)
    df["text"] = (
        df["turn1"].astype(str)
        + " [SEP] "
        + df["turn2"].astype(str)
        + " [SEP] "
        + df["turn3"].astype(str)
    )
    if "label" in df.columns:
        df["label_id"] = df["label"].map(EMOTION_TO_LABEL)
    return df


def load_emocontext_splits(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, dev, and test splits for EmoContext."""
    train_df = _read_emocontext_split("train", data_dir)
    dev_df = _read_emocontext_split("dev", data_dir)
    test_df = _read_emocontext_split("test", data_dir)
    return train_df, dev_df, test_df


def load_meld_splits(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, dev, and test annotation CSVs for MELD."""
    base_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    meld_dir = base_dir / "MELD.Raw"

    train_path = meld_dir / "train" / "train_sent_emo.csv"
    if not train_path.exists():
        train_path = base_dir / "train_sent_emo.csv"

    dev_path = meld_dir / "dev" / "dev_sent_emo.csv"
    if not dev_path.exists():
        dev_path = base_dir / "dev_sent_emo.csv"

    test_path = meld_dir / "test" / "test_sent_emo.csv"
    if not test_path.exists():
        test_path = base_dir / "test_sent_emo.csv"

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, dev_df, test_df]:
        if "Emotion" in df.columns:
            df["label"] = df["Emotion"].map(MELD_LABEL_MAP)

    return train_df, dev_df, test_df


def build_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def encode_split(df: pd.DataFrame, tokenizer, max_length: int = DEFAULT_MAX_LENGTH, text_col: str = "text"):
    return tokenizer(
        df[text_col].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def get_label_tensor(df: pd.DataFrame, label_col: str = "label_id"):
    if label_col not in df.columns:
        return None
    return torch.tensor(df[label_col].tolist(), dtype=torch.long)


def build_emocontext_datasets(
    tokenizer=None,
    data_dir: Optional[Path] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    model_name: str = DEFAULT_MODEL_NAME,
):
    train_df, dev_df, test_df = load_emocontext_splits(data_dir)
    tokenizer = tokenizer or build_tokenizer(model_name)

    train_dataset = EmoContextDataset(
        encode_split(train_df, tokenizer, max_length),
        get_label_tensor(train_df, "label_id"),
    )
    dev_dataset = EmoContextDataset(
        encode_split(dev_df, tokenizer, max_length),
        get_label_tensor(dev_df, "label_id"),
    )
    test_dataset = EmoContextDataset(
        encode_split(test_df, tokenizer, max_length),
        get_label_tensor(test_df, "label_id"),
    )

    return {
        "train_df": train_df,
        "dev_df": dev_df,
        "test_df": test_df,
        "train_dataset": train_dataset,
        "dev_dataset": dev_dataset,
        "test_dataset": test_dataset,
        "tokenizer": tokenizer,
    }


def build_emocontext_dataloaders(
    batch_size: int = 32,
    tokenizer=None,
    data_dir: Optional[Path] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    model_name: str = DEFAULT_MODEL_NAME,
):
    assets = build_emocontext_datasets(
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_length=max_length,
        model_name=model_name,
    )
    assets["train_loader"] = DataLoader(
        assets["train_dataset"], batch_size=batch_size, shuffle=True
    )
    assets["dev_loader"] = DataLoader(
        assets["dev_dataset"], batch_size=batch_size, shuffle=False
    )
    assets["test_loader"] = DataLoader(
        assets["test_dataset"], batch_size=batch_size, shuffle=False
    )
    return assets


def get_class_weights(train_df: pd.DataFrame, label_col: str = "label_id"):
    return compute_class_weight(
        class_weight="balanced",
        classes=sorted(train_df[label_col].dropna().unique()),
        y=train_df[label_col].dropna().to_numpy(),
    )


if __name__ == "__main__":
    try:
        assets = build_emocontext_dataloaders()
        print(f"Loaded EmoContext train/dev/test: {len(assets['train_df'])}/{len(assets['dev_df'])}/{len(assets['test_df'])}")
    except Exception as e:
        print(f"EmoContext loading note: {e}")
