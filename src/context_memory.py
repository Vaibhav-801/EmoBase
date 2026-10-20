import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


LABEL_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def build_emotion_history(df: pd.DataFrame, window_size: int = 5) -> np.ndarray:
    """Builds a sliding window of historical emotion labels for conversational context.

    For each utterance in a dialogue, collects the previous `window_size` emotion labels,
    padding with 0 for turns near the start of the dialogue.
    """
    histories = []
    dialogue_emotions: Dict[int, List[int]] = {}

    for _, row in df.iterrows():
        dialogue = row["Dialogue_ID"]
        label = row["label"]

        if dialogue not in dialogue_emotions:
            dialogue_emotions[dialogue] = []

        current_history = dialogue_emotions[dialogue][-window_size:]
        padding_size = window_size - len(current_history)
        padding = [0] * padding_size
        history = padding + current_history
        histories.append(history)

        dialogue_emotions[dialogue].append(int(label))

    return np.array(histories)


def load_and_align_features(
    features_dir: Optional[Path] = None, data_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads text, audio, and face pre-extracted features and aligns them with dialogue turns."""
    base = get_base_dir()
    f_dir = features_dir or (base / "features")
    d_dir = data_dir or (base / "Data")
    meld_dir = d_dir / "MELD.Raw"

    print("Loading precomputed features...")
    train_text = np.load(f_dir / "train_text.npy")
    train_audio = np.load(f_dir / "train_audio.npy")
    train_face = np.load(f_dir / "train_face.npy")
    train_features = np.concatenate((train_text, train_audio, train_face), axis=1)

    dev_text = np.load(f_dir / "dev_text.npy")
    dev_audio = np.load(f_dir / "dev_audio.npy")
    dev_face = np.load(f_dir / "dev_face.npy")
    dev_features = np.concatenate((dev_text, dev_audio, dev_face), axis=1)

    test_text = np.load(f_dir / "test_text.npy")
    test_audio = np.load(f_dir / "test_audio.npy")
    test_face = np.load(f_dir / "test_face.npy")
    test_features = np.concatenate((test_text, test_audio, test_face), axis=1)

    print(f"Concatenated shapes: Train={train_features.shape}, Dev={dev_features.shape}, Test={test_features.shape}")

    train_csv = meld_dir / "train" / "train_sent_emo.csv" if (meld_dir / "train" / "train_sent_emo.csv").exists() else d_dir / "train_sent_emo.csv"
    dev_csv = meld_dir / "dev" / "dev_sent_emo.csv" if (meld_dir / "dev" / "dev_sent_emo.csv").exists() else d_dir / "dev_sent_emo.csv"
    test_csv = meld_dir / "test" / "test_sent_emo.csv" if (meld_dir / "test" / "test_sent_emo.csv").exists() else d_dir / "test_sent_emo.csv"

    df_train = pd.read_csv(train_csv)
    df_dev = pd.read_csv(dev_csv)
    df_test = pd.read_csv(test_csv)

    for df in [df_train, df_dev, df_test]:
        df["label"] = df["Emotion"].str.lower().map(LABEL_MAP)

    # Sort in dialogue temporal order
    df_train = df_train.sort_values(["Dialogue_ID", "Utterance_ID"])
    train_features = train_features[df_train.index]
    df_train = df_train.reset_index(drop=True)

    df_dev = df_dev.sort_values(["Dialogue_ID", "Utterance_ID"])
    dev_features = dev_features[df_dev.index]
    df_dev = df_dev.reset_index(drop=True)

    df_test = df_test.sort_values(["Dialogue_ID", "Utterance_ID"])
    test_features = test_features[df_test.index]
    df_test = df_test.reset_index(drop=True)

    return train_features, dev_features, test_features, df_train, df_dev, df_test


class ConversationalContextDataset(Dataset):
    """Dataset delivering current fused utterance features + dialogue emotion history."""

    def __init__(self, features: np.ndarray, history: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.history = torch.tensor(history, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "features": self.features[idx],
            "history": self.history[idx],
            "labels": self.labels[idx],
        }


class ContextMemoryFusionModel(nn.Module):
    """Fuses utterance features with conversational emotion history via GRU / Embedding."""

    def __init__(
        self,
        feature_dim: int = 3584,
        history_window: int = 5,
        num_classes: int = 7,
        history_embed_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super(ContextMemoryFusionModel, self).__init__()
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        self.history_embed = nn.Embedding(num_classes, history_embed_dim)
        self.history_gru = nn.GRU(
            input_size=history_embed_dim,
            hidden_size=history_embed_dim * 2,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + history_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # 1. Feature projection
        f_proj = torch.relu(self.feature_proj(features))

        # 2. History embedding & GRU
        h_emb = self.history_embed(history)
        _, h_n = self.history_gru(h_emb)
        h_context = h_n.squeeze(0)

        # 3. Concatenate and classify
        combined = torch.cat([f_proj, h_context], dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Context Memory Model")
    parser.add_argument("--window-size", type=int, default=5)
    args = parser.parse_args()

    try:
        train_feat, dev_feat, test_feat, df_tr, df_dv, df_te = load_and_align_features()
        train_hist = build_emotion_history(df_tr, window_size=args.window_size)
        dev_hist = build_emotion_history(df_dv, window_size=args.window_size)
        test_hist = build_emotion_history(df_te, window_size=args.window_size)
        print("Emotion histories successfully generated!")
        print(f"Train history shape: {train_hist.shape}")
    except Exception as e:
        print(f"Note on features/data: {e}")
