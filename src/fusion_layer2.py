import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import librosa
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup,
)

LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "disgust": 6,
}
TARGET_NAMES = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


class FusionFeaturesDataset(Dataset):
    """Dataset loading pre-extracted numpy embeddings for Text, Audio, and Face."""

    def __init__(
        self,
        text_features: np.ndarray,
        audio_features: np.ndarray,
        face_features: np.ndarray,
        labels: np.ndarray,
    ):
        self.text = torch.tensor(text_features, dtype=torch.float32)
        self.audio = torch.tensor(audio_features, dtype=torch.float32)
        self.face = torch.tensor(face_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "face": self.face[idx],
            "labels": self.labels[idx],
        }


class SimpleFusionModel(nn.Module):
    """Linear projection fusion network with Dropout and ReLU activations."""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        face_dim: int = 2048,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 7,
    ):
        super(SimpleFusionModel, self).__init__()
        self.text_fc = nn.Linear(text_dim, proj_dim)
        self.audio_fc = nn.Linear(audio_dim, proj_dim)
        self.face_fc = nn.Linear(face_dim, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        text_out = torch.relu(self.text_fc(text))
        audio_out = torch.relu(self.audio_fc(audio))
        face_out = torch.relu(self.face_fc(face))
        combined = torch.cat([text_out, audio_out, face_out], dim=1)
        return self.classifier(combined)


def train_fusion_model(
    features_dir: Optional[str] = None,
    annotations_dir: Optional[str] = None,
    output_model_path: Optional[str] = None,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-5,
):
    base_dir = get_base_dir()
    f_dir = Path(features_dir) if features_dir else base_dir / "features"
    d_dir = Path(annotations_dir) if annotations_dir else base_dir / "Data"

    print(f"Loading pre-extracted features from: {f_dir}")
    train_text = np.load(f_dir / "train_text.npy")
    train_audio = np.load(f_dir / "train_audio.npy")
    train_face = np.load(f_dir / "train_face.npy")

    dev_text = np.load(f_dir / "dev_text.npy")
    dev_audio = np.load(f_dir / "dev_audio.npy")
    dev_face = np.load(f_dir / "dev_face.npy")

    test_text = np.load(f_dir / "test_text.npy")
    test_audio = np.load(f_dir / "test_audio.npy")
    test_face = np.load(f_dir / "test_face.npy")

    # Load labels
    meld_dir = d_dir / "MELD.Raw"
    train_csv = meld_dir / "train" / "train_sent_emo.csv" if (meld_dir / "train" / "train_sent_emo.csv").exists() else d_dir / "train_sent_emo.csv"
    dev_csv = meld_dir / "dev" / "dev_sent_emo.csv" if (meld_dir / "dev" / "dev_sent_emo.csv").exists() else d_dir / "dev_sent_emo.csv"
    test_csv = meld_dir / "test" / "test_sent_emo.csv" if (meld_dir / "test" / "test_sent_emo.csv").exists() else d_dir / "test_sent_emo.csv"

    df_train = pd.read_csv(train_csv)
    df_dev = pd.read_csv(dev_csv)
    df_test = pd.read_csv(test_csv)

    train_labels = df_train["Emotion"].map(LABEL_MAP).values
    dev_labels = df_dev["Emotion"].map(LABEL_MAP).values
    test_labels = df_test["Emotion"].map(LABEL_MAP).values

    train_dataset = FusionFeaturesDataset(train_text, train_audio, train_face, train_labels)
    dev_dataset = FusionFeaturesDataset(dev_text, dev_audio, dev_face, dev_labels)
    test_dataset = FusionFeaturesDataset(test_text, test_audio, test_face, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = SimpleFusionModel(
        text_dim=train_text.shape[1],
        audio_dim=train_audio.shape[1],
        face_dim=train_face.shape[1],
    ).to(device)

    # Balanced class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(7),
        y=train_labels,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print("Beginning training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(text, audio, face)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # Save model
    save_path = Path(output_model_path) if output_model_path else f_dir / "fusion_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"Model saved to {save_path}")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(text, audio, face)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nTest Set Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=TARGET_NAMES)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fusion model from pre-extracted features")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    train_fusion_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
