import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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


class TextEmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128, text_col: str = "Utterance"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = row["label"]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_text_classifier(
    train_csv: str,
    dev_csv: str,
    test_csv: str,
    model_name: str = "roberta-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    save_path: Optional[str] = None,
):
    df_train = pd.read_csv(train_csv)
    df_dev = pd.read_csv(dev_csv)
    df_test = pd.read_csv(test_csv)

    for df in [df_train, df_dev, df_test]:
        df["label"] = df["Emotion"].str.lower().map(LABEL_MAP)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABEL_MAP)
    )

    train_ds = TextEmotionDataset(df_train, tokenizer)
    dev_ds = TextEmotionDataset(df_dev, tokenizer)
    test_ds = TextEmotionDataset(df_test, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL_MAP)),
        y=df_train["label"].values,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    if save_path:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved model to {save_path}")
