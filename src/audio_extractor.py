import argparse
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, get_linear_schedule_with_warmup

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


def extract_mfcc(file_path: str, sr: int = 16000, n_mfcc: int = 40, max_len: int = 300) -> np.ndarray:
    """Extracts MFCC features from an audio file and pads/truncates to max_len."""
    try:
        y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc).T
        if mfcc.shape[0] < max_len:
            pad = np.zeros((max_len - mfcc.shape[0], n_mfcc))
            mfcc = np.vstack((mfcc, pad))
        else:
            mfcc = mfcc[:max_len, :]
        return mfcc
    except Exception:
        return np.zeros((max_len, n_mfcc))


def load_audio_waveform(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Loads a raw audio waveform."""
    try:
        y, _ = librosa.load(file_path, sr=target_sr)
        if len(y) == 0:
            return np.zeros(target_sr, dtype=np.float32)
        return y.astype(np.float32)
    except Exception:
        return np.zeros(target_sr, dtype=np.float32)


class Wav2VecAudioDataset(Dataset):
    """Dataset for training Wav2Vec2 on utterance audio."""

    def __init__(self, df: pd.DataFrame, audio_folder: str, processor, max_audio_len: int = 48000):
        self.df = df.reset_index(drop=True)
        self.audio_folder = audio_folder
        self.processor = processor
        self.max_audio_len = max_audio_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dia_id = row["Dialogue_ID"]
        utt_id = row["Utterance_ID"]
        filename = f"dia{dia_id}_utt{utt_id}.wav"
        full_path = os.path.join(self.audio_folder, filename)

        audio = load_audio_waveform(full_path)
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_audio_len,
            truncation=True,
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


def extract_wav2vec_embeddings(
    df: pd.DataFrame,
    audio_folder: str,
    processor,
    model,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Extracts mean-pooled hidden state embeddings from Wav2Vec2."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch_audio = []
            for _, row in batch_df.iterrows():
                path = os.path.join(audio_folder, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav")
                audio = load_audio_waveform(path)
                batch_audio.append(audio)

            inputs = processor(
                batch_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=48000,
                truncation=True,
            )
            input_values = inputs["input_values"].to(device)
            outputs = model(input_values=input_values)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(features)

    return np.vstack(all_embeddings)
