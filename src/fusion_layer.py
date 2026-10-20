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


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Safely loads an audio file or returns a 3-second zero-padding on error."""
    try:
        if os.path.exists(file_path):
            y, _ = librosa.load(file_path, sr=target_sr)
            if len(y) > 0:
                return y.astype(np.float32)
    except Exception:
        pass
    return np.zeros(target_sr * 3, dtype=np.float32)


def extract_middle_frame(video_path: str) -> Image.Image:
    """Safely extracts the middle frame from an utterance video file."""
    try:
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                middle_frame = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame)
            cap.release()
    except Exception:
        pass
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def get_utterance_paths(dia_id: int, utt_id: int, audio_folder: str, video_folder: str) -> Tuple[str, str]:
    filename = f"dia{dia_id}_utt{utt_id}"
    audio_path = os.path.join(audio_folder, f"{filename}.wav")
    video_path = os.path.join(video_folder, f"{filename}.mp4")
    return audio_path, video_path


class MultimodalMELDDataset(Dataset):
    """Loads text utterance, audio wav, and video middle frame for MELD on the fly."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_folder: str,
        video_folder: str,
        text_tokenizer,
        audio_processor,
        image_processor,
        max_text_len: int = 128,
        max_audio_len: int = 48000,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_folder = audio_folder
        self.video_folder = video_folder
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dia_id = row["Dialogue_ID"]
        utt_id = row["Utterance_ID"]
        label = torch.tensor(row["label"], dtype=torch.long)

        # 1. Text tokenization
        text = str(row["Utterance"])
        text_inputs = self.text_tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 2. Audio processing
        audio_path, video_path = get_utterance_paths(
            dia_id, utt_id, self.audio_folder, self.video_folder
        )
        audio = load_audio(audio_path)
        audio_inputs = self.audio_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_audio_len,
            truncation=True,
        )

        # 3. Vision processing
        frame = extract_middle_frame(video_path)
        image_inputs = self.image_processor(images=frame, return_tensors="pt")

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "input_values": audio_inputs["input_values"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "labels": label,
        }


class OnlineFusionModel(nn.Module):
    """Fuses pretrained encoders for Text, Audio, and Face with linear projections."""

    def __init__(
        self,
        text_model,
        audio_model,
        image_model,
        hidden_proj_dim: int = 128,
        num_classes: int = 7,
    ):
        super(OnlineFusionModel, self).__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.image_model = image_model

        text_dim = getattr(text_model.config, "hidden_size", 768)
        audio_dim = getattr(audio_model.config, "hidden_size", 768)
        image_dim = getattr(image_model.config, "hidden_size", 2048)

        self.text_fc = nn.Linear(text_dim, hidden_proj_dim)
        self.audio_fc = nn.Linear(audio_dim, hidden_proj_dim)
        self.image_fc = nn.Linear(image_dim, hidden_proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_proj_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, input_values, pixel_values):
        t_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        t_feat = t_out.last_hidden_state[:, 0, :]  # CLS token

        a_out = self.audio_model(input_values=input_values)
        a_feat = a_out.last_hidden_state.mean(dim=1)

        img_out = self.image_model(pixel_values=pixel_values)
        img_feat = (
            img_out.pooler_output
            if hasattr(img_out, "pooler_output") and img_out.pooler_output is not None
            else img_out.last_hidden_state.mean(dim=1)
        )

        t_proj = torch.relu(self.text_fc(t_feat))
        a_proj = torch.relu(self.audio_fc(a_feat))
        i_proj = torch.relu(self.image_fc(img_feat))

        fused = torch.cat([t_proj, a_proj, i_proj], dim=1)
        return self.classifier(fused)
