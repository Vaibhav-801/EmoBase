import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification


def extract_middle_frame(video_path: str) -> Image.Image:
    """Extracts the middle video frame representing the speaker's facial expression."""
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


def load_images_paths(root_dir: str) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Crawls class folders for facial emotion images (e.g., FER2013)."""
    image_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_folder = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(cls_folder, fname))
                labels.append(class_to_idx[cls_name])
    return image_paths, labels, class_to_idx


class FaceImageDataset(Dataset):
    """PyTorch Dataset for standalone facial emotion images."""

    def __init__(self, image_paths: List[str], labels: List[int], processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def extract_face_embeddings(
    df: pd.DataFrame,
    video_folder: str,
    processor,
    model,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extracts 2048-dim feature embeddings from video clips using ResNet/ViT."""
    model.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            frames = []
            for _, row in batch_df.iterrows():
                v_path = os.path.join(video_folder, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
                frames.append(extract_middle_frame(v_path))

            inputs = processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output.squeeze(-1).squeeze(-1).cpu().numpy()
            else:
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            all_features.append(features)

    return np.vstack(all_features)
