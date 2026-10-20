"""
EmoBase: Multimodal Emotion Recognition Framework
Combines Text (RoBERTa), Audio (Wav2Vec2/MFCC), and Facial Vision (ResNet/ViT)
with Cross-Modal Attention and Conversational Context Memory.
"""

from .advanced_model import CrossAttentionBlock, SOTAFusionModel
from .datasets import (
    EMOTION_TO_LABEL,
    LABEL_TO_EMOTION,
    EmoContextDataset,
    build_emocontext_dataloaders,
    load_emocontext_splits,
)

__all__ = [
    "CrossAttentionBlock",
    "SOTAFusionModel",
    "EMOTION_TO_LABEL",
    "LABEL_TO_EMOTION",
    "EmoContextDataset",
    "build_emocontext_dataloaders",
    "load_emocontext_splits",
]
