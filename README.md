# EmoBase: Multimodal Conversational Emotion Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for **Multimodal Conversational Emotion Recognition** across Text, Audio, and Facial Video modalities using Cross-Modal Attention and Conversational Context Memory (GRU). Evaluated on the **MELD** and **EmoContext** benchmarks.

---

## 🏛️ Architecture Overview

The system processes multimodal dialogue turns through a 4-stage pipeline:

```
                  ┌──────────────────────┐
   Text Turn  ───►│  RoBERTa Encoder     ├───► Text Embedding (768)   ─┐
                  └──────────────────────┘                             │
                  ┌──────────────────────┐                             ▼
   Audio Wave ───►│  Wav2Vec 2.0 / MFCC  ├───► Audio Embedding (768) ──┼──► Cross-Attention Fusion
                  └──────────────────────┘                             ▲    (Text attends to Audio & Face)
                  ┌──────────────────────┐                             │               │
   Video Face ───►│  ResNet-50 / ViT     ├───► Face Embedding (2048) ─┘               ▼
                  └──────────────────────┘                                 Conversational GRU Memory
                                                                               (Dialogue Context)
                                                                                       │
                                                                                       ▼
                                                                           Emotion Classifier (7 Classes)
```

1. **Unimodal Feature Encoders:**
   - **Text:** Fine-tuned `roberta-base` extracting contextual sentence representations.
   - **Audio:** `facebook/wav2vec2-base` extracting temporal acoustic features + MFCC baselines.
   - **Vision:** `microsoft/resnet-50` extracting facial affective features from dialogue video middle frames.
2. **Cross-Modal Attention (`CrossAttentionBlock`):**
   - Enables text queries to attend directly to audio and facial visual cues to disambiguate sarcasm, stress, and subtle emotional undertones.
3. **Conversational Context Memory (GRU):**
   - Recurrent memory tracking emotional state transitions across sequential dialogue turns (`Dialogue_ID`, `Utterance_ID`).
4. **Classification Head:**
   - Softmax classifier with balanced class weighting predicting 7 emotion categories (`neutral`, `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`).

---

## 📁 Repository Structure

```
EmoBase/
├── .gitignore                    # Comprehensive rules excluding >100MB weights & data
├── README.md                     # Architecture, documentation, and usage guide
├── requirements.txt              # Core dependencies
│
├── src/                          # Modular Python source modules
│   ├── __init__.py               # Package initializer
│   ├── advanced_model.py         # CrossAttentionBlock, Conversational GRU & SOTAFusionModel
│   ├── datasets.py               # EmoContext and MELD dataset loaders & label mappings
│   ├── fusion_layer.py           # On-the-fly multimodal feature loader and classifier
│   ├── fusion_layer2.py          # Pre-extracted features fusion training & evaluation
│   ├── context_memory.py         # Temporal dialogue history builder & sequential modeling
│   ├── audio_extractor.py        # MFCC & Wav2Vec2 audio feature extraction
│   ├── face_extractor.py         # Video middle-frame extractor & ResNet visual pipeline
│   └── text_classifier.py        # RoBERTa dialogue text fine-tuning
│
├── results/                      # Benchmark reports and predictions
│   ├── results.txt               # Evaluation metrics (accuracy, precision, recall, F1)
│   └── predictions.csv           # Model inference outputs
│
├── scripts/                      # Utility scripts
│   └── download_assets.py        # Automated downloader for model weights & datasets
│
├── docs/                         # Documentation & technical reports
│   └── generate_pdf.py           # PDF report generator
│
├── Data/                         # Dataset splits and annotations (ignored from Git)
└── features/                     # Pre-extracted numpy embeddings (ignored from Git)
```

---

## 💾 Managing Large Data & Model Weights

Git and GitHub enforce a **100 MB per-file hard limit** and are not designed to host multi-gigabyte binary datasets or model checkpoints.

This repository follows standard machine learning best practices:
1. **Source Code & Pipelines** are tracked directly in Git (`< 50 MB` total repository size).
2. **Model Checkpoints** (`model.safetensors` ~500 MB each) and **Raw Media** (MELD audio/video ~630 MB) are kept outside Git and downloaded on demand:

### Automatic Weight Download
To download all required Hugging Face backbone models:
```bash
python scripts/download_assets.py
```

### Alternatively: Git LFS (Large File Storage)
If your organization requires binary checkpoints inside the repository, initialize Git LFS:
```bash
git lfs install
git lfs track "*.safetensors" "*.npy" "*.pth"
git add .gitattributes
```
*(Note: Requires GitHub LFS data pack if binary storage exceeds the 2 GB free quota).*

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/your-username/EmoBase.git
cd EmoBase
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training the Fusion Model (Precomputed Embeddings)
```bash
python src/fusion_layer2.py --epochs 60 --batch-size 64 --lr 1e-5
```

### 3. Evaluating Dialogue Context Memory
```bash
python src/context_memory.py --window-size 5
```

### 4. Running the SOTA Cross-Attention Architecture
```python
from src.advanced_model import SOTAFusionModel
import torch

model = SOTAFusionModel(text_dim=768, audio_dim=768, face_dim=2048)
text_feat = torch.randn(4, 768)
audio_feat = torch.randn(4, 768)
face_feat = torch.randn(4, 2048)

logits, hidden_state = model(text_feat, audio_feat, face_feat)
print("Predicted emotion logits:", logits.shape)
```

---

## 📊 Benchmark Results

Evaluated on the 3-turn conversational test benchmark:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **80.0%** |
| **Weighted F1** | **0.83** |
| **Macro F1** | **0.61** |

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
