"""
Download Assets Utility for EmoBase
===================================
Automates downloading and verifying large model checkpoints and datasets
from Hugging Face Hub or custom URLs to prevent repository bloat.
"""

import argparse
import os
import sys
from pathlib import Path

# Required model checkpoints
DEFAULT_MODELS = {
    "wav2vec_model": "facebook/wav2vec2-base",
    "face_model": "microsoft/resnet-50",
    "emocontext_model": "roberta-base",
}


def download_huggingface_model(model_name: str, target_dir: Path):
    """Downloads model weights directly from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_name} to {target_dir}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[SUCCESS] Downloaded {model_name}")
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download large model assets for EmoBase")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="loaded resorce",
        help="Target folder for pretrained models",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    target_models = base_dir / args.models_dir
    target_models.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EmoBase Asset Downloader")
    print("=" * 60)
    print(f"Target Directory: {target_models}")

    for local_name, hf_id in DEFAULT_MODELS.items():
        sub_dir = target_models / local_name
        if sub_dir.exists() and any(sub_dir.iterdir()):
            print(f"[FOUND] {local_name} already exists at {sub_dir}")
        else:
            print(f"[FETCHING] {local_name} from {hf_id}...")
            download_huggingface_model(hf_id, sub_dir)

    print("\nAssets check complete.")


if __name__ == "__main__":
    main()
