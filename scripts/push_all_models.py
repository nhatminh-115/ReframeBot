"""Upload all three ReframeBot model checkpoints to Hugging Face Hub.

Paths are read from .env so they work on any machine.

Usage:
    uv run python scripts/push_all_models.py

Required .env variables:
    HF_TOKEN          — HF write token
    SFT_ADAPTER_PATH  — local path to SFT checkpoint directory
    ADAPTER_PATH      — local path to DPO checkpoint directory
    GUARDRAIL_PATH    — local path to guardrail model directory
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

import os  # noqa: E402

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set in .env")
    sys.exit(1)

MODELS = [
    {
        "name": "SFT Adapter",
        "path": os.environ.get("SFT_ADAPTER_PATH", ""),
        "repo_id": "Nhatminh1234/ReframeBot-SFT-Llama3.1-8B",
    },
    {
        "name": "DPO Adapter",
        "path": os.environ.get("ADAPTER_PATH", ""),
        "repo_id": "Nhatminh1234/ReframeBot-DPO-Llama3.1-8B",
    },
    {
        "name": "Guardrail Classifier",
        "path": os.environ.get("GUARDRAIL_PATH", ""),
        "repo_id": "Nhatminh1234/ReframeBot-Guardrail-DistilBERT",
    },
    {
        "name": "Merged AWQ Model",
        "path": os.environ.get("AWQ_MODEL_PATH", str(_REPO_ROOT / "merged_model_awq")),
        "repo_id": "Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ",
    },
]

from huggingface_hub import HfApi 

api = HfApi(token=HF_TOKEN)


def main() -> None:
    try:
        import hf_transfer
        print("HF Transfer enabled - using high-speed Rust uploader.")
    except ImportError:
        print("hf-transfer not found - using standard uploader. (Install with 'pip install hf-transfer' for better reliability)")

    for model in MODELS:
        name, path, repo_id = model["name"], model["path"], model["repo_id"]
        if not path:
            print(f"SKIP {name}: path not set in .env")
            continue
        if not Path(path).exists():
            print(f"SKIP {name}: path does not exist — {path}")
            continue

        print(f"\nUploading {name} ...")
        print(f"   Source: {path}")
        print(f"   Target: https://huggingface.co/{repo_id}")
        
        try:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {name}",
            )
            print(f"Done: {name}")
        except Exception as exc:
            print(f"ERROR uploading {name}: {exc}")

    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()
