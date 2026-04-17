"""Export merged model to disk for vLLM serving.

Loads base model on CPU (avoids VRAM constraints), merges the DPO
adapter, and saves the result in bf16.  Run this once before building
the Docker image or starting vLLM.

Usage:
    uv run python scripts/export_merged_model.py --output ./merged_model

Requirements:
    ~16 GB system RAM (bf16 Llama 3.1 8B on CPU)
    No GPU required.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _download_model(repo_id: str) -> str:
    """Download model into the default HF cache (resumes across runs)."""
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s into HF cache (resumes if interrupted)", repo_id)
    # Ignore .bin shards — repo has both .bin and .safetensors (same weights, twice the size)
    return snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.bin", "*.pt", "original/*"],
    )


def export(base_model_name: str, adapter_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # If base_model_name is a HF repo ID, download explicitly so we get resume support
    if not Path(base_model_name).exists():
        base_model_name = _download_model(base_model_name)

    logger.info("Loading tokenizer from %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model on CPU in bf16 — this may take a few minutes")
    t0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    logger.info("Base model loaded in %.1fs", time.perf_counter() - t0)

    logger.info("Merging DPO adapter from %s", adapter_path)
    t1 = time.perf_counter()
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    logger.info("Adapter merged in %.1fs", time.perf_counter() - t1)

    logger.info("Saving merged model to %s", out)
    t2 = time.perf_counter()
    import gc
    gc.collect()
    # max_shard_size: serialize 2GB at a time instead of the full 16GB in one shot
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(out))
    logger.info("Saved in %.1fs", time.perf_counter() - t2)

    total = time.perf_counter() - t0
    logger.info("Done — total %.1fs. Model at: %s", total, out.resolve())


def _resolve_adapter(adapter: str) -> str:
    """Return a local path to the adapter, downloading from HF if needed."""
    p = Path(adapter)
    if p.exists():
        return str(p)

    # Looks like a HF repo ID (contains '/' but is not an absolute path)
    if "/" in adapter and not p.is_absolute():
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
            raise SystemExit(1)
        logger.info("Downloading adapter from HF: %s", adapter)
        local = snapshot_download(repo_id=adapter)
        logger.info("Downloaded to: %s", local)
        return local

    logger.error("Adapter not found at '%s'", adapter)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export merged ReframeBot model")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF model ID or local path (default: meta-llama/Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--adapter",
        default="Nhatminh1234/ReframeBot-DPO-Llama3.1-8B",
        help="Local path or HF repo ID of the DPO adapter "
             "(default: Nhatminh1234/ReframeBot-DPO-Llama3.1-8B)",
    )
    parser.add_argument(
        "--output",
        default="./merged_model",
        help="Output directory (default: ./merged_model)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for gated models (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.hf_token or __import__("os").environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face")

    adapter_path = _resolve_adapter(args.adapter)
    export(args.base_model, adapter_path, args.output)


if __name__ == "__main__":
    main()
