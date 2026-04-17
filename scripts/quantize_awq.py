"""Quantize merged model to AWQ 4-bit for vLLM serving.

AWQ (Activation-aware Weight Quantization) is the recommended format for
vLLM — better throughput than bitsandbytes, first-class vLLM support.

Run after export_merged_model.py:
    uv run python scripts/quantize_awq.py \
        --input ./merged_model \
        --output ./merged_model_awq

Requirements:
    pip install autoawq        (not in pyproject.toml — install manually)
    GPU with sufficient VRAM to load the model for calibration (~8 GB)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_calib_data(calib_data_path: str | None, n_samples: int) -> list[str]:
    """Load calibration strings from local JSONL or fall back to pile-val-backup."""
    if calib_data_path and Path(calib_data_path).exists():
        import json
        samples = []
        with open(calib_data_path) as f:
            for line in f:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                text = " ".join(m["content"] for m in msgs if m.get("content"))
                if text.strip():
                    samples.append(text)
                if len(samples) >= n_samples:
                    break
        logger.info("Using %d local calibration samples from %s", len(samples), calib_data_path)
        return samples
    logger.info("No local calib data — using pile-val-backup (requires download)")
    return "pileval"  # type: ignore[return-value]  # AutoAWQ accepts dataset name string


def quantize(
    input_dir: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    calib_data_path: str | None = None,
    n_samples: int = 128,
) -> None:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        logger.error(
            "autoawq not installed. Run: pip install autoawq\n"
            "Note: install inside WSL2 or Linux environment if on Windows."
        )
        raise SystemExit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    logger.info("Loading model from %s for calibration", input_dir)
    t0 = time.perf_counter()
    model = AutoAWQForCausalLM.from_pretrained(
        input_dir,
        safetensors=True,
        max_memory={0: "7500MiB"},
    )
    tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
    logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

    calib_data = _load_calib_data(calib_data_path, n_samples)

    logger.info("Running AWQ quantization (bits=%d, group_size=%d) — ~5-15 min", bits, group_size)
    t1 = time.perf_counter()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    logger.info("Quantization done in %.1fs", time.perf_counter() - t1)

    logger.info("Moving model to CPU and clearing VRAM before saving")
    import gc
    model.model = model.model.cpu()
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Saving AWQ model to %s", out)
    model.save_quantized(str(out), safetensors=True)
    tokenizer.save_pretrained(str(out))

    total = time.perf_counter() - t0
    logger.info("Done — total %.1fs. AWQ model at: %s", total, out.resolve())
    logger.info(
        "Next step: point MERGED_MODEL_PATH=%s in .env then run docker compose up",
        out.resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ quantize merged ReframeBot model")
    parser.add_argument("--input", required=True, help="Path to merged bf16 model (from export_merged_model.py)")
    parser.add_argument("--output", default="./merged_model_awq", help="Output directory for AWQ model")
    parser.add_argument("--bits", type=int, default=4, choices=[4], help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument(
        "--calib-data",
        default=str(Path(__file__).resolve().parents[1] / "data" / "dataset.jsonl"),
        help="Path to local JSONL calibration data (default: data/dataset.jsonl)",
    )
    parser.add_argument("--n-samples", type=int, default=32, help="Number of calibration samples (default: 32)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error("Input model not found at '%s'. Run export_merged_model.py first.", args.input)
        raise SystemExit(1)

    quantize(args.input, args.output, bits=args.bits, group_size=args.group_size,
             calib_data_path=args.calib_data, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
