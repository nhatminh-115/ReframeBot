"""Benchmark base+DPO model inference via in-process NF4 loading.

Loads the merged DPO model (./merged_model) with bitsandbytes NF4 4-bit
quantization and measures the same metrics as benchmark.py:
  - Sequential latency (p50, p95, p99)
  - Time to first token (TTFT) via streaming generation
  - Tokens per second

Usage:
    uv run python scripts/benchmark_inprocess.py
    uv run python scripts/benchmark_inprocess.py --model ./merged_model --n 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

_REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PROMPTS = [
    "I failed my midterm and I feel like I'm not smart enough for university.",
    "I can't stop procrastinating and now I'm 3 weeks behind on assignments.",
    "My professor gave me a C and I worked so hard. I feel like giving up.",
    "I'm so anxious before every exam that I blank out even when I studied.",
    "Everyone else seems to understand the material but I have to study twice as hard.",
    "I missed a deadline and now I don't know if I can pass the course.",
    "I feel stupid compared to my classmates. They all get it and I don't.",
    "I'm overwhelmed with 4 assignments due this week and I don't know where to start.",
]

SYSTEM_PROMPT = "You are ReframeBot, helping students with academic stress using CBT."


def load_model(base: str, adapter: str | None = None):
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if adapter:
        from peft import PeftModel
        print(f"Loading base model {base} in NF4 ...")
        tokenizer = AutoTokenizer.from_pretrained(base)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map={"": 0}
        )
        print(f"Applying DPO adapter {adapter} ...")
        model = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    else:
        print(f"Loading merged model {base} in NF4 ...")
        tokenizer = AutoTokenizer.from_pretrained(base)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map={"": 0}, low_cpu_mem_usage=True,
        )

    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def build_prompt(tokenizer, user_message: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt").to("cuda")


def generate_latency(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> tuple[float, int]:
    """Returns (latency_s, n_tokens_generated)."""
    inputs = build_prompt(tokenizer, prompt)
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    n_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    return latency, n_tokens


def generate_ttft(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> tuple[float, float, int]:
    """Returns (ttft_s, total_s, n_tokens) using streaming generation."""
    inputs = build_prompt(tokenizer, prompt)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    t0 = time.perf_counter()
    thread.start()

    ttft = 0.0
    n_tokens = 0
    first = True
    for token_text in streamer:
        if first and token_text.strip():
            ttft = time.perf_counter() - t0
            first = False
        n_tokens += len(token_text.split())

    thread.join()
    total = time.perf_counter() - t0
    return ttft, total, n_tokens


def run_benchmark(model, tokenizer, n: int) -> dict:
    latencies: List[float] = []
    token_counts: List[int] = []
    ttfts: List[float] = []
    tps_list: List[float] = []

    print(f"--- Sequential latency ({n} requests) ---")
    for i in tqdm(range(n)):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        lat, ntok = generate_latency(model, tokenizer, prompt)
        latencies.append(lat)
        token_counts.append(ntok)
        tps_list.append(ntok / lat if lat > 0 else 0)
        print(f"  [{i+1}/{n}] {lat:.2f}s  {ntok} tokens  {ntok/lat:.1f} tok/s")

    print(f"\n--- TTFT (5 requests) ---")
    for i in range(min(5, n)):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        ttft, total, ntok = generate_ttft(model, tokenizer, prompt)
        ttfts.append(ttft)
        print(f"  [{i+1}] TTFT={ttft:.3f}s  total={total:.2f}s  ~{ntok} words")

    sorted_lat = sorted(latencies)
    k = len(sorted_lat)
    results = {
        "model": "base+DPO (NF4, in-process)",
        "n": n,
        "latency_p50_s": round(statistics.median(sorted_lat), 3),
        "latency_p95_s": round(sorted_lat[int(k * 0.95)], 3),
        "latency_p99_s": round(sorted_lat[min(int(k * 0.99), k - 1)], 3),
        "latency_mean_s": round(statistics.mean(sorted_lat), 3),
        "tokens_per_sec_mean": round(statistics.mean(tps_list), 1),
        "tokens_per_sec_p50": round(statistics.median(tps_list), 1),
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_p95_s": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 3) if ttfts else None,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base model HF ID or local path")
    parser.add_argument("--adapter", default="Nhatminh1234/ReframeBot-DPO-Llama3.1-8B", help="DPO adapter HF ID or local path (leave empty to load merged model from --base)")
    parser.add_argument("--n", type=int, default=10, help="Number of requests")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base, args.adapter or None)
    results = run_benchmark(model, tokenizer, args.n)
    results["model"] = f"{args.base} + {args.adapter} (NF4)" if args.adapter else f"{args.base} (NF4)"

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k:<30} {v}")

    out = _REPO_ROOT / "benchmark_inprocess_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
