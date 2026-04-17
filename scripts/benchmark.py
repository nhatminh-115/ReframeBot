"""Benchmark ReframeBot API — latency, throughput, and streaming TTFT.

Fires N requests at the running server (sequential and concurrent) and
reports p50/p95/p99 latency, tokens/sec, and time-to-first-token for
the streaming endpoint.

Usage:
    # Run server first: python app.py  OR  docker compose up
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --url http://localhost:8000 --n 20 --concurrency 4

Output:
    Benchmark results table + saves benchmark_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import httpx

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

CHAT_HISTORY_TEMPLATE = [
    {"role": "user", "content": "{prompt}"},
]


@dataclass
class RequestResult:
    latency_s: float
    status_code: int
    error: str = ""


@dataclass
class StreamResult:
    ttft_s: float        # time to first token
    total_s: float
    token_count: int
    status_code: int
    error: str = ""

    @property
    def tokens_per_sec(self) -> float:
        return self.token_count / self.total_s if self.total_s > 0 else 0.0


@dataclass
class BenchmarkSummary:
    label: str
    latencies: List[float] = field(default_factory=list)
    errors: int = 0

    def add(self, r: RequestResult) -> None:
        if r.error:
            self.errors += 1
        else:
            self.latencies.append(r.latency_s)

    def report(self) -> dict:
        if not self.latencies:
            return {"label": self.label, "error": "all requests failed"}
        sorted_l = sorted(self.latencies)
        n = len(sorted_l)
        return {
            "label": self.label,
            "n_ok": n,
            "n_err": self.errors,
            "p50_s": round(statistics.median(sorted_l), 3),
            "p95_s": round(sorted_l[int(n * 0.95)], 3),
            "p99_s": round(sorted_l[min(int(n * 0.99), n - 1)], 3),
            "mean_s": round(statistics.mean(sorted_l), 3),
            "min_s": round(sorted_l[0], 3),
            "max_s": round(sorted_l[-1], 3),
        }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def post_chat(client: httpx.AsyncClient, url: str, prompt: str) -> RequestResult:
    history = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/chat",
            json={"history": history},
            timeout=120.0,
        )
        latency = time.perf_counter() - t0
        return RequestResult(latency_s=latency, status_code=resp.status_code)
    except Exception as exc:
        return RequestResult(latency_s=0, status_code=0, error=str(exc))


async def post_chat_stream(client: httpx.AsyncClient, url: str, prompt: str) -> StreamResult:
    history = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    ttft = 0.0
    token_count = 0
    first = True
    try:
        async with client.stream(
            "POST",
            f"{url}/chat/stream",
            json={"history": history},
            timeout=120.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    token = payload.get("token", "")
                    if token:
                        if first:
                            ttft = time.perf_counter() - t0
                            first = False
                        token_count += len(token.split())
                except json.JSONDecodeError:
                    pass
        total = time.perf_counter() - t0
        return StreamResult(
            ttft_s=round(ttft, 3),
            total_s=round(total, 3),
            token_count=token_count,
            status_code=resp.status_code,
        )
    except Exception as exc:
        return StreamResult(ttft_s=0, total_s=0, token_count=0, status_code=0, error=str(exc))


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

async def run_sequential(url: str, n: int) -> BenchmarkSummary:
    summary = BenchmarkSummary(label="sequential (chat)")
    async with httpx.AsyncClient() as client:
        for i in range(n):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            result = await post_chat(client, url, prompt)
            summary.add(result)
            status = f"ok ({result.latency_s:.2f}s)" if not result.error else f"err: {result.error[:40]}"
            print(f"  [{i+1}/{n}] {status}")
    return summary


async def run_concurrent(url: str, n: int, concurrency: int) -> BenchmarkSummary:
    summary = BenchmarkSummary(label=f"concurrent x{concurrency} (chat)")
    sem = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> RequestResult:
        async with sem:
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            async with httpx.AsyncClient() as client:
                return await post_chat(client, url, prompt)

    tasks = [bounded(i) for i in range(n)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    for r in results:
        summary.add(r)
    print(f"  Wall time for {n} requests @ concurrency={concurrency}: {wall:.2f}s")
    print(f"  Throughput: {n / wall:.1f} req/s")
    return summary


async def run_stream_benchmark(url: str, n: int) -> dict:
    ttfts = []
    tpss = []
    print(f"  Streaming {n} requests sequentially...")
    async with httpx.AsyncClient() as client:
        for i in range(n):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            r = await post_chat_stream(client, url, prompt)
            if not r.error:
                ttfts.append(r.ttft_s)
                tpss.append(r.tokens_per_sec)
                print(f"  [{i+1}/{n}] TTFT={r.ttft_s:.3f}s total={r.total_s:.2f}s ~{r.tokens_per_sec:.1f} tok/s")
            else:
                print(f"  [{i+1}/{n}] ERROR: {r.error[:60]}")

    if not ttfts:
        return {"error": "all stream requests failed"}

    return {
        "label": "streaming (chat/stream)",
        "n_ok": len(ttfts),
        "ttft_p50_s": round(statistics.median(ttfts), 3),
        "ttft_p95_s": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 3),
        "tokens_per_sec_mean": round(statistics.mean(tpss), 1),
        "tokens_per_sec_p50": round(statistics.median(tpss), 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(url: str, n: int, concurrency: int, skip_concurrent: bool) -> None:
    print(f"\n=== ReframeBot Benchmark ===")
    print(f"Target: {url}  |  n={n}  |  concurrency={concurrency}\n")

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, timeout=10)
            print(f"Health check: {r.status_code} {r.json()}\n")
        except Exception as exc:
            print(f"Health check FAILED — is the server running? ({exc})")
            return

    results = {}

    print("--- Sequential latency ---")
    seq = await run_sequential(url, n)
    results["sequential"] = seq.report()
    _print_table(results["sequential"])

    if not skip_concurrent:
        print(f"\n--- Concurrent throughput (concurrency={concurrency}) ---")
        conc = await run_concurrent(url, n, concurrency)
        results["concurrent"] = conc.report()
        _print_table(results["concurrent"])

    print("\n--- Streaming TTFT ---")
    stream = await run_stream_benchmark(url, min(n, 5))
    results["streaming"] = stream
    _print_table(stream)

    # Save results
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def _print_table(data: dict) -> None:
    print()
    for k, v in data.items():
        print(f"  {k:<25} {v}")
    print()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ReframeBot API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=10, help="Number of requests (default: 10)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent requests for throughput test (default: 4)",
    )
    parser.add_argument(
        "--skip-concurrent",
        action="store_true",
        help="Skip concurrent throughput test",
    )
    args = parser.parse_args()
    asyncio.run(main(args.url, args.n, args.concurrency, args.skip_concurrent))


if __name__ == "__main__":
    cli()
