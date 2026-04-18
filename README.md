# ReframeBot

ReframeBot is a CBT-oriented chatbot for supporting university students with academic stress. It combines a fine-tuned Llama 3.1 model with a guardrail router (TASK_1/TASK_2/TASK_3) and optional RAG grounding from a CBT knowledge base.

> For full training details, hyperparameters, and per-class metrics, see [MODEL_CARD.md](MODEL_CARD.md).

## Model Repositories

| Model | Repository | Use |
|---|---|---|
| AWQ Model | [ReframeBot-Llama3.1-8B-AWQ](https://huggingface.co/Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ) | **Inference** — merged + AWQ 4-bit, served by vLLM |
| Guardrail Classifier | [ReframeBot-Guardrail-DistilBERT](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT) | **Inference** — 3-class task router (CBT / Crisis / Out-of-scope) |
| DPO Adapter | [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B) | Training artifact — LoRA adapter before merging |
| SFT Adapter | [ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B) | Training artifact — intermediate SFT checkpoint |

The API container image is published on Docker Hub:

| Image | Repository |
|---|---|
| API container | [nhatminh115/reframebot-api](https://hub.docker.com/r/nhatminh115/reframebot-api) |

## Features
- Fine-tuned Llama 3.1 8B (SFT + DPO adapter, merged and served via vLLM)
- AWQ 4-bit quantization (autoawq) — runs on 8 GB VRAM
- vLLM serving with PagedAttention and continuous batching
- Guardrail routing with crisis detection and out-of-scope redirection
- Optional RAG grounding over a CBT knowledge base
- Dockerized stack (vLLM container + FastAPI container) and a lightweight static web UI

## Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU with 8 GB+ VRAM
- 32 GB RAM (for model export step)
- Docker Desktop with NVIDIA Container Toolkit
- WSL2 (for AWQ quantization step)

### Option A — Docker (recommended)

All models are pre-built and hosted on Hugging Face / Docker Hub — no training or quantization required.

1. Clone and configure:
```bash
git clone https://github.com/minhnghiem32131024429/ReframeBot.git
cd ReframeBot
cp .env.example .env
# Set HF_TOKEN in .env (required to download the AWQ model from HF)
```

2. Download models:
```bash
# AWQ model (~4 GB, served by vLLM)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ', local_dir='./merged_model_awq')
"

# Guardrail classifier (~250 MB, runs on CPU)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Nhatminh1234/ReframeBot-Guardrail-DistilBERT', local_dir='./guardrail_model_retrained/best')
"
```

3. Start the stack (pulls API image from Docker Hub automatically):
```bash
docker compose up
```

4. Serve the web UI:
```bash
cd web && python -m http.server 8080
```
Open: http://localhost:8080/

### Option B — In-process (no Docker)

```bash
pip install -e ".[inprocess]"
cp .env.example .env
# Set ADAPTER_PATH and GUARDRAIL_PATH in .env
python app.py
```
Note: this path uses the original transformers/PEFT in-process loading without vLLM.

### Building from source (advanced)

If you want to rebuild the AWQ model yourself from the DPO adapter:
```bash
# Step 1: merge base model + DPO adapter → bf16 safetensors (~16 GB RAM, no GPU needed)
uv run python scripts/export_merged_model.py --output ./merged_model

# Step 2: AWQ 4-bit quantization (requires GPU, run in WSL2)
# In WSL2: pip install autoawq
python scripts/quantize_awq.py --input ./merged_model --output ./merged_model_awq
```

## Project Structure

```
ReframeBot/
├── app.py                      # Entry point: python app.py
├── docker-compose.yml          # vLLM + API containers
├── docker/api.Dockerfile       # FastAPI container image
├── pyproject.toml              # Dependencies (runtime / inprocess / scripts / train)
├── train.ipynb                 # Training notebook (SFT + DPO + Guardrail)
├── src/
│   └── reframebot/
│       ├── config.py           # All settings via pydantic-settings + .env
│       ├── constants.py        # Hotlines, keywords, regex, prototype sentences
│       ├── router.py           # Task routing logic (TASK_1/2/3 priority chain)
│       ├── main.py             # FastAPI app, lifespan, /chat + /chat/stream endpoints
│       └── services/
│           ├── guardrail.py    # Guardrail classifier + crisis detection
│           ├── rag.py          # ChromaDB retrieval
│           └── llm.py          # vLLM client (OpenAI-compatible)
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   ├── dataset.jsonl           # SFT training data
│   ├── dataset_dpo.jsonl       # DPO training data
│   └── guardrail_dataset.jsonl
├── scripts/
│   ├── export_merged_model.py  # Merge base + DPO adapter → bf16 safetensors
│   ├── quantize_awq.py         # AWQ 4-bit quantization (run in WSL2/Linux)
│   ├── benchmark.py            # Latency / throughput / TTFT benchmark (vLLM)
│   ├── benchmark_inprocess.py  # Latency / throughput / TTFT benchmark (NF4)
│   ├── build_rag_db.py         # Build ChromaDB from knowledge.txt
│   ├── train_guardrail.py      # Retrain guardrail classifier
│   ├── evaluate_model.py       # Evaluation + metrics (--mode inprocess|vllm)
│   ├── push_all_models.py      # Upload all models to HF Hub
│   └── push_model_cards.py     # Sync model cards to HF Hub
├── docs/
│   └── SETUP.md
└── Utils/                      # Background audio/image assets
```

## UI

- Glassmorphism-style layout (HTML/CSS)
- Responsive chat UI

## Configuration

### Change API URL
Edit `web/script.js`:
```javascript
const API_URL = "http://your-domain.com/chat";
```

### All configuration via `.env`
Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `ADAPTER_PATH` | — | Path to DPO adapter checkpoint (required) |
| `GUARDRAIL_PATH` | auto-discover | Path to guardrail model directory |
| `BASE_MODEL_NAME` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HF model ID or local path |
| `RAG_DB_PATH` | `./rag_db` | ChromaDB directory |
| `GUARDRAIL_CONTEXT_TURNS` | `3` | Recent user turns fed to the classifier |
| `CRISIS_CONFIDENCE_THRESHOLD` | `0.90` | Guardrail score above which TASK_2 is high-confidence |
| `CRISIS_SEMANTIC_SIM_THRESHOLD` | `0.62` | Cosine sim threshold for semantic crisis detection |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins |

### Customize Colors
Edit `web/style.css` to change color scheme, glass effects, and more.

## Serving Architecture

```
User request
     ↓
FastAPI container  (port 8000)
  ├─ Guardrail classifier (DistilBERT, CPU)
  ├─ Crisis detector (regex + semantic similarity)
  ├─ RAG retrieval (ChromaDB)
  └─ HTTP → vLLM container (OpenAI-compatible API)
               ↓
         vLLM container  (port 8001)
           AWQ-Marlin 4-bit Llama 3.1 8B
           PagedAttention + continuous batching
```

The LLM is served as a separate vLLM process — the FastAPI app calls it like an external service via the OpenAI client. This separates inference infrastructure from application logic and enables concurrent request batching.

## Model Sizes

| Checkpoint | Format | Disk size |
|---|---|---|
| Merged model (base + DPO adapter) | bf16 safetensors | 15 GB |
| AWQ quantized (served by vLLM) | AWQ 4-bit | 5.4 GB |
| Guardrail classifier | DistilBERT fp32 | ~250 MB |

The bf16 merged model (15 GB) exceeds the 8 GB VRAM of the development GPU and cannot be served unquantized on this hardware. AWQ 4-bit quantization reduces the footprint to 5.4 GB (2.8x compression), enabling deployment on a consumer 8 GB card.

## Inference Performance

Measured on NVIDIA RTX 5070 (8 GB VRAM):

| Metric | AWQ 4-bit (vLLM) | Base + DPO (NF4, in-process) | Speedup |
|---|---|---|---|
| Latency p50 | 3.3s | 106.8s | **32x** |
| Latency p95 | 5.9s | 124.1s | **21x** |
| Time to First Token (TTFT) p50 | 1.09s | 12.3s | **11x** |
| Tokens/sec | ~54 tok/s | ~2.1 tok/s | **26x** |
| Throughput (4 concurrent) | 1.1 req/s | — | — |
| VRAM usage at runtime | ~5.4 GB dedicated | ~8 GB dedicated + ~7 GB shared (system RAM) | — |

AWQ + vLLM (PagedAttention, continuous batching, Marlin kernel) delivers 26–32x faster inference vs in-process NF4 loading. The NF4 path spills into shared VRAM (system RAM) on Windows, has no kernel optimization or batching, and is suitable for evaluation and offline use only.

Cold-start latency (~115s first request on vLLM) is due to CUDA kernel compilation; subsequent requests are warm.

To reproduce:
```bash
# AWQ via vLLM
docker compose up -d vllm
uv run python scripts/benchmark.py --n 20 --concurrency 4

# Base+DPO NF4 in-process
python scripts/benchmark_inprocess.py --n 10
```

## Evaluation Results

All metrics were measured on a held-out test set not seen during training.

**Guardrail classifier** (same model regardless of LLM serving mode):

| Metric | Score | Notes |
|---|---|---|
| Accuracy (held-out test set) | **93.3%** | Separate harder evaluation set |
| Accuracy (validation split) | **99.0%** | 20% split, 335 samples |
| F1 macro (validation split) | **0.99** | |

**LLM quality** (varies by serving mode):

| Metric | AWQ 4-bit (vLLM) | Base + DPO (NF4) |
|---|---|---|
| BERTScore Relevance | **0.865** | 0.832 |
| BERTScore Faithfulness | **0.858** | 0.849 |
| Response Consistency | **0.775** | 0.732 |

AWQ quantization does not degrade quality — all LLM metrics are equal to or better than the NF4 baseline. Faithfulness > Relevance in both modes suggests the model grounds well in retrieved CBT context when RAG is active.

Full report: [`evaluation_report.json`](evaluation_report.json) | Radar chart: [`evaluation_summary.png`](evaluation_summary.png)

To reproduce:
```bash
docker compose up -d vllm && uv run python scripts/evaluate_model.py --mode vllm
uv run python scripts/evaluate_model.py --mode inprocess
```

## Training

See `train.ipynb` for the complete training pipeline:
1. **SFT (Supervised Fine-Tuning)** - Base model adaptation
2. **DPO (Direct Preference Optimization)** - Response quality improvement
3. **Guardrail Training** - Task classification model

Optional scripts:
- `scripts/train_guardrail.py`: retrain the guardrail classifier from a JSONL dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Author

**Nghiem Nhat Minh**
- GitHub: [@nhatminh115](https://github.com/nhatminh-115)
- Hugging Face: [@Nhatminh1234](https://huggingface.co/Nhatminh1234)

## Acknowledgments

- Meta AI for Llama 3.1
- Hugging Face for transformers and PEFT libraries
- FastAPI team
