# ReframeBot

ReframeBot is a CBT-oriented chatbot for supporting university students with academic stress. It combines a fine-tuned Llama 3.1 model with a guardrail router (TASK_1/TASK_2/TASK_3) and optional RAG grounding from a CBT knowledge base.

> For full training details, hyperparameters, and per-class metrics, see [MODEL_CARD.md](MODEL_CARD.md).

## Model Repositories

All models are available on Hugging Face:

| Model | Repository | Description |
|---|---|---|
| AWQ Model | [ReframeBot-Llama3.1-8B-AWQ](https://huggingface.co/Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ) | Merged + AWQ 4-bit quantized — ready for vLLM serving |
| DPO Adapter | [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B) | Direct Preference Optimization LoRA adapter |
| SFT Adapter | [ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B) | Supervised Fine-Tuning LoRA adapter |
| Guardrail Classifier | [ReframeBot-Guardrail-DistilBERT](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT) | 3-class task router (CBT / Crisis / Out-of-scope) |

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

1. Clone and configure:
```bash
git clone https://github.com/minhnghiem32131024429/ReframeBot.git
cd ReframeBot
cp .env.example .env
# Set HF_TOKEN in .env (required for gated Llama 3.1 access)
```

2. Download guardrail model:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Nhatminh1234/ReframeBot-Guardrail-DistilBERT', local_dir='./guardrail_model_retrained/best')
"
```

3. Export and quantize the LLM (one-time setup):
```bash
# Export: merge base model + DPO adapter to bf16 (~16 GB RAM, no GPU needed)
uv run python scripts/export_merged_model.py --output ./merged_model

# Quantize to AWQ 4-bit (requires GPU, ~5-10 min in WSL2)
# In WSL2: pip install autoawq
python scripts/quantize_awq.py --input ./merged_model --output ./merged_model_awq
```

4. Start the stack:
```bash
docker compose up
```

5. Serve the web UI:
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
│   ├── benchmark.py            # Latency / throughput / TTFT benchmark
│   ├── build_rag_db.py         # Build ChromaDB from knowledge.txt
│   ├── train_guardrail.py      # Retrain guardrail classifier
│   ├── evaluate_model.py       # Evaluation + metrics
│   └── push_all_models.py      # Upload all models to HF Hub
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

## Inference Performance

Measured on NVIDIA RTX 5070 (8 GB VRAM), AWQ-Marlin 4-bit, `max_model_len=2048`, `--enforce-eager`:

| Metric | Value |
|---|---|
| Latency p50 (warm) | 3.3s |
| Latency p95 (warm) | 5.9s |
| Time to First Token (TTFT) p50 | 1.09s |
| Tokens/sec | ~54 tok/s |
| Throughput (4 concurrent) | 1.1 req/s |
| VRAM usage | ~5.4 GB / 8 GB |

Cold-start latency (~115s first request) is due to vLLM's initial compilation pass; subsequent requests are warm.

To reproduce:
```bash
uv run python scripts/benchmark.py --n 20 --concurrency 4
```

## Evaluation Results

All metrics were measured on a held-out test set not seen during training.

| Metric | Value | Description |
|---|---|---|
| Guardrail Accuracy | **91.1%** | Task classification on held-out evaluation set |
| Guardrail F1 (macro) | **0.99** | Precision/recall balance across all 3 task classes |
| BERTScore Relevance | **0.832** | Semantic similarity between generated and reference responses |
| BERTScore Faithfulness | **0.849** | Alignment between generated response and retrieved RAG context |
| Response Consistency | **0.732** | Cosine similarity between repeated responses to the same prompt |

**Notes:**
- Guardrail F1=0.99 was measured on the 20% validation split (335 samples) during training; 91.1% reflects a separate, harder evaluation set.
- Faithfulness > Relevance suggests the model grounds well in retrieved context when RAG is active.
- Full evaluation report: [`evaluation_report.json`](evaluation_report.json) | Radar chart: [`evaluation_summary.png`](evaluation_summary.png)

## Training

See `train.ipynb` for the complete training pipeline:
1. **SFT (Supervised Fine-Tuning)** - Base model adaptation
2. **DPO (Direct Preference Optimization)** - Response quality improvement
3. **Guardrail Training** - Task classification model

Optional scripts:
- `scripts/prepare_guardrail_data.py`: merge + deduplicate guardrail data (and add hard negatives)
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
