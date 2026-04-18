# Setup Guide

> The recommended setup is Docker — see the [Quick Start](../README.md#quick-start) in the README.
> This document covers advanced configuration and troubleshooting only.

## Prerequisites

- Python 3.11+
- CUDA-capable GPU with 8 GB+ VRAM
- Docker Desktop with NVIDIA Container Toolkit (for the recommended path)

## Configuration

All settings are read from `.env`. Copy the template and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Required to download the AWQ model from Hugging Face |
| `GUARDRAIL_PATH` | Path to the guardrail model directory (auto-discovered if unset) |
| `RAG_DB_PATH` | Path to the ChromaDB directory (default: `./rag_db`) |

## Testing the API

Once the stack is running (`docker compose up` or `python app.py`):

```bash
# Non-streaming
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"history": [{"role": "user", "content": "I failed my exam and feel hopeless"}]}'

# Streaming
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"history": [{"role": "user", "content": "I failed my exam and feel hopeless"}]}'
```

## Troubleshooting

### CUDA Out of Memory (in-process mode)
- The merged model is 15 GB on disk but loads in ~8 GB VRAM via NF4 quantization.
- If loading fails with a page-file error on Windows, increase virtual memory:
  Win + X → System → Advanced system settings → Performance → Virtual Memory → set to 40 GB+.

### vLLM container not healthy
- Cold-start takes ~115s on first launch (CUDA kernel compilation). Wait for the healthcheck to pass before the API container starts.
- Check logs: `docker compose logs vllm`

### Guardrail model not found
- Ensure `GUARDRAIL_PATH` in `.env` points to the directory containing `config.json`.
- Default auto-discovery checks `./guardrail_model_retrained/best`.

### CORS errors
- Edit `CORS_ORIGINS` in `.env` to add your frontend origin.
- Default is `*` (allow all) for development.
