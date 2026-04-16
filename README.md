# ReframeBot

ReframeBot is a CBT-oriented chatbot for supporting university students with academic stress. It combines a fine-tuned Llama 3.1 model with a guardrail router (TASK_1/TASK_2/TASK_3) and optional RAG grounding from a CBT knowledge base.

> For full training details, hyperparameters, and per-class metrics, see [MODEL_CARD.md](MODEL_CARD.md).

## Model Repositories

Our trained models are available on Hugging Face:

- **[SFT Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B)** - Supervised Fine-Tuning adapter for Llama 3.1 8B
- **[DPO Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B)** - Direct Preference Optimization adapter
- **[Guardrail Classifier](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT)** - Task classifier for message routing (CBT/Crisis/Out-of-scope)

## Features
- Fine-tuned Llama 3.1 8B (DPO adapter)
- Guardrail routing with crisis detection and out-of-scope redirection
- Optional RAG grounding over a CBT knowledge base
- FastAPI backend and a lightweight static web UI

## Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/minhnghiem32131024429/ReframeBot.git
cd ReframeBot
```

2. Install dependencies:
```bash
# Runtime only (to run the server)
pip install -e .

# Including evaluation / push scripts
pip install -e ".[scripts]"

# Including full training pipeline
pip install -e ".[train]"
```

3. Download models from Hugging Face:
   - **SFT Adapter**: [ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B)
   - **DPO Adapter**: [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B)
   - **Guardrail Model**: [ReframeBot-Guardrail-DistilBERT](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT)

4. Create a local config file:
   - Copy `.env.example` to `.env`
   - Set at minimum `ADAPTER_PATH` and `GUARDRAIL_PATH` to your local model directories:
```env
ADAPTER_PATH=D:\Work\AI\results_reframebot_DPO\checkpoint-90
GUARDRAIL_PATH=D:\Work\AI\guardrail_model_retrained\best
```
   Notes:
   - `.env` is git-ignored by design — never commit it.
   - See `.env.example` for the full list of configurable variables.

5. Run the FastAPI server:
```bash
python app.py
```

6. Serve the web UI (in a new terminal):
```bash
cd web
python -m http.server 8080
```
   - Open: http://localhost:8080/

The UI will call the backend at `http://127.0.0.1:8000/chat`.

## Project Structure

```
ReframeBot/
├── app.py                      # Entry point: python app.py
├── pyproject.toml              # Dependencies (runtime / scripts / train groups)
├── train.ipynb                 # Training notebook (SFT + DPO + Guardrail)
├── src/
│   └── reframebot/
│       ├── config.py           # All settings via pydantic-settings + .env
│       ├── constants.py        # Hotlines, keywords, regex, prototype sentences
│       ├── router.py           # Task routing logic (TASK_1/2/3 priority chain)
│       ├── main.py             # FastAPI app, lifespan, /chat endpoint
│       └── services/
│           ├── guardrail.py    # Guardrail classifier + crisis detection
│           ├── rag.py          # ChromaDB retrieval
│           └── llm.py          # LLM loading + inference
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   ├── dataset.jsonl           # SFT training data
│   ├── dataset_dpo.jsonl       # DPO training data
│   └── guardrail_dataset.jsonl
├── scripts/
│   ├── build_rag_db.py         # Build ChromaDB from knowledge.txt
│   ├── train_guardrail.py      # Retrain guardrail classifier
│   ├── evaluate_model.py       # Evaluation + metrics
│   ├── prepare_guardrail_data.py
│   ├── push_to_hub.py          # Upload single model to HF Hub
│   └── push_all_models.py      # Upload all models
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
- GitHub: [@minhnghiem32131024429](https://github.com/nhatminh-115)
- Hugging Face: [@Nhatminh1234](https://huggingface.co/Nhatminh1234)

## Acknowledgments

- Meta AI for Llama 3.1
- Hugging Face for transformers and PEFT libraries
- FastAPI team
