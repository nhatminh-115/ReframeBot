"""Push README.md model cards to all three ReframeBot HF repositories.

Usage:
    uv run python scripts/push_model_cards.py

Requires HF_TOKEN in .env (write access to Nhatminh1234/* repos).
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

import os  # noqa: E402

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set in .env")
    sys.exit(1)

from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# Model card content
# ---------------------------------------------------------------------------

_SFT_CARD = """\
---
language:
- en
license: apache-2.0
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- lora
- peft
- sft
- cbt
- mental-health
- academic-stress
- chatbot
pipeline_tag: text-generation
---

# ReframeBot-SFT-Llama3.1-8B

LoRA adapter for `meta-llama/Meta-Llama-3.1-8B-Instruct`, fine-tuned with
Supervised Fine-Tuning (SFT) to support CBT-style Socratic questioning for
university students under academic stress.

This is **stage 1** of the ReframeBot training pipeline. The DPO adapter
([ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B))
was initialised from this checkpoint.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter = "Nhatminh1234/ReframeBot-SFT-Llama3.1-8B"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| LoRA rank (r) | 6 |
| LoRA alpha | 12 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Effective batch size | 8 (1 × grad_accum 8) |
| Epochs | 3 |
| Max sequence length | 384 |
| Quantization | 4-bit NF4, bfloat16 compute |
| Hardware | NVIDIA RTX 5070 (laptop, 8 GB VRAM) |

**Dataset:** 4,500 synthetic multi-turn dialogues generated with GPT-4,
covering academic stress scenarios (exam anxiety, GPA pressure, deadline
overwhelm, imposter syndrome, burnout). All conversations follow CBT
Socratic questioning patterns.

## Intended Use

Designed as a component in the ReframeBot system — not a standalone
mental-health tool. Must not be used for clinical intervention or crisis
support without human oversight.

## Project

GitHub: [ReframeBot](https://github.com/minhnghiem32131024429/ReframeBot)
"""

_DPO_CARD = """\
---
language:
- en
license: apache-2.0
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- lora
- peft
- dpo
- rlhf
- cbt
- mental-health
- academic-stress
- chatbot
pipeline_tag: text-generation
---

# ReframeBot-DPO-Llama3.1-8B

LoRA adapter for `meta-llama/Meta-Llama-3.1-8B-Instruct`, further aligned
with Direct Preference Optimisation (DPO) on top of the SFT checkpoint
([ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B)).

DPO training steered the model towards empathetic, open-ended Socratic
responses and away from direct advice, dismissiveness, or unsafe content.
**This is the production adapter used in the ReframeBot system.**

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter = "Nhatminh1234/ReframeBot-DPO-Llama3.1-8B"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Training Details

| Hyperparameter | Value |
|---|---|
| Starting checkpoint | ReframeBot-SFT-Llama3.1-8B |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 5e-6 |
| Optimizer | paged_adamw_8bit |
| Effective batch size | 48 (2 × grad_accum 24) |
| Epochs | 3 |
| Beta (KL penalty) | 0.1 |
| Max sequence length | 512 |
| Quantization | 4-bit NF4, bfloat16 compute |
| Hardware | NVIDIA RTX 5070 (laptop, 8 GB VRAM) |

**Dataset:** 1,400 preference pairs `{prompt, chosen, rejected}` generated
with GPT-4. Chosen responses demonstrate empathy + open-ended questioning;
rejected responses contain direct advice, dismissiveness, or unsafe content.

## Evaluation

| Metric | Score |
|---|---|
| BERTScore Relevance (F1) | 0.832 |
| BERTScore Faithfulness (F1) | 0.849 |
| Response Consistency | 0.732 |

## Intended Use

Designed as a component in the ReframeBot system — not a standalone
mental-health tool. Must not be used for clinical intervention or crisis
support without human oversight.

## Project

GitHub: [ReframeBot](https://github.com/minhnghiem32131024429/ReframeBot)
"""

_GUARDRAIL_CARD = """\
---
language:
- en
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
tags:
- text-classification
- guardrail
- safety
- cbt
- mental-health
- academic-stress
pipeline_tag: text-classification
---

# ReframeBot-Guardrail-DistilBERT

A 3-class text classifier fine-tuned from `distilbert-base-uncased` that
routes conversation turns to one of three task modes:

| Label | Meaning |
|---|---|
| `TASK_1` | CBT / academic stress — engage with Socratic questioning |
| `TASK_2` | Crisis / self-harm signal — redirect to emergency hotlines |
| `TASK_3` | Out-of-scope — validate feeling, pivot back to academics |

This model is the guardrail component of the ReframeBot system. Its output
is combined with a dual-signal crisis detector (regex + cosine similarity)
before the final routing decision is made.

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Nhatminh1234/ReframeBot-Guardrail-DistilBERT",
)

classifier("I'm really stressed about my finals next week")
# [{'label': 'TASK_1', 'score': 0.97}]

classifier("What's a good recipe for pasta?")
# [{'label': 'TASK_3', 'score': 0.94}]
```

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Number of labels | 3 (TASK_1, TASK_2, TASK_3) |
| Learning rate | 2e-6 |
| Batch size | 16 |
| Max epochs | 20 (early stopping, patience=3) |
| Weight decay | 0.01 |
| Max token length | 128 |
| Best model criterion | macro F1 |
| Hardware | NVIDIA RTX 5070 (laptop, 8 GB VRAM) |

**Dataset:** 1,674 labelled samples (80/20 train/val split). Includes hard
negatives — benign metaphors that superficially resemble crisis language
(e.g., "dying of embarrassment after that presentation").

## Evaluation

Per-class results on the validation split (335 samples):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| TASK_1 | 0.99 | 1.00 | 1.00 | 107 |
| TASK_2 | 0.98 | 1.00 | 0.99 | 91 |
| TASK_3 | 1.00 | 0.98 | 0.99 | 137 |
| **macro avg** | **0.99** | **0.99** | **0.99** | **335** |

Accuracy on a separate, harder held-out test set: **88.3%** (includes
boundary cases not present in the training distribution).

## Intended Use

Designed as a routing component in the ReframeBot system. The TASK_2
output alone is not sufficient for crisis intervention — the full system
also applies a regex + semantic similarity layer before acting on a
crisis signal.

## Project

GitHub: [ReframeBot](https://github.com/minhnghiem32131024429/ReframeBot)
"""

_AWQ_CARD = """\
---
language:
- en
license: apache-2.0
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- awq
- quantized
- vllm
- cbt
- mental-health
- academic-stress
- chatbot
pipeline_tag: text-generation
---

# ReframeBot-Llama3.1-8B-AWQ

4-bit AWQ quantized version of the merged [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B).
Optimized for high-throughput serving with **vLLM**.

This model combines the base Llama 3.1 8B Instruct model with the DPO-aligned CBT adapter, then compresses it using Activation-aware Weight Quantization (AWQ) for efficient production deployment.

## Usage

### vLLM (Recommended)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ", quantization="awq")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

prompts = ["I'm feeling so overwhelmed with my thesis..."]
outputs = llm.generate(prompts, sampling_params)
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

## Quantization Details

| Parameter | Value |
|---|---|
| Quantization Method | AWQ (Activation-aware Weight Quantization) |
| Bits | 4-bit |
| Group Size | 128 |
| Version | GEMM |
| Calibration Dataset | ReframeBot Socratic Dialogue Dataset (32 samples) |
| Hardware used | NVIDIA RTX 5070 (laptop, 8 GB VRAM) |

## Model Pipeline

1. **Base Model**: Llama 3.1 8B Instruct
2. **Stage 1 (SFT)**: Fine-tuned on 4.5k CBT dialogues.
3. **Stage 2 (DPO)**: Aligned with 1.4k preference pairs for empathy.
4. **Stage 3 (Merge)**: Merged adapter into base model.
5. **Stage 4 (Quantize)**: AWQ 4-bit quantization for serving.

## Intended Use

Designed for production deployment in the ReframeBot system. Must be used with the accompanying Guardrail and RAG components for safe and accurate operation. Not a substitute for professional mental health care.

## Project

GitHub: [ReframeBot](https://github.com/minhnghiem32131024429/ReframeBot)
"""

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

CARDS = {
    "Nhatminh1234/ReframeBot-SFT-Llama3.1-8B": _SFT_CARD,
    "Nhatminh1234/ReframeBot-DPO-Llama3.1-8B": _DPO_CARD,
    "Nhatminh1234/ReframeBot-Guardrail-DistilBERT": _GUARDRAIL_CARD,
    "Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ": _AWQ_CARD,
}


def main() -> None:
    for repo_id, content in CARDS.items():
        print(f"Pushing model card to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"  done: https://huggingface.co/{repo_id}")
    print("\nAll model cards pushed.")


if __name__ == "__main__":
    main()
