# Model Card — ReframeBot

This document covers all three models published as part of the ReframeBot project:

| Model | HF Repository | Base |
|---|---|---|
| SFT Adapter | [ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B) | Llama 3.1 8B Instruct |
| DPO Adapter | [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B) | SFT adapter above |
| Guardrail Classifier | [ReframeBot-Guardrail-DistilBERT](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT) | distilbert-base-uncased |

---

## Model Description

ReframeBot is a CBT-oriented conversational assistant designed for university students dealing with academic stress. The system consists of two model components:

- **LLM (Llama 3.1 8B):** Fine-tuned with SFT then further aligned with DPO to produce empathetic, Socratic-style CBT responses. Does not give direct advice or diagnoses.
- **Guardrail classifier (DistilBERT):** A 3-class text classifier that routes each conversation turn to one of three task modes: `TASK_1` (CBT/academic stress), `TASK_2` (crisis/self-harm), `TASK_3` (out-of-scope). The classifier output is combined with a regex + semantic similarity crisis detector before the final routing decision is made.

---

## Intended Use

**Intended for:**
- Supporting university students in reflecting on academic stress using CBT Socratic questioning techniques
- Research and educational demonstrations of fine-tuning + RLHF pipelines on mental health-adjacent tasks
- As a component in a larger system with human oversight

**Not intended for:**
- Clinical or therapeutic use — this is not a licensed mental health tool
- Replacing professional counselors or psychologists
- Deployment to vulnerable populations without human moderation in the loop
- High-stakes crisis intervention (the crisis path redirects to hotlines, but the LLM is not a crisis counselor)

---

## Training Data

### SFT Dataset (4,500 samples)
- **Format:** Multi-turn conversations in Llama 3.1 chat template format
- **Content:** Simulated student–therapist dialogues covering academic stress scenarios (exam anxiety, GPA pressure, deadline overwhelm, imposter syndrome, burnout)
- **Generation:** Synthetically generated using GPT-4 via structured prompting; each dialogue was prompted to follow CBT Socratic questioning patterns
- **Language:** English

### DPO Dataset (1,400 preference pairs)
- **Format:** `{prompt, chosen, rejected}` triplets
- **Content:** Same scenario distribution as SFT; `chosen` responses demonstrate empathy + open-ended questioning, `rejected` responses contain direct advice, dismissiveness, or unsafe content
- **Generation:** Synthetically generated using GPT-4

### Guardrail Dataset (1,674 samples)
- **Format:** `{text, label}` where label is one of `TASK_1`, `TASK_2`, `TASK_3`
- **Content:** Single-turn and multi-turn context windows; includes hard negatives (benign metaphors that superficially resemble crisis language, e.g., "dying of embarrassment")
- **Split:** 80% train (1,339 samples) / 20% validation (335 samples)

**Dataset limitations:** All training data is synthetically generated. The distribution may not reflect the full range of real student language, cultural expressions of distress, or code-switching (e.g., English/Vietnamese mixed turns).

---

## Training Procedure

### Hardware
- GPU: NVIDIA RTX 5070 (laptop, 8 GB VRAM)
- Quantization: 4-bit NF4 (bitsandbytes) with bfloat16 compute, enabling training on consumer-grade hardware

### SFT (Supervised Fine-Tuning)

| Hyperparameter | Value |
|---|---|
| Base model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| LoRA rank (r) | 6 |
| LoRA alpha | 12 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down proj |
| Learning rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Batch size (effective) | 8 (1 × grad_accum 8) |
| Epochs | 3 |
| Max sequence length | 384 |
| BF16 | enabled |
| Gradient checkpointing | enabled |

### DPO (Direct Preference Optimization)

| Hyperparameter | Value |
|---|---|
| Starting checkpoint | SFT final adapter |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 5e-6 |
| Optimizer | paged_adamw_8bit |
| Batch size (effective) | 48 (2 × grad_accum 24) |
| Epochs | 3 |
| Beta (KL penalty) | 0.1 |
| Max sequence length | 512 |
| BF16 | enabled |

### Guardrail Classifier

| Hyperparameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Learning rate | 2e-6 |
| Batch size | 16 |
| Max epochs | 20 (early stopping, patience=3) |
| Weight decay | 0.01 |
| Max token length | 128 |
| Best model criterion | macro F1 |

---

## Evaluation Results

### Guardrail Classifier

Measured on a held-out evaluation set separate from the training/validation split:

| Metric | Score |
|---|---|
| Accuracy (held-out test set) | 91.1% |
| Accuracy (validation split) | 99.0% |
| F1 macro (validation split) | 0.99 |

Per-class performance on validation split (335 samples):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| TASK_1 | 0.99 | 1.00 | 1.00 | 107 |
| TASK_2 | 0.98 | 1.00 | 0.99 | 91 |
| TASK_3 | 1.00 | 0.98 | 0.99 | 137 |

The gap between validation accuracy (99%) and held-out test accuracy (91%) is expected — the held-out evaluation set was designed to include harder boundary cases not represented in the training distribution.

### LLM System (End-to-End)

| Metric | Score | Method |
|---|---|---|
| BERTScore Relevance (F1) | 0.832 | Generated response vs. reference answer |
| BERTScore Faithfulness (F1) | 0.849 | Generated response vs. retrieved RAG context |
| Response Consistency | 0.732 | Cosine similarity between 2 independent runs on the same prompt |

Faithfulness (0.849) exceeding Relevance (0.832) suggests the model grounds well in RAG-retrieved CBT context when it is available.

---

## Limitations and Risks

- **Synthetic training data:** All data was GPT-generated. The model may produce responses that sound fluent but miss cultural nuance or real student language patterns.
- **Crisis detection is not fail-safe:** The dual-signal crisis detector (regex + semantic similarity) improves precision over a single-signal approach, but false negatives remain possible. This system must not be deployed as a standalone crisis intervention tool.
- **English-primary:** The training data is English. Vietnamese crisis keywords are partially supported in the regex layer but the LLM itself is not fine-tuned on Vietnamese dialogue.
- **8B parameter constraint:** The base model is a 7–8B instruction-tuned LLM. It can be inconsistent on ambiguous or multi-intent turns.
- **No human evaluation:** All evaluation metrics are automated (BERTScore, cosine similarity). No human rater study was conducted.

---

## Citation

If you use ReframeBot in academic work, please cite:

```
@misc{reframebot2025,
  author       = {Nghiem Nhat Minh},
  title        = {ReframeBot: A CBT-Oriented Chatbot for Academic Stress Support},
  year         = {2025},
  howpublished = {\url{https://github.com/minhnghiem32131024429/ReframeBot}},
}
```

---

## Author

**Nghiem Nhat Minh**
- GitHub: [@minhnghiem32131024429](https://github.com/minhnghiem32131024429)
- Hugging Face: [@Nhatminh1234](https://huggingface.co/Nhatminh1234)
