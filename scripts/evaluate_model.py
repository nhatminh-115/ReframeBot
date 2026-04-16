"""End-to-end evaluation script for the ReframeBot system.

Metrics:
  1. Accuracy        — guardrail classifier on held-out test set
  2. Consistency     — cosine similarity between repeated LLM responses
  3. Relevance       — BERTScore F1 (generated response vs. reference answer)
  4. Faithfulness    — BERTScore F1 (generated response vs. RAG context)
  5. Complexity      — Gaussian score penalising responses far from target length

Usage:
    uv run python scripts/evaluate_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

import os  # noqa: E402 — must follow dotenv load

BASE_MODEL_NAME: str = os.environ.get("BASE_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER_PATH: str = os.environ.get("ADAPTER_PATH", "")
GUARDRAIL_PATH: str = os.environ.get("GUARDRAIL_PATH", "")
RAG_DB_PATH: str = os.environ.get("RAG_DB_PATH", str(_REPO_ROOT / "rag_db"))
REPORT_FILE: Path = _REPO_ROOT / "evaluation_report.json"
SUMMARY_IMAGE: Path = _REPO_ROOT / "evaluation_summary.png"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_guardrail(path: str):
    from transformers import pipeline
    return pipeline("text-classification", model=path, tokenizer=path, device=-1)


def _load_llm(base: str, adapter: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    model.eval()
    return model, tokenizer


def _load_rag(db_path: str) -> Optional[object]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        return client.get_collection(name="cbt_knowledge")
    except Exception as exc:
        print(f"  [warn] RAG not available: {exc}")
        return None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def retrieve_context(collection, query: str) -> str:
    if collection is None:
        return ""
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        emb = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=emb, n_results=1)
        docs = results.get("documents", [[]])
        return docs[0][0] if docs and docs[0] else ""
    except Exception:
        return ""


def generate_response(
    model,
    tokenizer,
    user_message: str,
    task_label: str = "TASK_1",
    rag_context: str = "",
) -> str:
    if task_label == "TASK_2":
        return "I am deeply concerned for your safety. Please reach out: National Hotline 1900 1267."

    system_prompt = "You are ReframeBot, helping students with academic stress using CBT."
    if task_label == "TASK_3":
        system_prompt = "You are ReframeBot. Politely decline non-academic topics and redirect."
    elif rag_context:
        system_prompt += f"\n\nKNOWLEDGE BASE:\n{rag_context}"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    def __init__(self, model, tokenizer, guardrail_pipeline, rag_collection) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.guardrail = guardrail_pipeline
        self.rag = rag_collection
        self.results: Dict[str, float] = {}

    def evaluate_accuracy(self, test_data: List[Dict]) -> float:
        print("\n[1] Accuracy (guardrail classifier) ...")
        y_true = [item["expected_label"] for item in test_data]
        y_pred = [self.guardrail(item["text"])[0]["label"] for item in tqdm(test_data)]
        acc = float(accuracy_score(y_true, y_pred))
        self.results["accuracy"] = acc
        print(f"    -> {acc:.2%}")
        return acc

    def evaluate_consistency(self, prompts: List[str], num_samples: int = 2) -> float:
        print("\n[2] Consistency (response cosine similarity) ...")
        from sentence_transformers import SentenceTransformer, util
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        scores = []
        for prompt in tqdm(prompts):
            responses = [generate_response(self.model, self.tokenizer, prompt) for _ in range(num_samples)]
            emb = embedder.encode(responses)
            scores.append(float(util.cos_sim(emb[0], emb[1]).item()))
        avg = float(np.mean(scores))
        self.results["consistency"] = avg
        print(f"    -> {avg:.3f}")
        return avg

    def evaluate_semantic_relevance(self, test_data: List[Dict]) -> float:
        print("\n[3] Relevance (BERTScore F1, response vs. reference) ...")
        from bert_score import score as bert_score_func
        cands, refs = [], []
        for item in tqdm(test_data):
            cands.append(generate_response(self.model, self.tokenizer, item["question"]))
            refs.append(item.get("expected_answer", item["question"]))
        try:
            _, _, f1 = bert_score_func(cands, refs, lang="en", verbose=False)
            score = float(f1.mean().item())
        except Exception as exc:
            print(f"    [error] {exc}")
            score = 0.0
        self.results["relevance_bert"] = score
        print(f"    -> {score:.4f}")
        return score

    def evaluate_faithfulness(self, test_data: List[Dict]) -> float:
        print("\n[4] Faithfulness (BERTScore F1, response vs. RAG context) ...")
        from bert_score import score as bert_score_func
        cands, refs = [], []
        for item in tqdm(test_data):
            context = retrieve_context(self.rag, item["question"])
            if not context:
                continue
            cands.append(generate_response(self.model, self.tokenizer, item["question"], rag_context=context))
            refs.append(context)
        if not cands:
            print("    [warn] No RAG context available — skipping.")
            return 0.0
        try:
            _, _, f1 = bert_score_func(cands, refs, lang="en", verbose=False)
            score = float(f1.mean().item())
        except Exception as exc:
            print(f"    [error] {exc}")
            score = 0.0
        self.results["faithfulness_bert"] = score
        print(f"    -> {score:.4f}")
        return score

    def evaluate_complexity(self, prompts: List[str], target: int = 100, sigma: int = 80) -> float:
        print("\n[5] Response length score (Gaussian, target=100 words) ...")
        scores = []
        for prompt in tqdm(prompts):
            length = len(generate_response(self.model, self.tokenizer, prompt).split())
            scores.append(float(np.exp(-0.5 * ((length - target) / sigma) ** 2)))
        avg = float(np.mean(scores))
        self.results["complexity"] = avg
        print(f"    -> {avg:.3f}")
        return avg

    def save_report(self) -> None:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        labels = ["Accuracy", "Consistency", "Relevance (BERT)", "Faithfulness (BERT)", "Length Score"]
        keys = ["accuracy", "consistency", "relevance_bert", "faithfulness_bert", "complexity"]
        values = [self.results.get(k, 0.0) for k in keys]
        values_closed = values + [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        ax.plot(angles, values_closed, "o-", linewidth=2, color="#FF5722")
        ax.fill(angles, values_closed, alpha=0.25, color="#FF5722")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11, weight="bold")
        ax.set_ylim(0, 1)
        for a, v in zip(angles[:-1], values):
            ax.text(a, v + 0.1, f"{v:.2f}", ha="center")
        plt.title("ReframeBot Evaluation", y=1.08, weight="bold")
        plt.tight_layout()
        plt.savefig(SUMMARY_IMAGE, dpi=300)

        print(f"\nReport: {REPORT_FILE}")
        print(f"Chart:  {SUMMARY_IMAGE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not ADAPTER_PATH:
        print("ERROR: ADAPTER_PATH is not set in .env")
        sys.exit(1)
    if not GUARDRAIL_PATH:
        print("ERROR: GUARDRAIL_PATH is not set in .env")
        sys.exit(1)

    data_path = _REPO_ROOT / "data" / "evaluation_test_data.json"
    if not data_path.exists():
        print(f"ERROR: Test data not found at {data_path}")
        sys.exit(1)
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    print("--- Loading models ---")
    print("[1/3] Guardrail ...")
    guardrail_pipeline = _load_guardrail(GUARDRAIL_PATH)
    print("[2/3] LLM ...")
    model, tokenizer = _load_llm(BASE_MODEL_NAME, ADAPTER_PATH)
    print("[3/3] RAG ...")
    rag_collection = _load_rag(RAG_DB_PATH)
    print("--- Ready ---\n")

    evaluator = ModelEvaluator(model, tokenizer, guardrail_pipeline, rag_collection)
    evaluator.evaluate_accuracy(data["accuracy_test"])
    evaluator.evaluate_consistency(data["consistency_prompts"])
    evaluator.evaluate_semantic_relevance(data["relevance_test"])
    evaluator.evaluate_faithfulness(data["faithfulness_test"])
    evaluator.evaluate_complexity(data["complexity_prompts"])
    evaluator.save_report()


if __name__ == "__main__":
    main()
