import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chromadb
import re
from tqdm import tqdm
from typing import List, Dict

# Model & Evaluation Libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, confusion_matrix
from bert_score import score as bert_score_func

# ==================== CONFIGURATION ====================
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = r"D:\Work\AI\results_reframebot_DPO\checkpoint-90"
GUARDRAIL_PATH = r"D:\Work\AI\guardrail_model_1\checkpoint-840"
RAG_DB_PATH = "./rag_db"

REPORT_FILE = "evaluation_report.json"
SUMMARY_IMAGE = "evaluation_summary.png"

# ==================== LOAD MODELS ====================
print("--- LOADING SYSTEM COMPONENTS ---")

# 1. Load Guardrail
try:
    print("[1/4] Loading Guardrail...")
    guardrail_pipeline = pipeline("text-classification", model=GUARDRAIL_PATH, tokenizer=GUARDRAIL_PATH, device=-1)
except Exception as e:
    print(f"Error loading Guardrail: {e}")
    exit()

# 2. Load LLM
try:
    print("[2/4] Loading LLM...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, quantization_config=bnb_config, device_map={"": 0},
    )
    llm_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    llm_model = llm_model.merge_and_unload()
    llm_model.eval()
except Exception as e:
    print(f"Error loading LLM: {e}")
    exit()

# 3. Load RAG
print("[3/4] Loading RAG...")
try:
    rag_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    rag_client = chromadb.PersistentClient(path=RAG_DB_PATH)
    rag_collection = rag_client.get_collection(name="cbt_knowledge")
except:
    print("   -> Warning: RAG not found.")
    rag_collection = None

# 4. Metric Models
print("[4/4] Loading Auxiliary Models...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2') # Chỉ dùng cho Consistency

print("--- SYSTEM READY ---\n")

# ==================== HELPER FUNCTIONS ====================

def retrieve_knowledge(user_query: str) -> str:
    if not rag_collection: return ""
    try:
        query_embedding = rag_embedder.encode([user_query]).tolist()
        results = rag_collection.query(query_embeddings=query_embedding, n_results=1)
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0][0]
    except:
        pass
    return ""

def generate_response(user_message: str, task_label: str = "TASK_1") -> str:
    # ... (Giữ nguyên logic sinh text của bạn) ...
    if task_label == "TASK_2":
        return "I am deeply concerned for your safety. Please reach out to these resources in Vietnam: National Protection Hotline: 111."
    if task_label == "TASK_3":
        system_prompt = "You are ReframeBot. Politely decline non-academic topics."
    else:
        rag_context = retrieve_knowledge(user_message)
        system_prompt = "You are ReframeBot, helping students with academic stress using CBT."
        if rag_context:
            system_prompt += f"\n\nKNOWLEDGE BASE:\n{rag_context}"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(llm_model.device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=256, eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            do_sample=True, temperature=0.6, top_p=0.9,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

# ==================== EVALUATION CLASS ====================

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_accuracy(self, test_data: List[Dict]) -> float:
        print("\n[1] Evaluating ACCURACY (Guardrail)...")
        y_true = [item['expected_label'] for item in test_data]
        y_pred = [guardrail_pipeline(item['text'])[0]['label'] for item in tqdm(test_data)]
        acc = accuracy_score(y_true, y_pred)
        self.results['accuracy'] = acc
        print(f"   -> Accuracy: {acc:.2%}")
        return acc

    def evaluate_consistency(self, test_prompts: List[str], num_samples: int = 2) -> float:
        print("\n[2] Evaluating CONSISTENCY (Vector Similarity)...")
        scores = []
        for prompt in tqdm(test_prompts):
            resps = [generate_response(prompt, "TASK_1") for _ in range(num_samples)]
            emb = semantic_model.encode(resps)
            scores.append(util.cos_sim(emb[0], emb[1]).item())
        avg = np.mean(scores)
        self.results['consistency'] = avg
        print(f"   -> Consistency: {avg:.3f}")
        return avg

    def evaluate_semantic_relevance(self, test_data: List[Dict]) -> float:
        """
        Metric 3: Semantic Relevance using BERTScore.
        Compares [Generated Response] vs [Ground Truth / User Query]
        """
        print("\n[3] Evaluating SEMANTIC RELEVANCE (BERTScore)...")
        cands, refs = [], []
        for item in tqdm(test_data):
            resp = generate_response(item['question'], "TASK_1")
            cands.append(resp)
            # Dùng câu hỏi gốc làm tham chiếu ngữ cảnh nếu không có câu trả lời mẫu
            refs.append(item.get('expected_answer', item['question'])) 
            
        try:
            P, R, F1 = bert_score_func(cands, refs, lang="en", verbose=False)
            score = F1.mean().item()
            self.results['relevance_bert'] = score
            print(f"   -> Relevance (F1): {score:.4f}")
            return score
        except Exception as e:
            print(f"Error: {e}")
            return 0.0

    def evaluate_faithfulness(self, test_data: List[Dict]) -> float:
        """
        Metric 4: Faithfulness using BERTScore (Standardized).
        Compares [Generated Response] vs [Retrieved RAG Context]
        """
        print("\n[4] Evaluating FAITHFULNESS (BERTScore)...")
        cands, refs = [], []
        for item in tqdm(test_data):
            resp = generate_response(item['question'], "TASK_1")
            context = retrieve_knowledge(item['question'])
            
            if context:
                cands.append(resp)
                refs.append(context)
            
        if not cands: return 0.0
        
        try:
            # Đo độ tương đồng giữa Context và Response
            P, R, F1 = bert_score_func(cands, refs, lang="en", verbose=False)
            score = F1.mean().item()
            self.results['faithfulness_bert'] = score
            print(f"   -> Faithfulness (F1): {score:.4f}")
            return score
        except Exception as e:
            print(f"Error: {e}")
            return 0.0

    def evaluate_complexity(self, test_prompts: List[str]) -> float:
        print("\n[5] Evaluating COMPLEXITY (Gaussian)...")
        scores = []
        TARGET, SIGMA = 100, 80 # Đã chỉnh Sigma lên 80 như thảo luận
        for prompt in tqdm(test_prompts):
            length = len(generate_response(prompt, "TASK_1").split())
            scores.append(np.exp(-0.5 * ((length - TARGET) / SIGMA)**2))
        avg = np.mean(scores)
        self.results['complexity'] = avg
        print(f"   -> Complexity: {avg:.3f}")
        return avg

    def generate_report(self):
        # Save JSON
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
            
        # Plot Radar
        labels = ['Accuracy', 'Consistency', 'Relevance (BERT)', 'Faithfulness (BERT)', 'Complexity']
        keys = ['accuracy', 'consistency', 'relevance_bert', 'faithfulness_bert', 'complexity']
        values = [self.results.get(k, 0) for k in keys]
        
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, color='#FF5722')
        ax.fill(angles, values, alpha=0.25, color='#FF5722')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11, weight='bold')
        ax.set_ylim(0, 1)
        
        for a, v in zip(angles[:-1], values[:-1]):
            ax.text(a, v + 0.1, f"{v:.2f}", ha='center')
            
        plt.title('Standardized Model Evaluation', y=1.08, weight='bold')
        plt.tight_layout()
        plt.savefig(SUMMARY_IMAGE, dpi=300)
        print(f"\n✅ Report saved: {REPORT_FILE}, Image: {SUMMARY_IMAGE}")

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        with open('data/evaluation_test_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        print("Data file not found!")
        exit()

    evaluator = ModelEvaluator()
    evaluator.evaluate_accuracy(data['accuracy_test'])
    evaluator.evaluate_consistency(data['consistency_prompts'])
    evaluator.evaluate_semantic_relevance(data['relevance_test'])
    evaluator.evaluate_faithfulness(data['faithfulness_test'])
    evaluator.evaluate_complexity(data['complexity_prompts'])
    evaluator.generate_report()