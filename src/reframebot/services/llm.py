"""LLM inference service.

Responsibilities:
- Load the base Llama model + DPO adapter once at startup.
- Generate CBT/OOS responses with optional RAG context injection.
- Generate short empathy responses for crisis turns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Thread
from typing import Dict, Iterator, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from reframebot.config import Settings
from reframebot.constants import ACCIDENTAL_CRISIS_TRIGGERS

logger = logging.getLogger(__name__)

_model = None
_tokenizer: Optional[AutoTokenizer] = None
_terminators: Optional[List[int]] = None

_BASE_SYSTEM_PROMPT = """\
You are ReframeBot, a specialized AI assistant. Your primary goal is to help university students with academic stress using CBT Socratic questioning.
You MUST follow these 3 rules at all times:
1.  **TASK 1 (CBT):** If the user is discussing **academic stress**... you MUST respond with (1) Empathy, then (2) Socratic Questions.
2.  **TASK 2 (CRISIS):** If the user expresses **ANY** thought of suicide... you MUST **STOP**! and redirect to a hotline.
3.  **TASK 3 (OUT-OF-SCOPE):** If the user discusses **non-academic** topics... you MUST **STOP**! (1) Validate their feeling, then (2) Gently state your limitation and pivot back to academics.
Do not give direct advice. Do not diagnose.\
"""

_TASK3_OVERRIDE = (
    "\n\n**CRITICAL INSTRUCTION:** The user's last message was identified as **Out-of-Scope (TASK 3)**. "
    "You MUST follow TASK 3 rules. **DO NOT** ask follow-up questions about their non-academic topic. "
    "Validate the feeling, state your limitation, and pivot back to academics NOW."
)

_CRISIS_EMPATHY_PROMPT = (
    "You are an empathetic listener. A user is in severe crisis. "
    "Your ONLY job is to respond with **one or two sentences** that validates their pain and shows deep concern. "
    "DO NOT ask questions. DO NOT give advice. DO NOT use the word 'hotline' or 'resources'."
)


def load(settings: Settings) -> None:
    global _model, _tokenizer, _terminators

    adapter = settings.adapter_path
    if not adapter or not Path(adapter).exists():
        raise FileNotFoundError(
            f"DPO adapter not found at '{adapter}'. "
            "Set ADAPTER_PATH in .env pointing to your local checkpoint."
        )

    logger.info("Loading tokenizer from: %s", settings.base_model_name)
    _tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
    _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logger.info("Loading base model: %s", settings.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        settings.base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    logger.info("Loading DPO adapter from: %s", adapter)
    merged = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    merged.eval()
    _model = merged

    _terminators = [
        _tokenizer.eos_token_id,
        _tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    logger.info("LLM ready on device: %s", next(_model.parameters()).device)


# ---------------------------------------------------------------------------
# Internal generation helper
# ---------------------------------------------------------------------------

def _generate(messages: List[Dict[str, str]], max_new_tokens: int, temperature: float) -> str:
    assert _model is not None and _tokenizer is not None
    prompt = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = _tokenizer(prompt, return_tensors="pt", padding=False).to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=_terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    response_ids = outputs[0][inputs.input_ids.shape[-1]:]
    return _tokenizer.decode(response_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_response(
    history: List[Dict[str, str]],
    task_label: str,
    rag_context: str = "",
) -> str:
    """Generate a CBT (TASK_1) or out-of-scope (TASK_3) response."""
    system_prompt = _BASE_SYSTEM_PROMPT

    if rag_context:
        system_prompt += (
            f"\n\n**KNOWLEDGE BASE REFERENCE:**\n"
            f"The following information from the CBT knowledge base may help guide your response:\n\n"
            f"{rag_context}\n\n"
            "Use this information to explain the concept to the student clearly. "
            "You CAN define terms and explain steps if the user asks 'What is...'. "
            "However, after explaining, always try to link it back to their feelings or ask if they want to try it."
        )

    if task_label == "TASK_3":
        system_prompt += _TASK3_OVERRIDE

    messages = [{"role": "system", "content": system_prompt}, *history]
    response = _generate(messages, max_new_tokens=512, temperature=0.6)

    # Safeguard: if the LLM accidentally echoed crisis-style content on a
    # non-crisis task, strip it so callers can handle crisis uniformly.
    if task_label != "TASK_2":
        response_lower = response.lower()
        if any(t in response_lower for t in ACCIDENTAL_CRISIS_TRIGGERS):
            logger.warning("LLM produced accidental crisis content for %s — suppressing.", task_label)
            response = "I hear that things feel overwhelming right now. Would you like to talk more about what's stressing you academically?"

    return response


def get_crisis_empathy(history: List[Dict[str, str]]) -> str:
    """Generate a short empathetic acknowledgement for a crisis turn."""
    messages = [
        {"role": "system", "content": _CRISIS_EMPATHY_PROMPT},
        *history[-2:],
    ]
    return _generate(messages, max_new_tokens=64, temperature=0.5)


def stream_response(
    history: List[Dict[str, str]],
    task_label: str,
    rag_context: str = "",
) -> Iterator[str]:
    """Stream tokens for a CBT (TASK_1) or out-of-scope (TASK_3) response."""
    assert _model is not None and _tokenizer is not None

    system_prompt = _BASE_SYSTEM_PROMPT
    if rag_context:
        system_prompt += (
            f"\n\n**KNOWLEDGE BASE REFERENCE:**\n"
            f"The following information from the CBT knowledge base may help guide your response:\n\n"
            f"{rag_context}\n\n"
            "Use this information to explain the concept to the student clearly. "
            "You CAN define terms and explain steps if the user asks 'What is...'. "
            "However, after explaining, always try to link it back to their feelings or ask if they want to try it."
        )
    if task_label == "TASK_3":
        system_prompt += _TASK3_OVERRIDE

    messages = [{"role": "system", "content": system_prompt}, *history]
    prompt = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = _tokenizer(prompt, return_tensors="pt", padding=False).to(_model.device)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs: Dict = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 512,
        "eos_token_id": _terminators,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "streamer": streamer,
    }
    thread = Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()
    yield from streamer
    thread.join()
