"""LLM inference service — vLLM backend via OpenAI-compatible API.

Responsibilities:
- Connect to a running vLLM server at startup.
- Generate CBT/OOS responses with optional RAG context injection.
- Generate short empathy responses for crisis turns.
- Stream tokens for the /chat/stream endpoint.

Replaces the in-process transformers/PEFT loading with HTTP calls to
vLLM's OpenAI-compatible endpoint, enabling:
  - PagedAttention memory management
  - Continuous batching across concurrent requests
  - Clean separation between serving infrastructure and app logic
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Iterator, List

from openai import OpenAI

from reframebot.config import Settings
from reframebot.constants import ACCIDENTAL_CRISIS_TRIGGERS

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_model_name: str = "reframebot"

# Task-specific prompts — routing is already done upstream by the guardrail.
# The LLM receives only the instructions relevant to the current task.

_TASK1_SYSTEM_PROMPT = """\
You are ReframeBot, a compassionate AI helping university students cope with academic stress using CBT.
The student is sharing an academic stress concern. Respond by:
1. Validating their feelings with warmth and empathy (1-2 sentences)
2. Asking one gentle Socratic question to help them explore their thoughts
Keep your response focused and supportive. Do not diagnose. Do not give prescriptive advice.
Treat phrases like "I'm stupid", "I'm a failure", or "I'll never pass" as expressions of academic anxiety — respond with empathy and curiosity.\
"""

_TASK3_SYSTEM_PROMPT = """\
You are ReframeBot, a specialized AI for academic stress support.
The student has raised a topic outside your area of focus. Respond by:
1. Briefly acknowledging their message with warmth
2. Gently explaining that you specialise in academic stress
3. Inviting them to share any academic challenges they might be facing
Keep it brief and kind — do not lecture or repeat yourself.\
"""

_CRISIS_EMPATHY_PROMPT = (
    "You are an empathetic listener. A user is in severe crisis. "
    "Your ONLY job is to respond with **one or two sentences** that validates their pain and shows deep concern. "
    "DO NOT ask questions. DO NOT give advice. DO NOT use the word 'hotline' or 'resources'."
)


def load(settings: Settings) -> None:
    global _client, _model_name

    base_url = settings.vllm_base_url
    logger.info("Connecting to vLLM at %s", base_url)

    _client = OpenAI(base_url=base_url, api_key="ignored")

    # Verify the server is up and the model is available
    try:
        models = _client.models.list()
        available = [m.id for m in models.data]
        logger.info("vLLM models available: %s", available)
        if _model_name not in available and available:
            _model_name = available[0]
            logger.info("Using model: %s", _model_name)
    except Exception as exc:
        logger.error("vLLM health check failed: %s", exc)
        raise RuntimeError(
            f"Cannot reach vLLM server at {base_url}. "
            "Ensure the vLLM container is running (docker compose up vllm)."
        ) from exc

    logger.info("LLM ready — vLLM backend at %s (model: %s)", base_url, _model_name)


# ---------------------------------------------------------------------------
# Internal generation helper
# ---------------------------------------------------------------------------

def _generate(messages: List[Dict[str, str]], max_new_tokens: int, temperature: float) -> str:
    assert _client is not None
    t0 = time.perf_counter()
    response = _client.chat.completions.create(
        model=_model_name,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
    )
    elapsed = time.perf_counter() - t0
    content = response.choices[0].message.content or ""
    tokens = response.usage.completion_tokens if response.usage else 0
    tps = tokens / elapsed if elapsed > 0 else 0
    logger.debug("Generated %d tokens in %.2fs (%.1f tok/s)", tokens, elapsed, tps)
    return content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_system_prompt(task_label: str, rag_context: str) -> str:
    base = _TASK3_SYSTEM_PROMPT if task_label == "TASK_3" else _TASK1_SYSTEM_PROMPT
    if rag_context and task_label != "TASK_3":
        base += (
            f"\n\nKNOWLEDGE BASE REFERENCE:\n"
            f"{rag_context}\n\n"
            "Use this information to explain concepts clearly. "
            "After explaining, link it back to the student's situation or ask if they want to try it."
        )
    return base


def get_response(
    history: List[Dict[str, str]],
    task_label: str,
    rag_context: str = "",
) -> str:
    """Generate a CBT (TASK_1) or out-of-scope (TASK_3) response."""
    system_prompt = _build_system_prompt(task_label, rag_context)
    messages = [{"role": "system", "content": system_prompt}, *history]
    response = _generate(messages, max_new_tokens=512, temperature=0.6)

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
    assert _client is not None

    system_prompt = _build_system_prompt(task_label, rag_context)
    messages = [{"role": "system", "content": system_prompt}, *history]

    t0 = time.perf_counter()
    first_token = True
    token_count = 0

    with _client.chat.completions.create(
        model=_model_name,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=512,
        temperature=0.6,
        top_p=0.9,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token:
                    ttft = time.perf_counter() - t0
                    logger.debug("Time to first token: %.3fs", ttft)
                    first_token = False
                token_count += 1
                yield delta

    elapsed = time.perf_counter() - t0
    logger.debug("Stream complete: %d tokens in %.2fs (%.1f tok/s)", token_count, elapsed, token_count / elapsed if elapsed > 0 else 0)
