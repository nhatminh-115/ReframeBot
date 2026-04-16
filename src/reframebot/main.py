"""FastAPI application — entry point for the ReframeBot API.

Startup (lifespan):
  1. Load guardrail classifier + shared embedder
  2. Load RAG database (optional — gracefully disabled if missing)
  3. Load LLM (base model + DPO adapter)

Endpoints:
  GET  /         — health check
  POST /chat     — main conversation endpoint
"""
from __future__ import annotations

import json as _json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Iterator, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from reframebot.config import settings
from reframebot.constants import VIETNAMESE_HOTLINES
from reframebot.router import resolve_task
from reframebot.services import guardrail, llm, rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ReframeBot startup ===")
    guardrail.load(settings)
    rag.load(settings, embedder=guardrail.get_embedder())
    llm.load(settings)
    logger.info("=== All models ready ===")
    yield
    logger.info("=== ReframeBot shutdown ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="ReframeBot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    history: List[Dict[str, str]]


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Shared routing logic
# ---------------------------------------------------------------------------

def _resolve(
    history: List[Dict[str, str]],
) -> Tuple[Optional[str], str]:
    """Run crisis detection + guardrail routing.

    Returns (effective_label, rag_context).
    effective_label is None when a hard crisis is detected — caller must
    handle the crisis path (empathy + hotlines) separately.
    """
    last_user_prompt = history[-1]["content"]

    crisis_info = guardrail.detect_crisis(
        last_user_prompt,
        sim_threshold=settings.crisis_semantic_sim_threshold,
        sim_margin=settings.crisis_semantic_sim_margin,
    )
    if crisis_info["is_crisis"]:
        logger.info(
            "Crisis detected — keyword=%s semantic=%s crisis_sim=%.3f academic_sim=%.3f",
            crisis_info["keyword"],
            crisis_info["semantic"],
            crisis_info["crisis_sim"],
            crisis_info["academic_sim"],
        )
        return None, ""

    guardrail_text = guardrail.build_guardrail_input(
        history,
        context_turns=settings.guardrail_context_turns,
        max_chars=settings.guardrail_context_max_chars,
    )
    classifier_result = guardrail.classify(guardrail_text)
    label: str = classifier_result["label"]
    score: float = classifier_result["score"]
    logger.info("Guardrail: label=%s score=%.4f", label, score)

    effective_label = resolve_task(
        guardrail_label=label,
        guardrail_score=score,
        crisis_confidence_threshold=settings.crisis_confidence_threshold,
        history=history,
    )
    logger.info("Effective label: %s", effective_label)

    rag_context = ""
    if effective_label == "TASK_1":
        rag_context = rag.retrieve_knowledge(last_user_prompt, top_k=2)
        if rag_context:
            logger.info("RAG: retrieved %d chars of context", len(rag_context))

    return effective_label, rag_context


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "ReframeBot API is running."}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    history = request.history
    if not history:
        return ChatResponse(response="Hello! Please start the conversation.")

    logger.info("Request: %s", history[-1]["content"][:100])
    effective_label, rag_context = _resolve(history)

    if effective_label is None:
        empathy = llm.get_crisis_empathy(history)
        return ChatResponse(response=f"{empathy}\n\n{VIETNAMESE_HOTLINES}")

    return ChatResponse(response=llm.get_response(history, effective_label, rag_context=rag_context))


@app.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """SSE endpoint — yields `data: {"token": "..."}` chunks, ends with `data: [DONE]`."""
    history = request.history

    if not history:
        def _empty() -> Iterator[str]:
            yield f"data: {_json.dumps({'token': 'Hello! Please start the conversation.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    logger.info("Stream request: %s", history[-1]["content"][:100])
    effective_label, rag_context = _resolve(history)

    if effective_label is None:
        empathy = llm.get_crisis_empathy(history)
        full_response = f"{empathy}\n\n{VIETNAMESE_HOTLINES}"
        def _crisis() -> Iterator[str]:
            yield f"data: {_json.dumps({'token': full_response})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_crisis(), media_type="text/event-stream")

    def _tokens() -> Iterator[str]:
        for token in llm.stream_response(history, effective_label, rag_context):
            if token:
                yield f"data: {_json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_tokens(), media_type="text/event-stream")
