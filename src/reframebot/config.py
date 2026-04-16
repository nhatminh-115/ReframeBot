from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/reframebot -> src -> repo root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter_path: str = ""

    # --- Guardrail ---
    guardrail_path: str = ""
    guardrail_context_turns: int = 3
    guardrail_context_max_chars: int = 700

    # --- RAG ---
    rag_db_path: str = str(_REPO_ROOT / "rag_db")

    # --- Embedder (shared by router + RAG) ---
    router_embed_model: str = "all-MiniLM-L6-v2"

    # --- Crisis detection thresholds ---
    crisis_semantic_sim_threshold: float = 0.62
    crisis_semantic_sim_margin: float = 0.08
    crisis_confidence_threshold: float = 0.90

    # --- API ---
    cors_origins: list[str] = ["*"]
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v  # type: ignore[return-value]

    @field_validator("adapter_path", mode="before")
    @classmethod
    def resolve_adapter_path(cls, v: str) -> str:
        if v:
            return v
        # Fallback: look for common local checkpoint directories
        candidates = [
            _REPO_ROOT / "results_reframebot_DPO" / "checkpoint-90",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return v

    @field_validator("guardrail_path", mode="before")
    @classmethod
    def resolve_guardrail_path(cls, v: str) -> str:
        if v:
            return v
        candidates = [
            _REPO_ROOT / "guardrail_model_retrained" / "best",
            _REPO_ROOT / "guardrail_model" / "checkpoint-950",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # Return the first candidate as the expected path (will fail later with a clear error)
        return str(candidates[0])


settings = Settings()
