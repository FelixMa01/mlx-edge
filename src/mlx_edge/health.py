"""Health probing — connect to a local OpenAI-compatible server and inspect state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class EngineState:
    model_count: int
    loaded_count: int
    current_model_memory_gb: float
    final_ceiling_gb: float
    default_model: str | None


def fetch_health(base_url: str = "http://localhost:1234", timeout: float = 5.0) -> EngineState | None:
    """Hit /health; return EngineState or None if server is down."""
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        r.raise_for_status()
    except (httpx.HTTPError, httpx.RequestError):
        return None

    data: dict[str, Any] = r.json()
    pool = data.get("engine_pool", {}) or {}
    return EngineState(
        model_count=int(pool.get("model_count", 0)),
        loaded_count=int(pool.get("loaded_count", 0)),
        current_model_memory_gb=round(int(pool.get("current_model_memory", 0)) / 1e9, 2),
        final_ceiling_gb=round(int(pool.get("final_ceiling", 0)) / 1e9, 2),
        default_model=data.get("default_model"),
    )


def fetch_models(base_url: str = "http://localhost:1234", timeout: float = 5.0) -> list[str]:
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=timeout)
        r.raise_for_status()
    except (httpx.HTTPError, httpx.RequestError):
        return []
    return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]