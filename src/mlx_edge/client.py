"""OpenAI-compatible chat client with preflight + auto-swap."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from mlx_edge.preflight import PreflightResult, check_prompt


class PreflightError(RuntimeError):
    """Raised when preflight blocks the request."""


def chat_stream(
    prompt: str,
    *,
    base_url: str = "http://localhost:1234",
    model: str | None = None,
    available_memory_gb: float = 16.0,
    model_size_gb: float = 4.0,
    timeout: float = 120.0,
) -> Iterator[str]:
    """Stream a chat completion, with preflight check.

    Yields content deltas as they arrive.
    Raises PreflightError if the request would OOM during prefill.
    """
    pre: PreflightResult = check_prompt(
        prompt, available_memory_gb=available_memory_gb, model_size_gb=model_size_gb
    )
    if pre.severity == "block":
        raise PreflightError(pre.reason)

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": pre.suggested_max_tokens,
    }
    if model:
        payload["model"] = model

    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            try:
                import json
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue