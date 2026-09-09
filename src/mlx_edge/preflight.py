"""Preflight guardrails.

Hard-won lesson (2026-08-30, M5 Air 16GB + qwen3-4b-4bit):
the real bottleneck for local inference is **prefill memory**, not the
advertised context window. A 30k-token prompt threw ChunkedEncodingError
even though the model advertises 40960. Setting `max_context_window=32768`
and `max_tokens=16384` fixed it.

This module encodes that lesson as a hard guardrail so a fresh install
won't silently produce a half-streamed response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Conservative defaults that worked on M5 Air 16GB with qwen3-4b-4bit.
DEFAULT_MAX_CONTEXT_WINDOW = 32768
DEFAULT_MAX_TOKENS = 16384

# Preflight threshold: warn user before they waste tokens on a request
# that's likely to OOM during prefill.
PREFLIGHT_PROMPT_WARN = 8_000  # tokens


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    severity: Literal["ok", "warn", "block"]
    reason: str
    suggested_max_tokens: int
    suggested_max_context: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English/code, 1.5 chars for CJK."""
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return (cjk + 1) // 2 + other // 4


def check_prompt(
    prompt: str,
    *,
    available_memory_gb: float = 16.0,
    advertised_context: int | None = None,
    model_size_gb: float = 4.0,
) -> PreflightResult:
    """Decide whether to run a prompt, warn, or block.

    Heuristic: prefill memory ≈ model_size_gb × (prompt_tokens / max_context).
    If estimated prefill > 50% of available memory, warn.
    If > 80%, block.
    """
    tokens = estimate_tokens(prompt)
    max_ctx = advertised_context or DEFAULT_MAX_CONTEXT_WINDOW
    safe_ctx = min(max_ctx, DEFAULT_MAX_CONTEXT_WINDOW)

    if tokens == 0:
        return PreflightResult(
            ok=True,
            severity="ok",
            reason="empty prompt",
            suggested_max_tokens=DEFAULT_MAX_TOKENS,
            suggested_max_context=safe_ctx,
        )

    # Conservative prefill estimate
    prefill_gb = model_size_gb * (tokens / safe_ctx)
    ratio = prefill_gb / available_memory_gb

    if tokens > safe_ctx:
        return PreflightResult(
            ok=False,
            severity="block",
            reason=f"prompt ~{tokens} tokens exceeds safe context {safe_ctx}",
            suggested_max_tokens=DEFAULT_MAX_TOKENS,
            suggested_max_context=safe_ctx,
        )

    if ratio > 0.8:
        return PreflightResult(
            ok=False,
            severity="block",
            reason=(
                f"prefill ~{prefill_gb:.1f}GB / {available_memory_gb}GB available "
                f"= {ratio:.0%}; too tight, will likely OOM"
            ),
            suggested_max_tokens=DEFAULT_MAX_TOKENS,
            suggested_max_context=safe_ctx,
        )

    if ratio > 0.5 or tokens > PREFLIGHT_PROMPT_WARN:
        return PreflightResult(
            ok=True,
            severity="warn",
            reason=(
                f"prefill ~{prefill_gb:.1f}GB / {available_memory_gb}GB "
                f"({ratio:.0%}); large prompt, expect slower response"
            ),
            suggested_max_tokens=DEFAULT_MAX_TOKENS,
            suggested_max_context=safe_ctx,
        )

    return PreflightResult(
        ok=True,
        severity="ok",
        reason=f"prompt ~{tokens} tokens, comfortable",
        suggested_max_tokens=DEFAULT_MAX_TOKENS,
        suggested_max_context=safe_ctx,
    )
