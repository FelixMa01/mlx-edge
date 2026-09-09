"""Auto-swap: pick a model based on current engine memory pressure.

When /health reports current_model_memory is climbing, the right move is to swap
to a smaller quantization (or a different family entirely) before prefill fails.
This module reads /health, applies a policy, returns the recommended model id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlx_edge.health import EngineState


@dataclass(frozen=True)
class SwapPolicy:
    """Memory thresholds (fraction of ceiling) for swapping to smaller models."""

    warn_at: float = 0.5
    swap_at: float = 0.7
    block_at: float = 0.9

    @property
    def levels(self) -> tuple[float, ...]:
        return (self.warn_at, self.swap_at, self.block_at)


@dataclass(frozen=True)
class SwapDecision:
    severity: Literal["ok", "warn", "swap", "block"]
    reason: str
    recommended_model: str | None
    ratio: float


# Default fallback ladder — MLX quantized, smaller → larger.
DEFAULT_LADDER: tuple[str, ...] = (
    "mlx-community--Qwen2-VL-2B-Instruct-4bit",
    "qwen3-4b-4bit",
    "qwen3-4b-cmd-router",
    "qwen3-4b-cmd-v2",
)


def decide(state: EngineState, policy: SwapPolicy | None = None) -> SwapDecision:
    """Pick the right model for current memory pressure.

    Returns ok/warn (no change), swap (use smaller), or block (refuse to load more).
    """
    pol = policy or SwapPolicy()
    if state.final_ceiling_gb <= 0:
        return SwapDecision("ok", "no ceiling info; trust caller", None, 0.0)

    ratio = state.current_model_memory_gb / state.final_ceiling_gb

    if ratio >= pol.block_at:
        # Already over the line — recommend the smallest available
        return SwapDecision(
            severity="block",
            reason=f"memory {ratio:.0%} of ceiling; refuse new loads",
            recommended_model=DEFAULT_LADDER[0],
            ratio=ratio,
        )

    if ratio >= pol.swap_at:
        # Pick the next-smaller ladder rung
        current_idx = _find_ladder_index(state.default_model)
        if current_idx is not None and current_idx > 0:
            smaller = DEFAULT_LADDER[current_idx - 1]
            return SwapDecision(
                severity="swap",
                reason=f"memory {ratio:.0%}; swap to {smaller} to free headroom",
                recommended_model=smaller,
                ratio=ratio,
            )
        return SwapDecision(
            severity="warn",
            reason=f"memory {ratio:.0%}; already at smallest known model",
            recommended_model=state.default_model,
            ratio=ratio,
        )

    if ratio >= pol.warn_at:
        return SwapDecision(
            severity="warn",
            reason=f"memory {ratio:.0%}; under pressure but within limits",
            recommended_model=None,
            ratio=ratio,
        )

    return SwapDecision(
        severity="ok",
        reason=f"memory {ratio:.0%}; comfortable",
        recommended_model=None,
        ratio=ratio,
    )


def _find_ladder_index(model: str | None) -> int | None:
    if not model:
        return None
    for i, rung in enumerate(DEFAULT_LADDER):
        if model in rung or rung in model:
            return i
    return None
