"""Tests for auto-swap policy."""

from mlx_edge.autoswap import DEFAULT_LADDER, decide
from mlx_edge.health import EngineState


def _state(
    memory_gb: float, ceiling_gb: float, model: str | None = "qwen3-4b-4bit"
) -> EngineState:
    return EngineState(
        model_count=4,
        loaded_count=1,
        current_model_memory_gb=memory_gb,
        final_ceiling_gb=ceiling_gb,
        default_model=model,
    )


def test_comfortable_state_is_ok():
    decision = decide(_state(2.0, 10.0))
    assert decision.severity == "ok"
    assert decision.recommended_model is None


def test_warning_state_no_swap():
    decision = decide(_state(6.0, 10.0))  # 60% — over warn (0.5), under swap (0.7)
    assert decision.severity == "warn"


def test_swap_to_smaller_model():
    decision = decide(_state(8.0, 10.0))  # 80% — over swap threshold
    assert decision.severity == "swap"
    assert decision.recommended_model == DEFAULT_LADDER[0]  # qwen2-VL-2b


def test_block_when_memory_critical():
    decision = decide(_state(9.5, 10.0))  # 95%
    assert decision.severity == "block"
    assert decision.recommended_model is not None


def test_zero_ceiling_falls_back_to_ok():
    decision = decide(_state(1.0, 0.0))
    assert decision.severity == "ok"


def test_unknown_model_at_smallest_still_warns():
    decision = decide(_state(8.0, 10.0, model="unknown-model"))
    # Unknown not in ladder; can't suggest smaller, just warn
    assert decision.severity == "warn"
