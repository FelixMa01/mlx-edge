"""Tests for health probing (with mocked HTTP)."""

from unittest.mock import Mock, patch

from mlx_edge.health import EngineState, fetch_health


def test_fetch_health_unreachable():
    import httpx

    with patch("mlx_edge.health.httpx.get") as mock_get:
        mock_get.side_effect = httpx.ConnectError("connection refused")
        assert fetch_health("http://nope:9999") is None


def test_fetch_health_ok():
    fake = Mock()
    fake.json.return_value = {
        "status": "healthy",
        "default_model": "qwen3-4b-4bit",
        "engine_pool": {
            "model_count": 4,
            "loaded_count": 1,
            "current_model_memory": 2_376_173_655,
            "final_ceiling": 8_800_021_120,
        },
    }
    fake.raise_for_status = Mock()
    with patch("mlx_edge.health.httpx.get", return_value=fake):
        state = fetch_health()
        assert state is not None
        assert isinstance(state, EngineState)
        assert state.default_model == "qwen3-4b-4bit"
        assert state.model_count == 4
        assert state.current_model_memory_gb == 2.38
        assert state.final_ceiling_gb == 8.8
