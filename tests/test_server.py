"""Tests for server lifecycle helpers (no real mlx-lm spawn)."""

from mlx_edge.server import ServeSpec, build_serve_command, resolve_model_path


def test_build_serve_command_bakes_in_prefill_clamp():
    spec = ServeSpec(model_path="mlx-community/Qwen2-VL-2B-Instruct-4bit", port=9999)
    cmd = build_serve_command(spec)
    assert "--max-context-window" in cmd
    assert "32768" in cmd  # the safe clamp
    assert "--port" in cmd
    assert "9999" in cmd
    assert cmd[-1] != ""  # no trailing empties


def test_resolve_model_path_accepts_local():
    spec = resolve_model_path(".")
    # "." resolves to the current dir which exists; should return absolute path
    assert spec.startswith("/")


def test_resolve_model_path_passes_through_hf_id():
    # Non-existent path; treat as HF repo id
    spec = resolve_model_path("mlx-community/Some-Model-Not-Local")
    assert spec == "mlx-community/Some-Model-Not-Local"


def test_check_mlx_lm_returns_honest_message():
    # Don't actually import mlx_lm — just verify the message string when missing
    import mlx_edge.server as srv

    avail, msg = srv.check_mlx_lm_available()
    # Either mlx-lm is installed (true) or we get the install hint
    if avail:
        assert "available" in msg
    else:
        assert "pip install mlx-edge[server]" in msg