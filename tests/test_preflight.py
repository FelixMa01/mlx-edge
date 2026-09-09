"""Tests for preflight guardrails."""

from mlx_edge.preflight import check_prompt, estimate_tokens


def test_empty_prompt_ok():
    r = check_prompt("", available_memory_gb=16.0)
    assert r.ok
    assert r.severity == "ok"


def test_small_prompt_ok():
    r = check_prompt("hello world", available_memory_gb=16.0)
    assert r.ok
    assert r.severity == "ok"


def test_huge_prompt_blocks():
    huge = "x" * 200_000  # ~50k tokens, far over 32k safe context
    r = check_prompt(huge, available_memory_gb=16.0)
    assert not r.ok
    assert r.severity == "block"


def test_medium_prompt_warns():
    medium = "y" * 40_000  # ~10k tokens, over warn threshold
    r = check_prompt(medium, available_memory_gb=16.0, model_size_gb=4.0)
    assert r.ok
    assert r.severity == "warn"


def test_cjk_token_estimate():
    # CJK chars count roughly 2 per char (more dense than English)
    cjk = "你好世界" * 100  # 400 chars
    en = "hello world " * 100  # 1200 chars
    assert estimate_tokens(cjk) > estimate_tokens(en) // 2  # CJK denser


def test_block_when_exceeds_safe_context():
    prompt = "z" * (32768 * 4 + 100)  # >32k tokens
    r = check_prompt(prompt, available_memory_gb=16.0, advertised_context=40960)
    assert not r.ok
    assert r.severity == "block"
