# mlx-edge

[![CI](https://github.com/FelixMa01/mlx-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/FelixMa01/mlx-edge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CLI-first local LLM inference for Apple Silicon via MLX. OpenAI-compatible server with **preflight guardrails** baked in.

## Why

Local inference on M-series Macs hits a real bottleneck that's not in the advertised context window: **prefill memory**. A 30k-token prompt can stream half-way and then throw `ChunkedEncodingError` with no server-side error. This tool encodes the workaround (clamp `max_context_window` to 32768, `max_tokens` to 16384, preflight at 8k tokens) so a fresh install won't get bitten.

## Install

```bash
# Server mode (needs MLX + a model)
pip install mlx-edge[server]

# CLI-only (just client + preflight + doctor)
pip install mlx-edge
```

## CLI

```bash
mlx-edge doctor                       # hit /health, print model + memory
mlx-edge preflight "your prompt"      # check if it'll fit in prefill memory
mlx-edge tokens "hello world"         # estimate token count
mlx-edge config-init                  # write a starter mlx-edge.yaml
```

## Python API

```python
from mlx_edge.client import chat_stream
from mlx_edge.preflight import check_prompt

# Pre-flight check before sending
result = check_prompt(long_prompt, available_memory_gb=16.0, model_size_gb=4.0)
if result.severity == "block":
    raise SystemExit(f"prompt too big: {result.reason}")

for chunk in chat_stream(long_prompt, base_url="http://localhost:1234"):
    print(chunk, end="", flush=True)
```

## Differentiators vs `mlx-omni-server` (742⭐)

- **CLI-first**, not API-first — same workflow as `ollama run`
- **Preflight guardrails** built in (the omlx prefill trap)
- **Auto-swap** model based on `/health engine_pool` (in progress)
- Single binary, no daemon config to maintain

## Status

v0.1.0 Day-1 scaffold. Components:

- `preflight.py` — token estimate + memory check
- `health.py` — server state inspection
- `client.py` — OpenAI-compatible streaming client
- `cli.py` — `typer` entry point

## Run tests

```bash
pip install -e .[test]
pytest -v
```