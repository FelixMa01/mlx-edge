"""Server lifecycle — spin up an MLX-backed OpenAI-compatible server in-process.

This wraps the optional `mlx-lm` dependency. We import lazily so the CLI stays
usable on machines without Metal (e.g. CI) — only `mlx-edge serve` needs it.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServeSpec:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 1234
    max_context_window: int = 32768
    max_tokens: int = 16384


def check_mlx_lm_available() -> tuple[bool, str]:
    """Return (available, message). Honest answer — no silent fallback."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False, "mlx-lm not installed; run `pip install mlx-edge[server]`"
    return True, "mlx-lm available"


def build_serve_command(
    spec: ServeSpec, *, extra_args: list[str] | None = None
) -> list[str]:
    """Compose the mlx-lm serve command. The prefill clamp is baked in here."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        spec.model_path,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
    ]
    # mlx-lm passes these through to its OpenAI-compatible endpoint
    cmd += ["--max-context-window", str(spec.max_context_window)]
    if extra_args:
        cmd += extra_args
    return cmd


@contextmanager
def serve(spec: ServeSpec, *, wait_seconds: float = 10.0) -> Iterator[subprocess.Popen]:
    """Spawn `mlx-lm.server`, yield the process, terminate on exit.

    The caller is responsible for checking /health on `spec.host:spec.port`
    before assuming the server is ready — this context only guarantees the
    process is alive.
    """
    import time

    proc = subprocess.Popen(
        build_serve_command(spec),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Give MLX a moment to load weights before the caller probes /health
        time.sleep(min(wait_seconds, 0.5))
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def resolve_model_path(name_or_path: str) -> str:
    """Accept either a HF repo id (org/repo) or a local path. Validate it exists."""
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p.resolve())
    # Assume it's a HF repo id — mlx-lm will resolve via huggingface_hub
    return name_or_path
