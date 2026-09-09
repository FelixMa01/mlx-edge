"""CLI entry — `mlx-edge` command."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from mlx_edge import __version__
from mlx_edge.autoswap import SwapDecision, decide as autoswap_decide
from mlx_edge.health import EngineState, fetch_health, fetch_models
from mlx_edge.preflight import DEFAULT_MAX_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, check_prompt, estimate_tokens

app = typer.Typer(help="CLI-first local LLM inference for Apple Silicon via MLX.")
console = Console()


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"mlx-edge [bold cyan]{__version__}[/]")


@app.command()
def doctor(
    base_url: str = typer.Option("http://localhost:1234", help="OpenAI-compatible base URL"),
) -> None:
    """Check whether a local server is reachable and report engine state."""
    state = fetch_health(base_url)
    if state is None:
        console.print(f"[red]✗[/] server not reachable at {base_url}")
        raise typer.Exit(1)

    t = Table(title="Engine Health", show_header=False)
    t.add_row("default_model", str(state.default_model))
    t.add_row("models", str(state.model_count))
    t.add_row("loaded", str(state.loaded_count))
    t.add_row(
        "memory",
        f"{state.current_model_memory_gb}GB / {state.final_ceiling_gb}GB",
    )
    console.print(t)
    console.print(f"\nModels available: {', '.join(fetch_models(base_url))}")


@app.command()
def preflight(
    prompt: str = typer.Argument(..., help="Prompt text to check"),
    memory: float = typer.Option(16.0, "--memory", help="Available RAM in GB"),
    model_size: float = typer.Option(4.0, "--model-size", help="Model size in GB"),
) -> None:
    """Estimate whether a prompt will fit in prefill memory."""
    result = check_prompt(prompt, available_memory_gb=memory, model_size_gb=model_size)
    icon = {"ok": "✓", "warn": "⚠", "block": "✗"}[result.severity]
    color = {"ok": "green", "warn": "yellow", "block": "red"}[result.severity]
    console.print(f"[{color}]{icon}[/] {result.reason}")
    console.print(
        f"  suggested max_context={result.suggested_max_context} "
        f"max_tokens={result.suggested_max_tokens}"
    )
    if result.severity == "block":
        raise typer.Exit(2)


@app.command()
def tokens(
    text: str = typer.Argument(..., help="Text to estimate"),
) -> None:
    """Estimate token count for a string."""
    console.print(f"[cyan]{estimate_tokens(text)}[/] tokens (rough)")


@app.command()
def config_init(
    output: str = typer.Option("mlx-edge.yaml", help="Output config path"),
) -> None:
    """Write a starter config file with safe defaults."""
    cfg = {
        "server": {
            "base_url": "http://localhost:1234",
            "max_context_window": DEFAULT_MAX_CONTEXT_WINDOW,
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        "preflight": {
            "available_memory_gb": 16.0,
            "model_size_gb": 4.0,
        },
    }
    try:
        import yaml  # type: ignore[import-not-found]
        with open(output, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except ImportError:
        import json
        with open(output, "w") as f:
            json.dump(cfg, f, indent=2)
        console.print(f"[yellow]⚠[/] PyYAML not installed; wrote JSON to {output}")
        return
    console.print(f"[green]✓[/] wrote {output}")


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Prompt to send"),
    base_url: str = typer.Option("http://localhost:1234", help="OpenAI-compatible base URL"),
    model: str | None = typer.Option(None, "--model", help="Override model id"),
    memory: float = typer.Option(16.0, "--memory", help="Available RAM in GB"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming (wait for full reply)"),
) -> None:
    """Pre-flight check, then send a prompt and print the response."""
    pre = check_prompt(prompt, available_memory_gb=memory)
    icon = {"ok": "✓", "warn": "⚠", "block": "✗"}[pre.severity]
    color = {"ok": "green", "warn": "yellow", "block": "red"}[pre.severity]
    console.print(f"[{color}]{icon} preflight:[/] {pre.reason}")
    if pre.severity == "block":
        raise typer.Exit(2)

    import httpx

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": pre.suggested_max_tokens,
    }
    if model:
        payload["model"] = model

    if no_stream:
        r = httpx.post(f"{base_url}/v1/chat/completions", json={**payload, "stream": False}, timeout=120)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        console.print(content)
    else:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={**payload, "stream": True},
            timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        console.print(delta, end="")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            console.print()


@app.command()
def autoswap(
    base_url: str = typer.Option("http://localhost:1234", help="OpenAI-compatible base URL"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of pretty output"),
) -> None:
    """Inspect /health and recommend a model swap if memory is tight."""
    state = fetch_health(base_url)
    if state is None:
        console.print(f"[red]✗[/] server not reachable at {base_url}")
        raise typer.Exit(1)
    decision: SwapDecision = autoswap_decide(state)

    if json_output:
        console.print_json(
            data={
                "engine_state": {
                    "default_model": state.default_model,
                    "current_model_memory_gb": state.current_model_memory_gb,
                    "final_ceiling_gb": state.final_ceiling_gb,
                },
                "decision": {
                    "severity": decision.severity,
                    "reason": decision.reason,
                    "recommended_model": decision.recommended_model,
                    "ratio": decision.ratio,
                },
            }
        )
        return

    icon = {"ok": "✓", "warn": "⚠", "swap": "↻", "block": "✗"}[decision.severity]
    color = {
        "ok": "green",
        "warn": "yellow",
        "swap": "cyan",
        "block": "red",
    }[decision.severity]
    console.print(f"[{color}]{icon}[/] {decision.reason}")
    if decision.recommended_model:
        console.print(f"  recommended: [bold]{decision.recommended_model}[/]")


@app.command()
def serve_check() -> None:
    """Verify whether mlx-lm is installed (server mode prerequisite)."""
    from mlx_edge.server import check_mlx_lm_available

    ok, msg = check_mlx_lm_available()
    icon = "✓" if ok else "✗"
    color = "green" if ok else "red"
    console.print(f"[{color}]{icon}[/] {msg}")
    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    sys.exit(app())