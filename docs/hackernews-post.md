# Show HN: mlx-edge — CLI-first MLX inference with preflight guardrails

## Title options (pick one — short, leads with hook):

A) **Show HN: mlx-edge — Apple Silicon LLM inference with prefill guardrails**
B) **Show HN: mlx-edge — A 30k-token prompt will silently half-stream; I built a CLI to stop that**
C) **Show HN: mlx-edge — `ollama run` for MLX, with the omlx 32k-context trap fixed**

Recommended: **B** (specific pain point → curiosity).

---

## Post body (final draft):

I shipped [mlx-edge](https://github.com/FelixMa01/mlx-edge) — a CLI for running local LLMs on Apple Silicon via MLX. There's already `madroidmaq/mlx-omni-server` (742 stars) which does OpenAI-compatible serving, but I kept hitting a problem nobody had codified:

**A prompt over ~30k tokens on an M5 Air would stream half-way, then throw `ChunkedEncodingError` with no server-side error.** Not a 5xx, not a 400 — just the stream terminating mid-SSE. No `omlx` log line, no Python traceback. After ~3 days of debugging I realized: it's prefill memory, not the advertised 40960-token context window. The model claims it can take it; the hardware disagrees.

`mlx-edge` ships with this fix baked in:

```
$ mlx-edge preflight "..." 
✓ estimated 8500 tokens; 32% of prefill budget; safe to send

$ mlx-edge run "..."
(Streams the response)

$ mlx-edge autoswap
✓ memory 22%; comfortable
```

Three things I'm doing differently from the 742-star incumbent:

1. **CLI-first**, not API-first — same workflow as `ollama run` or `docker run`. No docker-compose, no daemon config to maintain.
2. **Preflight guardrails built in.** Token estimate + memory check before you send anything. The clamp (`max_context_window=32768`, `max_tokens=16384`) is the default, not an option to discover after a 4-hour debugging session.
3. **Auto-swap.** Reads `/health engine_pool.current_model_memory` and recommends a smaller-quantized model when the ceiling is tight. Defaults to a 4-step ladder: 2b-4bit → 4b-4bit → 8b-4bit.

18 tests, ruff-clean, GitHub Actions matrix (3.10/3.11/3.12 on macos-14). MIT licensed.

Honest caveats:
- Doesn't include a model zoo — you bring your own HF repo id or local path.
- `mlx-lm` is an optional dep (`pip install mlx-edge[server]`) so the CLI works on Linux/Intel too.
- I haven't benchmarked on M1/M2 — only M5 Air 16GB.

Would love feedback on:
- Is the auto-swap ladder (4 quantization levels) the right granularity?
- Should the preflight token estimate use a real tokenizer (slower but accurate) instead of the heuristic (4.0 KB/prompt)?

---

## Submission tips:

- **Best time to post**: Tuesday-Thursday 8-10am US Eastern (HN peak). Avoid Friday/weekends.
- **Reply to every comment in the first 2 hours** — HN's algorithm weighs early engagement.
- **Don't link to PyPI** until you have one — link to GitHub directly. Comments like "why not on PyPI?" are fine but don't pre-empt.
- **If a founder of `mlx-omni-server` shows up**: thank them, link to their work, don't get defensive. "I built the missing CLI layer I wished existed" lands better than "this is better."

---

## Tags: #show #apple-silicon #mlx #llm #cli #local-ai