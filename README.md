# bonsai-claude

[![PyPI](https://img.shields.io/pypi/v/bonsai-claude.svg)](https://pypi.org/project/bonsai-claude/)
[![Python](https://img.shields.io/pypi/pyversions/bonsai-claude.svg)](https://pypi.org/project/bonsai-claude/)
[![License](https://img.shields.io/github/license/braininahat/bonsai-claude.svg)](https://github.com/braininahat/bonsai-claude/blob/master/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/bonsai-claude.svg)](https://pypi.org/project/bonsai-claude/)

Run [Claude Code](https://docs.claude.com/en/docs/claude-code) locally on [Bonsai 8B 1-bit](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) — [PrismML](https://prismml.com)'s 1-bit quantized Qwen3-8B — via Apple [MLX](https://github.com/ml-explore/mlx). No Anthropic API key; no tokens leave your Mac.

## Install

```bash
uv tool install bonsai-claude
```

Then:

```bash
bonsai-claude
```

(First run auto-downloads the 55 MB PrismML-fork MLX wheel + the Bonsai model weights from HuggingFace.)

Run ephemerally without installing:

```bash
uvx bonsai-claude
```

## Requirements

- **Apple Silicon Mac** (M1 or newer)
- **macOS 26+** (the prebuilt fork wheel is tagged `macosx_26_0_arm64`)
- [`uv`](https://docs.astral.sh/uv/) on PATH — install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [`claude`](https://docs.claude.com/en/docs/claude-code/setup) CLI on PATH

Python 3.12 is managed by uv automatically.

## How it works

Claude Code speaks the **Anthropic** API shape (`POST /v1/messages`). MLX's server only speaks the **OpenAI** shape. So `ANTHROPIC_BASE_URL` can't point directly at it — a translator sits between.

```
claude CLI ──POST /v1/messages──▶ anthropic_shim :11434 ──POST /v1/chat/completions──▶ mlx_lm.server :8080 ──▶ Bonsai
            (Anthropic shape)       (direct adapter)         (OpenAI shape)
```

The adapter is ported from [ollama/anthropic/anthropic.go](https://github.com/ollama/ollama/blob/main/anthropic/anthropic.go) (MIT — attribution in [`NOTICE`](NOTICE)). It handles request/response translation and the streaming state machine — including the `input_json_delta` events for tool_calls that LiteLLM's chat→anthropic adapter fails to emit.

## Usage

```bash
bonsai-claude                         # interactive: pick context + --bare, then launch
bonsai-claude --non-interactive       # skip prompts, use saved prefs or defaults
bonsai-claude --smoke                 # headless HTTP round-trip test, then exit
bonsai-claude --panes                 # also open iTerm2 windows: log tail + macmon
bonsai-claude <claude args passed through>
```

Per-project preferences (`max_kv_size`, `--bare` choice) are saved at `~/.mlx_claude/prefs.json` keyed by CWD.

## Why Bonsai + 1-bit?

Bonsai is an 8B-parameter model in ~1 GB of weights — a ~8× memory reduction vs fp16. It fits in system RAM on M1 Macs that normally can't serve 8B models. The PrismML fork of `mlx` adds the 1-bit quant kernels needed to run it; the wheel is pinned and auto-fetched.

Prefill rate: ~100-150 tok/s on M-series chips (1-bit saves memory bandwidth but not FLOPs, so prefill is compute-bound). Generation: faster. `--bare` strips Claude Code's default context to keep turn-1 fast.

## Caveats

- **Tool-call quality**: Bonsai scores ~65.7 on the Berkeley Function Calling Leaderboard. Good enough for most Claude Code flows but weaker than frontier models on complex tool orchestration.
- **Large-context slowness**: turn-1 with full context can take minutes on 1-bit quant. Use `--bare` (the TUI's default) to shrink Claude Code's system prompt 10-20×.
- **Prefix KV cache is in-memory only**: restart the stack, the cache resets. Turn 2+ within a session reuses automatically.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attributions.
