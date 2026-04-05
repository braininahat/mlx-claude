# mlx-claude

A tiny TUI launcher that lets [Claude Code](https://docs.claude.com/en/docs/claude-code) talk to local [MLX](https://github.com/ml-explore/mlx) models on Apple Silicon.

Pick a model from a menu → the script spins up `mlx_lm.server` + a [LiteLLM](https://github.com/BerriAI/litellm) translating proxy → launches `claude --model <alias>`. Everything runs through `uvx` (no persistent installs). Cleans up on exit.

## Why this exists

Claude Code speaks the **Anthropic** API shape (`POST /v1/messages`). `mlx_lm.server` only speaks the **OpenAI** shape (`/v1/chat/completions`). So `ANTHROPIC_BASE_URL` can't point directly at `mlx_lm.server` — there needs to be a translator in between. LiteLLM's `/v1/messages` endpoint does exactly that.

```
claude CLI ──POST /v1/messages──▶ LiteLLM :11434 ──POST /v1/chat/completions──▶ mlx_lm.server :8080 ──▶ MLX model
            (Anthropic shape)       (translator)           (OpenAI shape)
```

## Requirements

- Apple Silicon Mac
- [`uv`](https://docs.astral.sh/uv/) on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [`claude`](https://docs.claude.com/en/docs/claude-code/setup) CLI on PATH
- Python 3.12 (managed by `uv` automatically)

That's it. `mlx-lm`, `litellm`, and the TUI's own deps (`questionary`, `rich`, `httpx`) are fetched on demand by `uvx`.

## Usage

```bash
./mlx-claude.py              # menu → pick → claude launches
./mlx-claude.py --help       # extra args forward to claude
```

First run of each profile will download the model weights and populate `uv`'s cache. Subsequent runs are fast.

## Bundled profiles

| Profile | Model | Notes |
|---|---|---|
| `qwen` | `mlx-community/Qwen3.5-9B-MLX-4bit` | Stock upstream `mlx-lm` |
| `bonsai` | `prism-ml/Bonsai-8B-mlx-1bit` | Needs the PrismML `mlx` fork for 1-bit quant — see below |

## Using the Bonsai (1-bit) profile

The Bonsai model uses 1-bit quantization which isn't in upstream `mlx` yet. The [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx) fork adds support for it. Until that lands upstream, the profile injects the fork wheel via `uvx --with`.

**Prebuilt wheel (Apple Silicon, Python 3.12, macOS 26+):**
Download from the [latest release](../../releases/latest). Place it at the path the script expects:

```bash
mkdir -p ~/Desktop
curl -L -o ~/Desktop/mlx-0.31.2.dev20260404+72ec298f-cp312-cp312-macosx_26_0_arm64.whl \
  <release-asset-url>
```

Or edit `FORK_WHEEL` at the top of `mlx-claude.py` to point wherever you keep it.

**Building the wheel yourself** (if you're on a different Python/macOS combo):

```bash
# needs full Xcode + Metal Toolchain:
# xcode-select -s /Applications/Xcode.app/Contents/Developer
# xcodebuild -downloadComponent MetalToolchain

uv tool install mlx-lm \
  --with "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism" \
  --force --reinstall-package mlx --python 3.12
# wheel ends up in ~/.cache/uv/sdists-v9/git/...
```

## Adding more model profiles

Append a `Profile(...)` entry to `PROFILES` in `mlx-claude.py`:

```python
Profile(
    alias="mistral",
    hf_model_id="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    label="Mistral 7B 4bit (stock)",
),
```

For profiles that need a custom mlx build, set `extra_withs=("<wheel-path-or-git-url>",)`.

## Verification

With a profile running, in another shell:

```bash
# mlx backend:
curl -s http://127.0.0.1:8080/v1/models | jq .

# litellm translator (Anthropic shape):
curl -s http://127.0.0.1:11434/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: dummy' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}' | jq .
```

Ctrl-C or `/exit` in Claude Code → script tears down both backends. Confirm:

```bash
pgrep -fl 'mlx_lm.server|litellm' || echo clean
```

## Caveats

- **Context window**: Claude Code prefers ≥64k tokens. Default `mlx_lm.server` context may be smaller; pass `--max-kv-size` into `mlx_cmd(...)` if you hit truncation.
- **Tool-use quality**: Claude Code leans hard on tool-call formatting. Small local models may not keep up — model capability limit, not a proxy bug.
- **First-run download time**: multi-GB model weights; the 600s readiness timeout covers typical broadband.

## License

MIT
