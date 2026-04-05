# mlx-claude

A tiny TUI launcher that lets [Claude Code](https://docs.claude.com/en/docs/claude-code) talk to local [MLX](https://github.com/ml-explore/mlx) models on Apple Silicon.

Pick a model from a menu → the script spins up `mlx_lm.server` (text) or `mlx_vlm.server` (vision+text) + a [LiteLLM](https://github.com/BerriAI/litellm) translating proxy → launches `claude --model <alias>`. Everything runs through `uvx` (no persistent installs). Cleans up on exit.

## Why this exists

Claude Code speaks the **Anthropic** API shape (`POST /v1/messages`). MLX's servers only speak the **OpenAI** shape (`/v1/chat/completions`). So `ANTHROPIC_BASE_URL` can't point directly at them — a translator sits between. LiteLLM's `/v1/messages` endpoint does exactly that.

```
claude CLI ──POST /v1/messages──▶ LiteLLM :11434 ──POST /v1/chat/completions──▶ sse-sanitizer :8081 ──▶ mlx_{lm,vlm}.server :8080 ──▶ MLX model
            (Anthropic shape)       (translator)         (OpenAI shape)         (strips tool_calls:[])
```

The `sse-sanitizer` strips `tool_calls: []` from streaming chunks to work around a LiteLLM bug ([BerriAI/litellm#25172](https://github.com/BerriAI/litellm/issues/25172)) that drops text content when that field is present but empty.

## Install

```bash
git clone https://github.com/braininahat/mlx-claude
cd mlx-claude
./install.sh
```

Installs `~/.local/bin/mlx-claude` as a symlink to the clone, so the repo is your source of truth.

**Update:** `cd mlx-claude && git pull && ./install.sh`

**Uninstall:** `./uninstall.sh` from the clone.

## Requirements

- Apple Silicon Mac
- [`uv`](https://docs.astral.sh/uv/) on PATH — install first: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [`claude`](https://docs.claude.com/en/docs/claude-code/setup) CLI on PATH
- Python 3.12 (managed by `uv` automatically)

`mlx-lm` / `mlx-vlm` / `litellm` / TUI deps (`questionary`, `rich`, `httpx`) are fetched on demand by `uvx` on first run.

## Usage

```bash
mlx-claude              # menu → pick → claude launches
mlx-claude --help       # extra args forward to claude
```

## Bundled profiles

| Profile | Model | Backend | Notes |
|---|---|---|---|
| `qwen` | `mlx-community/Qwen3.5-9B-MLX-4bit` | `mlx_vlm.server` | Vision+text |
| `bonsai` | `prism-ml/Bonsai-8B-mlx-1bit` | `mlx_lm.server` | Text. Needs PrismML fork wheel (see below) |

## Parity with Ollama's Claude Code integration

What we carry over:

- **Env-var handshake** (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY=""`) — set automatically before launching `claude`.
- **Anthropic → OpenAI shape translation** via LiteLLM: tools, streaming, system prompts, vision (for VLM profiles).
- **Per-profile sampling defaults** (Ollama-Modelfile equivalent) baked into each `Profile` and injected as LiteLLM request defaults, so every request gets model-author-recommended decoding without the user tweaking anything.
- **Context window**: passed as `--max-kv-size` to `mlx_vlm.server`. For `mlx_lm.server` it's advisory only until upstream [ml-explore/mlx-lm#615](https://github.com/ml-explore/mlx-lm/issues/615) adds the CLI flag; a yellow warning fires if the profile asks for < 64k.
- **Startup transparency**: the launcher prints `temp/top_p/top_k/min_p/rep/max_tokens` and `max_kv_size` for the chosen profile so you know exactly what's about to serve.

## Using the Bonsai (1-bit) profile

Bonsai uses 1-bit quantization that isn't in upstream `mlx` yet. The [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx) fork adds it. Until upstream lands, the Bonsai profile injects the fork wheel via `uvx --with`.

**Prebuilt wheel** (Apple Silicon, Python 3.12, macOS 26+): download from the [latest release](../../releases/latest) and place at the path the script expects:

```bash
mkdir -p ~/Desktop
curl -L -o ~/Desktop/mlx-0.31.2.dev20260404+72ec298f-cp312-cp312-macosx_26_0_arm64.whl \
  <release-asset-url>
```

Or edit `FORK_WHEEL` at the top of `mlx-claude.py` to point wherever you keep it.

**Build the wheel yourself** (different Python/macOS combo):

```bash
# needs full Xcode + Metal Toolchain:
# sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
# xcodebuild -downloadComponent MetalToolchain

uv tool install mlx-lm \
  --with "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism" \
  --force --reinstall-package mlx --python 3.12
# wheel ends up in ~/.cache/uv/sdists-v9/git/...
```

## Adding more profiles

Append a `Profile(...)` in `mlx-claude.py`:

```python
Profile(
    alias="mistral",
    hf_model_id="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    label="Mistral 7B 4-bit (mlx-lm)",
    backend="mlx_lm",
    sampling=SamplingParams(
        temperature=0.7, top_p=0.95, top_k=50,
        repetition_penalty=1.0, max_tokens=4096,
    ),
    max_kv_size=32768,
),
```

For profiles needing a custom mlx build: `extra_withs=("<wheel-path-or-git-url>",)`.

## Verification

With a profile running, in another shell:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .

curl -s http://127.0.0.1:11434/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: dummy' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}' | jq .
```

Ctrl-C or `/exit` in Claude Code → script tears down both backends:

```bash
pgrep -fl 'mlx_lm.server|mlx_vlm.server|litellm' || echo clean
```

## Caveats

- **Tool-use quality**: Claude Code leans hard on tool-call formatting. Small local models may not keep up — model capability limit, not a proxy bug.
- **Chat template drift**: MLX uses the HF tokenizer's embedded chat template; usually correct, occasionally differs from GGUF/llama.cpp's template for the same model.
- **First-run download time**: multi-GB model weights on first invocation.

## License

MIT
