#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "questionary>=2.0",
#   "rich>=13.0",
#   "httpx>=0.27",
# ]
# ///
"""mlx-claude — TUI launcher for Claude Code against local MLX models.

Pick a profile, the script spins up `mlx_lm.server` (text) or
`mlx_vlm.server` (vision-language) + LiteLLM proxy via `uvx`, sets
ANTHROPIC_* env vars, and execs `claude --model <alias>`. Cleans up
backend processes on exit. No persistent installs — everything goes
through uv's cache.

Each profile carries its model-author-recommended sampling defaults
(the Ollama-Modelfile equivalent) — they're injected into every
Claude Code request via the LiteLLM config.
"""

from __future__ import annotations
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import questionary
from rich.console import Console

console = Console()

MLX_PORT = 8080        # mlx_lm.server / mlx_vlm.server backend
SANITIZER_PORT = 8081  # strips tool_calls:[] from SSE chunks (LiteLLM bug workaround)
PROXY_PORT = 11434     # LiteLLM /v1/messages endpoint for Claude Code
PYTHON_VER = "3.12"
MIN_CONTEXT_WARN = 65536   # Ollama-docs recommended ≥64k for Claude Code

# Prebuilt PrismML-fork MLX wheel for the Bonsai 1-bit profile.
# Fetched lazily the first time a profile needs it.
FORK_WHEEL_NAME = "mlx-0.31.2.dev20260404+72ec298f-cp312-cp312-macosx_26_0_arm64.whl"
FORK_WHEEL_URL = (
    "https://github.com/braininahat/mlx-claude/releases/download/v0.1.0/"
    + FORK_WHEEL_NAME
)
FORK_WHEEL = Path.home() / ".local" / "share" / "mlx-claude" / "wheels" / FORK_WHEEL_NAME


@dataclass(frozen=True)
class SamplingParams:
    """Per-profile decoding defaults — the Ollama-Modelfile equivalent.

    Injected into every `/v1/chat/completions` request via LiteLLM.
    Request-level values from Claude Code (if any) override these.
    """
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class Profile:
    alias: str                      # what `claude --model <alias>` sends
    hf_model_id: str                # HuggingFace model id
    label: str                      # menu label
    backend: str = "mlx_lm"         # "mlx_lm" (text) or "mlx_vlm" (vision+text)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    max_kv_size: int = 65536        # enforced on mlx_vlm.server; advisory for mlx_lm.server
    extra_withs: tuple[str, ...] = ()   # extra --with args for uvx


PROFILES: list[Profile] = [
    Profile(
        alias="qwen",
        hf_model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        label="Qwen 3.5 9B 4-bit VLM    (mlx-vlm, vision+text)",
        backend="mlx_vlm",
        # Qwen3 family author-recommended decoding (non-thinking mode)
        sampling=SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
            repetition_penalty=1.05, max_tokens=8192,
        ),
        max_kv_size=65536,
    ),
    Profile(
        alias="bonsai",
        hf_model_id="prism-ml/Bonsai-8B-mlx-1bit",
        label="Bonsai 8B 1-bit           (mlx-lm + PrismML fork wheel)",
        backend="mlx_lm",
        # Per HF model card "Best Practices → Generation Parameters"
        sampling=SamplingParams(
            temperature=0.5, top_p=0.9, top_k=20, min_p=0.0,
            repetition_penalty=1.0, max_tokens=4096,
        ),
        max_kv_size=65536,
        extra_withs=(str(FORK_WHEEL),),
    ),
]


def die(msg: str, code: int = 1) -> None:
    console.print(f"[red]ERROR:[/] {msg}")
    sys.exit(code)


def check_prereqs(p: Profile) -> None:
    from shutil import which
    for cmd in ("uv", "uvx", "claude"):
        if which(cmd) is None:
            die(f"'{cmd}' not on PATH.")
    for extra in p.extra_withs:
        if os.path.sep in extra and not extra.startswith(("http://", "https://", "git+")):
            path_part = extra.split("@")[-1].strip() if "@" in extra else extra
            if not Path(path_part).is_file():
                die(f"wheel/path not found: {path_part}")


def write_litellm_config(p: Profile) -> Path:
    s = p.sampling
    cfg = Path(tempfile.mkstemp(prefix="litellm-", suffix=".yaml")[1])
    cfg.write_text(f"""\
model_list:
  - model_name: {p.alias}
    litellm_params:
      model: openai/{p.hf_model_id}
      api_base: http://127.0.0.1:{SANITIZER_PORT}/v1
      api_key: none
      temperature: {s.temperature}
      top_p: {s.top_p}
      max_tokens: {s.max_tokens}
      extra_body:
        top_k: {s.top_k}
        min_p: {s.min_p}
        repetition_penalty: {s.repetition_penalty}
litellm_settings:
  drop_params: true
""")
    return cfg


def wait_ready(url: str, timeout_s: int, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    die(f"{what} did not become ready at {url} within {timeout_s}s. Check logs.")


def start_proc(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    f = log_path.open("w")
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=proc_env,
    )


def shutdown(procs: list[subprocess.Popen]) -> None:
    for pr in procs:
        if pr.poll() is None:
            try:
                os.killpg(os.getpgid(pr.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    for pr in procs:
        try:
            pr.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def mlx_cmd(p: Profile) -> list[str]:
    """uvx invocation for mlx_lm.server (text) or mlx_vlm.server (vision)."""
    if p.backend == "mlx_vlm":
        cmd = ["uvx", "--python", PYTHON_VER, "--from", "mlx-vlm"]
        for w in p.extra_withs:
            cmd += ["--with", w]
        cmd += ["mlx_vlm.server", "--model", p.hf_model_id,
                "--port", str(MLX_PORT), "--host", "127.0.0.1"]
        if p.max_kv_size:
            cmd += ["--max-kv-size", str(p.max_kv_size)]
    else:  # mlx_lm
        cmd = ["uvx", "--python", PYTHON_VER, "--from", "mlx-lm"]
        for w in p.extra_withs:
            cmd += ["--with", w]
        cmd += ["mlx_lm.server", "--model", p.hf_model_id,
                "--port", str(MLX_PORT), "--host", "127.0.0.1"]
        # Server-side sampling defaults. LiteLLM sends per-request overrides
        # anyway, but these cover any request that reaches the server directly.
        cmd += [
            "--temp", str(p.sampling.temperature),
            "--top-p", str(p.sampling.top_p),
            "--top-k", str(p.sampling.top_k),
            "--min-p", str(p.sampling.min_p),
        ]
        # mlx_lm.server has no --max-kv-size flag yet (tracks ml-explore/mlx-lm#615)
    return cmd


def litellm_cmd(cfg: Path) -> list[str]:
    return [
        "uvx", "--python", PYTHON_VER, "--from", "litellm[proxy]",
        "litellm", "--config", str(cfg),
        "--port", str(PROXY_PORT), "--host", "127.0.0.1",
    ]


def sanitizer_cmd() -> list[str]:
    """Run sse_sanitizer.py as a uv script — it's a sibling file.
    `resolve()` handles the symlinked install (bootstrap.sh puts the
    bin symlink in ~/.local/bin and the real file in ~/.local/share)."""
    shim_path = Path(__file__).resolve().with_name("sse_sanitizer.py")
    return ["uv", "run", "--quiet", str(shim_path)]


def print_sampling_status(p: Profile) -> None:
    s = p.sampling
    console.print(
        f"[dim]  sampling:[/] temp={s.temperature} top_p={s.top_p} "
        f"top_k={s.top_k} min_p={s.min_p} rep={s.repetition_penalty} "
        f"max_tokens={s.max_tokens}"
    )
    console.print(f"[dim]  context:[/] max_kv_size={p.max_kv_size}")
    if p.max_kv_size and p.max_kv_size < MIN_CONTEXT_WARN:
        console.print(
            f"[yellow]  WARN:[/] max_kv_size < {MIN_CONTEXT_WARN} — "
            "Claude Code recommends ≥64k context"
        )


def run_smoke(choice: Profile) -> int:
    """Headless e2e test: hit /v1/messages non-stream + stream, assert valid."""
    url = f"http://127.0.0.1:{PROXY_PORT}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "dummy",
        "anthropic-version": "2023-06-01",
    }
    # Non-streaming
    body = {
        "model": choice.alias,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "reply with the single word OK"}],
    }
    console.print("[cyan]smoke: non-stream POST /v1/messages ...[/]")
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=300)
    except httpx.HTTPError as e:
        console.print(f"[red]SMOKE FAIL[/] non-stream request error: {e}")
        return 1
    if r.status_code != 200:
        console.print(
            f"[red]SMOKE FAIL[/] non-stream: HTTP {r.status_code} {r.text[:400]}"
        )
        return 1
    try:
        text = r.json()["content"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        console.print(f"[red]SMOKE FAIL[/] non-stream: bad body ({e}): {r.text[:400]}")
        return 1
    if not text.strip():
        console.print("[red]SMOKE FAIL[/] non-stream: empty content text")
        return 1
    preview = text.strip().replace("\n", " ")[:80]
    console.print(f"[green]SMOKE OK[/] non-stream: {preview!r}")

    # Streaming
    stream_body = {
        "model": choice.alias,
        "max_tokens": 20,
        "stream": True,
        "messages": [{"role": "user", "content": "count from one to three"}],
    }
    console.print("[cyan]smoke: stream POST /v1/messages ...[/]")
    saw_message_start = False
    saw_text_delta = False
    try:
        with httpx.stream(
            "POST", url, json=stream_body, headers=headers, timeout=300,
        ) as r:
            if r.status_code != 200:
                console.print(
                    f"[red]SMOKE FAIL[/] stream: HTTP {r.status_code}"
                )
                return 1
            for line in r.iter_lines():
                if line.startswith("event: message_start"):
                    saw_message_start = True
                if line.startswith("data: ") and "text_delta" in line:
                    saw_text_delta = True
    except httpx.HTTPError as e:
        console.print(f"[red]SMOKE FAIL[/] stream request error: {e}")
        return 1
    if not saw_message_start:
        console.print("[red]SMOKE FAIL[/] stream: no 'event: message_start'")
        return 1
    if not saw_text_delta:
        console.print("[red]SMOKE FAIL[/] stream: no text_delta events")
        return 1
    console.print("[green]SMOKE OK[/] stream: message_start + text_delta observed")
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="TUI launcher for Claude Code against local MLX models",
        add_help=True,
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="run headless e2e HTTP tests against the stack, then exit (no claude UI)",
    )
    parser.add_argument(
        "--profile", help="select profile by alias (skips the menu)",
    )
    args, claude_args = parser.parse_known_args()

    choice: Profile | None = None
    if args.profile:
        choice = next((p for p in PROFILES if p.alias == args.profile), None)
        if choice is None:
            aliases = ", ".join(p.alias for p in PROFILES)
            die(f"profile {args.profile!r} not found. available: {aliases}")
    else:
        choice = questionary.select(
            "Pick an MLX model for Claude Code:",
            choices=[questionary.Choice(title=p.label, value=p) for p in PROFILES],
        ).ask()
        if choice is None:
            return 0

    check_prereqs(choice)
    cfg = write_litellm_config(choice)
    mlx_log = Path(f"/tmp/mlx-{MLX_PORT}.log")
    proxy_log = Path(f"/tmp/litellm-{PROXY_PORT}.log")
    procs: list[subprocess.Popen] = []

    try:
        console.print(f"[cyan]Starting {choice.backend}.server[/] "
                      f"([dim]{choice.hf_model_id}[/]) on :{MLX_PORT} ...")
        print_sampling_status(choice)
        procs.append(start_proc(mlx_cmd(choice), mlx_log))
        wait_ready(
            f"http://127.0.0.1:{MLX_PORT}/v1/models", 600,
            f"{choice.backend}.server",
        )
        console.print("  [green]ready[/]")

        sanitizer_log = Path(f"/tmp/sanitizer-{SANITIZER_PORT}.log")
        console.print(f"[cyan]Starting SSE sanitizer[/] on :{SANITIZER_PORT} "
                      "[dim](strips tool_calls:[] from stream chunks)[/] ...")
        procs.append(start_proc(
            sanitizer_cmd(), sanitizer_log,
            env={"BACKEND_URL": f"http://127.0.0.1:{MLX_PORT}",
                 "SANITIZER_PORT": str(SANITIZER_PORT)},
        ))
        wait_ready(
            f"http://127.0.0.1:{SANITIZER_PORT}/v1/models", 60, "sse sanitizer",
        )
        console.print("  [green]ready[/]")

        console.print(f"[cyan]Starting LiteLLM proxy[/] on :{PROXY_PORT} ...")
        # LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES forces
        # /v1/messages -> /v1/chat/completions on the backend, instead of
        # the default /v1/responses routing for openai/ models (which MLX
        # servers don't expose). See litellm handler.py:36-44.
        procs.append(start_proc(
            litellm_cmd(cfg), proxy_log,
            env={"LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES": "true"},
        ))
        wait_ready(
            f"http://127.0.0.1:{PROXY_PORT}/v1/models", 120, "litellm proxy",
        )
        console.print("  [green]ready[/]")

        if args.smoke:
            return run_smoke(choice)

        env = os.environ.copy()
        env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"

        console.print(f"\n[bold]Launching:[/] claude --model {choice.alias}")
        console.print(f"[dim](logs: {mlx_log}  {proxy_log})[/]\n")

        return subprocess.run(
            ["claude", "--model", choice.alias, *claude_args], env=env
        ).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        console.print("[dim]stopping backends ...[/]")
        shutdown(procs)
        cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
