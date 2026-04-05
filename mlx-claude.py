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

Pick a profile, the script spins up `mlx_lm.server` + LiteLLM proxy via
`uvx`, sets ANTHROPIC_* env vars, and execs `claude --model <alias>`.
Cleans up backend processes on exit. No persistent installs — everything
goes through uv's cache.
"""

from __future__ import annotations
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import questionary
from rich.console import Console

console = Console()

MLX_PORT = 8080
PROXY_PORT = 11434
PYTHON_VER = "3.12"
FORK_WHEEL = Path.home() / "Desktop" / \
    "mlx-0.31.2.dev20260404+72ec298f-cp312-cp312-macosx_26_0_arm64.whl"


@dataclass(frozen=True)
class Profile:
    alias: str                          # what `claude --model <alias>` sends
    hf_model_id: str                    # HuggingFace model id
    label: str                          # menu label
    extra_withs: tuple[str, ...] = ()   # extra --with args for uvx


PROFILES: list[Profile] = [
    Profile(
        alias="qwen",
        hf_model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        label="Qwen 3.5 9B MLX-4bit  (stock mlx via uvx)",
    ),
    Profile(
        alias="bonsai",
        hf_model_id="prism-ml/Bonsai-8B-mlx-1bit",
        label="Bonsai 8B MLX-1bit    (PrismML fork wheel via uvx --with)",
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
    cfg = Path(tempfile.mkstemp(prefix="litellm-", suffix=".yaml")[1])
    cfg.write_text(f"""\
model_list:
  - model_name: {p.alias}
    litellm_params:
      model: openai/{p.hf_model_id}
      api_base: http://127.0.0.1:{MLX_PORT}/v1
      api_key: none
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


def start_proc(cmd: list[str], log_path: Path) -> subprocess.Popen:
    f = log_path.open("w")
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    return subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
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
    cmd = ["uvx", "--python", PYTHON_VER, "--from", "mlx-lm"]
    for w in p.extra_withs:
        cmd += ["--with", w]
    cmd += ["mlx_lm.server", "--model", p.hf_model_id, "--port", str(MLX_PORT)]
    return cmd


def litellm_cmd(cfg: Path) -> list[str]:
    return [
        "uvx", "--python", PYTHON_VER, "--from", "litellm[proxy]",
        "litellm", "--config", str(cfg),
        "--port", str(PROXY_PORT), "--host", "127.0.0.1",
    ]


def main() -> int:
    choice: Profile | None = questionary.select(
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
        console.print(f"[cyan]Starting mlx_lm.server[/] "
                      f"([dim]{choice.hf_model_id}[/]) on :{MLX_PORT} ...")
        procs.append(start_proc(mlx_cmd(choice), mlx_log))
        wait_ready(f"http://127.0.0.1:{MLX_PORT}/v1/models", 600, "mlx_lm.server")
        console.print("  [green]ready[/]")

        console.print(f"[cyan]Starting LiteLLM proxy[/] on :{PROXY_PORT} ...")
        procs.append(start_proc(litellm_cmd(cfg), proxy_log))
        wait_ready(f"http://127.0.0.1:{PROXY_PORT}/v1/models", 120, "litellm proxy")
        console.print("  [green]ready[/]")

        env = os.environ.copy()
        env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"

        console.print(f"\n[bold]Launching:[/] claude --model {choice.alias}")
        console.print(f"[dim](logs: {mlx_log}  {proxy_log})[/]\n")

        return subprocess.run(
            ["claude", "--model", choice.alias, *sys.argv[1:]], env=env
        ).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        console.print("[dim]stopping backends ...[/]")
        shutdown(procs)
        cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
