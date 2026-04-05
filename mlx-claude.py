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
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import httpx
import questionary
from rich.console import Console

console = Console()

MLX_PORT = 8080     # mlx_lm.server / mlx_vlm.server backend
PROXY_PORT = 11434  # anthropic_shim /v1/messages endpoint for Claude Code
PYTHON_VER = "3.12"
MIN_CONTEXT_WARN = 65536   # Ollama-docs recommended ≥64k for Claude Code
PREFS_PATH = Path.home() / ".mlx_claude" / "prefs.json"

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


# (LiteLLM + sanitizer removed; stack is now claude → anthropic_shim → mlx server)


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


def shim_cmd() -> list[str]:
    """Run anthropic_shim.py as a uv script — it's a sibling file.
    `resolve()` handles the symlinked install (install.sh puts the
    bin symlink in ~/.local/bin pointing at the repo checkout)."""
    shim_path = Path(__file__).resolve().with_name("anthropic_shim.py")
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


_CONTEXT_CHOICES: list[tuple[str, int | str]] = [
    ("4k  (4096)",   4096),
    ("16k (16384)",  16384),
    ("32k (32768)",  32768),
    ("65k (65536)  — default / Claude Code recommended", 65536),
    ("128k (131072)", 131072),
    ("custom ...",   "custom"),
]


def pick_context_size(default: int) -> int:
    """Second interactive prompt: override profile's max_kv_size. Returns
    the picked integer (falls back to `default` if the user aborts)."""
    # questionary's `default` wants one of the choice VALUES, not a label.
    # If the profile's default isn't in our preset list, pass None and let
    # questionary highlight the first option.
    default_value = next(
        (v for _, v in _CONTEXT_CHOICES if v == default), None,
    )
    picked = questionary.select(
        "Context window (max_kv_size):",
        choices=[
            questionary.Choice(title=lbl, value=v) for lbl, v in _CONTEXT_CHOICES
        ],
        default=default_value,
    ).ask()
    if picked is None:      # Ctrl-C
        return default
    if picked != "custom":  # preset selected
        return int(picked)
    raw = questionary.text(
        "Enter context size (tokens):",
        default=str(default),
        validate=lambda s: s.isdigit() and int(s) > 0 or "positive integer, please",
    ).ask()
    if raw is None or not raw.isdigit():
        return default
    return int(raw)


def load_project_prefs() -> dict | None:
    """Return {profile, max_kv_size, bare} for the current CWD, or None."""
    if not PREFS_PATH.is_file():
        return None
    try:
        data = json.loads(PREFS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != 1:
        return None
    return data.get("projects", {}).get(str(Path.cwd().resolve()))


def save_project_prefs(profile_alias: str, max_kv_size: int, bare: bool) -> None:
    """Merge current CWD's prefs into the file. Silent on any IO error."""
    key = str(Path.cwd().resolve())
    try:
        data = (json.loads(PREFS_PATH.read_text())
                if PREFS_PATH.is_file() else {"version": 1, "projects": {}})
        if data.get("version") != 1:
            data = {"version": 1, "projects": {}}
    except (json.JSONDecodeError, OSError):
        data = {"version": 1, "projects": {}}
    data.setdefault("projects", {})[key] = {
        "profile": profile_alias,
        "max_kv_size": max_kv_size,
        "bare": bare,
        "last_used": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def _iterm_spawn_window(cmd: str, title: str) -> bool:
    """Open a new iTerm2 window and run `cmd`. Returns True on success."""
    # AppleScript uses " as string delimiter — escape any " or \ in cmd.
    safe_cmd = cmd.replace("\\", "\\\\").replace('"', '\\"')
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "iTerm2"\n'
        "  activate\n"
        "  set newWin to (create window with default profile)\n"
        "  tell current session of newWin\n"
        f'    set name to "{safe_title}"\n'
        f'    write text "{safe_cmd}"\n'
        "  end tell\n"
        "end tell\n"
    )
    r = subprocess.run(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    return r.returncode == 0


def open_aux_panes(mlx_log: Path) -> None:
    """Spawn iTerm2 windows tailing the server log + macmon GPU monitor."""
    from shutil import which
    term = os.environ.get("TERM_PROGRAM", "unset")
    if term != "iTerm.app":
        console.print(
            f"[yellow]  --panes:[/] TERM_PROGRAM={term} (iTerm2 required), skipping"
        )
        return
    if _iterm_spawn_window(f"tail -f {mlx_log}", "mlx-claude: server log"):
        console.print(f"  [green]pane:[/] tailing {mlx_log}")
    if which("macmon"):
        if _iterm_spawn_window("macmon", "mlx-claude: GPU"):
            console.print("  [green]pane:[/] macmon")
    elif which("asitop"):
        if _iterm_spawn_window("sudo asitop", "mlx-claude: GPU"):
            console.print("  [green]pane:[/] asitop (needs sudo)")
    else:
        console.print(
            "[yellow]  --panes:[/] no GPU monitor on PATH "
            "(install: brew install vladkens/tap/macmon)"
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
    parser.add_argument(
        "--panes", action="store_true",
        help="open iTerm2 windows tailing the server log + macmon GPU monitor",
    )
    args, claude_args = parser.parse_known_args()

    choice: Profile | None = None
    use_bare = False
    if args.profile:
        choice = next((p for p in PROFILES if p.alias == args.profile), None)
        if choice is None:
            aliases = ", ".join(p.alias for p in PROFILES)
            die(f"profile {args.profile!r} not found. available: {aliases}")
    else:
        prefs = load_project_prefs()
        if prefs is not None:
            console.print(
                f"[dim]  prefs: {prefs.get('profile')} / "
                f"{prefs.get('max_kv_size')} / bare={prefs.get('bare')} "
                f"(last used {prefs.get('last_used', '?')})[/]"
            )
        default_profile = next(
            (p for p in PROFILES if prefs and p.alias == prefs.get("profile")), None,
        )
        choice = questionary.select(
            "Pick an MLX model for Claude Code:",
            choices=[questionary.Choice(title=p.label, value=p) for p in PROFILES],
            default=default_profile,
        ).ask()
        if choice is None:
            return 0
        # Second prompt: context-window override (skipped for --profile/--smoke).
        default_kv = prefs.get("max_kv_size") if prefs else choice.max_kv_size
        kv = pick_context_size(default_kv)
        if kv != choice.max_kv_size:
            choice = replace(choice, max_kv_size=kv)
        # Third prompt: --bare mode. Strongly recommended for non-Anthropic
        # backends — skips CLAUDE.md auto-discovery, hooks, plugin sync, MCP
        # discovery, auto-memory. Without it Claude Code slams the full
        # context into turn 1 and you wait minutes for prefill.
        bare_default = prefs.get("bare") if prefs else True
        bare_ans = questionary.confirm(
            "Launch with --bare? (skips CLAUDE.md/hooks/plugins/MCP autoloading — huge prefill win)",
            default=bool(bare_default),
        ).ask()
        if bare_ans is None:
            return 0
        use_bare = bool(bare_ans)
        save_project_prefs(choice.alias, choice.max_kv_size, use_bare)

    check_prereqs(choice)
    mlx_log = Path(f"/tmp/mlx-{MLX_PORT}.log")
    shim_log = Path(f"/tmp/anthropic-shim-{PROXY_PORT}.log")
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

        console.print(f"[cyan]Starting anthropic_shim[/] on :{PROXY_PORT} "
                      "[dim](Anthropic↔OpenAI adapter)[/] ...")
        procs.append(start_proc(
            shim_cmd(), shim_log,
            env={
                "BACKEND_URL": f"http://127.0.0.1:{MLX_PORT}",
                "MODEL_ALIAS": choice.alias,
                "MODEL_ID": choice.hf_model_id,
                "SHIM_PORT": str(PROXY_PORT),
                "SHIM_HOST": "127.0.0.1",
            },
        ))
        wait_ready(
            f"http://127.0.0.1:{PROXY_PORT}/v1/models", 60, "anthropic_shim",
        )
        console.print("  [green]ready[/]")

        if args.smoke:
            return run_smoke(choice)

        if args.panes:
            open_aux_panes(mlx_log)

        env = os.environ.copy()
        env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"

        claude_launch = ["claude", "--model", choice.alias]
        if use_bare:
            claude_launch.append("--bare")
        claude_launch.extend(claude_args)

        console.print(f"\n[bold]Launching:[/] {' '.join(claude_launch)}")
        console.print(f"[dim](logs: {mlx_log}  {shim_log})[/]\n")

        return subprocess.run(claude_launch, env=env).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        console.print("[dim]stopping backends ...[/]")
        shutdown(procs)


if __name__ == "__main__":
    raise SystemExit(main())
