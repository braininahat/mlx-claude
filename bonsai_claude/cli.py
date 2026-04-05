"""bonsai-claude — Run Claude Code on the Bonsai 8B 1-bit MLX model.

Boots mlx_lm.server with the PrismML fork (required for 1-bit quant)
plus a direct Anthropic↔OpenAI adapter, then execs `claude` pointed at
the adapter. All backends clean up on exit.
"""

from __future__ import annotations
import argparse
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

MLX_PORT = 8080      # mlx_lm.server backend
PROXY_PORT = 11434   # anthropic_shim /v1/messages endpoint for Claude Code
PYTHON_VER = "3.12"
MIN_CONTEXT_WARN = 65536
PREFS_PATH = Path.home() / ".mlx_claude" / "prefs.json"

# Prebuilt PrismML-fork MLX wheel (required for 1-bit quantization support).
FORK_WHEEL_NAME = "mlx-0.31.2.dev20260404+72ec298f-cp312-cp312-macosx_26_0_arm64.whl"
FORK_WHEEL_URL = (
    "https://github.com/braininahat/bonsai-claude/releases/download/v0.1.0/"
    + FORK_WHEEL_NAME
)
FORK_WHEEL = Path.home() / ".local" / "share" / "mlx-claude" / "wheels" / FORK_WHEEL_NAME


@dataclass(frozen=True)
class SamplingParams:
    """Per-Bonsai HF-card-recommended generation defaults."""
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class Profile:
    alias: str
    hf_model_id: str
    label: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    max_kv_size: int = 65536


BONSAI = Profile(
    alias="bonsai",
    hf_model_id="prism-ml/Bonsai-8B-mlx-1bit",
    label="Bonsai 8B 1-bit",
)


def die(msg: str, code: int = 1) -> None:
    console.print(f"[red]ERROR:[/] {msg}")
    sys.exit(code)


# --- Fork wheel: check, auto-download on first run --------------------------

def ensure_fork_wheel() -> None:
    if FORK_WHEEL.exists():
        return
    console.print(
        f"[cyan]Fetching PrismML-fork MLX wheel[/] "
        f"[dim]({FORK_WHEEL_NAME})[/] ..."
    )
    FORK_WHEEL.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream(
            "GET", FORK_WHEEL_URL, follow_redirects=True, timeout=300,
        ) as r:
            r.raise_for_status()
            with FORK_WHEEL.open("wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    f.write(chunk)
    except httpx.HTTPError as e:
        FORK_WHEEL.unlink(missing_ok=True)
        die(f"failed to download fork wheel: {e}\n"
            f"manual download: {FORK_WHEEL_URL}\n"
            f"save to: {FORK_WHEEL}")
    size_mb = FORK_WHEEL.stat().st_size / (1024 * 1024)
    console.print(f"  [green]saved[/] {FORK_WHEEL} ({size_mb:.1f} MB)")


def check_prereqs() -> None:
    from shutil import which
    for cmd in ("uv", "uvx", "claude"):
        if which(cmd) is None:
            die(
                f"'{cmd}' not on PATH.\n"
                "  uv:     curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                "  claude: https://docs.claude.com/en/docs/claude-code/setup"
            )


# --- Project preferences ----------------------------------------------------

def load_project_prefs() -> dict | None:
    if not PREFS_PATH.is_file():
        return None
    try:
        data = json.loads(PREFS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != 1:
        return None
    return data.get("projects", {}).get(str(Path.cwd().resolve()))


def save_project_prefs(max_kv_size: int, bare: bool) -> None:
    key = str(Path.cwd().resolve())
    try:
        data = (json.loads(PREFS_PATH.read_text())
                if PREFS_PATH.is_file() else {"version": 1, "projects": {}})
        if data.get("version") != 1:
            data = {"version": 1, "projects": {}}
    except (json.JSONDecodeError, OSError):
        data = {"version": 1, "projects": {}}
    data.setdefault("projects", {})[key] = {
        "max_kv_size": max_kv_size,
        "bare": bare,
        "last_used": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


# --- Subprocess lifecycle ---------------------------------------------------

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
    cmd: list[str], log_path: Path, env: dict[str, str] | None = None,
) -> subprocess.Popen:
    f = log_path.open("w")
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=proc_env,
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
    s = p.sampling
    return [
        "uvx", "--python", PYTHON_VER, "--from", "mlx-lm",
        "--with", str(FORK_WHEEL),
        "mlx_lm.server", "--model", p.hf_model_id,
        "--port", str(MLX_PORT), "--host", "127.0.0.1",
        "--temp", str(s.temperature), "--top-p", str(s.top_p),
        "--top-k", str(s.top_k), "--min-p", str(s.min_p),
    ]


def shim_cmd() -> list[str]:
    """Launch the Anthropic↔OpenAI adapter as a sibling module."""
    return [sys.executable, "-m", "bonsai_claude.shim"]


def print_sampling_status(p: Profile) -> None:
    s = p.sampling
    console.print(
        f"[dim]  sampling:[/] temp={s.temperature} top_p={s.top_p} "
        f"top_k={s.top_k} min_p={s.min_p} rep={s.repetition_penalty} "
        f"max_tokens={s.max_tokens}"
    )
    console.print(f"[dim]  context:[/] max_kv_size={p.max_kv_size}")
    if p.max_kv_size < MIN_CONTEXT_WARN:
        console.print(
            f"[yellow]  WARN:[/] max_kv_size < {MIN_CONTEXT_WARN} — "
            "Claude Code recommends ≥64k context"
        )


# --- TUI prompts ------------------------------------------------------------

_CONTEXT_CHOICES: list[tuple[str, int | str]] = [
    ("4k  (4096)",   4096),
    ("16k (16384)",  16384),
    ("32k (32768)",  32768),
    ("65k (65536)  — default / Claude Code recommended", 65536),
    ("128k (131072)", 131072),
    ("custom ...",   "custom"),
]


def pick_context_size(default: int) -> int:
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
    if picked is None:
        return default
    if picked != "custom":
        return int(picked)
    raw = questionary.text(
        "Enter context size (tokens):",
        default=str(default),
        validate=lambda s: s.isdigit() and int(s) > 0 or "positive integer, please",
    ).ask()
    if raw is None or not raw.isdigit():
        return default
    return int(raw)


# --- Optional iTerm2 monitoring windows ------------------------------------

def _iterm_spawn_window(cmd: str, title: str) -> bool:
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
    from shutil import which
    term = os.environ.get("TERM_PROGRAM", "unset")
    if term != "iTerm.app":
        console.print(
            f"[yellow]  --panes:[/] TERM_PROGRAM={term} (iTerm2 required), skipping"
        )
        return
    if _iterm_spawn_window(f"tail -f {mlx_log}", "bonsai-claude: server log"):
        console.print(f"  [green]pane:[/] tailing {mlx_log}")
    if which("macmon"):
        if _iterm_spawn_window("macmon", "bonsai-claude: GPU"):
            console.print("  [green]pane:[/] macmon")
    elif which("asitop"):
        if _iterm_spawn_window("sudo asitop", "bonsai-claude: GPU"):
            console.print("  [green]pane:[/] asitop (needs sudo)")
    else:
        console.print(
            "[yellow]  --panes:[/] no GPU monitor on PATH "
            "(install: brew install vladkens/tap/macmon)"
        )


# --- Headless smoke test ----------------------------------------------------

def run_smoke(profile: Profile) -> int:
    url = f"http://127.0.0.1:{PROXY_PORT}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "dummy",
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": profile.alias,
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
        console.print(f"[red]SMOKE FAIL[/] non-stream: HTTP {r.status_code} {r.text[:400]}")
        return 1
    try:
        text = r.json()["content"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        console.print(f"[red]SMOKE FAIL[/] non-stream: bad body ({e}): {r.text[:400]}")
        return 1
    if not text.strip():
        console.print("[red]SMOKE FAIL[/] non-stream: empty content")
        return 1
    preview = text.strip().replace("\n", " ")[:80]
    console.print(f"[green]SMOKE OK[/] non-stream: {preview!r}")

    stream_body = {
        "model": profile.alias,
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
                console.print(f"[red]SMOKE FAIL[/] stream: HTTP {r.status_code}")
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


# --- Main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bonsai-claude",
        description="Run Claude Code on the Bonsai 8B 1-bit MLX model.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="run headless HTTP tests against the stack, then exit",
    )
    parser.add_argument(
        "--panes", action="store_true",
        help="open iTerm2 windows tailing the server log + macmon GPU monitor",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="skip interactive prompts; use prefs or defaults",
    )
    args, claude_args = parser.parse_known_args()

    check_prereqs()
    ensure_fork_wheel()

    profile = BONSAI
    use_bare = True

    if args.smoke or args.non_interactive:
        prefs = load_project_prefs()
        if prefs:
            profile = replace(profile, max_kv_size=int(prefs.get("max_kv_size") or profile.max_kv_size))
            use_bare = bool(prefs.get("bare", True))
    else:
        prefs = load_project_prefs()
        if prefs is not None:
            console.print(
                f"[dim]  prefs: {profile.alias} / "
                f"{prefs.get('max_kv_size')} / bare={prefs.get('bare')} "
                f"(last used {prefs.get('last_used', '?')})[/]"
            )
        default_kv = int(prefs.get("max_kv_size")) if prefs else profile.max_kv_size
        kv = pick_context_size(default_kv)
        if kv != profile.max_kv_size:
            profile = replace(profile, max_kv_size=kv)
        bare_default = prefs.get("bare") if prefs else True
        bare_ans = questionary.confirm(
            "Launch with --bare? (skips CLAUDE.md/hooks/plugins/MCP autoloading — huge prefill win)",
            default=bool(bare_default),
        ).ask()
        if bare_ans is None:
            return 0
        use_bare = bool(bare_ans)
        save_project_prefs(profile.max_kv_size, use_bare)

    mlx_log = Path(f"/tmp/bonsai-mlx-{MLX_PORT}.log")
    shim_log = Path(f"/tmp/bonsai-shim-{PROXY_PORT}.log")
    procs: list[subprocess.Popen] = []

    try:
        console.print(
            f"[cyan]Starting mlx_lm.server[/] "
            f"([dim]{profile.hf_model_id}[/]) on :{MLX_PORT} ..."
        )
        print_sampling_status(profile)
        procs.append(start_proc(mlx_cmd(profile), mlx_log))
        wait_ready(
            f"http://127.0.0.1:{MLX_PORT}/v1/models", 600, "mlx_lm.server",
        )
        console.print("  [green]ready[/]")

        console.print(
            f"[cyan]Starting anthropic_shim[/] on :{PROXY_PORT} "
            "[dim](Anthropic↔OpenAI adapter)[/] ..."
        )
        procs.append(start_proc(
            shim_cmd(), shim_log,
            env={
                "BACKEND_URL": f"http://127.0.0.1:{MLX_PORT}",
                "MODEL_ALIAS": profile.alias,
                "MODEL_ID": profile.hf_model_id,
                "SHIM_PORT": str(PROXY_PORT),
                "SHIM_HOST": "127.0.0.1",
            },
        ))
        wait_ready(
            f"http://127.0.0.1:{PROXY_PORT}/v1/models", 60, "anthropic_shim",
        )
        console.print("  [green]ready[/]")

        if args.smoke:
            return run_smoke(profile)

        if args.panes:
            open_aux_panes(mlx_log)

        env = os.environ.copy()
        env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"

        claude_launch = ["claude", "--model", profile.alias]
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
