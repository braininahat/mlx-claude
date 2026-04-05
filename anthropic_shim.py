#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastapi>=0.110",
#   "uvicorn[standard]>=0.29",
#   "httpx>=0.27",
# ]
# ///
"""anthropic_shim.py — Direct Anthropic Messages API <-> OpenAI Chat
Completions adapter. Replaces LiteLLM + sse_sanitizer in the stack.

Ported from ollama/anthropic/anthropic.go (MIT license, attribution in
NOTICE). Adapted from Ollama's internal api.ChatRequest/Response to
OpenAI chat/completions shape so it can sit in front of mlx_lm.server
and mlx_vlm.server.

Env:
  BACKEND_URL  (default http://127.0.0.1:8080)
  MODEL_ALIAS  — the alias the client will send in "model" field
  MODEL_ID     — the HF id forwarded to the backend's "model" field
  SHIM_HOST    (default 127.0.0.1)
  SHIM_PORT    (default 11434)
"""

from __future__ import annotations
import json
import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "mlx")
MODEL_ID = os.environ.get("MODEL_ID", MODEL_ALIAS)
HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "11434"))

app = FastAPI()


# --- Request translation: Anthropic Messages -> OpenAI chat/completions -----

def _flatten_system(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    # list of {type:"text", text:"..."}
    if isinstance(system, list):
        parts = []
        for b in system:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts) or None
    return None


def _content_to_openai(content: Any) -> tuple[str, list[dict], list[dict]]:
    """Convert an Anthropic message.content payload to OpenAI equivalents.
    Returns (text, image_parts, tool_use_parts, tool_result_parts)."""
    # Normalize: Anthropic allows string or [content blocks].
    if isinstance(content, str):
        return content, [], [], []
    text_parts: list[str] = []
    image_parts: list[dict] = []
    tool_use_parts: list[dict] = []
    tool_result_parts: list[dict] = []
    if not isinstance(content, list):
        return "", [], [], []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            text_parts.append(block.get("text") or "")
        elif t == "image":
            # Anthropic: {type:"image", source:{type:"base64",media_type,data}}
            src = block.get("source") or {}
            if src.get("type") == "base64":
                url = f"data:{src.get('media_type','image/png')};base64,{src.get('data','')}"
            else:
                url = src.get("url", "")
            if url:
                image_parts.append({"type": "image_url", "image_url": {"url": url}})
        elif t == "tool_use":
            tool_use_parts.append({
                "id": block.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
        elif t == "tool_result":
            c = block.get("content")
            if isinstance(c, list):
                c_text = "".join(
                    b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif isinstance(c, str):
                c_text = c
            else:
                c_text = json.dumps(c) if c is not None else ""
            tool_result_parts.append({
                "tool_use_id": block.get("tool_use_id"),
                "content": c_text,
                "is_error": bool(block.get("is_error")),
            })
    return "\n".join(p for p in text_parts if p), image_parts, tool_use_parts, tool_result_parts


def _msg_to_openai(msg: dict) -> list[dict]:
    """Anthropic MessageParam -> list of OpenAI messages."""
    role = msg.get("role", "user")
    text, images, tool_uses, tool_results = _content_to_openai(msg.get("content"))
    out: list[dict] = []
    if role == "assistant":
        m: dict[str, Any] = {"role": "assistant"}
        if text:
            m["content"] = text
        if tool_uses:
            m["tool_calls"] = tool_uses
            if "content" not in m:
                m["content"] = None
        out.append(m)
    elif role == "user":
        # Tool results become tool messages, one per tool_use_id.
        for tr in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"] or "",
                "content": tr["content"],
            })
        # Remaining text + images in a user message.
        if text or images:
            if images:
                content_parts: list[dict] = []
                if text:
                    content_parts.append({"type": "text", "text": text})
                content_parts.extend(images)
                out.append({"role": "user", "content": content_parts})
            else:
                out.append({"role": "user", "content": text})
    else:
        # fallback: treat like user text
        if text:
            out.append({"role": role, "content": text})
    return out


def _tool_to_openai(tool: dict) -> dict:
    """Anthropic Tool -> OpenAI function tool."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {
                "type": "object", "properties": {},
            },
        },
    }


def _tool_choice_to_openai(tc: Any) -> Any:
    if not isinstance(tc, dict):
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool" and tc.get("name"):
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def anthropic_to_openai(body: dict) -> dict:
    msgs: list[dict] = []
    sys_text = _flatten_system(body.get("system"))
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    for m in body.get("messages") or []:
        msgs.extend(_msg_to_openai(m))
    out: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": msgs,
        "max_tokens": body.get("max_tokens", 4096),
    }
    for k_src, k_dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
    ):
        if k_src in body:
            out[k_dst] = body[k_src]
    stops = body.get("stop_sequences")
    if stops:
        out["stop"] = stops
    if body.get("stream"):
        out["stream"] = True
    tools = body.get("tools")
    if tools:
        out["tools"] = [_tool_to_openai(t) for t in tools]
    tc = _tool_choice_to_openai(body.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc
    return out


# --- Response translation (non-stream): OpenAI -> Anthropic -----------------

def _map_stop_reason(finish: str | None, had_tools: bool) -> str:
    if had_tools:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    if finish == "stop":
        return "end_turn"
    if finish:
        return "stop_sequence"
    return "end_turn"


def _parse_args(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s) if s else {}
        except json.JSONDecodeError:
            return {"_raw": s}
    return {}


def openai_to_anthropic_response(oai: dict) -> dict:
    choices = oai.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    content: list[dict] = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "name": fn.get("name", ""),
            "input": _parse_args(fn.get("arguments")),
        })
    usage = oai.get("usage") or {}
    return {
        "id": oai.get("id") or f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": MODEL_ALIAS,
        "content": content,
        "stop_reason": _map_stop_reason(
            choice.get("finish_reason"), bool(tool_calls)
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# --- Stream converter: OpenAI SSE chunks -> Anthropic SSE events -------------

class StreamConverter:
    """State machine ported from ollama/anthropic StreamConverter.Process.
    Consumes OpenAI chat.completion.chunk events, emits Anthropic SSE events.
    """

    def __init__(self, message_id: str) -> None:
        self.id = message_id
        self.first_write = True
        self.content_index = 0
        self.text_started = False
        self.tool_calls_started: dict[int, bool] = {}  # idx -> started
        self.tool_calls_index: dict[int, int] = {}     # openai idx -> our content_index
        self.had_tool_call = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason: str | None = None

    def _event(self, event: str, data: dict) -> bytes:
        return (f"event: {event}\ndata: {json.dumps(data)}\n\n").encode("utf-8")

    def process(self, chunk: dict) -> list[bytes]:
        out: list[bytes] = []
        choices = chunk.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")
        usage = chunk.get("usage") or {}
        if usage:
            self.input_tokens = usage.get("prompt_tokens", self.input_tokens) or 0
            self.output_tokens = usage.get("completion_tokens", self.output_tokens) or 0

        if self.first_write:
            self.first_write = False
            out.append(self._event("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL_ALIAS,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 0,
                    },
                },
            }))

        # Text content
        content_piece = delta.get("content")
        if isinstance(content_piece, str) and content_piece:
            if not self.text_started:
                self.text_started = True
                out.append(self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self.content_index,
                    "content_block": {"type": "text", "text": ""},
                }))
            out.append(self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.content_index,
                "delta": {"type": "text_delta", "text": content_piece},
            }))

        # Tool calls
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            oai_idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            args_frag = fn.get("arguments") or ""
            # Close the text block before opening any tool_use block
            if self.text_started and oai_idx not in self.tool_calls_started:
                out.append(self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self.content_index,
                }))
                self.content_index += 1
                self.text_started = False
            if oai_idx not in self.tool_calls_started:
                self.tool_calls_started[oai_idx] = True
                self.tool_calls_index[oai_idx] = self.content_index
                self.had_tool_call = True
                out.append(self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self.content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "name": fn.get("name", ""),
                        "input": {},
                    },
                }))
                if oai_idx != list(self.tool_calls_started.keys())[0]:
                    # New tool call index means a new content block
                    pass
            if args_frag:
                out.append(self._event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.tool_calls_index[oai_idx],
                    "delta": {"type": "input_json_delta", "partial_json": args_frag},
                }))
            # Advance content_index only when we transition to a new tool call
            # mlx_lm.server buffers; one chunk = one complete tool_call.

        if finish:
            # Close any open content block
            if self.text_started:
                out.append(self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self.content_index,
                }))
                self.text_started = False
            for oai_idx, started in list(self.tool_calls_started.items()):
                if started:
                    out.append(self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self.tool_calls_index[oai_idx],
                    }))
                    self.tool_calls_started[oai_idx] = False
            self.stop_reason = _map_stop_reason(finish, self.had_tool_call)
            out.append(self._event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
            }))
            out.append(self._event("message_stop", {"type": "message_stop"}))
        return out


# --- HTTP routes -------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ALIAS, "object": "model", "owned_by": "mlx-claude"}],
    }


async def _forward_nonstream(openai_body: dict) -> dict:
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{BACKEND_URL}/v1/chat/completions",
            json=openai_body,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()


async def _forward_stream(openai_body: dict):
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    conv = StreamConverter(message_id)
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(
        "POST", f"{BACKEND_URL}/v1/chat/completions",
        json=openai_body, headers={"Content-Type": "application/json"},
    )
    resp = await client.send(req, stream=True)
    try:
        buf = b""
        async for piece in resp.aiter_bytes():
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", errors="replace").strip()
                if not s.startswith("data:"):
                    continue
                payload = s[len("data:"):].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for ev in conv.process(chunk):
                    yield ev
    finally:
        await resp.aclose()
        await client.aclose()


@app.post("/v1/messages")
async def messages(req: Request):
    raw = await req.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return Response(
            content=json.dumps({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad json"},
            }).encode(),
            status_code=400,
            media_type="application/json",
        )
    openai_body = anthropic_to_openai(body)
    if openai_body.get("stream"):
        return StreamingResponse(
            _forward_stream(openai_body),
            media_type="text/event-stream",
        )
    try:
        oai = await _forward_nonstream(openai_body)
    except httpx.HTTPStatusError as e:
        return Response(
            content=e.response.content,
            status_code=e.response.status_code,
            media_type=e.response.headers.get("content-type", "application/json"),
        )
    return Response(
        content=json.dumps(openai_to_anthropic_response(oai)).encode(),
        status_code=200,
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
