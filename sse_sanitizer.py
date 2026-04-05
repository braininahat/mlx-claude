#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastapi>=0.110",
#   "uvicorn[standard]>=0.29",
#   "httpx>=0.27",
# ]
# ///
"""SSE sanitizer — strips `tool_calls: []` from mlx_{lm,vlm}.server streaming
chunks so LiteLLM's chat-completions → anthropic-messages adapter doesn't
misclassify them as tool-call deltas.

The bug in LiteLLM (litellm/llms/anthropic/experimental_pass_through/
adapters/transformation.py :: _translate_streaming_openai_chunk_to_anthropic):
if `choice.delta.tool_calls is not None` — True for empty list `[]` — it
sets `partial_json=""` and returns `input_json_delta`, dropping text content.
mlx servers emit `"tool_calls": []` on every chunk (OpenAI-standard field),
so all streamed text gets dropped.

This proxy reads each SSE `data:` line from the backend, parses the JSON,
drops the `tool_calls` field when it's an empty list, and re-emits. Also
passes through non-streaming requests and the /v1/models listing.

Env:
  SANITIZER_HOST (default 127.0.0.1)
  SANITIZER_PORT (default 8081)
  BACKEND_URL    (default http://127.0.0.1:8080)
"""

from __future__ import annotations
import json
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
HOST = os.environ.get("SANITIZER_HOST", "127.0.0.1")
PORT = int(os.environ.get("SANITIZER_PORT", "8081"))

app = FastAPI()


def _sanitize_chunk(data_json: str) -> str:
    """Remove `tool_calls: []` (empty list) from each choice's delta."""
    try:
        obj = json.loads(data_json)
    except json.JSONDecodeError:
        return data_json
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        tc = delta.get("tool_calls")
        if isinstance(tc, list) and len(tc) == 0:
            delta.pop("tool_calls", None)
    return json.dumps(obj)


async def _proxy_stream(backend_resp: httpx.Response):
    """Re-emit SSE lines with sanitized `data:` payloads."""
    buf = b""
    async for chunk in backend_resp.aiter_bytes():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace")
            if text.startswith("data: ") and not text.startswith("data: [DONE]"):
                payload = text[len("data: "):]
                sanitized = _sanitize_chunk(payload)
                yield ("data: " + sanitized + "\n").encode("utf-8")
            else:
                yield (text + "\n").encode("utf-8")
    if buf:
        yield buf


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: Request):
    body_bytes = await req.body()
    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        return Response(content=b"bad json", status_code=400)
    is_stream = bool(body.get("stream"))
    # Preserve critical headers; drop host.
    fwd_headers = {
        k: v for k, v in req.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    fwd_headers["content-type"] = "application/json"
    url = f"{BACKEND_URL}/v1/chat/completions"

    if is_stream:
        client = httpx.AsyncClient(timeout=None)
        req_builder = client.build_request(
            "POST", url, content=body_bytes, headers=fwd_headers,
        )
        backend_resp = await client.send(req_builder, stream=True)

        async def gen():
            try:
                async for part in _proxy_stream(backend_resp):
                    yield part
            finally:
                await backend_resp.aclose()
                await client.aclose()

        return StreamingResponse(
            gen(),
            status_code=backend_resp.status_code,
            media_type=backend_resp.headers.get("content-type", "text/event-stream"),
        )

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, content=body_bytes, headers=fwd_headers)
    return Response(
        content=r.content, status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.get("/v1/models")
@app.get("/models")
async def models():
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BACKEND_URL}/v1/models")
    return Response(
        content=r.content, status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
