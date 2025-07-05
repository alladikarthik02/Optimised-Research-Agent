"""
FastAPI wrapper that streams answers over Server-Sent Events (SSE).

POST /ask  →  stream
GET  /health → {"status":"ok"}
"""

from __future__ import annotations
from typing import AsyncGenerator

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles

from llamascholar.graph_runner import agent_graph
from llamascholar.memory import get_memory  # ensures Redis is initialised

# ───── FastAPI init ─────────────────────────────────────────── #
app = FastAPI(title="LlamaScholar")

# serve demo page if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
async def health():
    return {"status": "ok"}

# ------- body model so Swagger shows a textbox ---------------- #
class AskPayload(BaseModel):
    query: str
    thread_id: str | None = None
    stream: bool = True

@app.post("/ask", response_class=EventSourceResponse, summary="Ask (SSE)")
async def ask(payload: AskPayload):
    if not payload.query.strip():
        raise HTTPException(422, "query must be non-empty")

    # blocking mode for Swagger or simple clients
    if not payload.stream:
        answer = agent_graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": "You are LlamaScholar."},
                    {"role": "user", "content": payload.query},
                ]
            },
            {"configurable": {"thread_id": payload.thread_id or "cli"}},
        )["messages"][-1].content
        return JSONResponse({"answer": answer})

    # streaming mode (SSE)
    async def event_stream() -> AsyncGenerator[dict, None]:
        async for chunk in agent_graph.astream_events(
            {
                "messages": [
                    {"role": "system", "content": "You are LlamaScholar."},
                    {"role": "user", "content": payload.query},
                ]
            },
            {"configurable": {"thread_id": payload.thread_id or "web"}},
        ):
            if chunk.get("event") == "token":
                yield {"event": "token", "data": chunk["data"]}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_stream())
s