# api.py (only the changed parts shown)
from typing import AsyncGenerator
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# 1️⃣  Body schema so Swagger knows what to render
class AskPayload(BaseModel):
    query: str
    thread_id: str | None = None
    stream: bool = True  # future-proof flag

app = FastAPI(title="LlamaScholar API")

@app.post("/ask", response_class=EventSourceResponse, summary="Ask (SSE)")
async def ask(payload: AskPayload):
    """
    Stream the answer token-by-token using Server-Sent Events.
    Set `stream=false` to get a blocking JSON instead.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="query must be non-empty")

    async def event_stream() -> AsyncGenerator[dict, None]:
        thread_id = payload.thread_id or "web"
        async for chunk in agent_graph.astream_events(
            {
                "messages": [
                    { "role": "system",
                      "content": "You are LlamaScholar, a concise research assistant." },
                    { "role": "user", "content": payload.query }
                ]
            },
            { "configurable": { "thread_id": thread_id } },
        ):
            if chunk.get("event") == "token":
                yield { "event": "token", "data": chunk["data"] }
        yield { "event": "done", "data": "[DONE]" }

    # Allow a blocking JSON mode if someone wants it
    if not payload.stream:
        final = await agent_graph.invoke(
            {
                "messages": [
                    { "role": "system",
                      "content": "You are LlamaScholar, a concise research assistant." },
                    { "role": "user", "content": payload.query }
                ]
            },
            { "configurable": { "thread_id": payload.thread_id or "cli" } },
        )
        return JSONResponse({ "answer": final["messages"][-1].content })

    return EventSourceResponse(event_stream())
