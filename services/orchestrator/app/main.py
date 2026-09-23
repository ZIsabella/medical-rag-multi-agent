import json
import logging
import asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from app.agent import MedicalAgent
except ModuleNotFoundError:
    from agent import MedicalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="Medical RAG Orchestrator", version="1.0.0")
agent = MedicalAgent()


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None


async def log_interaction(query: str, response: str):
    logger.info("Query handled: len(query)=%d, len(resp)=%d", len(query), len(response))


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "orchestrator"}


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator():
        yield ": ping\n\n"
        full_response = ""
        try:
            async for item in agent.run_stream(request.query, request.chat_history):
                if isinstance(item, dict):
                    yield f"data: {json.dumps(item)}\n\n"
                elif isinstance(item, str):
                    full_response += item
                    yield f"data: {item}\n\n"

            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            logger.warning("Client disconnected from SSE stream.")
            raise
        except Exception as e:
            logger.exception("Error while streaming response")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if full_response:
                asyncio.create_task(log_interaction(request.query, full_response))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
