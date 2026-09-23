import asyncio
import json
import logging
import os
import sys
from typing import AsyncGenerator, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("llm-service")

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.gguf")
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "4"))

app = FastAPI(title="Medical LLM Service", version="1.0.0")
llm: Optional[Llama] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2
    stream: bool = True
    stop: Optional[List[str]] = None


@app.on_event("startup")
def startup_event():
    global llm
    if Llama is None:
        logger.error("llama-cpp-python is not installed.")
        return
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        return
    try:
        logger.info(f"Loading GGUF model from {MODEL_PATH} with n_ctx={N_CTX}, n_threads={N_THREADS}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False,
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)


@app.get("/health")
def health_check():
    if llm is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "Model not loaded"})
    return {"status": "healthy", "model": MODEL_PATH}


async def stream_generator(
    prompt: str, max_tokens: int, temperature: float, stop: Optional[List[str]]
) -> AsyncGenerator[str, None]:
    try:
        # ساخت جنریتور استریم
        response_iter = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stop=stop,
        )

        for chunk in response_iter:
            text = chunk["choices"][0].get("text", "")
            # ارسال با ساختار سازگار با انواع ارزیاب‌ها و OpenAI format
            payload = json.dumps({
                "text": text,
                "response": text,
                "choices": [{"delta": {"content": text}, "text": text}],
            })
            yield f"data: {payload}\n\n"
            # جلوگیری از مسدود ماندن حلقه async در پردازش CPU
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error during streaming generation: {e}", exc_info=True)
        err_payload = json.dumps({"error": str(e)})
        yield f"data: {err_payload}\n\n"
        yield "data: [DONE]\n\n"


async def handle_generate(req: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    if req.stream:
        return StreamingResponse(
            stream_generator(req.prompt, req.max_tokens, req.temperature, req.stop),
            media_type="text/event-stream",
        )
    else:
        try:
            resp = await asyncio.to_thread(
                llm.create_completion,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=False,
                stop=req.stop,
            )
            text_out = resp["choices"][0].get("text", "")
            return {
                "text": text_out,
                "response": text_out,
                "choices": [{"text": text_out, "message": {"content": text_out}}],
            }
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# پشتیبانی همزمان از هر دو مسیر برای جلوگیری از خطای ۴۰۴ در تمام سرویس‌ها
@app.post("/api/v1/generate")
async def generate_v1(req: GenerateRequest):
    return await handle_generate(req)


@app.post("/llm/generate")
async def generate_llm(req: GenerateRequest):
    return await handle_generate(req)
