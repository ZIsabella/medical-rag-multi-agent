import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
import redis.asyncio as redis

from app.infrastructure.repositories.postgres_repository import PostgresRepository
from app.application.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data-service")


class Settings(BaseSettings):
    database_url: str = "postgresql://medical_user:medical_password@postgres:5432/medical_rag"
    redis_url: str = "redis://redis:6379/0"
    embedding_model_name: str = "/models/bge-small-en-v1.5"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()


class InteractionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    query: str
    answer: str
    session_id: Optional[str] = "default_session"
    safety_reason: Optional[str] = None
    is_medical: Optional[bool] = False
    is_safe: Optional[bool] = True
    safety_status: Optional[str] = "unknown"
    status: Optional[str] = "completed"
    retrieval_items: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Data Service resources...")
    app.state.embedding_service = EmbeddingService(model_name=settings.embedding_model_name)
    app.state.repository = PostgresRepository(dsn=settings.database_url)
    app.state.redis = redis.from_url(settings.redis_url)

    try:
        await asyncio.to_thread(app.state.repository.ensure_schema)
        logger.info("Database schema verified.")
    except Exception as e:
        logger.error(f"Error ensuring schema: {e}")

    yield

    logger.info("Closing Data Service resources...")
    await app.state.repository.close()
    await app.state.redis.close()


app = FastAPI(title="Medical RAG - Data Service", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "data-service", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/interactions")
async def save_interaction(interaction: InteractionRequest, request: Request):
    try:
        repo: PostgresRepository = request.app.state.repository
        meta_payload = {
            "safety_reason": interaction.safety_reason,
            "is_medical": interaction.is_medical,
            "is_safe": interaction.is_safe,
            "safety_status": interaction.safety_status,
            "status": interaction.status,
            "retrieval_items": interaction.retrieval_items,
            "latency_ms": interaction.latency_ms,
            "extra_metadata": interaction.metadata,
        }
        try:
            metadata_json = json.dumps(meta_payload, ensure_ascii=False)
        except Exception:
            metadata_json = "{}"

        await repo.log_interaction(
            session_id=interaction.session_id or "default_session",
            query=interaction.query,
            response=f"{interaction.answer}\n\n[METADATA]{metadata_json}",
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/interactions")
async def get_interactions(
    request: Request,
    limit: int = 50,
    offset: int = 0,
):
    try:
        repo: PostgresRepository = request.app.state.repository
        total_count = await repo.count_interactions()
        raw_rows = await repo.get_interactions(limit=limit, offset=offset)

        parsed_items = []
        for row in raw_rows:
            raw_resp = row.get("response", "")
            clean_answer = raw_resp
            metadata_dict = {}

            if "[METADATA]" in raw_resp:
                parts = raw_resp.split("[METADATA]", 1)
                clean_answer = parts[0].strip()
                try:
                    metadata_dict = json.loads(parts[1].strip())
                except Exception:
                    metadata_dict = {}

            parsed_items.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "query": row["query"],
                "answer": clean_answer,
                "created_at": row["created_at"],
                "metadata": metadata_dict,
            })

        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "items": parsed_items,
        }
    except Exception as e:
        logger.error(f"Failed to fetch interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieval/search")
async def search(retrieval_req: RetrievalRequest, request: Request):
    try:
        repo: PostgresRepository = request.app.state.repository
        emb_service: EmbeddingService = request.app.state.embedding_service

        query_embedding = emb_service.embed_query(retrieval_req.query)

        if retrieval_req.retrieval_mode == "hybrid":
            results = await repo.search_hybrid(
                query_text=retrieval_req.query,
                query_embedding=query_embedding,
                top_k=retrieval_req.top_k,
            )
        else:
            results = await repo.search_similar(
                query_embedding=query_embedding,
                top_k=retrieval_req.top_k,
            )
        return {"query": retrieval_req.query, "results": results}
    except Exception as e:
        logger.error(f"Search retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
