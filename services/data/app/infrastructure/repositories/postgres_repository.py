import asyncio
import logging
from typing import Any, Dict, List, Optional
import json

import asyncpg
import psycopg2

logger = logging.getLogger(__name__)


def _embedding_to_pg_string(embedding: Any) -> Optional[str]:
    """تبدیل لیست یا وکتور به رشته [x,y,...] برای asyncpg"""
    if embedding is None:
        return None
    if isinstance(embedding, (list, tuple)):
        return "[" + ",".join(str(float(x)) for x in embedding) + "]"
    if isinstance(embedding, str):
        s = embedding.strip()
        if s.startswith("[") and s.endswith("]"):
            return s
        if s.startswith("(") and s.endswith(")"):
            return "[" + s[1:-1] + "]"
        return "[" + s + "]"
    try:
        return "[" + ",".join(str(float(x)) for x in embedding) + "]"
    except (TypeError, ValueError):
        return None


class PostgresRepository:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None
        self._fts_column = "content_tsv"

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn)
        return self._pool

    def ensure_schema(self) -> None:
        """
        Ensures that the required PostgreSQL schema exists.
        """
        conn = psycopg2.connect(self.dsn)

        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id VARCHAR(255) PRIMARY KEY,
                        question TEXT NOT NULL DEFAULT '',
                        context TEXT,
                        answer TEXT NOT NULL DEFAULT '',
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        source_dataset VARCHAR(255) NOT NULL DEFAULT '',
                        chunk_index INTEGER NOT NULL DEFAULT 0,
                        content TEXT NOT NULL DEFAULT '',
                        embedding VECTOR(384),
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT fk_document_chunks_document
                            FOREIGN KEY (document_id)
                            REFERENCES documents(document_id) ON DELETE CASCADE
                    );
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS question
                    TEXT NOT NULL DEFAULT '';
                    """
                )
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_document_chunks_doc_chunk "
                    "ON document_chunks (document_id, chunk_index);"
                )

                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS context
                    TEXT;
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS answer
                    TEXT NOT NULL DEFAULT '';
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS metadata
                    JSONB NOT NULL DEFAULT '{}'::jsonb;
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS created_at
                    TIMESTAMPTZ NOT NULL DEFAULT NOW();
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS updated_at
                    TIMESTAMPTZ NOT NULL DEFAULT NOW();
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS source_dataset
                    VARCHAR(255) NOT NULL DEFAULT '';
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS chunk_index
                    INTEGER NOT NULL DEFAULT 0;
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS content
                    TEXT NOT NULL DEFAULT '';
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS created_at
                    TIMESTAMPTZ NOT NULL DEFAULT NOW();
                    """
                )

                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'document_chunks'
                      AND column_name = 'content_tsv';
                    """
                )

                content_tsv_exists = cur.fetchone() is not None

                if not content_tsv_exists:
                    cur.execute(
                        """
                        ALTER TABLE document_chunks
                        ADD COLUMN content_tsv TSVECTOR
                        GENERATED ALWAYS AS (
                            to_tsvector(
                                'english',
                                COALESCE(content, '')
                            )
                        ) STORED;
                        """
                    )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_document_chunks_document_id
                    ON document_chunks (document_id);
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    document_chunks_content_tsv_idx
                    ON document_chunks
                    USING GIN (content_tsv);
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    document_chunks_embedding_hnsw_idx
                    ON document_chunks
                    USING HNSW (embedding vector_cosine_ops);
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interactions (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )

            conn.commit()
            logger.info("PostgreSQL schema ensured successfully.")

        except Exception:
            conn.rollback()
            logger.exception("Failed to ensure PostgreSQL schema.")
            raise

        finally:
            conn.close()

    def _get_fts_column(self) -> str:
        return self._fts_column

    async def insert_documents(
            self,
            documents: List[Dict[str, Any]],
    ) -> None:
        if not documents:
            return

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO documents (
                    document_id,
                    question,
                    context,
                    answer,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (document_id)
                DO NOTHING;
                """,
                [
                    (
                        str(document.get("document_id") or document.get("id")),
                        document.get("question", ""),
                        document.get("context"),
                        document.get("answer", ""),
                        json.dumps(document.get("metadata", {}) or {}, ensure_ascii=False),
                    )
                    for document in documents
                ],
            )

    async def insert_many(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        query = """
        INSERT INTO document_chunks (
            chunk_id,
            document_id,
            source_dataset,
            chunk_index,
            content,
            metadata,
            embedding
        ) VALUES (
            $1,
            $2,
            $3,
            $4,
            $5,
            $6::jsonb,
            $7::vector
        )
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            chunk_id = EXCLUDED.chunk_id,
            source_dataset = EXCLUDED.source_dataset,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            created_at = NOW();
        """

        records = []
        for chunk in chunks:
            metadata_val = chunk.get("metadata", {})
            if isinstance(metadata_val, (dict, list)):
                metadata_json = json.dumps(metadata_val, ensure_ascii=False)
            elif isinstance(metadata_val, str):
                metadata_json = metadata_val
            else:
                metadata_json = "{}"

            embedding_val = chunk.get("embedding")
            if isinstance(embedding_val, (list, tuple)):
                embedding_str = f"[{','.join(str(float(x)) for x in embedding_val)}]"
            elif isinstance(embedding_val, str):
                embedding_str = embedding_val
            else:
                embedding_str = None

            chunk_id = str(chunk.get("chunk_id") or chunk.get("id"))
            doc_id = str(chunk.get("document_id") or chunk.get("doc_id") or "")
            source_dataset = str(chunk.get("source_dataset") or chunk.get("dataset") or "")
            chunk_index = int(chunk.get("chunk_index", 0))
            content = str(chunk.get("content") or chunk.get("text") or "")

            records.append(
                (
                    chunk_id,
                    doc_id,
                    source_dataset,
                    chunk_index,
                    content,
                    metadata_json,
                    embedding_str,
                )
            )

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(query, records)

    async def hybrid_search_rrf_async(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()

        vector_str = _embedding_to_pg_string(query_embedding)
        if vector_str is None:
            raise ValueError("query_embedding must be a list of floats or a string")

        query = """
            WITH vector_matches AS (
                SELECT
                    chunk_id,
                    ROW_NUMBER() OVER (
                        ORDER BY embedding <=> $1::vector
                    ) AS rank
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            ),
            keyword_matches AS (
                SELECT
                    chunk_id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            content_tsv,
                            websearch_to_tsquery(
                                'english',
                                $4
                            )
                        ) DESC
                    ) AS rank
                FROM document_chunks
                WHERE content_tsv @@ websearch_to_tsquery(
                    'english',
                    $4
                )
                ORDER BY ts_rank(
                    content_tsv,
                    websearch_to_tsquery(
                        'english',
                        $4
                    )
                ) DESC
                LIMIT $2
            ),
            combined_matches AS (
                SELECT
                    COALESCE(v.chunk_id, k.chunk_id) AS chunk_id,
                    COALESCE(
                        1.0 / ($3 + v.rank),
                        0.0
                    )
                    +
                    COALESCE(
                        1.0 / ($3 + k.rank),
                        0.0
                    ) AS score
                FROM vector_matches AS v
                FULL OUTER JOIN keyword_matches AS k
                    ON v.chunk_id = k.chunk_id
            )
            SELECT
                c.chunk_id,
                c.document_id,
                c.content,
                c.metadata,
                m.score
            FROM combined_matches AS m
            INNER JOIN document_chunks AS c
                ON c.chunk_id = m.chunk_id
            ORDER BY m.score DESC
            LIMIT $2;
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                vector_str,
                top_k,
                rrf_k,
                query_text,
            )

            return [dict(row) for row in rows]

    async def hybrid_search_rrf(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return await self.hybrid_search_rrf_async(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def search_hybrid(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Alias for hybrid_search_rrf to maintain API compatibility."""
        return await self.hybrid_search_rrf(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def similarity_search(
            self,
            query_embedding: List[float],
            limit: int,
    ) -> List[Dict[str, Any]]:
        """Pure vector similarity search."""
        pool = await self._get_pool()

        vector_str = _embedding_to_pg_string(query_embedding)
        if vector_str is None:
            raise ValueError("query_embedding must be a list of floats or a string")

        sql = """
            SELECT chunk_id, document_id, content, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, vector_str, limit)
            return [dict(row) for row in rows]

    async def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Alias for similarity_search to maintain API compatibility."""
        return await self.similarity_search(query_embedding=query_embedding, limit=top_k)

    async def log_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
    ) -> None:
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interactions (
                    session_id,
                    query,
                    response
                )
                VALUES ($1, $2, $3);
                """,
                session_id,
                query,
                response,
            )

    async def get_interactions(
            self,
            limit: int = 50,
            offset: int = 0,
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        fetch_query = """
            SELECT id, session_id, query, response, created_at
            FROM interactions
            ORDER BY created_at DESC, id DESC
            LIMIT $1 OFFSET $2;
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(fetch_query, limit, offset)
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "query": r["query"],
                    "response": r["response"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                })
            return results

    async def count_interactions(self) -> int:
        pool = await self._get_pool()
        query = "SELECT COUNT(*) FROM interactions;"
        async with pool.acquire() as conn:
            val = await conn.fetchval(query)
            return int(val or 0)

    async def close_async(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def close(self) -> None:
        if self._pool is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close_async())
        else:
            loop.create_task(self.close_async())
