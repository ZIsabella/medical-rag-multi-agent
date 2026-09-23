from asyncpg import Pool
from app.embedder import embed


async def save_document(pool: Pool, title: str, source: str, chunks: list[str],
                        metadata: dict = None) -> int:
    async with pool.acquire() as conn:
        doc_id = await conn.fetchval(
            "INSERT INTO documents (title, source, metadata) "
            "VALUES ($1, $2, $3) RETURNING id",
            title, source, metadata or {},
        )
        vectors = embed(chunks)                 # [n_chunks x 1024]
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            await conn.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, content, embedding) "
                "VALUES ($1, $2, $3, $4)",
                doc_id, i, chunk, vec,
            )
        return doc_id
