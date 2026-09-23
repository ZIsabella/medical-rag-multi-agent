import asyncio
import math
from typing import List, Optional
from app.domain.entities import DocumentChunk
from app.infrastructure.repositories.postgres_repository import PostgresRepository
from app.application.embedding_service import EmbeddingService

# --- Configuration ---
# تغییر postgres به localhost برای دسترسی از ویندوز
DATABASE_URL = "postgresql://medical_user:medical_password@localhost:5432/medical_rag"
TEST_QUERY = "What is Adult Acute Lymphoblastic Leukemia?"
SEARCH_LIMIT = 3


async def run_test():
    """Runs a comprehensive test for the retrieval functionality."""
    print(f"\n--- Running Retrieval Test ---")
    print(f"Test Query: '{TEST_QUERY}'")
    print(f"Search Limit: {SEARCH_LIMIT}")

    repo = PostgresRepository(dsn=DATABASE_URL)
    embedder = EmbeddingService()

    try:
        # 1. Generate query embedding
        print("[Test Info] Generating embedding for query...")
        query_embedding = embedder.embed_query(TEST_QUERY)

        # 2. Perform similarity search
        print("[Test Info] Performing similarity search...")
        results = await repo.similarity_search(query_embedding, limit=SEARCH_LIMIT)

        # 3. Output results
        print(f"[Test Info] Received {len(results)} results.")
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"  Content snippet: {str(result.get('content', ''))[:100]}...")
            print(f"  Metadata: {result.get('metadata', {})}")

        if len(results) == 0:
            print("\n[Test Info] No results found! Check if data was actually inserted.")

    except Exception as e:
        print(f"\n[Test Failed] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await repo.close_async()


if __name__ == "__main__":
    asyncio.run(run_test())
