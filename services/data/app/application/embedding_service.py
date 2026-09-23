import math
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.domain.entities import DocumentChunk
import os
from typing import List, Optional

MODEL_PATH = (
    os.getenv("EMBEDDING_MODEL_PATH")
    or os.getenv("EMBEDDING_MODEL_NAME")
    or "/models/bge-small-en-v1.5"
)

class EmbeddingService:
    """
    Service to generate embeddings for document chunks using Sentence Transformers.
    Includes validation and device management.
    """
    # Default dimension for bge-small-en-v1.5
    DIM = 384

    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', batch_size: int = 32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        print(f"Initializing EmbeddingService on device: {self.device}")
        print(f"Loading embedding model from: {MODEL_PATH}")

        try:
            self.model = SentenceTransformer(
                MODEL_PATH,
                device=self.device,
                local_files_only=True,
            )
        except Exception as e:
            print(f"[Error] Failed to load model {self.model_name}: {e}")

            raise

    def _validate_embedding(self, vec: List[float]) -> bool:
        """Validates an embedding vector."""
        if vec is None or len(vec) != self.DIM:
            return False
        # Check for NaN, Inf
        if any(not math.isfinite(float(x)) for x in vec):
            return False
        # Check for zero norm (L2 norm squared)
        norm2 = sum(float(x) * float(x) for x in vec)
        return norm2 > 1e-9 # Use a small epsilon to avoid floating point issues

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generates embeddings for a list of DocumentChunks.
        Skips chunks with empty content or invalid embeddings.
        """
        valid_chunks: List[DocumentChunk] = []
        texts_to_embed: List[str] = []

        # 1. Pre-filter chunks
        for chunk in chunks:
            if chunk.content and chunk.content.strip():
                valid_chunks.append(chunk)
                texts_to_embed.append(chunk.content)
            else:
                print(f"[Warning] Skipping empty chunk content for document_id: {chunk.document_id}, chunk_id: {chunk.chunk_id}")

        if not texts_to_embed:
            print("[Info] No texts to embed after filtering.")
            return []

        # 2. Generate embeddings
        try:
            print(f"[Info] Embedding {len(texts_to_embed)} texts with batch size {self.batch_size}...")
            embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                normalize_embeddings=True, # L2 normalization is applied here
                show_progress_bar=True
            )
            print("[Info] Embedding complete.")
        except Exception as e:
            print(f"[Error] Failed to generate embeddings: {e}")
            # Return chunks without embeddings if embedding fails critically
            return valid_chunks

        # 3. Validate and assign embeddings
        embedded_chunks: List[DocumentChunk] = []
        for i, chunk in enumerate(valid_chunks):
            embedding = embeddings[i].tolist() # Convert numpy array to list

            if self._validate_embedding(embedding):
                chunk.embedding = embedding
                embedded_chunks.append(chunk)
            else:
                print(f"[Warning] Invalid embedding generated for document_id: {chunk.document_id}, chunk_id: {chunk.chunk_id}. Skipping.")

        print(f"[Info] Successfully embedded and validated {len(embedded_chunks)} chunks.")
        return embedded_chunks
    def embed_query(self, query: str) -> List[float]:
        """
        Create a normalized 384-dimensional embedding for one retrieval query.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            query.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        if not self._validate_embedding(embedding):
            raise ValueError(
                "Generated query embedding failed validation."
            )

        return embedding

    def get_embedding(self, query: str) -> List[float]:
        """
        Backward-compatible alias for single-query embedding.
        """
        return self.embed_query(query)
