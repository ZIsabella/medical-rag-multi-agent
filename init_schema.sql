-- ============================================================
-- Medical RAG Multi-Agent — Data Service database schema
-- PostgreSQL + pgvector
-- ============================================================

-- Enable pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------
-- Development reset
-- WARNING: deletes all previously ingested records.
-- ------------------------------------------------------------
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- ------------------------------------------------------------
-- Parent table: raw QA documents
-- Matches QADocument:
-- document_id, question, context, answer, metadata
-- ------------------------------------------------------------
CREATE TABLE documents (
    document_id    VARCHAR(255) PRIMARY KEY,
    question       TEXT NOT NULL,
    context        TEXT,
    answer         TEXT NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Child table: embedded chunks
-- Matches DocumentChunk:
-- chunk_id, document_id, source_dataset, chunk_index,
-- content, embedding, metadata
-- ------------------------------------------------------------
CREATE TABLE document_chunks (
    chunk_id        VARCHAR(255) PRIMARY KEY,
    document_id     VARCHAR(255) NOT NULL,
    source_dataset  VARCHAR(255) NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       VECTOR(384) NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_document_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_document_chunks_document_index
        UNIQUE (document_id, chunk_index)
);

-- ------------------------------------------------------------
-- Full-text search support for hybrid retrieval
-- ------------------------------------------------------------
ALTER TABLE document_chunks
ADD COLUMN content_tsv TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('english', COALESCE(content, ''))
) STORED;

-- Fast parent-child join / document filtering.
CREATE INDEX idx_document_chunks_document_id
    ON document_chunks (document_id);

-- BM25-like lexical/full-text component of hybrid retrieval.
CREATE INDEX document_chunks_content_tsv_idx
    ON document_chunks
    USING GIN (content_tsv);

-- Semantic vector retrieval using cosine distance.
CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks
    USING HNSW (embedding vector_cosine_ops);
