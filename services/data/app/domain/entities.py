"""Domain entities shared across the data ingestion and retrieval layers.

The entities in this module intentionally remain lightweight dataclasses.
They are used by loaders, chunking, embedding, and persistence components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# JSON-compatible metadata stored alongside documents/chunks.
Metadata = Dict[str, Any]

# Suggested metadata keys (all optional):
#
# QADocument.metadata:
# - source_dataset: str        e.g. "medquad" or "pubmedqa"
# - source_id: str             original dataset record identifier
# - source_url: str            source/reference URL, when available
# - category: str              medical category/specialty
# - title: str                 source title, when available
# - language: str              e.g. "en"
# - split: str                 e.g. "train", "validation", "test"
#
# DocumentChunk.metadata:
# - source_dataset: str
# - source_id: str
# - source_url: str
# - category: str
# - title: str
# - language: str
# - chunk_start: int           character offset in the source text
# - chunk_end: int             character offset in the source text
# - token_count: int           optional estimated token count
# - ingestion_version: str     pipeline/version traceability


@dataclass
class QADocument:
    """A normalized question-answer document loaded from a medical dataset.

    Attributes:
        document_id: Stable unique identifier for the original document.
        question: Medical question.
        context: Supporting context or passage associated with the question.
        answer: Reference answer from the source dataset.
        metadata: Optional source-specific and traceability metadata.
    """

    document_id: str
    question: str
    context: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """A text chunk ready for embedding and persistence in pgvector.

    Attributes:
        chunk_id: Stable unique identifier for this chunk.
        document_id: Identifier of the parent ``QADocument``.
        source_dataset: Dataset name, such as ``medquad`` or ``pubmedqa``.
        chunk_index: Zero-based position of this chunk in its parent document.
        content: The text that is embedded and later retrieved.
        embedding: Optional dense embedding vector. It is populated after
            the embedding step and before insertion into the vector database.
        metadata: Optional retrieval, provenance, and traceability metadata.
    """

    chunk_id: str
    document_id: str
    source_dataset: str
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: Metadata = field(default_factory=dict)
