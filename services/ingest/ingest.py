"""
Medical RAG ingestion pipeline.

Responsibilities:
1. Download MedQuAD and PubMedQA raw datasets.
2. Convert the official MedQuAD ZIP archive into normalized XML.
3. Load Q&A documents through dataset-specific loaders.
4. Chunk documents.
5. Generate embeddings.
6. Store embedded chunks in PostgreSQL/pgvector.
"""

from __future__ import annotations

import asyncio
import logging
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from app.application.chunker import Chunker
from app.application.dataset_downloader import DatasetDownloader, DatasetSpec
from app.application.embedding_service import EmbeddingService
from app.domain.entities import DocumentChunk, QADocument
from app.infrastructure.loaders.medquad_loader import MedQuADLoader
from app.infrastructure.loaders.pubmedqa_loader import PubMedQALoader
from app.infrastructure.repositories.postgres_repository import PostgresRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ingestion_pipeline")

BATCH_SIZE = 100

DEFAULT_MEDQUAD_URL = (
    "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
)

DEFAULT_PUBMEDQA_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)


def get_clean_text(element: Optional[ET.Element]) -> str:
    """
    Extract all visible textual content from an XML element recursively.
    """
    if element is None:
        return ""

    return " ".join(
        text.strip()
        for text in element.itertext()
        if text and text.strip()
    ).strip()


def build_medquad_complete_xml(zip_path: Path, output_path: Path) -> int:
    """
    Convert the official MedQuAD GitHub ZIP archive to a single normalized XML
    file compatible with MedQuADLoader.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"MedQuAD archive does not exist: {zip_path}"
        )

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(
            f"MedQuAD source is not a valid ZIP archive: {zip_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_root = ET.Element("MedQuAD")
    total_entries = 0
    skipped_malformed_files = 0
    skipped_invalid_pairs = 0

    logger.info("Normalizing MedQuAD archive: %s", zip_path)

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        xml_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".xml")
        ]

        if not xml_members:
            raise RuntimeError(
                "No XML files were found inside the MedQuAD ZIP archive."
            )

        logger.info(
            "Found %d XML files inside MedQuAD archive.",
            len(xml_members),
        )

        for member in xml_members:
            try:
                source_xml_bytes = archive.read(member)
                source_root = ET.fromstring(source_xml_bytes)
            except ET.ParseError:
                skipped_malformed_files += 1
                logger.warning(
                    "Skipping malformed XML file inside MedQuAD archive: %s",
                    member.filename,
                )
                continue
            except OSError as exc:
                logger.warning(
                    "Could not read XML file from MedQuAD archive: %s | %s",
                    member.filename,
                    exc,
                )
                continue

            focus = get_clean_text(source_root.find(".//Focus"))
            qa_pairs = source_root.findall(".//QAPair")

            for qa_pair in qa_pairs:
                question = get_clean_text(qa_pair.find("./Question"))
                answer = get_clean_text(qa_pair.find("./Answer"))

                if not question or not answer:
                    skipped_invalid_pairs += 1
                    continue

                entry = ET.SubElement(normalized_root, "Entry")

                question_element = ET.SubElement(entry, "Question")
                question_element.text = question

                answer_element = ET.SubElement(entry, "Answer")
                answer_element.text = answer

                context_element = ET.SubElement(entry, "Context")
                context_element.text = focus

                # نگهداری نام فایل اصلی به عنوان مرجع منبع در ساختار XML
                source_elem = ET.SubElement(entry, "Source")
                source_elem.text = member.filename

                total_entries += 1

    if total_entries == 0:
        raise RuntimeError(
            "MedQuAD normalization completed, but no valid question-answer entries were found."
        )

    tree = ET.ElementTree(normalized_root)
    ET.indent(tree, space="  ")

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    logger.info(
        "MedQuAD normalization finished | entries=%d | malformed_files=%d "
        "| skipped_invalid_pairs=%d | output=%s",
        total_entries,
        skipped_malformed_files,
        skipped_invalid_pairs,
        output_path,
    )

    return total_entries


def _get_object_value(
    obj: Any,
    *field_names: str,
    default: Any = None,
) -> Any:
    """Read a value from a dataclass, object, or dictionary."""
    if isinstance(obj, dict):
        for field_name in field_names:
            if field_name in obj:
                return obj[field_name]
        return default

    for field_name in field_names:
        if hasattr(obj, field_name):
            return getattr(obj, field_name)

    return default


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass or mapping-like object to a dictionary."""
    if isinstance(obj, dict):
        return dict(obj)

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return dict(vars(obj))

    raise TypeError(
        f"Expected a dictionary, dataclass, or object with __dict__; "
        f"received {type(obj).__name__}."
    )


def _document_to_repository_row(doc: Any, default_dataset: str = "MedQuAD") -> Dict[str, Any]:
    """تبدیل QADocument به دیکشنری برای درج در دیتابیس با متادیتای اصلاح‌شده"""
    metadata = getattr(doc, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    source = metadata.get("source") or metadata.get("source_dataset") or default_dataset
    metadata["source"] = source
    metadata["source_dataset"] = default_dataset

    return {
        "document_id": getattr(doc, "document_id", None) or getattr(doc, "id", None),
        "question": getattr(doc, "question", "") or "",
        "context": getattr(doc, "context", "") or "",
        "answer": getattr(doc, "answer", "") or "",
        "metadata": metadata,
    }


def _chunk_to_repository_row(chunk: DocumentChunk, default_source: str = "MedQuAD") -> Dict[str, Any]:
    """
    Convert a DocumentChunk entity to the row shape expected by the repository.
    تضمین می‌کند فیلد source هرگز 'unknown' نباشد.
    """
    chunk_id = _get_object_value(chunk, "id", "chunk_id")
    document_id = _get_object_value(chunk, "document_id", "doc_id")
    text = _get_object_value(chunk, "text", "chunk_text", "content")
    metadata = _get_object_value(chunk, "metadata", "meta", default={})
    embedding = _get_object_value(chunk, "embedding", "vector")

    missing_fields = []
    if chunk_id is None:
        missing_fields.append("id")
    if document_id is None:
        missing_fields.append("document_id")
    if text is None:
        missing_fields.append("text")
    if embedding is None:
        missing_fields.append("embedding")

    if missing_fields:
        raise ValueError(
            "DocumentChunk is missing required fields: "
            + ", ".join(missing_fields)
        )

    # پاک‌سازی و تضمین متادیتای منبع
    if not isinstance(metadata, dict):
        metadata = {}

    current_source = str(metadata.get("source") or metadata.get("source_dataset") or "").strip()
    if not current_source or current_source.lower() == "unknown":
        metadata["source"] = default_source
        metadata["source_dataset"] = default_source

    return {
        "id": chunk_id,
        "document_id": document_id,
        "text": text,
        "metadata": metadata,
        "embedding": embedding,
    }


class IngestionPipeline:
    """Coordinates loading, chunking, embedding, and database persistence."""

    def __init__(
        self,
        raw_data_dir: Path,
        chunker: Chunker,
        embedder: EmbeddingService,
        repository: PostgresRepository,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.chunker = chunker
        self.embedder = embedder
        self.repository = repository

    def _get_medquad_path(self) -> Path:
        return self.raw_data_dir / "medquad-complete.xml"

    def _get_pubmedqa_path(self) -> Path:
        return self.raw_data_dir / "pubmedqa.json"

    def _load_medquad(self, file_path: Path) -> List[QADocument]:
        logger.info("Loading MedQuAD from: %s", file_path)
        loader = MedQuADLoader(min_q_len=20, min_a_len=40)
        loaded = loader.load(str(file_path))
        docs = list(loaded) if not isinstance(loaded, list) else loaded
        # تزریق صریح منبع به تک تک اسناد لود شده
        for doc in docs:
            setattr(doc, "source_dataset", "MedQuAD")
            setattr(doc, "source", "MedQuAD")
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                doc.metadata["source"] = "MedQuAD"
                doc.metadata["source_dataset"] = "MedQuAD"
        return docs

    def _load_pubmedqa(self, file_path: Path) -> List[QADocument]:
        logger.info("Loading PubMedQA from: %s", file_path)
        loader = PubMedQALoader(min_q_len=20, min_a_len=40)
        loaded = loader.load(str(file_path))
        docs = list(loaded) if not isinstance(loaded, list) else loaded
        # تزریق صریح منبع برای اسناد پاب‌مد
        for doc in docs:
            setattr(doc, "source_dataset", "PubMedQA")
            setattr(doc, "source", "PubMedQA")
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                doc.metadata["source"] = "PubMedQA"
                doc.metadata["source_dataset"] = "PubMedQA"
        return docs

    def _process_in_batches(
        self,
        documents: Iterable[QADocument],
    ) -> Generator[List[DocumentChunk], None, None]:
        current_batch: List[DocumentChunk] = []

        for chunk in self.chunker.chunk_many(documents):
            current_batch.append(chunk)

            if len(current_batch) >= BATCH_SIZE:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch

    async def run_pipeline(self, dataset_name: str) -> None:
        dataset_name = dataset_name.strip().lower()
        dataset_display_name = "MedQuAD" if dataset_name == "medquad" else "PubMedQA"

        logger.info(
            "========== Starting ingestion: %s ==========",
            dataset_name,
        )

        try:
            if dataset_name == "medquad":
                source_path = self._get_medquad_path()
                documents = self._load_medquad(source_path)
            elif dataset_name == "pubmedqa":
                source_path = self._get_pubmedqa_path()
                documents = self._load_pubmedqa(source_path)
            else:
                raise ValueError(
                    f"Unsupported dataset name: {dataset_name}. "
                    "Supported datasets are: medquad, pubmedqa."
                )

            if not documents:
                logger.warning(
                    "No documents loaded for dataset=%s. Skipping ingestion.",
                    dataset_name,
                )
                return

            logger.info(
                "Documents loaded successfully | dataset=%s | documents=%d",
                dataset_name,
                len(documents),
            )

            # Ensure schema
            self.repository.ensure_schema()

            # 1) Insert parent documents
            logger.info(
                "Inserting parent documents | dataset=%s | count=%d",
                dataset_name,
                len(documents),
            )

            document_rows = [
                _document_to_repository_row(doc, default_dataset=dataset_display_name)
                for doc in documents
            ]
            await self.repository.insert_documents(document_rows)

            # 2) Chunk, embed and insert chunks
            total_chunks_inserted = 0
            total_batches = 0

            for chunk_batch in self._process_in_batches(documents):
                if not chunk_batch:
                    continue

                total_batches += 1

                logger.info(
                    "Embedding chunks | dataset=%s | batch=%d | chunks=%d",
                    dataset_name,
                    total_batches,
                    len(chunk_batch),
                )

                embedded_chunks = self.embedder.embed_chunks(chunk_batch)

                if not embedded_chunks:
                    logger.warning(
                        "Embedding returned no chunks | dataset=%s | batch=%d",
                        dataset_name,
                        total_batches,
                    )
                    continue

                chunk_rows = [
                    _chunk_to_repository_row(chunk, default_source=dataset_display_name)
                    for chunk in embedded_chunks
                ]

                await self.repository.insert_many(chunk_rows)

                inserted_count = len(chunk_rows)
                total_chunks_inserted += inserted_count

                logger.info(
                    "Stored chunk batch | dataset=%s | batch=%d "
                    "| inserted=%d | total_inserted=%d",
                    dataset_name,
                    total_batches,
                    inserted_count,
                    total_chunks_inserted,
                )

            logger.info(
                "========== Ingestion completed: %s | documents=%d "
                "| batches=%d | chunks_inserted=%d ==========",
                dataset_name,
                len(documents),
                total_batches,
                total_chunks_inserted,
            )

        except Exception:
            logger.exception(
                "Ingestion pipeline failed for dataset=%s",
                dataset_name,
            )
            raise


async def main() -> None:
    database_url = os.getenv(
        "DATABASE_URL",
        "secret-b7e1935epostgres:5432/medical_rag",
    )

    raw_data_dir = Path(
        os.getenv("RAW_DATA_DIR", "data/raw")
    )

    medquad_url = os.getenv(
        "MEDQUAD_URL",
        DEFAULT_MEDQUAD_URL,
    )

    pubmedqa_url = os.getenv(
        "PUBMEDQA_URL",
        DEFAULT_PUBMEDQA_URL,
    )

    medquad_sha256 = os.getenv("MEDQUAD_SHA256") or None
    pubmedqa_sha256 = os.getenv("PUBMEDQA_SHA256") or None

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    medquad_zip_path = raw_data_dir / "medquad-official.zip"
    medquad_xml_path = raw_data_dir / "medquad-complete.xml"
    pubmedqa_json_path = raw_data_dir / "pubmedqa.json"

    repository: Optional[PostgresRepository] = None

    try:
        logger.info("Initializing Data Service ingestion dependencies.")

        downloader = DatasetDownloader()
        repository = PostgresRepository(dsn=database_url)
        chunker = Chunker()
        embedder = EmbeddingService()

        pipeline = IngestionPipeline(
            raw_data_dir=raw_data_dir,
            chunker=chunker,
            embedder=embedder,
            repository=repository,
        )

        logger.info("Ensuring PostgreSQL schema and pgvector extension.")
        repository.ensure_schema()

        dataset_specs = [
            DatasetSpec(
                name="medquad",
                url=medquad_url,
                output_path=medquad_zip_path,
                sha256=medquad_sha256,
            ),
            DatasetSpec(
                name="pubmedqa",
                url=pubmedqa_url,
                output_path=pubmedqa_json_path,
                sha256=pubmedqa_sha256,
            ),
        ]

        logger.info("Checking and downloading raw datasets if required.")
        for spec in dataset_specs:
            downloaded_path = downloader.ensure(spec)
            logger.info(
                "Dataset ready | name=%s | path=%s",
                spec.name,
                downloaded_path,
            )

        if (
            not medquad_xml_path.exists()
            or medquad_xml_path.stat().st_size == 0
        ):
            build_medquad_complete_xml(
                zip_path=medquad_zip_path,
                output_path=medquad_xml_path,
            )
        else:
            logger.info(
                "Normalized MedQuAD XML already exists: %s",
                medquad_xml_path,
            )

        await pipeline.run_pipeline("medquad")
        await pipeline.run_pipeline("pubmedqa")

        logger.info("All ingestion datasets completed successfully.")

    except Exception:
        logger.critical(
            "Fatal error in Data Service ingestion process.",
            exc_info=True,
        )
        raise

    finally:
        if repository is not None:
            repository.close()


if __name__ == "__main__":
    asyncio.run(main())
