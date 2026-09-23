import xml.etree.ElementTree as ET
import hashlib
import html
import re
import unicodedata
from typing import Generator, Dict, Any, Set
from app.domain.entities import QADocument


class MedQuADLoader:
    """
    Loader for MedQuAD dataset with robust normalization,
    filtering, and deduplication.
    """

    # Patterns for cleaning
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    _WS_RE = re.compile(r"\s+")

    # Boilerplate answers to filter out
    _BOILERPLATE_ANSWERS = {
        "n/a", "na", "none", "null", "unknown",
        "not available", "no answer", "not provided"
    }

    def __init__(self, min_q_len: int = 15, min_a_len: int = 30):
        self.min_q_len = min_q_len
        self.min_a_len = min_a_len

    def _normalize_text(self, text: Any) -> str:
        """Performs deep normalization on input text."""
        if text is None:
            return ""

        # Convert to string and unescape HTML
        text = html.unescape(str(text))

        # Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters
        text = self._CONTROL_RE.sub(" ", text)

        # Collapse multiple whitespaces/newlines
        text = self._WS_RE.sub(" ", text).strip()

        return text

    def _is_valid(self, question: str, answer: str) -> bool:
        """Validates if the Q&A pair meets quality thresholds."""
        if len(question) < self.min_q_len:
            return False
        if len(answer) < self.min_a_len:
            return False
        if answer.lower() in self._BOILERPLATE_ANSWERS:
            return False
        return True

    def load(self, file_path: str) -> Generator[QADocument, None, None]:
        """Parses MedQuAD XML and yields cleaned QADocuments."""
        seen_hashes: Set[str] = set()

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[Error] Failed to parse XML file {file_path}: {e}")
            return

        for entry in root.findall(".//Entry"):
            raw_q = entry.findtext("Question")
            raw_a = entry.findtext("Answer")
            # Context is often empty in MedQuAD XML structure, but we handle it
            raw_c = entry.findtext("Context") or ""

            # 1. Normalize
            q = self._normalize_text(raw_q)
            a = self._normalize_text(raw_a)
            c = self._normalize_text(raw_c)

            # 2. Quality Filter
            if not self._is_valid(q, a):
                continue

            # 3. Deduplication (Content-based)
            content_hash = hashlib.sha256(f"{q}|{c}|{a}".encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # 4. Metadata preparation (JSON-safe)
            metadata = {
                "source": "medquad",
                "source_file": file_path,
                "original_length_q": len(raw_q) if raw_q else 0
            }

            # 5. Create Entity
            # document_id can be the hash itself for uniqueness
            yield QADocument(
                document_id=content_hash,
                question=q,
                context=c,
                answer=a,
                metadata=metadata
            )
