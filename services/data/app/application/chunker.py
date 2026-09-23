import hashlib
import re
from typing import Generator, List, Optional, Tuple

from app.domain.entities import DocumentChunk, QADocument


class Chunker:
    """
    Chunker for medical QA documents with PDQ boilerplate stripping
    and enlarged chunk sizes for medical context preservation.
    """

    _WS_RE = re.compile(r"\s+")
    _SENTENCE_END_RE = re.compile(
        r"""(?:[.!?]|[.!?]["'”’»)\]])(?=\s|$)"""
    )

    # بهبود الگو برای شناسایی لینک‌ها
    _PDQ_LINK_ITEM_RE = re.compile(
        r"^\s*[-•*]?\s*(See the following|Types of leukemia|"
        r"(Childhood|Adult|Chronic|Acute) [A-Za-z ,'()/-]+(Treatment|Malignancies|Leukemia Treatment)\.?)\s*$",
        re.IGNORECASE,
    )

    # گسترش لیست TOC و عناوین زائد PDQ
    _PDQ_TOC_HEADER_RE = re.compile(
        r"^\s*(Table of Contents|Contents|Key Points|General Information|Stages|"
        r"Treatment Options?|Risk Factors?|Related Resources?|Get More Information|"
        r"Changes to this summary|About this PDQ summary|Alternatives)\s*:?\s*$",
        re.IGNORECASE,
    )

    def __init__(
            self,
            size: int = 1500,  # افزایش برای حفظ پاراگراف‌ها
            overlap: int = 200,
            min_chunk_len: int = 80,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        if min_chunk_len < 0:
            raise ValueError("min_chunk_len cannot be negative")

        self.size = size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len
        self.stride = size - overlap

    @classmethod
    def _clean_text(cls, text: str) -> str:
        if not text:
            return ""
        return cls._WS_RE.sub(" ", text).strip()

    @classmethod
    def _strip_pdq_boilerplate(cls, text: str) -> str:
        """
        حذف خطوط زائد PDQ (TOC و لینک‌ها) - نیاز به کاراکترهای newline دارد.
        """
        if not text:
            return ""
        lines = text.split("\n")
        kept = []
        for line in lines:
            s = line.strip()
            if not s:
                kept.append("")
                continue
            # اگر خط با الگوهای TOC یا لینک‌ها مطابقت داشت، حذف شود
            if cls._PDQ_LINK_ITEM_RE.match(s) or cls._PDQ_TOC_HEADER_RE.match(s):
                continue
            kept.append(line)
        cleaned = "\n".join(kept)
        # فشرده‌سازی خطوط خالی اضافه
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    @classmethod
    def _generate_chunk_id(
            cls,
            document_id: str,
            chunk_index: int,
            content: str,
    ) -> str:
        raw_key = f"{document_id}:{chunk_index}:{content}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def _build_structured_text(cls, doc: QADocument) -> str:
        """
        اصلاح ترتیب عملیات: ابتدا حذف boilerplate روی متن خام، سپس استفاده از متن تمیز شده.
        """
        sections: List[str] = []

        # کوئری کوتاه است، clean کردن آن بلامانع است
        question = cls._clean_text(getattr(doc, "question", "") or "")

        # حذف بویلرپلیت ابتدا روی متن خام (با حفظ ساختار خطوط)
        raw_context = getattr(doc, "context", "") or ""
        context = cls._strip_pdq_boilerplate(raw_context)

        raw_answer = getattr(doc, "answer", "") or ""
        answer = cls._strip_pdq_boilerplate(raw_answer)

        if question:
            sections.append(f"Question: {question}")
        if context:
            sections.append(f"Context: {context}")
        if answer:
            sections.append(f"Answer: {answer}")

        return "\n\n".join(sections).strip()

    # ... سایر متدها (_normalize_for_chunking, _find_paragraph_boundary و غیره بدون تغییر باقی می‌مانند)

    @staticmethod
    def _normalize_for_chunking(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _find_paragraph_boundary(cls, text: str, start: int, preferred_end: int) -> int:
        search_region = text[start:preferred_end]
        last_break = search_region.rfind("\n\n")
        if last_break != -1:
            boundary = start + last_break + 2
            if boundary - start >= int((preferred_end - start) * 0.35):
                return boundary
        return -1

    @classmethod
    def _find_sentence_boundary(cls, text: str, start: int, preferred_end: int) -> int:
        search_region = text[start:preferred_end]
        matches = list(cls._SENTENCE_END_RE.finditer(search_region))
        if matches:
            last_match = matches[-1]
            boundary = start + last_match.end()
            if boundary - start >= int((preferred_end - start) * 0.35):
                return boundary
        return -1

    @classmethod
    def _find_word_boundary(cls, text: str, start: int, preferred_end: int) -> int:
        search_region = text[start:preferred_end]
        last_space = search_region.rfind(" ")
        if last_space != -1:
            boundary = start + last_space + 1
            if boundary - start >= int((preferred_end - start) * 0.5):
                return boundary
        return -1

    def _select_chunk_end(self, text: str, start: int, max_end: int) -> int:
        if max_end >= len(text):
            return len(text)
        boundary = self._find_paragraph_boundary(text, start, max_end)
        if boundary != -1:
            return boundary
        boundary = self._find_sentence_boundary(text, start, max_end)
        if boundary != -1:
            return boundary
        boundary = self._find_word_boundary(text, start, max_end)
        if boundary != -1:
            return boundary
        return max_end

    def _split_text(self, text: str) -> List[Tuple[str, int, int]]:
        normalized = self._normalize_for_chunking(text)
        if not normalized:
            return []

        text_len = len(normalized)
        if text_len <= self.size:
            return [(normalized, 0, text_len)]

        chunks: List[Tuple[str, int, int]] = []
        start = 0

        while start < text_len:
            max_end = min(start + self.size, text_len)
            end = self._select_chunk_end(normalized, start, max_end)
            if end <= start:
                end = min(start + self.size, text_len)

            chunk_text = normalized[start:end].strip()
            if len(chunk_text) >= self.min_chunk_len or (end == text_len and chunk_text):
                chunks.append((chunk_text, start, end))

            if end >= text_len:
                break

            next_start = max(0, end - self.overlap)
            space_pos = normalized.find(" ", next_start, end)
            if space_pos != -1:
                next_start = space_pos + 1

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def chunk_document(self, doc: QADocument) -> List[DocumentChunk]:
        structured_text = self._build_structured_text(doc)
        if not structured_text:
            return []

        doc_id = getattr(doc, "document_id", None) or getattr(doc, "id", None)
        if not doc_id:
            doc_id = hashlib.sha256(structured_text[:100].encode("utf-8")).hexdigest()[:16]

        source_dataset = getattr(doc, "source_dataset", None) or getattr(doc, "source", "unknown")
        text_pieces = self._split_text(structured_text)
        chunks: List[DocumentChunk] = []

        for idx, (piece, start, end) in enumerate(text_pieces):
            chunk_id = self._generate_chunk_id(doc_id, idx, piece)
            meta = dict(getattr(doc, "metadata", {}) or {})
            meta.update({
                "source": source_dataset,
                "chunk_index": idx,
                "chunk_size": self.size,
                "chunk_overlap": self.overlap,
                "char_start": start,
                "char_end": end,
            })

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    source_dataset=source_dataset,
                    chunk_index=idx,
                    content=piece,
                    metadata=meta,
                )
            )
        return chunks

    def chunk_many(
        self, documents: List[QADocument]
    ) -> Generator[DocumentChunk, None, None]:
        for doc in documents:
            for chunk in self.chunk_document(doc):
                yield chunk
