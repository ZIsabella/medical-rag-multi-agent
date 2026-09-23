import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Generator, Set, Union

from app.domain.entities import QADocument


class PubMedQALoader:
    """
    Loader for PubMedQA dataset with robust normalization and filtering.
    """

    _CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    _WS_RE = re.compile(r"\s+")
    _BOILERPLATE_ANSWERS = {
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "not available",
        "no answer",
        "not provided",
    }

    def __init__(self, min_q_len: int = 15, min_a_len: int = 30):
        self.min_q_len = min_q_len
        self.min_a_len = min_a_len

    def _normalize_text(self, text: Any) -> str:
        if text is None:
            return ""

        text = html.unescape(str(text))
        text = unicodedata.normalize("NFKC", text)
        text = self._CONTROL_RE.sub(" ", text)
        text = self._WS_RE.sub(" ", text).strip()
        return text

    def _is_valid(self, question: str, answer: str) -> bool:
        if len(question) < self.min_q_len:
            return False

        if len(answer) < self.min_a_len:
            return False

        if answer.lower() in self._BOILERPLATE_ANSWERS:
            return False

        return True

    def load(
            self,
            source: Union[str, Path, Dict[str, Any], list],
    ) -> Generator[QADocument, None, None]:

        """
        Load PubMedQA records from a JSON file path, dict, or list.

        Standard PubMedQA structure:
        {
            "<PMID>": {
                "QUESTION": "...",
                "CONTEXTS": ["...", "..."],
                "LONG_ANSWER": "..."
            }
        }
        """

        # ingest.py passes the JSON file path.
        if isinstance(source, (str, Path)):
            file_path = Path(source)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"PubMedQA file not found: {file_path}"
                )

            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            data = source

        # Standard PubMedQA JSON root is a dict indexed by PubMed ID.
        if isinstance(data, dict):
            records = data.values()
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError(
                "Invalid PubMedQA payload. Expected a JSON object or list; "
                f"received {type(data).__name__}."
            )

        seen_hashes: Set[str] = set()

        for item in records:
            # Defensive validation: a malformed record is skipped safely.
            if not isinstance(item, dict):
                continue

            raw_q = item.get("QUESTION", "")
            raw_c = item.get("CONTEXTS", "")
            raw_a = (
                item.get("LONG_ANSWER", "")
                or item.get("final_decision", "")
            )

            # CONTEXTS in PubMedQA normally is list[str].
            if isinstance(raw_c, list):
                raw_c = " ".join(
                    str(context_part)
                    for context_part in raw_c
                )

            q = self._normalize_text(raw_q)
            c = self._normalize_text(raw_c)
            a = self._normalize_text(raw_a)

            if not self._is_valid(q, a):
                continue

            content_hash = hashlib.sha256(
                f"{q}|{c}|{a}".encode("utf-8")
            ).hexdigest()

            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)

            yield QADocument(
                document_id=content_hash,
                question=q,
                context=c,
                answer=a,
                metadata={
                    "source": "PubMedQA",
                },
            )
