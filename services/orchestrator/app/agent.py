import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger("medical_rag_agent")
logger.setLevel(logging.INFO)

BOILERPLATE_PATTERN = re.compile(
    r"\bSee the following PDQ summaries\b"
    r"|\bfor information about other types of leukemia\b"
    r"|\bDiagnostic Tests\s*-\s*Drug Therapy\b"
    r"|\bSurgery and Rehabilitation\b",
    re.IGNORECASE,
)

NON_MEDICAL_PATTERN = re.compile(
    r"\b(capital\s+of|president\s+of|prime\s+minister|population\s+of|weather\s+in|"
    r"currency\s+of|who\s+won|score\s+of|movie|actor|actress|director|"
    r"write\s+(?:a\s+)?code|python\s+script|solve\s+math|calculate|"
    r"what\s+is\s+the\s+capital)\b",
    re.IGNORECASE,
)

PERSONAL_QUERY_PATTERN = re.compile(
    r"\b(who\s+am\s+i|what\s+is\s+my\s+name|where\s+do\s+i\s+live|my\s+name\s+is|"
    r"what\s+is\s+.*'s\s+name|who\s+is\s+.*'s\s+mother|who\s+is\s+.*'s\s+father|tell\s+me\s+about\s+my|my\s+friend)\b",
    re.IGNORECASE,
)

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data-service:8001")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8002")

STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>",
    "<|im_sep|>",
    "<|endoftext|>",
    "\nContext Information:",
    "\nClinical Question:",
]

ABSTENTION_MESSAGE = "Not enough information in the provided context."
NON_MEDICAL_MESSAGE = "I am an expert clinical AI assistant, and I do not know the answer to non-medical questions."


class MedicalAgent:
    def __init__(
        self,
        data_service_url: str = DATA_SERVICE_URL,
        llm_service_url: str = LLM_SERVICE_URL,
        timeout: float = 120.0,
    ):
        self.data_service_url = data_service_url.rstrip("/")
        self.llm_service_url = llm_service_url.rstrip("/")
        self.timeout = timeout

        self._special_tags_regex = re.compile(
            r"(<\|im_start\|>[a-z]*|<\|im_end\|>|<\|im_sep\|>|<\|endoftext\|>)",
            re.IGNORECASE,
        )
        self._clean_prefix_regex = re.compile(
            r"^(\s*\[?Source:[^\]\n]*\]?"
            r"|\s*Medical Context:[^\n]*"
            r"|\s*Context Information:[^\n]*"
            r"|\s*Clinical Question:[^\n]*"
            r"|\s*(?:What are|How to|What is)[^\n\?]*\?\s*:?[^\n]*"
            r"|\s*(?:Question|Context|Answer|User|Assistant|Input|Output):\s*"
            r"|\s*[a-zA-Z\s]+\?Answer:\s*)+",
            re.IGNORECASE,
        )

    def _clean_chunk_text(self, text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"^(Question:\s*|Context:\s*|Answer:\s*)+", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _clean_final_text(self, text: str) -> str:
        if not text:
            return ""
        text = self._special_tags_regex.sub("", text)
        text = self._clean_prefix_regex.sub("", text).strip()
        if "You are an expert clinical AI assistant" in text:
            parts = text.split("Assistant:\n")
            if len(parts) > 1:
                text = parts[-1].strip()
            else:
                text = re.sub(
                    r"You are an expert clinical AI assistant.*?Clinical Question:[^\n]*",
                    "",
                    text,
                    flags=re.DOTALL,
                ).strip()
        return text.strip()

    async def retrieve_context(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> Dict[str, Any]:
        url = f"{self.data_service_url}/api/v1/retrieval/search"
        payload = {"query": query, "top_k": top_k}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                logger.error(f"Error retrieving context from data service: {exc}")
                return {"context": "", "sources": [], "top_score": 0.0}

        raw_results = data.get("results") or data.get("documents") or []
        contexts: List[str] = []
        sources: List[str] = []
        top_score = 0.0

        for item in raw_results:
            raw_score = item.get("score")
            current_score = 0.0
            if raw_score is not None:
                try:
                    current_score = float(raw_score)
                    if current_score > top_score:
                        top_score = current_score
                except (ValueError, TypeError):
                    pass

            if raw_score is not None and current_score < score_threshold:
                continue

            text = (
                item.get("content")
                or item.get("text")
                or item.get("chunk_text")
                or ""
            )
            clean_text = self._clean_chunk_text(text)
            if not clean_text or BOILERPLATE_PATTERN.search(clean_text):
                continue

            source: Optional[str] = None
            meta_raw = item.get("metadata")

            if isinstance(meta_raw, dict):
                source = meta_raw.get("source") or meta_raw.get("source_dataset") or meta_raw.get("source_file")
            elif isinstance(meta_raw, str) and meta_raw.strip():
                try:
                    meta_dict = json.loads(meta_raw)
                    if isinstance(meta_dict, dict):
                        source = meta_dict.get("source") or meta_dict.get("source_dataset") or meta_dict.get("source_file")
                except Exception:
                    source = None
            elif item.get("source"):
                source = str(item.get("source"))

            source = (source or "").strip()
            if not source or source.lower() == "unknown":
                source = None

            if source and source not in sources:
                sources.append(source)

            contexts.append(clean_text)

            if len(contexts) >= top_k:
                break

        logger.info(f"Retrieved {len(contexts)} contexts for query. Top score: {top_score}")

        return {
            "context": "\n\n---\n\n".join(contexts),
            "sources": sources,
            "top_score": top_score,
        }

    def build_prompt(self, query: str, context: str) -> str:
        system_instructions = (
            "You are an expert clinical AI assistant.\n"
            "Your task is to answer the clinical question strictly using ONLY the medical context provided below.\n"
            "Rules:\n"
            "1. Answer based purely and exclusively on the context facts.\n"
            "2. If the question is non-medical, personal, or context is empty, reply with EXACTLY: Not enough information in the provided context.\n"
            "3. Do NOT make assumptions, extrapolations, or use general pre-trained knowledge.\n"
            "4. Never output personal opinions or non-clinical chatter."
        )
        return (
            f"<|im_start|>system\n{system_instructions}<|im_end|>\n"
            f"<|im_start|>user\nContext Information:\n{context}\n\nClinical Question: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    async def _call_llm_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        endpoints = ["/llm/generate", "/api/v1/generate"]
        payload = {
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": True,
            "stop": STOP_SEQUENCES,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_exc: Optional[Exception] = None

            for ep in endpoints:
                url = f"{self.llm_service_url}{ep}"
                try:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code == 404:
                            continue
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not line:
                                continue

                            clean_line = line.strip()
                            if clean_line.startswith("data:"):
                                clean_line = clean_line[5:].strip()

                            if not clean_line or clean_line == "[DONE]":
                                continue

                            try:
                                payload_data = json.loads(clean_line)
                                token = payload_data.get("token") or payload_data.get("text") or ""
                                if token:
                                    yield token
                            except json.JSONDecodeError:
                                yield clean_line

                    return

                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    last_exc = exc
                    continue

            logger.error(f"LLM stream failed on all endpoints. Last error: {last_exc}")
            return

    async def _call_llm_direct_fallback(self, prompt: str) -> str:
        endpoints = ["/llm/generate", "/api/v1/generate"]
        payload = {
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
            "stop": STOP_SEQUENCES,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for ep in endpoints:
                url = f"{self.llm_service_url}{ep}"
                try:
                    response = await client.post(url, json=payload)
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    data = response.json()

                    if isinstance(data, dict):
                        choices = data.get("choices", [])
                        if choices:
                            return str(choices[0].get("text") or choices[0].get("message", {}).get("content") or "")
                        return str(data.get("text") or data.get("response") or data.get("content") or "")
                    return str(data)
                except Exception as exc:
                    logger.warning(f"Fallback endpoint {url} failed: {exc}")
                    continue
        return ""

    async def run_stream(
        self, query: str, chat_history: Optional[list] = None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        cleaned_query = query.strip()

        if PERSONAL_QUERY_PATTERN.search(cleaned_query):
            yield {"retrieved_context": "", "sources": []}
            yield NON_MEDICAL_MESSAGE
            return

        if NON_MEDICAL_PATTERN.search(cleaned_query):
            yield {"retrieved_context": "", "sources": []}
            yield NON_MEDICAL_MESSAGE
            return

        retrieval = await self.retrieve_context(query=cleaned_query, top_k=5, score_threshold=0.0)
        context = retrieval.get("context", "")
        sources = retrieval.get("sources", [])

        yield {
            "retrieved_context": context,
            "sources": sources,
        }

        if not context or len(context.strip()) < 10:
            yield ABSTENTION_MESSAGE
            return

        prompt = self.build_prompt(query=cleaned_query, context=context)

        collected_tokens: List[str] = []
        async for token in self._call_llm_stream(prompt):
            collected_tokens.append(token)

        raw_output = "".join(collected_tokens)
        cleaned_output = self._clean_final_text(raw_output)

        words_count = len(cleaned_output.split())
        if words_count < 3 or len(cleaned_output) < 15:
            logger.warning(
                f"Generated answer too short or invalid ('{cleaned_output}'). Running direct fallback..."
            )
            fallback_text = await self._call_llm_direct_fallback(prompt)
            cleaned_output = self._clean_final_text(fallback_text)

        if not cleaned_output or len(cleaned_output.split()) < 3:
            cleaned_output = ABSTENTION_MESSAGE

        yield cleaned_output
