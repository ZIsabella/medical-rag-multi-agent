from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class RerankerServiceError(Exception):
    """Raised when the Cross-Encoder reranker cannot be loaded or run."""


class RerankerService:
    """
    Offline-safe Cross-Encoder reranking service.

    The model must be available locally inside the container. By default,
    the expected mounted directory is /models/bge-reranker-base.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.enabled = enabled

        self.model_path = (
            model_path
            or os.getenv(
                "RERANKER_MODEL_PATH",
                "/models/bge-reranker-base",
            )
        )

        self.device = (
            device
            or os.getenv("RERANKER_DEVICE", "cpu")
        )

        self._model: Any | None = None
        self.is_loaded = False

    def _load_model_sync(self) -> None:
        """Load Cross-Encoder only from the mounted local model directory."""
        if not self.enabled:
            return

        if self._model is not None:
            return

        if not os.path.isdir(self.model_path):
            raise RerankerServiceError(
                "Reranker model directory does not exist: "
                f"{self.model_path}"
            )

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading reranker from local path '%s' on device '%s'.",
                self.model_path,
                self.device,
            )

            self._model = CrossEncoder(
                self.model_path,
                device=self.device,
                local_files_only=True,
            )

            self.is_loaded = True

            logger.info(
                "Reranker model loaded successfully from '%s'.",
                self.model_path,
            )

        except Exception as exc:
            self._model = None
            self.is_loaded = False

            logger.exception(
                "Could not load local reranker model '%s': %s",
                self.model_path,
                exc,
            )

            raise RerankerServiceError(
                "Could not load local reranker model "
                f"'{self.model_path}': {exc}"
            ) from exc

    async def warmup(self) -> None:
        """Load the Cross-Encoder during startup when enabled."""
        if not self.enabled:
            return

        await asyncio.to_thread(self._load_model_sync)

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Score candidate chunks using Cross-Encoder and return reranked items.

        Original hybrid score is retained in `retrieval_score`.
        Final API `score` becomes the Cross-Encoder score.
        """
        if not self.enabled or not candidates:
            return candidates[:top_k]

        if not self.is_loaded:
            await asyncio.to_thread(self._load_model_sync)

        def predict_sync() -> list[float]:
            pairs = [
                (
                    query,
                    str(candidate.get("content", "")),
                )
                for candidate in candidates
            ]

            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
            )

            return [float(score) for score in scores]

        try:
            scores = await asyncio.to_thread(predict_sync)

            scored_candidates: list[dict[str, Any]] = []

            for candidate, reranker_score in zip(
                candidates,
                scores,
                strict=True,
            ):
                updated_candidate = dict(candidate)

                # Score before Cross-Encoder reranking is preserved.
                updated_candidate["retrieval_score"] = (
                    updated_candidate.get(
                        "hybrid_score",
                        updated_candidate.get(
                            "rrf_score",
                            updated_candidate.get("score"),
                        ),
                    )
                )

                updated_candidate["reranker_score"] = reranker_score
                updated_candidate["score"] = reranker_score

                scored_candidates.append(updated_candidate)

            reranked_candidates = sorted(
                scored_candidates,
                key=lambda item: item["reranker_score"],
                reverse=True,
            )

            for rank, candidate in enumerate(
                reranked_candidates,
                start=1,
            ):
                candidate["reranker_rank"] = rank

            return reranked_candidates[:top_k]

        except Exception as exc:
            logger.exception("Reranker inference failed: %s", exc)

            raise RerankerServiceError(
                f"Reranker inference failed: {exc}"
            ) from exc
