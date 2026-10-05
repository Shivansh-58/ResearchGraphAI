from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from app.infrastructure.local_models import ModelUnavailableError, OllamaClient


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client, self.model = client, model
        self.dimension = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 16):
            result = self.client.request(
                "/api/embed",
                {
                    "model": self.model,
                    "input": list(texts[start : start + 16]),
                    "truncate": False,
                    "keep_alive": "0",
                },
            )
            try:
                batch = np.asarray(result.get("embeddings", []), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ModelUnavailableError(
                    "Local embedding model returned invalid vectors."
                ) from exc
            if (
                batch.ndim != 2
                or len(batch) != len(texts[start : start + 16])
                or batch.shape[1] == 0
                or not np.isfinite(batch).all()
            ):
                raise ModelUnavailableError("Local embedding model returned invalid vectors.")
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if np.any(norms == 0) or self.dimension not in {0, batch.shape[1]}:
                raise ModelUnavailableError("Embedding dimensions changed or vectors were empty.")
            self.dimension = batch.shape[1]
            vectors.extend((batch / norms).tolist())
        return vectors


class HashingEmbedder:
    """Fast deterministic local embedding with no model download required."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = re.findall(r"[a-z0-9]+", text.lower())
            for term in terms:
                digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimension
                sign = 1.0 if value & 1 else -1.0
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix.tolist()


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'semantic' extra to use sentence-transformer embeddings."
            ) from exc
        from pathlib import Path

        if not Path(model_name).is_dir():
            raise ValueError("Sentence-transformer weights must be an existing local directory.")
        self._model = SentenceTransformer(model_name, local_files_only=True)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.astype("float32").tolist()
