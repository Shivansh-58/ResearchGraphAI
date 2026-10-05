from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Sequence

import numpy as np

from app.domain.models import Chunk

STOP_WORDS = frozenset(
    "a an and are as at be been by can could did do does for from had has have how "
    "i if in into is it its me of on or our she should that the their them there these "
    "they this those to was we were what when where which who why will with would you your".split()
)


def _tokens(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in STOP_WORDS]


class VectorIndex:
    """FAISS inner-product index with a NumPy fallback for constrained environments."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._faiss_index = None
        self.backend = "numpy"
        self._lock = threading.RLock()

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            if not chunks:
                self._matrix = np.empty((0, 0), dtype=np.float32)
                self._faiss_index = None
                return
            self._matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
            try:
                import faiss

                self._faiss_index = faiss.IndexFlatIP(self._matrix.shape[1])
                self._faiss_index.add(self._matrix)
                self.backend = "faiss"
            except ImportError:
                self._faiss_index = None
                self.backend = "numpy"

    def search(self, query_vector: Sequence[float], limit: int) -> list[tuple[str, float]]:
        with self._lock:
            if not self._chunks:
                return []
            top_n = min(limit, len(self._chunks))
            query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            if self._faiss_index is not None:
                scores, positions = self._faiss_index.search(query, top_n)
                return [
                    (self._chunks[position].id, float(score))
                    for score, position in zip(scores[0], positions[0], strict=True)
                    if position >= 0
                ]
            scores = self._matrix @ query[0]
            positions = np.argsort(scores)[::-1][:top_n]
            return [(self._chunks[position].id, float(scores[position])) for position in positions]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._average_length = 0.0
        self._lock = threading.RLock()

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            tokenized = [_tokens(chunk.text) for chunk in chunks]
            self._term_frequencies = [Counter(tokens) for tokens in tokenized]
            self._document_frequencies = Counter(
                term for tokens in tokenized for term in set(tokens)
            )
            self._average_length = (
                sum(len(tokens) for tokens in tokenized) / len(tokenized) if tokenized else 0.0
            )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        with self._lock:
            if not self._chunks:
                return []
            query_terms = _tokens(query)
            corpus_size = len(self._chunks)
            scored: list[tuple[str, float]] = []
            for chunk, frequencies in zip(self._chunks, self._term_frequencies, strict=True):
                length = sum(frequencies.values())
                score = 0.0
                for term in query_terms:
                    frequency = frequencies.get(term, 0)
                    if not frequency:
                        continue
                    document_frequency = self._document_frequencies[term]
                    inverse_frequency = math.log(
                        1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
                    )
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / max(self._average_length, 1.0)
                    )
                    score += inverse_frequency * frequency * (self.k1 + 1) / denominator
                if score > 0:
                    scored.append((chunk.id, score))
            return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
