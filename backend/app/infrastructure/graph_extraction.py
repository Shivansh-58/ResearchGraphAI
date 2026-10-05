from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence

from app.domain.models import Chunk, Entity, Relation


class RuleBasedGraphExtractor:
    """Entity/co-occurrence baseline that can later be replaced by an LLM extractor."""

    _entity_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9+#.-]*)(?:\s+(?:[A-Z][A-Za-z0-9+#.-]*|AI|ML)){0,3}\b"
    )
    _excluded = {
        "the",
        "this",
        "that",
        "these",
        "those",
        "a",
        "an",
        "and",
        "but",
        "for",
    }

    @staticmethod
    def _entity_id(normalized_name: str) -> str:
        return hashlib.sha256(normalized_name.encode()).hexdigest()[:20]

    def extract(
        self, chunks: Sequence[Chunk]
    ) -> tuple[list[Entity], list[tuple[str, str]], list[Relation]]:
        entities: dict[str, Entity] = {}
        chunk_entities: list[tuple[str, str]] = []
        relation_weights: dict[tuple[str, str, str], float] = defaultdict(float)
        for chunk in chunks:
            names: dict[str, str] = {}
            for match in self._entity_pattern.findall(chunk.text):
                name = match.strip(" .,-")
                normalized = re.sub(r"\s+", " ", name.lower())
                if len(name) < 3 or normalized in self._excluded:
                    continue
                names[normalized] = name
            chunk_ids: list[str] = []
            for normalized, name in names.items():
                entity_id = self._entity_id(normalized)
                entities.setdefault(entity_id, Entity(entity_id, name, normalized))
                chunk_entities.append((chunk.id, entity_id))
                chunk_ids.append(entity_id)
            for index, source_id in enumerate(chunk_ids):
                for target_id in chunk_ids[index + 1 :]:
                    source_key, target_key = sorted((source_id, target_id))
                    relation_weights[(source_key, target_key, chunk.id)] += 1.0
        relations = [
            Relation(source, target, "CO_OCCURS_WITH", weight, chunk_id)
            for (source, target, chunk_id), weight in relation_weights.items()
        ]
        return list(entities.values()), chunk_entities, relations
