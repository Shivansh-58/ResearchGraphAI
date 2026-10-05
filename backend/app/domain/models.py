from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Document:
    id: str
    filename: str
    media_type: str
    byte_size: int
    content_path: str
    created_at: datetime
    status: str = "ready"
    chunk_count: int = 0
    content_hash: str = ""
    workspace: str = "default"
    owner_id: str = "local"
    allowed_groups: list[str] = field(default_factory=list)
    version: int = 1
    is_current: bool = True
    supersedes_id: str | None = None


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    embedding: list[float] = field(default_factory=list)
    filename: str = ""
    page: int | None = None
    section: str = ""
    version: int = 1


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    name: str
    normalized_name: str
    kind: str = "concept"


@dataclass(frozen=True, slots=True)
class Relation:
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    evidence_chunk_id: str
    evidence_text: str = ""


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    matched_entities: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    chunk_id: str
    document_id: str
    filename: str
    excerpt: str
    score: float
    page: int | None = None
    section: str = ""
    version: int = 1


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation]
    confidence: float
    trace: dict[str, object] = field(default_factory=dict)
