from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    byte_size: int
    created_at: datetime
    status: str
    chunk_count: int
    content_hash: str
    owner_id: str
    allowed_groups: list[str]
    version: int
    is_current: bool


class IngestionResponse(BaseModel):
    document: DocumentResponse
    created: bool
    message: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    graph_hops: int = Field(default=1, ge=0, le=3)
    include_debug: bool = False

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class CitationResponse(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    filename: str
    excerpt: str
    score: float
    page: int | None = None
    section: str = ""
    version: int = 1


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    retrieval_score: float
    answer_mode: str
    citation_validation: str = (
        "Declared sources checked; single-source formatting normalized; "
        "claim entailment requires evaluation."
    )
    citations: list[CitationResponse]
    trace: dict[str, object] | None = None
