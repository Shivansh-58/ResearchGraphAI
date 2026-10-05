from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.infrastructure.answer_generators import ExtractiveAnswerGenerator
from app.infrastructure.document_processing import DocumentParser, SlidingWindowChunker
from app.infrastructure.embeddings import (
    HashingEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
)
from app.infrastructure.graph_extraction import RuleBasedGraphExtractor
from app.infrastructure.local_answers import OllamaAnswerGenerator, TransformersAnswerGenerator
from app.infrastructure.local_models import OllamaClient
from app.infrastructure.repository import SQLiteKnowledgeRepository
from app.infrastructure.search_indexes import BM25Index, VectorIndex
from app.infrastructure.structured_graph import StructuredGraphExtractor
from app.services.agents.graph_agent import GraphExpansionAgent
from app.services.agents.retrieval_agent import RetrievalAgent
from app.services.agents.synthesis_agent import SynthesisAgent
from app.services.ingestion import IngestionService
from app.services.jobs import IngestionJobs
from app.services.orchestrator import ResearchGraphOrchestrator
from app.services.retrieval import HybridRetriever


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    repository: SQLiteKnowledgeRepository
    vector_index: VectorIndex
    ingestion: IngestionService
    orchestrator: ResearchGraphOrchestrator
    jobs: IngestionJobs
    model_client: OllamaClient


def build_container(settings: Settings) -> ApplicationContainer:
    settings.validate()
    settings.ensure_directories()
    repository = SQLiteKnowledgeRepository(settings.database_path)
    repository.ensure_embedding_signature(settings.embedding_signature)
    client = OllamaClient(settings.ollama_base_url, settings.model_timeout)
    if settings.embedding_provider == "sentence-transformers":
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    elif settings.embedding_provider == "ollama":
        embedder = OllamaEmbedder(client, settings.embedding_model)
    else:
        embedder = HashingEmbedder(settings.embedding_dimension)
    vector_index, keyword_index = VectorIndex(), BM25Index()
    retriever = HybridRetriever(
        repository,
        embedder,
        vector_index,
        keyword_index,
        min_similarity=settings.min_vector_similarity,
    )
    retriever.refresh()

    if settings.llm_provider == "ollama":
        generator = OllamaAnswerGenerator(client, settings.llm_model, settings.max_context_chars)
    elif settings.llm_provider == "transformers":
        generator = TransformersAnswerGenerator(
            settings.model_dir, settings.adapter_dir, settings.max_context_chars
        )
    else:
        generator = ExtractiveAnswerGenerator()
    orchestrator = ResearchGraphOrchestrator(
        RetrievalAgent(retriever),
        GraphExpansionAgent(repository),
        SynthesisAgent(generator),
    )
    ingestion = IngestionService(
        repository=repository,
        parser=DocumentParser(),
        chunker=SlidingWindowChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        graph_extractor=StructuredGraphExtractor(client, settings.llm_model)
        if settings.graph_provider == "ollama"
        else RuleBasedGraphExtractor(),
        document_directory=settings.data_dir / "documents",
        refresh_indexes=retriever.refresh,
    )
    jobs = IngestionJobs(repository, ingestion, settings.data_dir / "staging")
    return ApplicationContainer(
        settings, repository, vector_index, ingestion, orchestrator, jobs, client
    )


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container
