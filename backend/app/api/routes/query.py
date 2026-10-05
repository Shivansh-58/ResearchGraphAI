from fastapi import APIRouter, Depends

from app.api.dependencies import ApplicationContainer, get_container
from app.api.schemas import CitationResponse, QueryRequest, QueryResponse
from app.core.security import Principal, get_principal

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(
    payload: QueryRequest,
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> QueryResponse:
    answer = await container.orchestrator.answer(
        payload.question.strip(),
        payload.top_k,
        payload.graph_hops,
        payload.include_debug,
        principal,
    )
    return QueryResponse(
        answer=answer.text,
        confidence=answer.confidence,
        retrieval_score=answer.confidence,
        answer_mode=container.settings.llm_provider,
        citations=[
            CitationResponse.model_validate(item, from_attributes=True) for item in answer.citations
        ],
        trace=answer.trace if payload.include_debug else None,
    )
