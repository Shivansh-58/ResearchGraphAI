from fastapi import APIRouter, Depends

from app.api.dependencies import ApplicationContainer, get_container

router = APIRouter(tags=["operations"])


@router.get("/health")
async def health(container: ApplicationContainer = Depends(get_container)) -> dict:
    return {"status": "ok", "version": "0.2.0"}
