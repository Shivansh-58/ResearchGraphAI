from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import ApplicationContainer, get_container
from app.core.security import Principal, get_principal, require_admin
from app.infrastructure.local_models import ModelUnavailableError

router = APIRouter(tags=["system"])


@router.get("/session")
def session(principal: Principal = Depends(get_principal)):
    return {
        "user_id": principal.user_id,
        "workspace": principal.workspace,
        "groups": principal.groups,
        "role": principal.role,
    }


@router.get("/system")
def status(
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    return {
        "mode": "private",
        "llm_provider": container.settings.llm_provider,
        "llm_model": container.settings.llm_model,
        "embedding_provider": container.settings.embedding_provider,
        "graph_provider": container.settings.graph_provider,
        "counts": container.repository.stats(principal),
    }


@router.get("/ready")
async def ready(
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    settings = container.settings
    if (
        settings.llm_provider == "transformers"
        and not (settings.model_dir / "config.json").is_file()
    ):
        raise HTTPException(503, "Local transformer weights are missing.")
    needed = set()
    if settings.llm_provider == "ollama" or settings.graph_provider == "ollama":
        needed.add(settings.llm_model)
    if settings.embedding_provider == "ollama":
        needed.add(settings.embedding_model)
    try:
        installed = (
            await asyncio.to_thread(container.model_client.installed_models) if needed else set()
        )
    except ModelUnavailableError:
        raise HTTPException(503, "Local model service is not ready.") from None
    missing = sorted(
        name for name in needed if name not in installed and f"{name}:latest" not in installed
    )
    if missing:
        raise HTTPException(503, {"missing_models": missing})
    return {"status": "ready", "models": sorted(needed)}


@router.get("/graph")
def graph(
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    return container.repository.graph_view(principal)


@router.get("/training-runs")
def training_runs(
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(require_admin),
):
    runs = []
    for path in sorted((container.settings.data_dir / "training-runs").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("workspace") == principal.workspace:
                runs.append(
                    {
                        key: manifest.get(key)
                        for key in ("run_id", "base_model", "status", "created_at", "metrics")
                    }
                )
        except (OSError, ValueError):
            continue
    return {"runs": runs}
