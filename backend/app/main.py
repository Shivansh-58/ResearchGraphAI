from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import build_container
from app.api.routes import documents, health, query, system
from app.core.config import settings
from app.core.logging import configure_logging
from app.infrastructure.local_models import ModelUnavailableError

configure_logging()
logger = logging.getLogger(__name__)


def create_app(app_settings=settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.container = build_container(app_settings)
        application.state.container.jobs.start()
        logger.info("ResearchGraph AI started")
        yield
        application.state.container.jobs.stop()
        logger.info("ResearchGraph AI stopped")

    application = FastAPI(
        title="ResearchGraph AI",
        description="Graph-RAG knowledge assistant with hybrid retrieval and grounded citations.",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.exception_handler(ModelUnavailableError)
    async def model_error(request: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    application.include_router(health.router)
    application.include_router(documents.router, prefix="/api/v1")
    application.include_router(query.router, prefix="/api/v1")
    application.include_router(system.router, prefix="/api/v1")
    if app_settings.frontend_dir.is_dir():
        application.mount(
            "/", StaticFiles(directory=app_settings.frontend_dir, html=True), name="web"
        )
    return application


app = create_app()
