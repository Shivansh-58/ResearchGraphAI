from pathlib import Path
from uuid import uuid4

import pytest

from app.api.dependencies import ApplicationContainer, build_container
from app.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    data_dir = Path(__file__).parent / "runtime" / uuid4().hex
    data_dir.mkdir(parents=True)
    return Settings(
        data_dir=data_dir,
        environment="test",
        auth_required=False,
        embedding_provider="hashing",
        llm_provider="extractive",
        graph_provider="rules",
        chunk_size=40,
        chunk_overlap=8,
        embedding_dimension=96,
    )


@pytest.fixture
def container(test_settings: Settings) -> ApplicationContainer:
    return build_container(test_settings)
