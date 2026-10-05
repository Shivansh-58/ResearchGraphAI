from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


def env(name: str, default: str) -> str:
    return os.getenv(f"RESEARCHGRAPH_{name}", default)


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str = field(default_factory=lambda: env("ENV", "private"))
    data_dir: Path = field(default_factory=lambda: Path(env("DATA_DIR", "./data")).resolve())
    auth_required: bool = field(default_factory=lambda: env("AUTH_REQUIRED", "true") == "true")
    users_file: Path = field(
        default_factory=lambda: Path(env("USERS_FILE", "./secrets/users.json"))
    )
    embedding_provider: str = field(default_factory=lambda: env("EMBEDDING_PROVIDER", "ollama"))
    embedding_model: str = field(default_factory=lambda: env("EMBEDDING_MODEL", "nomic-embed-text"))
    embedding_dimension: int = 384
    llm_provider: str = field(default_factory=lambda: env("LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: env("LLM_MODEL", "qwen3:1.7b"))
    ollama_base_url: str = field(
        default_factory=lambda: env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    local_model_hosts: list[str] = field(
        default_factory=lambda: env("LOCAL_MODEL_HOSTS", "127.0.0.1,localhost,ollama").split(",")
    )
    graph_provider: str = field(default_factory=lambda: env("GRAPH_PROVIDER", "ollama"))
    model_dir: Path = field(default_factory=lambda: Path(env("MODEL_DIR", "./models/base")))
    adapter_dir: str = field(default_factory=lambda: env("ADAPTER_DIR", ""))
    frontend_dir: Path = field(default_factory=lambda: Path(env("FRONTEND_DIR", "./frontend/dist")))
    cors_origins: list[str] = field(
        default_factory=lambda: env("CORS_ORIGINS", "http://localhost:5173").split(",")
    )
    chunk_size: int = 220
    chunk_overlap: int = 40
    max_upload_bytes: int = 25 * 1024 * 1024
    model_timeout: float = 180.0
    max_context_chars: int = 10000
    min_vector_similarity: float = 0.55
    graph_max_nodes: int = 100

    @property
    def database_path(self) -> Path:
        return self.data_dir / "researchgraph.db"

    @property
    def embedding_signature(self) -> str:
        model = self.embedding_model
        if self.embedding_provider == "hashing":
            model = str(self.embedding_dimension)
        return f"{self.embedding_provider}:{model}"

    def validate(self) -> None:
        if self.llm_provider not in {"ollama", "extractive", "transformers"}:
            raise ValueError("Only local model providers are supported.")
        if self.embedding_provider not in {"ollama", "hashing", "sentence-transformers"}:
            raise ValueError("Unknown embedding provider.")
        if self.graph_provider not in {"ollama", "rules"}:
            raise ValueError("Unknown graph provider.")
        parsed = urlparse(self.ollama_base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in self.local_model_hosts
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama must use an explicitly allowed internal HTTP host.")
        if self.auth_required and not self.users_file.is_file():
            raise ValueError("Missing users file. Run scripts/bootstrap.py before startup.")
        if not self.auth_required and self.environment != "test":
            raise ValueError("Authentication may be disabled only in the test environment.")
        if self.adapter_dir:
            import json

            manifest_path = Path(self.adapter_dir).parent / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Adapter needs a training manifest and held-out evaluation.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "evaluated" or not manifest.get("approved_for_serving"):
                raise ValueError("Evaluate and explicitly approve the adapter before serving it.")
            import hashlib

            config_file = self.model_dir / "config.json"
            if not config_file.is_file() or hashlib.sha256(
                config_file.read_bytes()
            ).hexdigest() != manifest.get("base_config_sha256"):
                raise ValueError("Adapter base-model configuration does not match local weights.")
            if self.auth_required:
                users = json.loads(self.users_file.read_text(encoding="utf-8"))["users"]
                if {user["workspace"] for user in users if not user.get("disabled")} != {
                    manifest.get("workspace")
                }:
                    raise ValueError(
                        "A company adapter must be served in its own workspace installation."
                    )

    def ensure_directories(self) -> None:
        for name in ("", "documents", "staging", "indexes", "training-runs"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


settings = Settings()
