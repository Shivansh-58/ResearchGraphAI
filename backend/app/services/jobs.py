from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.core.security import Principal
from app.infrastructure.local_models import ModelUnavailableError


class IngestionJobs:
    """Durable SQLite queue; a single worker processes uploads outside the event loop."""

    def __init__(self, repository, ingestion, staging_dir: Path) -> None:
        self.repository, self.ingestion, self.staging_dir = repository, ingestion, staging_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self.repository._connection() as connection:
            connection.execute(
                "UPDATE jobs SET status='queued', stage='queued' WHERE status='running'"
            )
        self._thread = threading.Thread(target=self._worker, name="ingestion", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def submit(
        self, filename: str, content: bytes, principal: Principal, groups: list[str]
    ) -> dict:
        if not principal.admin and not set(groups).issubset(principal.groups):
            raise ValueError("You can share only with your own groups.")
        job_id = uuid4().hex
        path = self.staging_dir / job_id
        path.write_bytes(content)
        payload = json.dumps(
            {
                "user_id": principal.user_id,
                "workspace": principal.workspace,
                "groups": list(principal.groups),
                "role": principal.role,
                "allowed_groups": groups,
            }
        )
        try:
            with self.repository._connection() as connection:
                connection.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, NULL, NULL)",
                    (
                        job_id,
                        principal.user_id,
                        principal.workspace,
                        filename,
                        str(path),
                        payload,
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return self.get(job_id, principal)

    def get(self, job_id: str, principal: Principal) -> dict | None:
        with self.repository._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=? AND workspace=?", (job_id, principal.workspace)
            ).fetchone()
        if row is None or (not principal.admin and row["owner_id"] != principal.user_id):
            return None
        return {
            key: row[key]
            for key in ("id", "filename", "status", "stage", "created_at", "document_id", "error")
        }

    def _worker(self) -> None:
        while not self._stop.is_set():
            if not self.process_one():
                self._stop.wait(0.5)

    def process_one(self) -> bool:
        with self.repository._lock, self.repository._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE jobs SET status='running', stage='parsing' WHERE id=?", (row["id"],)
            )
        payload = json.loads(row["payload"])
        principal = Principal(
            payload["user_id"], payload["workspace"], tuple(payload["groups"]), payload["role"]
        )
        path = Path(row["staged_path"])

        def progress(stage: str) -> None:
            with self.repository._connection() as connection:
                connection.execute("UPDATE jobs SET stage=? WHERE id=?", (stage, row["id"]))

        try:
            document, _ = self.ingestion.ingest(
                row["filename"],
                path.read_bytes(),
                principal=principal,
                allowed_groups=payload["allowed_groups"],
                progress=progress,
            )
            with self.repository._connection() as connection:
                connection.execute(
                    "UPDATE jobs SET status='completed', stage='ready', document_id=? WHERE id=?",
                    (document.id, row["id"]),
                )
        except Exception as exc:
            if isinstance(exc, (ValueError, ModelUnavailableError)):
                error = str(exc)[:250]
            else:
                error = (
                    "Ingestion failed. The original upload was removed; contact your administrator."
                )
            with self.repository._connection() as connection:
                connection.execute(
                    "UPDATE jobs SET status='failed', stage='failed', error=? WHERE id=?",
                    (error, row["id"]),
                )
        finally:
            path.unlink(missing_ok=True)
        return True
