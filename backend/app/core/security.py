from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str = "local"
    workspace: str = "default"
    groups: tuple[str, ...] = ()
    role: str = "member"

    @property
    def admin(self) -> bool:
        return self.role == "admin"

    def can_read(self, document) -> bool:
        return document.workspace == self.workspace and (
            self.admin
            or document.owner_id == self.user_id
            or bool(set(self.groups) & set(document.allowed_groups))
        )


bearer = HTTPBearer(auto_error=False)


def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    settings = request.app.state.container.settings
    if not settings.auth_required:
        return Principal(role="admin")
    if credentials is None:
        raise HTTPException(
            401, "Sign in with your access token.", headers={"WWW-Authenticate": "Bearer"}
        )
    digest = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    try:
        users = json.loads(settings.users_file.read_text(encoding="utf-8"))["users"]
        for user in users:
            if hmac.compare_digest(user["token_sha256"], digest) and not user.get(
                "disabled", False
            ):
                return Principal(
                    user["id"],
                    user["workspace"],
                    tuple(user.get("groups", [])),
                    user.get("role", "member"),
                )
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(503, "Access configuration unavailable.") from None
    raise HTTPException(401, "Invalid access token.", headers={"WWW-Authenticate": "Bearer"})


def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.admin:
        raise HTTPException(403, "Administrator access required.")
    return principal
