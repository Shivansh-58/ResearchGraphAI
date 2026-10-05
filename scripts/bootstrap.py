"""Generate local credentials/config without printing tokens or replacing existing users."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="admin")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--role", choices=["admin", "member"], default="admin")
    parser.add_argument("--groups", default="engineering,hr")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if any(
        not value or not all(char.isalnum() or char in "_-" for char in value)
        for value in (args.user, args.workspace)
    ):
        raise ValueError("User and workspace IDs must contain only letters, digits, '_' or '-'.")
    root = Path(__file__).resolve().parents[1]
    directory = root / "secrets"
    directory.mkdir(mode=0o700, exist_ok=True)
    users_file = directory / "users.json"
    users = (
        json.loads(users_file.read_text(encoding="utf-8")) if users_file.exists() else {"users": []}
    )
    if any(user["id"] == args.user for user in users["users"]):
        print(f"User {args.user} already exists; credentials were preserved.")
    else:
        token_path = directory / f"{args.user}.token"
        token = (
            token_path.read_text(encoding="utf-8").strip()
            if token_path.exists()
            else secrets.token_urlsafe(32)
        )
        users["users"].append(
            {
                "id": args.user,
                "workspace": args.workspace,
                "role": args.role,
                "groups": [group.strip() for group in args.groups.split(",") if group.strip()],
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }
        )
        if not token_path.exists():
            with token_path.open("x", encoding="utf-8") as stream:
                stream.write(token)
        token_path.chmod(0o600)
        users_file.write_text(json.dumps(users, indent=2), encoding="utf-8")
        users_file.chmod(0o644)
        print(
            "Credentials created. Access token is stored in "
            f"secrets/{args.user}.token (not printed)."
        )
    for name in ("models/ollama", "models/base", "training/datasets", "data/private"):
        (root / name).mkdir(parents=True, exist_ok=True)
    config = root / ".env.local"
    if not config.exists():
        config.write_text(
            "RESEARCHGRAPH_ENV=private\nRESEARCHGRAPH_DATA_DIR=./data/private\n"
            "RESEARCHGRAPH_USERS_FILE=./secrets/users.json\n"
            f"RESEARCHGRAPH_LLM_PROVIDER={'extractive' if args.demo else 'ollama'}\n"
            f"RESEARCHGRAPH_EMBEDDING_PROVIDER={'hashing' if args.demo else 'ollama'}\n"
            f"RESEARCHGRAPH_GRAPH_PROVIDER={'rules' if args.demo else 'ollama'}\n"
            "RESEARCHGRAPH_LLM_MODEL=qwen3:1.7b\nRESEARCHGRAPH_EMBEDDING_MODEL=nomic-embed-text\n"
            "RESEARCHGRAPH_OLLAMA_BASE_URL=http://127.0.0.1:11434\n",
            encoding="utf-8",
        )
    print(
        "Bootstrap complete. .env.local is for native startup; Docker has its own local defaults."
    )


if __name__ == "__main__":
    main()
