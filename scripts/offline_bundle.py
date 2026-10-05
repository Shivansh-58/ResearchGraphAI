"""Export/import runtime images and public default model weights, never company data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ["researchgraph-api:0.2.0", "researchgraph-web:0.2.0", "ollama/ollama:0.35.1"]
INSTALL_FILES = [
    "scripts/bootstrap.py",
    "scripts/install.ps1",
    "scripts/install.sh",
    "deploy/compose.yaml",
    "deploy/compose.gpu.yaml",
    "deploy/compose.setup.yaml",
    "docs/DEPLOYMENT.md",
    "docs/SECURITY.md",
    "docs/ARCHITECTURE.md",
    "docs/TRAINING.md",
    "docs/EVALUATION.md",
    "docs/INTERVIEW.md",
    "scripts/offline_bundle.py",
    "README.md",
]


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def public_model_files(root: Path) -> list[Path]:
    """Follow only the two public default manifests; exclude keys and custom models."""
    base = root / "models/ollama/models"
    manifests = [
        base / "manifests/registry.ollama.ai/library/qwen3/1.7b",
        base / "manifests/registry.ollama.ai/library/nomic-embed-text/latest",
    ]
    selected = set(manifests)
    for path in manifests:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        for layer in [metadata["config"], *metadata["layers"]]:
            name = layer["digest"]
            prefix, value = name.split(":", 1)
            if (
                prefix != "sha256"
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Unsafe model blob digest.")
            blob = base / "blobs" / f"sha256-{value}"
            if digest(blob) != value:
                raise ValueError("Model blob checksum mismatch.")
            selected.add(blob)
    return sorted(selected)


def create_bundle(output: Path) -> None:
    if output.exists():
        raise ValueError("Choose a new output file; existing bundles are never overwritten.")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="researchgraph-bundle-") as temporary:
        image_archive = Path(temporary) / "images.tar"
        subprocess.run(["docker", "image", "save", "-o", str(image_archive), *IMAGES], check=True)
        files = {name: ROOT / name for name in INSTALL_FILES}
        files.update({path.relative_to(ROOT).as_posix(): path for path in public_model_files(ROOT)})
        files["images.tar"] = image_archive
        manifest = {
            "format": 1,
            "version": "0.2.0",
            "images": IMAGES,
            "files": {
                name: {"sha256": digest(path), "size": path.stat().st_size}
                for name, path in files.items()
            },
        }
        manifest_file = Path(temporary) / "bundle-manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            with tarfile.open(output, "x") as archive:
                for name, path in files.items():
                    archive.add(path, arcname=name, recursive=False)
                archive.add(manifest_file, arcname="bundle-manifest.json")
        except Exception:
            output.unlink(missing_ok=True)
            raise
    print(f"Bundle created: {output}. Company documents, tokens, keys and adapters excluded.")


def safe_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (
            not member.isfile()
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in member.name
            or ":" in member.name
            or member.name in members
        ):
            raise ValueError("Unsafe or duplicate bundle member.")
        members[member.name] = member
    return members


def import_bundle(bundle: Path, destination: Path, *, load_images: bool = True) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Import into a new directory to preserve existing installations.")
    with tarfile.open(bundle, "r") as archive:
        members = safe_members(archive)
        info = members.get("bundle-manifest.json")
        if info is None or info.size > 1024 * 1024:
            raise ValueError("Missing or oversized bundle manifest.")
        manifest = json.load(archive.extractfile(info))
        expected = manifest["files"]
        if manifest.get("format") != 1 or set(members) != set(expected) | {"bundle-manifest.json"}:
            raise ValueError("Bundle inventory does not match its manifest.")
        # Verify the entire archive before creating files or invoking Docker.
        for name, details in expected.items():
            member = members[name]
            if member.size != details["size"]:
                raise ValueError("Bundle file size mismatch.")
            checksum = hashlib.sha256()
            with archive.extractfile(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != details["sha256"]:
                raise ValueError("Bundle checksum mismatch.")
        destination.mkdir(parents=True, exist_ok=False)
        for name, member in members.items():
            target = destination.joinpath(*PurePosixPath(name).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Bundle path escaped the installation directory.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if load_images:
        subprocess.run(
            ["docker", "image", "load", "-i", str(destination / "images.tar")], check=True
        )
    print(f"Verified bundle imported into {destination}. Run scripts/install with offline mode.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    install = commands.add_parser("import")
    install.add_argument("bundle", type=Path)
    install.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create_bundle(args.output)
    else:
        import_bundle(args.bundle, args.destination)


if __name__ == "__main__":
    main()
