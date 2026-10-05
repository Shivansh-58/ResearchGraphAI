from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from pypdf import PdfReader

from app.domain.models import Chunk


class UnsupportedDocumentError(ValueError):
    pass


class DocumentParser:
    supported_extensions = {".pdf", ".txt", ".md"}

    def parse(self, filename: str, content: bytes) -> str:
        return "\n\n".join(text for _, text in self.parse_pages(filename, content))

    def parse_pages(self, filename: str, content: bytes) -> list[tuple[int | None, str]]:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.supported_extensions:
            raise UnsupportedDocumentError(
                f"Unsupported file type '{suffix}'. Use PDF, TXT, or Markdown."
            )
        if suffix == ".pdf":
            try:
                reader = PdfReader(io.BytesIO(content))
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs must be unlocked before upload.")
                pages = [
                    (index, page.extract_text() or "") for index, page in enumerate(reader.pages, 1)
                ]
            except Exception as exc:
                raise ValueError(
                    "PDF could not be read. Unlock it or export a text-based PDF."
                ) from exc
        else:
            try:
                pages = [(None, content.decode("utf-8-sig"))]
            except UnicodeDecodeError as exc:
                raise ValueError("Text files must use UTF-8 encoding.") from exc
        result = []
        for page, text in pages:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                result.append((page, text))
        if not result:
            raise ValueError("No extractable text. Scanned PDFs need OCR before upload.")
        return result


class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 220, overlap: int = 40) -> None:
        if overlap >= chunk_size or overlap < 0 or chunk_size <= 0:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document_id: str, text: str) -> list[Chunk]:
        tokens = list(re.finditer(r"\S+", text))
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for ordinal, token_start in enumerate(range(0, len(tokens), step)):
            window = tokens[token_start : token_start + self.chunk_size]
            if not window:
                break
            start_char, end_char = window[0].start(), window[-1].end()
            chunk_text = text[start_char:end_char]
            chunk_id = hashlib.sha256(f"{document_id}:{ordinal}:{chunk_text}".encode()).hexdigest()[
                :24
            ]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    ordinal=ordinal,
                    text=chunk_text,
                    token_count=len(window),
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            if token_start + self.chunk_size >= len(tokens):
                break
        return chunks
