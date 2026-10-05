from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}.") from exc
        for key in ("question", "context", "answer"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Missing {key} at line {line_number}.")
        refs = set(re.findall(r"\[(\d+)\]", item["answer"]))
        available = set(re.findall(r"\[(\d+)\]", item["context"]))
        if not refs or not refs.issubset(available):
            raise ValueError(f"Invalid evidence citation at line {line_number}.")
        examples.append(item)
    if not examples:
        raise ValueError("Dataset is empty.")
    return examples


def validate_splits(train: list[dict], validation: list[dict]) -> None:
    def key(item):
        return re.sub(r"\s+", " ", item["question"].casefold()).strip()

    if len({key(item) for item in train}) != len(train):
        raise ValueError("Duplicate training questions.")
    if {key(item) for item in train} & {key(item) for item in validation}:
        raise ValueError("Training and validation questions overlap.")


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_hash(example: dict) -> str:
    normalized = re.sub(r"\s+", " ", example["question"].casefold()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def score_answer(answer: str, example: dict) -> dict[str, float]:
    refs = set(re.findall(r"\[(\d+)\]", answer))
    allowed = set(re.findall(r"\[(\d+)\]", example["context"]))
    facts = example.get("required_facts", [])
    return {
        "valid_citations": float(bool(refs) and refs.issubset(allowed)),
        "required_fact_recall": sum(fact.casefold() in answer.casefold() for fact in facts)
        / len(facts)
        if facts
        else 0,
    }
